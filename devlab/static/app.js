// app.js -- DPO Inspector orchestration: form -> run -> SSE -> state -> render.
// Live streaming and replaying a saved run share the exact same applyEvent()
// state-mutation path; the only difference is whether render() is called
// after each event (live) or once after the whole trace is loaded (replay).
import { DualMeshViewer } from "/static/viewer.js";

const { lineChart, histogram, groupedBars, trajectoryRail, COLOR, fmt } = window.DPOCharts;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
// One generation can now fork/steer/resume more than once across the same
// schedule (trellis_core.dpo_branch.DPOBranchConfig.num_branches) -- each
// fork gets its own entry in state.branches, in schedule order. Everything
// that used to be a single top-level field (iBranch, candidates, gradSteps,
// ...) now lives per-branch; state.currentBranchView picks which one the
// Candidate comparison / Score breakdown / Gradient steps / Geometry
// distributions panels are showing (see the Branches table, which both
// summarizes every fork and acts as the selector).
function freshBranch() {
  return {
    iBranch: null,
    k: null,
    branchT: null,
    resumeT: null,
    nVoxels: null,
    outOfWindow: false,
    trustRegion: null,
    branchNoiseScale: null,
    judgeWeights: null,
    preBranchSteps: 0,
    postBranchSteps: 0,
    candidates: { reference: null, delta_initial: null, delta_final: null },
    gradSteps: [],
    resumedFrom: null,
  };
}

function freshState() {
  return {
    runId: null,
    config: {},
    tPairs: [],
    stepsDoneAbsolute: 0, // cumulative no-grad sampler_step count, in absolute schedule position -- see applyEvent's sampler_step case
    nBranchesExpected: null,
    branches: [],
    currentBranchView: 0,
    phase: "idle",
    printableResult: null,
    rawResult: null,
    vanillaResult: null,
    stageName: null,
    notes: [],
    error: null,
    finished: false,
    ok: null,
  };
}
let state = freshState();
let rightShowing = "delta_initial"; // which candidate the right viewer pane shows
let viewer = null;
let currentSource = null; // active EventSource, if any
let loadedMeshFile = { left: null, right: null };

function ensureBranch(idx) {
  idx = idx ?? 0; // traces recorded before branch_index existed default to branch 0
  while (state.branches.length <= idx) state.branches.push(freshBranch());
  return state.branches[idx];
}

function currentBranch() {
  return state.branches[state.currentBranchView] || freshBranch();
}

// ---------------------------------------------------------------------------
// Event application (pure state mutation)
// ---------------------------------------------------------------------------
function applyEvent(evt) {
  const p = evt.payload || {};
  switch (evt.type) {
    case "session_start":
      state.config = p;
      break;
    case "stage":
      state.stageName = p.name;
      if (p.name === "preprocess_image" || p.name === "sample_sparse_structure") state.phase = "presample";
      if (p.name === "print_prep") state.phase = "post_process";
      break;
    case "run_start": // dpo_branch.py's internal schedule announcement
      state.tPairs = p.t_pairs || [];
      state.phase = "pre_branch";
      break;
    case "sampler_step": {
      const idx = p.branch_index ?? 0;
      const br = ensureBranch(idx);
      if (p.phase === "pre_branch") {
        br.preBranchSteps = p.index + 1;
        state.phase = "pre_branch";
        const prev = idx > 0 ? state.branches[idx - 1] : null;
        const prevEnd = prev && prev.iBranch !== null ? prev.iBranch + prev.k : 0;
        state.stepsDoneAbsolute = prevEnd + p.index + 1;
      } else if (p.phase === "post_branch") {
        br.postBranchSteps = p.index + 1;
        state.phase = "post_branch";
        if (br.iBranch !== null) state.stepsDoneAbsolute = br.iBranch + br.k + p.index + 1;
      }
      break;
    }
    case "branch_point": {
      const idx = p.branch_index ?? 0;
      const br = ensureBranch(idx);
      br.iBranch = p.i_branch;
      br.k = p.k;
      br.branchT = p.branch_t;
      br.resumeT = p.resume_t;
      br.nVoxels = p.n_voxels;
      br.outOfWindow = !!p.out_of_window;
      br.trustRegion = p.trust_region;
      br.branchNoiseScale = p.branch_noise_scale;
      br.judgeWeights = p.judge_weights || null;
      if (p.n_branches) state.nBranchesExpected = p.n_branches;
      state.currentBranchView = idx; // follow the branch currently being processed
      state.phase = "steering";
      break;
    }
    case "candidate_scored": {
      const br = ensureBranch(p.branch_index);
      if (p.which) br.candidates[p.which] = p;
      break;
    }
    case "grad_step":
      ensureBranch(p.branch_index).gradSteps.push(p);
      break;
    case "resume":
      ensureBranch(p.branch_index).resumedFrom = p.resumed_from;
      break;
    case "raw_result":
      state.rawResult = p;
      break;
    case "printable_result":
      state.printableResult = p;
      state.phase = "done";
      break;
    case "vanilla_result":
      state.vanillaResult = p;
      break;
    case "session_end":
      state.finished = true;
      state.ok = true;
      if (state.phase !== "done") state.phase = "done";
      break;
    case "error":
      state.finished = true;
      state.ok = false;
      state.error = p;
      state.phase = "error";
      break;
    // run_end (dpo_branch.py's own module-level event) deliberately not
    // treated as terminal here either -- print-prep still runs after it.
    default:
      break;
  }
  if (Array.isArray(p.notes) && p.notes.length) {
    p.notes.forEach((n) => { if (!state.notes.includes(n)) state.notes.push(n); });
  }
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------
function meshUrl(which) {
  const c = currentBranch().candidates[which];
  if (!c || !c.mesh || !c.mesh.file) return null;
  return `/api/mesh/${state.runId}/${c.mesh.file}`;
}

// Download helpers -----------------------------------------------------------
// Candidate meshes (reference/delta_initial/delta_final) are decoded straight
// off the sampler at/around the branch point -- before print-prep ever runs
// (see trellis_core/dpo_branch.py) -- so both formats here are already "no
// postprocessing" with no extra flag needed. glbFile/stlFile come straight
// from TraceWriter._export_mesh's summary dict (file / stl_file).
function candidateDownloads(which) {
  const c = currentBranch().candidates[which];
  const mesh = c && c.mesh;
  if (!mesh) return { glb: null, stl: null };
  return {
    glb: mesh.file ? `/api/mesh/${state.runId}/${mesh.file}` : null,
    stl: mesh.stl_file ? `/api/mesh/${state.runId}/${mesh.stl_file}` : null,
  };
}

// glb_path/stl_path/obj_path in printable_result/vanilla_result are the
// filesystem paths runner.py wrote to (<run_dir>/output/<name>.<ext>) --
// only the basename matters, /api/output/{run_id}/{filename} resolves it
// under this run's own output/ dir.
function outputUrl(path) {
  if (!path) return null;
  return `/api/output/${state.runId}/${path.split("/").pop()}`;
}

function scoreLine(which) {
  const c = currentBranch().candidates[which];
  if (!c) return "…";
  if (!c.score) return "failed to decode";
  const s = c.score;
  return `S=${fmt(s.total)}  detail=${fmt(s.detail_reward)}  oh=${fmt(s.overhang_penalty)}  th=${fmt(s.thickness_penalty)}${s.thickness_valid ? "" : "*"}  topo=${fmt(s.topology_penalty)}`;
}

async function updateViewer() {
  const leftUrl = meshUrl("reference");
  if (leftUrl && leftUrl !== loadedMeshFile.left) {
    loadedMeshFile.left = leftUrl;
    await viewer.loadLeft(leftUrl);
  }
  const rightUrl = meshUrl(rightShowing);
  if (rightUrl && rightUrl !== loadedMeshFile.right) {
    loadedMeshFile.right = rightUrl;
    await viewer.loadRight(rightUrl);
  }
  document.getElementById("scoreLeftLabel").textContent = scoreLine("reference");
  document.getElementById("scoreRightLabel").textContent = scoreLine(rightShowing);
  document.getElementById("rightLabel").textContent =
    rightShowing === "delta_final" ? "delta (final, post-steering)" : "delta (initial)";
}

function dlLink(url, filename, ext) {
  if (!url) return "";
  return `<a href="${url}" download="${filename}">${ext}</a>`;
}

function renderCandidateDownloads() {
  const specs = [
    { key: "reference", label: "reference" },
    { key: "delta_initial", label: "delta (initial)" },
    { key: "delta_final", label: "delta (final)" },
  ];
  const container = document.getElementById("candidateDownloads");
  container.innerHTML = specs.map(({ key, label }) => {
    const { glb, stl } = candidateDownloads(key);
    const links = dlLink(glb, `${key}.glb`, "glb") + dlLink(stl, `${key}.stl`, "stl");
    return `<div class="dl-group"><span class="dl-label">${label}</span>${links || '<span class="dl-empty">not decoded yet</span>'}</div>`;
  }).join("");
}

function renderRail() {
  const container = document.getElementById("rail");
  trajectoryRail(container, {
    tPairs: state.tPairs, branches: state.branches, stepsDone: state.stepsDoneAbsolute, phase: state.phase,
  });
  const nExpected = state.nBranchesExpected;
  const nSoFar = state.branches.filter((b) => b.iBranch !== null).length;
  const branchCountText = nExpected && nExpected > 1 ? `${nSoFar}/${nExpected} branches forked  ·  ` : "";
  const cur = currentBranch();
  const phaseText = cur.branchT !== null
    ? `${branchCountText}viewing branch ${state.currentBranchView + 1}: t=${fmt(cur.branchT, 3)}${cur.outOfWindow ? "  (outside [0.3,0.7] window!)" : ""}`
    : branchCountText;
  document.getElementById("railPhase").textContent = phaseText;
}

function renderBranchesTable() {
  const tbody = document.querySelector("#branchesTable tbody");
  if (!state.branches.length) {
    tbody.innerHTML = '<tr><td colspan="5" class="empty-state">waiting for the first branch point…</td></tr>';
    document.getElementById("branchesSub").textContent = "";
    return;
  }
  document.getElementById("branchesSub").textContent =
    state.branches.length > 1 ? `${state.branches.length} fork(s) -- click a row to inspect it below` : "";
  tbody.innerHTML = state.branches.map((br, idx) => {
    const ref = br.candidates.reference && br.candidates.reference.score;
    const delta = (br.candidates.delta_final || br.candidates.delta_initial);
    const deltaScore = delta && delta.score;
    const rowClass = idx === state.currentBranchView ? "selected" : "";
    return `<tr class="branch-row ${rowClass}" data-branch-idx="${idx}" style="cursor:pointer">
      <td>${idx + 1}${state.nBranchesExpected && state.nBranchesExpected > state.branches.length && idx === state.branches.length - 1 ? "…" : ""}</td>
      <td class="num" style="text-align:left">${br.branchT !== null ? fmt(br.branchT, 3) : "…"}</td>
      <td class="num" style="text-align:left">${br.resumedFrom || (br.iBranch !== null ? "…" : "–")}</td>
      <td class="num">${ref ? fmt(ref.total) : "–"}</td>
      <td class="num">${deltaScore ? fmt(deltaScore.total) : "–"}</td>
    </tr>`;
  }).join("");
  tbody.querySelectorAll(".branch-row").forEach((row) => {
    row.addEventListener("click", () => {
      state.currentBranchView = parseInt(row.dataset.branchIdx, 10);
      loadedMeshFile = { left: null, right: null }; // force the viewer to reload this branch's meshes
      render();
    });
  });
}

function renderScoreBars() {
  const br = currentBranch();
  const w = br.judgeWeights;
  const container = document.getElementById("scoreBars");
  if (!w) { container.innerHTML = '<div class="empty-state">waiting for branch point…</div>'; return; }
  const weighted = (which, term) => {
    const c = br.candidates[which];
    if (!c || !c.score) return null;
    const s = c.score;
    if (term === "detail") return w.alpha * s.detail_reward;
    if (term === "overhang") return -w.beta * s.overhang_penalty;
    if (term === "thickness") return -w.gamma * s.thickness_penalty;
    if (term === "topology") return -w.delta * s.topology_penalty;
    if (term === "total") return s.total;
    return null;
  };
  const series = [
    { key: "reference", label: "reference", color: COLOR.reference },
    { key: "delta_initial", label: "delta (initial)", color: COLOR.delta },
    { key: "delta_final", label: "delta (final)", color: COLOR.deltaFinal },
  ];
  const terms = ["detail", "overhang", "thickness", "topology", "total"];
  const groups = terms.map((t) => ({
    label: t === "detail" ? "α·R_Detail" : t === "overhang" ? "-β·L_OH" : t === "thickness" ? "-γ·L_Th" : t === "topology" ? "-δ·L_Topo" : "S (total)",
    values: Object.fromEntries(series.map((s) => [s.key, weighted(s.key, t)])),
  }));
  groupedBars(container, { groups, series, width: container.clientWidth || 700, height: 220 });

  document.getElementById("candidateSub").textContent =
    (state.branches.length > 1 ? `branch ${state.currentBranchView + 1}/${state.branches.length}` : "")
    + (br.resumedFrom ? `  ·  resumed from: ${br.resumedFrom}` : "");
}

function renderGradCharts() {
  const steps = currentBranch().gradSteps;
  const lossSeries = [{ values: steps.map((s) => s.loss), color: COLOR.delta }];
  const proxySeries = [{ values: steps.map((s) => s.proxy), color: COLOR.delta }];
  const rmsSeries = [{ values: steps.map((s) => s.rms), color: COLOR.delta }];
  const proxyRef = steps.length ? steps[0].proxy_reference : null;

  lineChart(document.getElementById("chartLoss"), {
    series: lossSeries, title: "DPO loss", width: (document.getElementById("chartLoss").clientWidth || 220), height: 150,
  });
  lineChart(document.getElementById("chartProxy"), {
    series: proxySeries, title: "detail proxy",
    refLines: proxyRef !== null ? [{ value: proxyRef, label: "reference", color: COLOR.reference }] : [],
    width: (document.getElementById("chartProxy").clientWidth || 220), height: 150,
  });
  lineChart(document.getElementById("chartRms"), {
    series: rmsSeries, title: "delta RMS",
    refLines: currentBranch().trustRegion !== null ? [{ value: currentBranch().trustRegion, label: "trust region", color: COLOR.bad }] : [],
    width: (document.getElementById("chartRms").clientWidth || 220), height: 150,
  });
}

function renderHistograms() {
  const which = document.getElementById("histCandidate").value;
  const c = currentBranch().candidates[which];
  const d = c && c.details;
  const w = currentBranch().judgeWeights;
  const grid = document.getElementById("histGrid");
  const specs = [
    { key: "laplacian_mag", title: "detail energy (per-vertex)", color: COLOR.reference },
    {
      key: "overhang_cos", title: "overhang (n·g, per-face)", color: COLOR.delta,
      refLines: w ? [{ value: Math.cos((w.theta_crit_deg * Math.PI) / 180), color: COLOR.bad }] : [],
    },
    {
      key: "wall_depths", title: "wall thickness (per-ray)", color: COLOR.deltaFinal,
      refLines: w ? [{ value: w.d_min, color: COLOR.bad }] : [],
    },
    { key: "edge_incidence", title: "edge incidence (1=open,2=ok)", color: COLOR.warn || "#FFC24B" },
  ];
  grid.innerHTML = "";
  specs.forEach((spec) => {
    const box = document.createElement("div");
    grid.appendChild(box);
    histogram(box, d ? d[spec.key] : null, {
      title: spec.title, color: spec.color, refLines: spec.refLines || [],
      width: (grid.clientWidth || 700) / 4 - 10, height: 120,
    });
  });
  document.getElementById("histSub").textContent = d
    ? `faces_scored=${d.faces_scored}  rays=${d.rays_hit}/${d.rays_requested}  open/nonmanifold/edges=${d.n_open}/${d.n_nonmanifold}/${d.n_edges}`
    : "";
}

function renderDiagnostics() {
  const tbody = document.querySelector("#diagTable tbody");
  const rows = [];
  const cfg = state.config;
  rows.push(["image", cfg.image || "–"]);
  rows.push(["pipeline_type / steps / seed", `${cfg.pipeline_type || "–"} / ${cfg.steps ?? "default"} / ${cfg.seed ?? "–"}`]);
  rows.push(["branches requested / forked", `${cfg.num_branches ?? 1} / ${state.branches.filter((b) => b.iBranch !== null).length}`]);
  const pr = state.printableResult;
  if (pr) {
    rows.push(["watertight (post-repair)", pr.watertight ? "yes" : "no"]);
    rows.push(["vertices / faces (final)", `${fmt(pr.vertex_count, 0)} / ${fmt(pr.face_count, 0)}`]);
    rows.push(["generation / postprocess time", `${fmt(pr.generation_seconds, 1)}s / ${fmt(pr.postprocess_seconds, 1)}s`]);
    if (pr.diagnostics) rows.push(["diagnostics", JSON.stringify(pr.diagnostics)]);
    if (pr.fidelity) rows.push(["fidelity", JSON.stringify(pr.fidelity)]);
  }
  if (state.vanillaResult) {
    rows.push(["vanilla comparison", `${fmt(state.vanillaResult.vertex_count, 0)}v / ${fmt(state.vanillaResult.face_count, 0)}f in ${fmt(state.vanillaResult.seconds, 1)}s`]);
  }
  if (state.error) rows.push(["ERROR", `${state.error.type}: ${state.error.message}`]);
  tbody.innerHTML = rows.map(([k, v]) => `<tr><td>${k}</td><td class="num" style="text-align:left;font-weight:600">${v}</td></tr>`).join("");

  const notesEl = document.getElementById("notesList");
  notesEl.innerHTML = state.notes.map((n) => `<li>${n}</li>`).join("");
}

function renderResultDownloads() {
  const container = document.getElementById("resultDownloads");
  const groups = [];

  const raw = state.rawResult && state.rawResult.mesh;
  if (raw) {
    groups.push({
      label: "raw final (no postprocessing)",
      links: dlLink(raw.file && `/api/mesh/${state.runId}/${raw.file}`, "raw_final.glb", "glb")
        + dlLink(raw.stl_file && `/api/mesh/${state.runId}/${raw.stl_file}`, "raw_final.stl", "stl"),
    });
  }
  const pr = state.printableResult;
  if (pr) {
    groups.push({
      label: "printable final (post-processed)",
      links: dlLink(outputUrl(pr.glb_path), "printable_final.glb", "glb")
        + dlLink(outputUrl(pr.stl_path), "printable_final.stl", "stl"),
    });
  }
  const vr = state.vanillaResult;
  if (vr) {
    groups.push({
      label: "vanilla (no DPO branch)",
      links: dlLink(outputUrl(vr.glb_path), "vanilla.glb", "glb")
        + dlLink(outputUrl(vr.obj_path), "vanilla.obj", "obj"),
    });
  }

  container.innerHTML = groups.length
    ? groups.map((g) => `<div class="dl-group"><span class="dl-label">${g.label}</span>${g.links || '<span class="dl-empty">–</span>'}</div>`).join("")
    : '<div class="empty-state">no downloadable meshes yet</div>';
}

function render() {
  document.getElementById("emptyState").style.display = state.runId ? "none" : "block";
  document.getElementById("content").style.display = state.runId ? "flex" : "none";
  if (!state.runId) return;
  renderRail();
  renderBranchesTable();
  updateViewer();
  renderCandidateDownloads();
  renderScoreBars();
  renderGradCharts();
  renderHistograms();
  renderDiagnostics();
  renderResultDownloads();
  updateRunButtons();
}

// ---------------------------------------------------------------------------
// Health / prod-server guard
// ---------------------------------------------------------------------------
async function pollHealth() {
  try {
    const r = await fetch("/api/health");
    const h = await r.json();
    const pill = document.getElementById("healthPill");
    const text = document.getElementById("healthText");
    pill.className = "health-pill " + (h.busy ? "busy" : "idle");
    text.textContent = h.busy ? `running: ${h.active_run_id}` : "idle";

    const banner = document.getElementById("prodWarning");
    if (h.prod_warning && !banner.dataset.dismissed) {
      banner.classList.remove("hidden");
      document.getElementById("prodWarningText").textContent = h.prod_warning;
    } else if (!h.prod_warning) {
      banner.classList.add("hidden");
      banner.dataset.dismissed = "";
    }
    updateRunButtons(h.busy);
  } catch (e) {
    // dev server itself unreachable -- rare, don't spam the console every 3s
  }
}
document.getElementById("dismissWarning").addEventListener("click", () => {
  const banner = document.getElementById("prodWarning");
  banner.classList.add("hidden");
  banner.dataset.dismissed = "1";
});
setInterval(pollHealth, 3000);
pollHealth();

function updateRunButtons(busyOverride) {
  const busy = busyOverride !== undefined ? busyOverride : !state.finished && state.runId && state.phase !== "error" && state.phase !== "done";
  document.getElementById("cancelBtn").style.display = busy ? "block" : "none";
  document.getElementById("runBtn").disabled = !!busy;
}

// ---------------------------------------------------------------------------
// Run list (sidebar)
// ---------------------------------------------------------------------------
let didAutoLoad = false;
async function refreshRunList() {
  try {
    const r = await fetch("/api/runs");
    const { runs } = await r.json();
    const list = document.getElementById("runList");
    if (!runs.length) { list.innerHTML = '<div class="empty-state">no runs yet</div>'; return; }
    if (!didAutoLoad && !state.runId) {
      // On first load (nothing selected yet), show the most recent run --
      // matches the usual "come back and see what you were looking at" dev
      // tool convention rather than an empty state you have to click past.
      didAutoLoad = true;
      loadRun(runs[0].run_id);
    }
    list.innerHTML = runs.map((run) => {
      const statusClass = !run.finished ? "status-pending" : run.ok ? "status-ok" : "status-err";
      const statusText = !run.finished ? "running…" : run.ok ? "done" : "error";
      const mins = (run.duration_seconds / 60).toFixed(1);
      return `<div class="run-item ${run.run_id === state.runId ? "active" : ""}" data-run-id="${run.run_id}">
        <div class="rid">${run.run_id}</div>
        <div>${run.image || "?"} · ${run.pipeline_type || "?"} · <span class="${statusClass}">${statusText}</span> · ${mins}m</div>
      </div>`;
    }).join("");
    list.querySelectorAll(".run-item").forEach((elm) => {
      elm.addEventListener("click", () => loadRun(elm.dataset.runId));
    });
  } catch (e) { /* ignore */ }
}
setInterval(refreshRunList, 5000);
refreshRunList();

// ---------------------------------------------------------------------------
// Loading a run (live stream for a fresh run_id, or replay for a past one)
// ---------------------------------------------------------------------------
function resetForRun(runId) {
  if (currentSource) { currentSource.close(); currentSource = null; }
  state = freshState();
  state.runId = runId;
  rightShowing = "delta_initial";
  loadedMeshFile = { left: null, right: null };
  if (!viewer) {
    viewer = new DualMeshViewer(document.getElementById("canvasLeft"), document.getElementById("canvasRight"));
  }
  render();
}

async function loadRun(runId) {
  resetForRun(runId);
  // Replay everything already on disk, then attach live ONLY if the run is
  // still in flight. /api/stream always replays from byte 0 (that's what
  // makes live-tail and full-replay the same code path server-side, see
  // trace.py) -- so for an already-finished run, attaching it here on top
  // of this /api/trace fetch would apply every event TWICE (a real bug
  // caught by watching a finished run's gradient-step charts render as a
  // zigzag with 2x the real point count instead of a clean decline).
  const r = await fetch(`/api/trace/${runId}`);
  const { events } = await r.json();
  events.forEach(applyEvent);
  render();
  const lastType = events.length ? events[events.length - 1].type : null;
  const isFinished = lastType === "session_end" || lastType === "error";
  if (!isFinished) attachStream(runId);
  refreshRunList();
}

function attachStream(runId) {
  const es = new EventSource(`/api/stream/${runId}`);
  currentSource = es;
  es.onmessage = (e) => {
    try {
      const evt = JSON.parse(e.data);
      applyEvent(evt);
      render();
    } catch (err) { console.warn("[app] bad SSE event", err); }
  };
  es.addEventListener("done", () => { es.close(); if (currentSource === es) currentSource = null; refreshRunList(); });
  es.onerror = () => { /* browser auto-retries; if the run truly finished the server closes cleanly via the "done" event above */ };
}

// ---------------------------------------------------------------------------
// New-run form
// ---------------------------------------------------------------------------
let selectedImage = null;
const dropzone = document.getElementById("dropzone");
const imageInput = document.getElementById("imageInput");
dropzone.addEventListener("click", () => imageInput.click());
dropzone.addEventListener("dragover", (e) => { e.preventDefault(); });
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  if (e.dataTransfer.files[0]) setImage(e.dataTransfer.files[0]);
});
imageInput.addEventListener("change", () => { if (imageInput.files[0]) setImage(imageInput.files[0]); });

function setImage(file) {
  selectedImage = file;
  const reader = new FileReader();
  reader.onload = () => {
    dropzone.innerHTML = `<img src="${reader.result}"><div>${file.name}</div>`;
    dropzone.classList.add("has-image");
  };
  reader.readAsDataURL(file);
}

async function submitRun(ack) {
  if (!selectedImage) { alert("choose an input image first"); return; }
  const fd = new FormData();
  fd.append("image", selectedImage);
  fd.append("pipeline_type", document.getElementById("pipelineType").value);
  const steps = document.getElementById("steps").value;
  if (steps) fd.append("steps", steps);
  fd.append("seed", document.getElementById("seed").value);
  fd.append("target_faces", document.getElementById("targetFaces").value);
  fd.append("t_branch", document.getElementById("tBranch").value);
  fd.append("num_branches", document.getElementById("numBranches").value);
  fd.append("branch_noise_scale", document.getElementById("noiseScale").value);
  fd.append("continuation_steps", document.getElementById("contSteps").value);
  fd.append("num_delta_grad_steps", document.getElementById("gradSteps").value);
  fd.append("dpo_beta", document.getElementById("dpoBeta").value);
  if (document.getElementById("vanillaToo").checked) fd.append("vanilla_too", "1");
  if (ack) fd.append("acknowledge_prod_warning", "1");

  const r = await fetch("/api/run", { method: "POST", body: fd });
  if (r.status === 409) {
    const err = await r.json();
    const proceed = confirm(err.detail + "\n\nRun anyway?");
    if (proceed) return submitRun(true);
    return;
  }
  if (!r.ok) { const err = await r.json().catch(() => ({})); alert("Failed to start run: " + (err.detail || r.statusText)); return; }
  const { run_id } = await r.json();
  resetForRun(run_id);
  attachStream(run_id);
  refreshRunList();
}

document.getElementById("runForm").addEventListener("submit", (e) => {
  e.preventDefault();
  submitRun(false);
});

document.getElementById("cancelBtn").addEventListener("click", async () => {
  if (!state.runId) return;
  await fetch(`/api/cancel/${state.runId}`, { method: "POST" });
  refreshRunList();
});

document.getElementById("showInitialBtn").addEventListener("click", () => { rightShowing = "delta_initial"; updateViewer(); });
document.getElementById("showFinalBtn").addEventListener("click", () => { rightShowing = "delta_final"; updateViewer(); });
document.getElementById("histCandidate").addEventListener("change", renderHistograms);

render();

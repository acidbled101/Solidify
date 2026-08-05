// run_view.js -- the run-specific half of the Concept Proof page.
//
// Everything here is read from ONE run's trace.jsonl via /api/trace/{id}.
// Nothing is modelled or interpolated. Where a run predates an instrument
// (older traces have no `latent` on their sampler_step events), the panel says
// so rather than substituting a plausible-looking curve -- a dashboard that
// silently fabricates the missing half is worse than one that reports a gap.
//
// Run selection is shared with the Run inspector through localStorage +
// ?run=<id>, so picking a run in either place follows you to the other.
"use strict";

const RV = {
  pre: "#6C7C86",
  reference: "#3ED6C4",
  delta_initial: "#FF9F45",
  delta_final: "#FF6B4A",
  good: "#7CE38B",
  bad: "#FF6B6B",
  warn: "#FFC24B",
  dim: "#8B99A4",
};
const SELECTED_KEY = "dpoInspector.selectedRun";
// Bumped by every loadRun() call before its first await; a response that
// resolves after a newer selection has superseded it must be discarded.
// Same race as devlab/static/app.js's loadRun -- this file has no `state`
// object, it writes straight into DOM elements by id, which makes a stale
// write even more directly visible (a slower run's trajectory/table can
// silently replace what a faster, more-recently-selected run already drew).
let loadGeneration = 0;

// Fail loudly and actionably if charts.js is older than this file. Without
// this the symptom is a bare ReferenceError naming one function, which looks
// like a scoping bug rather than what it is: a browser serving a cached
// charts.js from before these functions existed.
function requireCharts(names, where) {
  const missing = names.filter((n) => typeof window[n] !== "function");
  if (!missing.length) return true;
  const msg =
    `<div class="banner"><strong>Stale charts.js.</strong> ${where} needs ` +
    `<code>${missing.join("</code>, <code>")}</code>, which this browser's cached ` +
    `copy of <code>/static/charts.js</code> does not define ` +
    `(loaded build: <code>${(window.DPOCharts && window.DPOCharts.BUILD) || "pre-v2"}</code>). ` +
    `Hard-reload the page (Cmd-Shift-R) to pick up the current file. The server now sends ` +
    `<code>Cache-Control: no-cache</code>, so this should not recur.</div>`;
  const host = document.querySelector("main") || document.body;
  host.insertAdjacentHTML("afterbegin", msg);
  return false;
}


function rn(v, d = 3) {
  if (v === null || v === undefined || Number.isNaN(v)) return "–";
  return Number(v).toFixed(d);
}
function stat(k, v, color) {
  return `<div><span class="k">${k}</span><span class="v"${color ? ` style="color:${color}"` : ""}>${v}</span></div>`;
}

// ---------------------------------------------------------------------------
// Build a linear timeline of "frames" from the raw event list.
//
// The trace is chronological, but the three fork branches are NOT interleaved
// in it: each continuation runs to completion before the next starts, so the
// events arrive as (all reference steps, all delta_initial steps, all
// delta_final steps). They are re-keyed by (branch, index) here so a single
// slider position shows all three candidates at the SAME timestep, which is
// the only way the divergence is readable.
// ---------------------------------------------------------------------------
function buildFrames(events) {
  const frames = [];
  const forkOrigin = {};   // branch_index -> [x,y] at the fork
  const branchLatents = {}; // branch_index -> { which -> [ {index, t, ...} ] }

  events.forEach((e) => {
    const p = e.payload || {};
    if (e.type === "branch_latent") {
      const b = p.branch_index ?? 0;
      if (p.which === "fork") {
        if (p.latent && p.latent.proj) forkOrigin[b] = p.latent.proj;
      } else {
        (branchLatents[b] = branchLatents[b] || {});
        (branchLatents[b][p.which] = branchLatents[b][p.which] || []).push({ ...p.latent, t: p.t, index: p.index });
      }
    }
  });

  const emittedFork = {};
  events.forEach((e) => {
    const p = e.payload || {};
    if (e.type === "sampler_step") {
      frames.push({
        kind: "step", phase: p.phase, branch: p.branch_index ?? 0, t: p.t,
        proj: p.latent && p.latent.proj ? p.latent.proj : null,
        v_rms: p.latent ? p.latent.v_rms : undefined,
        rms: p.latent ? p.latent.rms : undefined,
      });
    } else if (e.type === "branch_point") {
      const b = p.branch_index ?? 0;
      if (emittedFork[b]) return;
      emittedFork[b] = true;
      const lat = branchLatents[b] || {};
      const k = Math.max(
        (lat.reference || []).length, (lat.delta_initial || []).length, (lat.delta_final || []).length, 0
      );
      for (let i = 0; i < k; i++) {
        const at = (which) => {
          const arr = lat[which] || [];
          const hit = arr.find((r) => r.index === i) || arr[i];
          return hit && hit.proj ? hit.proj : null;
        };
        const anyRow = (lat.reference || [])[i] || (lat.delta_initial || [])[i] || {};
        frames.push({
          kind: "fork", branch: b, forkIndex: i, t: anyRow.t ?? p.branch_t,
          forkOrigin: forkOrigin[b] || null,
          points: { reference: at("reference"), delta_initial: at("delta_initial"), delta_final: at("delta_final") },
          v: {
            reference: ((lat.reference || [])[i] || {}).v_rms,
            delta_initial: ((lat.delta_initial || [])[i] || {}).v_rms,
            delta_final: ((lat.delta_final || [])[i] || {}).v_rms,
          },
          branchPoint: p,
        });
      }
      if (k === 0) {
        frames.push({ kind: "fork", branch: b, forkIndex: 0, t: p.branch_t,
                      forkOrigin: forkOrigin[b] || null, points: {}, v: {}, branchPoint: p });
      }
    }
  });
  return frames;
}

function collectBranches(events) {
  const out = {};
  const slot = (i) => (out[i] = out[i] || { index: i, scores: {}, cmp: {}, loss: [], proxy: [], rms: [] });
  events.forEach((e) => {
    const p = e.payload || {};
    const b = p.branch_index ?? 0;
    if (e.type === "branch_point") Object.assign(slot(b), { point: p });
    else if (e.type === "branch_perturbation") Object.assign(slot(b), { perturbation: p });
    else if (e.type === "candidate_scored") {
      slot(b).scores[p.which] = p.score ? p.score.total : null;
      slot(b).cmp[p.which] = p.cmp;
    } else if (e.type === "grad_step") {
      slot(b).loss.push(p.loss); slot(b).proxy.push(p.proxy); slot(b).rms.push(p.rms);
      slot(b).proxyRef = p.proxy_reference;
    } else if (e.type === "resume") slot(b).resumedFrom = p.resumed_from;
  });
  return Object.values(out).sort((a, b) => a.index - b.index);
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

function renderRunSummary(events) {
  const first = events.find((e) => e.type === "session_start");
  const end = [...events].reverse().find((e) => e.type === "session_end");
  const err = [...events].reverse().find((e) => e.type === "error");
  const pr = [...events].reverse().find((e) => e.type === "printable_result");
  const s = (first && first.payload) || {};
  const status = end ? (end.payload.ok ? "ok" : "failed") : err ? "error" : "incomplete";
  const dur = events.length > 1 ? events[events.length - 1].t - events[0].t : null;

  document.getElementById("runSummary").innerHTML = [
    stat("status", status, status === "ok" ? RV.good : status === "incomplete" ? RV.warn : RV.bad),
    stat("image", s.image || "–"),
    stat("pipeline", s.pipeline_type || "–"),
    stat("seed", s.seed ?? "–"),
    stat("branches", s.num_branches ?? "–"),
    stat("events", events.length),
    stat("duration", dur ? `${(dur / 60).toFixed(1)} min` : "–"),
    pr ? stat("watertight", String(pr.payload.watertight),
              pr.payload.watertight ? RV.good : RV.bad) : "",
    pr ? stat("faces", (pr.payload.face_count || 0).toLocaleString()) : "",
  ].join("");

  if (err) {
    document.getElementById("runNotice").innerHTML =
      `<div class="callout" style="border-left-color:${RV.bad}"><strong>This run errored.</strong>
       <code>${(err.payload.message || err.payload.error || JSON.stringify(err.payload)).slice(0, 400)}</code></div>`;
  } else {
    document.getElementById("runNotice").innerHTML = "";
  }
  return status;
}

function renderTrajectory(frames, events) {
  const slider = document.getElementById("stepSlider");
  const hasLatent = frames.some((f) => (f.proj) || (f.kind === "fork" && Object.values(f.points).some(Boolean)));
  slider.max = Math.max(0, frames.length - 1);
  slider.disabled = !hasLatent;

  if (!hasLatent) {
    document.getElementById("latentPathChart").innerHTML =
      `<div class="callout" style="border-left-color:${RV.warn}">
        <strong>This run predates the latent instrumentation.</strong> Its
        <code>sampler_step</code> events carry only <code>t</code>/<code>t_prev</code> — no latent
        projection or velocity was recorded, so there is no trajectory to draw and none will be
        invented. Runs started after the <code>LatentProbe</code> change contain it; re-run this
        input to populate this panel.
       </div>`;
    document.getElementById("scrubReadout").textContent = "no latent telemetry";
    ["velocityChart", "latentRmsChart", "branchDivergenceChart"].forEach((id) => {
      document.getElementById(id).innerHTML = "";
    });
    return false;
  }

  const draw = () => {
    const at = Number(slider.value);
    latentPath(document.getElementById("latentPathChart"), {
      frames, upTo: at, width: 720, height: 400,
      title: "latent trajectory (fixed random projection of the full SLat state)",
      colors: RV,
    });
    const f = frames[at] || {};
    const kind = f.kind === "fork"
      ? `fork ${f.branch + 1}, continuation step ${f.forkIndex + 1}`
      : `${(f.phase || "").replace("_", "-")} step`;
    document.getElementById("scrubReadout").innerHTML =
      `<span class="mono">${at + 1}/${frames.length}</span> · ${kind} · t=<span class="mono">${rn(f.t, 4)}</span>` +
      (f.v_rms !== undefined ? ` · ‖v‖<sub>rms</sub>=<span class="mono">${rn(f.v_rms, 4)}</span>` : "") +
      (f.kind === "fork" && f.v.reference !== undefined
        ? ` · ‖v‖ ref <span class="mono">${rn(f.v.reference, 4)}</span> / δ <span class="mono">${rn(f.v.delta_initial, 4)}</span>`
        : "");
    renderVelocity(frames, at);
  };
  slider.oninput = draw;
  slider.value = String(frames.length - 1);
  draw();
  return true;
}

function renderVelocity(frames, cursor) {
  // Main-path velocity over the whole schedule, with the fork positions and
  // the slider cursor marked so this panel lines up with the trajectory above.
  const stepFrames = frames.map((f, i) => ({ ...f, i })).filter((f) => f.kind === "step");
  const vSeries = stepFrames.map((f) => (f.v_rms === undefined ? null : f.v_rms)).filter((v) => v !== null);
  const rmsSeries = stepFrames.map((f) => (f.rms === undefined ? null : f.rms)).filter((v) => v !== null);

  // Map global frame indices onto this chart's x axis (which counts only step
  // frames, not fork frames) so the markers land in the right place.
  const stepXOf = (globalIdx) => {
    let n = -1;
    for (let i = 0; i < frames.length && i <= globalIdx; i++) if (frames[i].kind === "step") n++;
    return n;
  };
  const forkMarks = [];
  frames.forEach((f, i) => {
    if (f.kind === "fork" && f.forkIndex === 0) {
      forkMarks.push({ index: stepXOf(i), color: RV.good, label: `fork ${f.branch + 1}` });
    }
  });
  const cursorMark = frames[cursor] && frames[cursor].kind === "step"
    ? [{ index: stepXOf(cursor), color: "#DCE4EA", solid: true }]
    : [];

  lineChart(document.getElementById("velocityChart"), {
    series: [{ values: vSeries, color: RV.pre }],
    width: 720, height: 200, title: "‖v_θ‖ RMS per sampler step (main trajectory)", yLabel: "‖v‖ rms",
    vLines: [...forkMarks, ...cursorMark],
  });
  lineChart(document.getElementById("latentRmsChart"), {
    series: [{ values: rmsSeries, color: RV.reference }],
    width: 350, height: 180, title: "latent ‖x‖ RMS per step", yLabel: "‖x‖ rms",
    vLines: forkMarks,
  });

  // Divergence between the branches inside each fork window: how far apart the
  // candidates actually got, in the projection.
  const forks = frames.filter((f) => f.kind === "fork");
  const dist = (a, b) => (a && b ? Math.hypot(a[0] - b[0], a[1] - b[1]) : null);
  const dRandom = forks.map((f) => dist(f.points.reference, f.points.delta_initial)).filter((v) => v !== null);
  const dSteered = forks.map((f) => dist(f.points.reference, f.points.delta_final)).filter((v) => v !== null);
  const dGrad = forks.map((f) => dist(f.points.delta_initial, f.points.delta_final)).filter((v) => v !== null);

  if (dRandom.length) {
    lineChart(document.getElementById("branchDivergenceChart"), {
      series: [
        { values: dRandom, color: RV.delta_initial },
        { values: dSteered, color: RV.delta_final },
        { values: dGrad, color: RV.good },
      ],
      width: 350, height: 180, yLabel: "distance",
      title: "branch separation in the projection",
    });
  } else {
    document.getElementById("branchDivergenceChart").innerHTML = "";
  }
}

function renderForkDetail(branches, frames) {
  if (!branches.length) {
    document.getElementById("forkDetail").innerHTML = "";
    return;
  }
  const parts = branches.map((b) => {
    const p = b.point || {};
    const pert = b.perturbation || {};
    const forkFrames = frames.filter((f) => f.kind === "fork" && f.branch === b.index);
    const dist = (a, c) => (a && c ? Math.hypot(a[0] - c[0], a[1] - c[1]) : null);
    const last = forkFrames[forkFrames.length - 1];
    const sep = last ? dist(last.points.reference, last.points.delta_final) : null;
    return `<div style="margin-bottom:6px"><strong>Fork ${b.index + 1}</strong> at
      t=<code>${rn(p.branch_t, 3)}</code> → resume t=<code>${rn(p.resume_t, 3)}</code>,
      ${p.n_voxels ? `${p.n_voxels.toLocaleString()} voxels` : "voxel count not recorded"}.
      ${pert.relative !== undefined
        ? `Perturbation is <strong>${(pert.relative * 100).toFixed(2)}%</strong> of the latent's own RMS
           (ε=<code>${rn(pert.eps_rms, 4)}</code> against <code>${rn(pert.base_rms, 4)}</code>)`
        : "Perturbation magnitude not recorded for this run"}${sep !== null
        ? `, and by the end of the window the steered candidate sits <code>${rn(sep, 4)}</code> away
           from the reference in the projection.` : "."}
      Resumed from <strong style="color:${b.resumedFrom === "delta" ? RV.delta_final : RV.reference}">${b.resumedFrom || "?"}</strong>.</div>`;
  });
  document.getElementById("forkDetail").innerHTML =
    parts.join("") +
    `<div class="dim" style="margin-top:6px">The perturbation is deliberately tiny — the trust region is
     a hard RMS cap set at <code>branch_noise_scale × delta_max_norm_ratio</code>. If the branches look
     nearly coincident above, that is the design working, not a rendering artifact.</div>`;
}

function renderRunBranches(branches) {
  const el = document.getElementById("runBranchTable");
  if (!branches.length) {
    el.innerHTML = `<tr><td class="dim">no forks recorded in this run</td></tr>`;
    document.getElementById("runLossChart").innerHTML = "";
    document.getElementById("runRmsChart").innerHTML = "";
    return;
  }
  el.innerHTML =
    `<tr><th>fork</th><th class="num">t</th><th class="num">S ref</th><th class="num">S δ random</th>
      <th class="num">S δ steered</th><th class="num">gain (random)</th><th class="num">gain (steering)</th>
      <th>resumed</th></tr>` +
    branches.map((b) => {
      const s = b.scores;
      const gr = (s.reference != null && s.delta_initial != null) ? s.delta_initial - s.reference : null;
      const gs = (s.delta_initial != null && s.delta_final != null) ? s.delta_final - s.delta_initial : null;
      const col = (v) => (v == null ? RV.dim : v > 0 ? RV.good : RV.bad);
      return `<tr>
        <td>${b.index + 1}</td>
        <td class="num">${rn((b.point || {}).branch_t, 3)}</td>
        <td class="num">${rn(s.reference, 4)}</td>
        <td class="num">${rn(s.delta_initial, 4)}</td>
        <td class="num">${rn(s.delta_final, 4)}</td>
        <td class="num" style="color:${col(gr)}">${gr == null ? "–" : (gr > 0 ? "+" : "") + rn(gr, 4)}</td>
        <td class="num" style="color:${col(gs)}">${gs == null ? "–" : (gs > 0 ? "+" : "") + rn(gs, 4)}</td>
        <td style="color:${b.resumedFrom === "delta" ? RV.delta_final : RV.reference}">${b.resumedFrom || "–"}</td>
      </tr>`;
    }).join("");

  const withGrad = branches.filter((b) => b.loss.length);
  lineChart(document.getElementById("runLossChart"), {
    series: withGrad.map((b, i) => ({ values: b.loss, color: i === 0 ? RV.delta_initial : RV.delta_final })),
    width: 350, height: 180, title: "DPO loss per gradient step (one line per fork)", yLabel: "loss",
  });
  lineChart(document.getElementById("runRmsChart"), {
    series: withGrad.map((b, i) => ({ values: b.rms, color: i === 0 ? RV.delta_initial : RV.delta_final })),
    width: 350, height: 180, title: "‖delta‖ RMS per gradient step", yLabel: "rms",
    refLines: withGrad.length && withGrad[0].point
      ? [{ value: withGrad[0].point.trust_region, label: "trust region", color: RV.bad }] : [],
  });
}

// ---------------------------------------------------------------------------
// Cross-run pool
// ---------------------------------------------------------------------------
async function renderPool() {
  let m;
  try {
    m = await (await fetch("/api/metrics")).json();
  } catch (e) {
    document.getElementById("poolNote").innerHTML = `could not load /api/metrics — ${e.message}`;
    return;
  }
  const s = m.summary;
  document.getElementById("poolSummary").innerHTML = [
    stat("real runs", `${s.n_runs_ok}/${s.n_runs - s.n_synthetic_excluded}`),
    stat("forks pooled", s.n_branches_scored),
    stat("verdict flipped", `${s.n_verdict_flipped}/${s.n_branches_scored}`),
    stat("flip rate", s.flip_rate == null ? "–" : rn(s.flip_rate, 3)),
    stat("mean gain: random", s.mean_gain_perturbation == null ? "–" : rn(s.mean_gain_perturbation, 4),
         s.mean_gain_perturbation > 0 ? RV.good : RV.bad),
    stat("mean gain: steering", s.mean_gain_steering == null ? "–" : rn(s.mean_gain_steering, 4),
         s.mean_gain_steering > 0 ? RV.good : RV.bad),
    stat("± SE", s.stderr_gain_steering == null ? "–" : rn(s.stderr_gain_steering, 4)),
    stat("significant?", String(s.steering_significant), s.steering_significant ? RV.warn : RV.dim),
  ].join("");

  const real = (m.branch_rows || []).filter((r) => !r.is_synthetic && r.gain_steering != null);
  if (real.length) {
    groupedBars(document.getElementById("poolChart"), {
      groups: real.map((r) => ({
        label: `${r.run_id.slice(9, 15)}·${r.branch_index}`,
        values: { random: r.gain_perturbation, steering: r.gain_steering },
      })),
      series: [
        { key: "random", label: "random perturbation", color: RV.delta_initial },
        { key: "steering", label: "gradient steering", color: RV.delta_final },
      ],
      width: 900, height: 230,
      title: "per-fork score gain: what the random draw bought vs what the gradient added",
    });
  }

  document.getElementById("poolTable").innerHTML =
    `<tr><th>run</th><th class="num">fork</th><th class="num">S ref</th><th class="num">S δ rand</th>
      <th class="num">S δ steer</th><th class="num">gain rand</th><th class="num">gain steer</th>
      <th>flipped</th></tr>` +
    (m.branch_rows || []).map((r) => `<tr${r.is_synthetic ? ' style="opacity:.45"' : ""}>
      <td>${r.run_id}${r.is_synthetic ? " <span class='dim'>(synthetic, excluded)</span>" : ""}</td>
      <td class="num">${r.branch_index}</td>
      <td class="num">${rn(r.score_reference, 4)}</td>
      <td class="num">${rn(r.score_delta_initial, 4)}</td>
      <td class="num">${rn(r.score_delta_final, 4)}</td>
      <td class="num" style="color:${r.gain_perturbation > 0 ? RV.good : RV.bad}">${rn(r.gain_perturbation, 4)}</td>
      <td class="num" style="color:${r.gain_steering > 0 ? RV.good : RV.bad}">${rn(r.gain_steering, 4)}</td>
      <td style="color:${r.verdict_flipped ? RV.warn : RV.dim}">${String(r.verdict_flipped)}</td>
    </tr>`).join("");

  const dir = s.mean_gain_steering == null ? "unmeasured"
    : s.mean_gain_steering > 0 ? "positive" : "negative";
  document.getElementById("poolNote").innerHTML =
    `<strong>Across ${s.n_branches_scored} real forks, gradient steering is worth
     ${rn(s.mean_gain_steering, 4)} ± ${rn(s.stderr_gain_steering, 4)} (SE) — ${dir} in sign and
     <em>not</em> statistically distinguishable from zero.</strong>
     The random perturbation alone is worth ${rn(s.mean_gain_perturbation, 4)}.
     Steering improved ${s.n_steering_positive} forks and hurt ${s.n_steering_negative}.
     <span class="dim">This is the cheapest decisive measurement available on this pipeline and it
     needed no new GPU time — only aggregation over traces already on disk. It does not show the
     gradient is useless; it shows the current evidence cannot tell it apart from noise, and that
     n must grow (or the proxy must change) before the extra ~3 backward passes per fork are
     justified. The re-rank gate means the cost so far is compute, not output quality.</span>`;
}

// ---------------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------------
async function loadRun(runId) {
  const myGeneration = ++loadGeneration;
  const res = await fetch(`/api/trace/${runId}`);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  const data = await res.json();
  if (myGeneration !== loadGeneration) return; // superseded while this fetch was in flight
  const events = data.events || data;
  document.getElementById("runSub").textContent = runId;
  document.getElementById("openInspector").href = `/?run=${encodeURIComponent(runId)}`;
  renderRunSummary(events);
  const frames = buildFrames(events);
  const branches = collectBranches(events);
  renderTrajectory(frames, events);
  renderForkDetail(branches, frames);
  renderRunBranches(branches);
}

async function initRunView() {
  if (!requireCharts(["latentPath", "lineChart", "groupedBars"], "The run panels")) return;
  let runs = [];
  try {
    runs = (await (await fetch("/api/runs")).json()).runs || [];
  } catch (e) {
    document.getElementById("runNotice").innerHTML =
      `<div class="banner">could not list runs — ${e.message}</div>`;
    return;
  }
  const sel = document.getElementById("runSelect");
  // Newest first, synthetic demos last -- they exist to exercise the UI and
  // should never be what the page opens on.
  runs.sort((a, b) => (Number(a.is_synthetic) - Number(b.is_synthetic))
    || String(b.run_id).localeCompare(String(a.run_id)));
  // list_runs reports finished/ok, not a single status string.
  runs.forEach((r) => { r.status = !r.finished ? "running" : r.ok ? "ok" : "error"; });
  sel.innerHTML = runs.map((r) => {
    const tags = [
      r.status,
      r.grad_step_count ? `${r.grad_step_count} grad steps` : null,
      r.has_latent_telemetry ? "latent ✓" : "no latent",
      r.is_synthetic ? "synthetic" : null,
    ].filter(Boolean);
    return `<option value="${r.run_id}">${r.run_id} — ${tags.join(" · ")}</option>`;
  }).join("");

  const params = new URLSearchParams(location.search);
  const wanted = params.get("run") || localStorage.getItem(SELECTED_KEY);
  // Default to the most informative run available: real, has latent telemetry,
  // completed. Falling back through progressively weaker requirements rather
  // than landing on whatever happens to sort first.
  const pick = (f) => runs.find(f);
  const initial = (runs.find((r) => r.run_id === wanted) && wanted)
    || (pick((r) => !r.is_synthetic && r.has_latent_telemetry && r.status === "ok")
     || pick((r) => !r.is_synthetic && r.has_latent_telemetry)
     || pick((r) => !r.is_synthetic && r.status === "ok")
     || pick((r) => !r.is_synthetic)
     || runs[0] || {}).run_id;
  if (!initial) return;
  sel.value = initial;

  const go = async (id) => {
    localStorage.setItem(SELECTED_KEY, id);
    history.replaceState(null, "", `?run=${encodeURIComponent(id)}`);
    try {
      await loadRun(id);
    } catch (e) {
      document.getElementById("runNotice").innerHTML =
        `<div class="banner">could not load ${id} — ${e.message}</div>`;
    }
  };
  sel.onchange = () => go(sel.value);
  await go(initial);
  await renderPool();
}

initRunView();

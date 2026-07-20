"use strict";

// -------------------------------------------------------------------------
// Elements
// -------------------------------------------------------------------------
const form = document.getElementById("upload-form");
const imageInput = document.getElementById("image-input");
const preview = document.getElementById("preview");
const submitBtn = document.getElementById("submit-btn");
const queueBanner = document.getElementById("queue-banner");
const modelNote = document.getElementById("model-note");

const statusPanel = document.getElementById("status-panel");
const statusText = document.getElementById("status-text");
const statusSub = document.getElementById("status-sub");
const spinner = document.getElementById("spinner");
const stagesEl = document.getElementById("stages");

const errorPanel = document.getElementById("error-panel");
const errorMessage = document.getElementById("error-message");
const errorHint = document.getElementById("error-hint");
const retryBtn = document.getElementById("retry-btn");

const resultPanel = document.getElementById("result-panel");
const viewer = document.getElementById("viewer");
const downloads = document.getElementById("downloads");
const meshTable = document.getElementById("mesh-table");
const diagTable = document.getElementById("diag-table");
const diagWrap = document.getElementById("diag-wrap");
const newBtn = document.getElementById("new-btn");

const STAGES = [
  ["queued", "Queued"],
  ["loading_model", "Loading model"],
  ["preprocessing", "Preprocessing"],
  ["generating", "Generating mesh"],
  ["making_printable", "Print-prep"],
  ["done", "Done"],
];

const LS_KEY = "trellis_job_id";
let pollTimer = null;
let lastFormValues = null;
let serverConfig = null;

// -------------------------------------------------------------------------
// Init: load config, restore any in-flight job
// -------------------------------------------------------------------------
async function init() {
  try {
    serverConfig = await (await fetch("/api/config")).json();
    applyConfig(serverConfig);
  } catch (e) {
    console.warn("Could not load /api/config", e);
  }
  await refreshQueueBanner();

  const savedId = localStorage.getItem(LS_KEY);
  if (savedId) {
    showPanel("status");
    startPolling(savedId);
  }
}

function applyConfig(cfg) {
  document.getElementById("seed").value = cfg.default_seed;
  document.getElementById("target_faces").value = cfg.default_target_faces;
  document.getElementById("skip_printable").checked = !!cfg.skip_printable_by_default;
  const sel = document.getElementById("pipeline_type");
  sel.innerHTML = "";
  (cfg.pipeline_types || []).forEach((pt) => {
    const opt = document.createElement("option");
    opt.value = pt;
    opt.textContent = pt;
    if (pt === cfg.default_pipeline_type) opt.selected = true;
    sel.appendChild(opt);
  });
  if (modelNote) {
    modelNote.textContent =
      "Model: " + (cfg.model_id || "?") +
      (cfg.no_texture ? " · textures skipped (vertex-colored geometry)" : "");
  }
}

async function refreshQueueBanner() {
  try {
    const h = await (await fetch("/api/health")).json();
    if (h.queue_depth && h.queue_depth > 0) {
      queueBanner.style.display = "block";
      queueBanner.textContent =
        h.queue_depth + " job(s) already in the queue. Yours will start once they finish.";
    } else if (!h.pipeline_loaded) {
      queueBanner.style.display = "block";
      queueBanner.textContent = "Model is still warming up (~100s). Your first job may wait for it.";
    } else {
      queueBanner.style.display = "none";
    }
  } catch (e) {
    /* health failing shouldn't block the UI */
  }
}

// -------------------------------------------------------------------------
// Submit
// -------------------------------------------------------------------------
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!imageInput.files || imageInput.files.length === 0) return;

  const fd = new FormData();
  fd.append("image", imageInput.files[0]);
  fd.append("seed", document.getElementById("seed").value);
  fd.append("target_faces", document.getElementById("target_faces").value);
  fd.append("pipeline_type", document.getElementById("pipeline_type").value);
  fd.append("skip_printable", document.getElementById("skip_printable").checked ? "1" : "0");
  lastFormValues = fd;

  submitBtn.disabled = true;
  submitBtn.textContent = "Submitting...";
  hidePanel("error");
  hidePanel("result");

  try {
    const res = await fetch("/api/jobs", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.error || "Upload failed.");
    }
    localStorage.setItem(LS_KEY, data.job_id);
    showPanel("status");
    startPolling(data.job_id);
  } catch (err) {
    submitBtn.disabled = false;
    submitBtn.textContent = "Generate 3D model";
    showError({ message: String(err.message || err), hint: "" });
  }
});

retryBtn.addEventListener("click", () => {
  hidePanel("error");
  form.requestSubmit();
});

newBtn.addEventListener("click", () => {
  localStorage.removeItem(LS_KEY);
  hidePanel("result");
  hidePanel("status");
  submitBtn.disabled = false;
  submitBtn.textContent = "Generate 3D model";
  refreshQueueBanner();
});

imageInput.addEventListener("change", () => {
  const f = imageInput.files && imageInput.files[0];
  if (f) {
    preview.src = URL.createObjectURL(f);
    preview.style.display = "block";
  } else {
    preview.style.display = "none";
  }
});

// -------------------------------------------------------------------------
// Polling
// -------------------------------------------------------------------------
function startPolling(jobId) {
  if (pollTimer) clearInterval(pollTimer);
  submitBtn.disabled = true;
  poll(jobId);
  pollTimer = setInterval(() => poll(jobId), 2000);
}

function stopPolling() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = null;
}

async function poll(jobId) {
  let job;
  try {
    const res = await fetch("/api/jobs/" + jobId);
    if (res.status === 404) {
      // Job no longer known (server restarted). Clear and reset.
      stopPolling();
      localStorage.removeItem(LS_KEY);
      hidePanel("status");
      submitBtn.disabled = false;
      submitBtn.textContent = "Generate 3D model";
      return;
    }
    job = await res.json();
  } catch (e) {
    return; // transient; keep polling
  }

  renderStatus(job);

  if (job.status === "done") {
    stopPolling();
    localStorage.removeItem(LS_KEY);
    submitBtn.disabled = false;
    submitBtn.textContent = "Generate 3D model";
    renderResult(jobId, job);
  } else if (job.status === "error") {
    stopPolling();
    localStorage.removeItem(LS_KEY);
    submitBtn.disabled = false;
    submitBtn.textContent = "Generate 3D model";
    showError(job.error || { message: "Unknown error", hint: "" });
  }
}

// -------------------------------------------------------------------------
// Rendering
// -------------------------------------------------------------------------
function renderStatus(job) {
  showPanel("status");
  statusText.textContent = job.status_text || job.status;
  const elapsed = job.elapsed_seconds != null ? Math.round(job.elapsed_seconds) : 0;
  statusSub.textContent = "Elapsed: " + formatDuration(elapsed);
  spinner.classList.remove("hidden");

  const activeIdx = STAGES.findIndex((s) => s[0] === job.status);
  stagesEl.innerHTML = "";
  STAGES.forEach(([key, label], idx) => {
    const div = document.createElement("div");
    div.className = "stage";
    if (activeIdx >= 0 && idx < activeIdx) div.className += " done";
    if (key === job.status) div.className += " active";
    div.innerHTML = '<span class="dot"></span><span>' + label + "</span>";
    stagesEl.appendChild(div);
  });
}

function renderResult(jobId, job) {
  hidePanel("status");
  showPanel("result");

  const result = job.result || {};
  const files = result.files || [];
  const base = "/api/jobs/" + jobId + "/files/";

  // Preview: printable glb preferred, else raw glb.
  const previewName = result.preview_filename;
  if (previewName) {
    viewer.src = base + previewName;
    viewer.style.display = "block";
  } else {
    viewer.style.display = "none";
  }

  // Download links.
  downloads.innerHTML = "";
  files.forEach((f) => {
    const a = document.createElement("a");
    a.href = base + f.filename;
    a.textContent = "↓ " + (f.label || f.filename);
    a.download = f.filename;
    downloads.appendChild(a);
  });

  // Mesh stats.
  const stats = result.mesh_stats || {};
  meshTable.innerHTML = "";
  addRow(meshTable, "Vertices", fmtNum(stats.vertices));
  addRow(meshTable, "Faces", fmtNum(stats.faces));
  if (result.watertight != null) {
    addRow(meshTable, "Watertight", result.watertight ? "Yes" : "No");
  }

  // Diagnostics + fidelity.
  const d = job.diagnostics;
  const fdel = job.fidelity;
  if (!d && !fdel) {
    diagWrap.style.display = "none";
  } else {
    diagWrap.style.display = "block";
    diagTable.innerHTML = "";
    if (d) {
      addRow(diagTable, "Overhang area", pct(d.overhang_pct));
      addRow(diagTable, "Overhang angle", d.overhang_angle_deg != null ? d.overhang_angle_deg + "°" : "–");
      addRow(diagTable, "Thin-wall warnings", fmtNum(d.thin_wall_warnings));
      if (d.thin_wall_threshold_mm != null) {
        addRow(diagTable, "Thin-wall threshold", d.thin_wall_threshold_mm + " mm");
      }
    }
    if (fdel) {
      addRow(diagTable, "Chamfer distance", pct(fdel.chamfer_pct));
      addRow(diagTable, "Hausdorff distance", pct(fdel.hausdorff_pct));
      addRow(diagTable, "Volume change", pct(fdel.vol_change_pct));
      addRow(diagTable, "Face ratio", pct(fdel.face_ratio_pct));
    }
  }
}

function showError(err) {
  hidePanel("status");
  showPanel("error");
  errorMessage.textContent = err.message || "An error occurred.";
  // hint may be the full multi-line watchdog workaround text -- render verbatim.
  errorHint.textContent = err.hint || "";
  errorHint.style.display = err.hint ? "block" : "none";
}

// -------------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------------
function showPanel(name) {
  ({ status: statusPanel, error: errorPanel, result: resultPanel })[name].style.display = "block";
}
function hidePanel(name) {
  ({ status: statusPanel, error: errorPanel, result: resultPanel })[name].style.display = "none";
}
function addRow(table, label, value) {
  const tr = document.createElement("tr");
  tr.innerHTML = "<th>" + label + '</th><td class="val">' + value + "</td>";
  table.appendChild(tr);
}
function fmtNum(n) {
  if (n == null || isNaN(n)) return "–";
  return Number(n).toLocaleString();
}
function pct(n) {
  if (n == null || isNaN(n)) return "–";
  return Number(n).toFixed(2) + "%";
}
function formatDuration(sec) {
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return m > 0 ? m + "m " + s + "s" : s + "s";
}

init();

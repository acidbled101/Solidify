/* Training dashboard client.
 *
 * Incremental by design: the server hands back a byte offset into
 * metrics.jsonl and we only ever ask for what is past it. A dashboard left
 * open on a phone all weekend therefore transfers kilobytes per poll rather
 * than re-downloading a metrics file that has grown to tens of megabytes.
 *
 * No chart library: a phone on a tailnet cannot reach a CDN, and the CSP on
 * anything self-hosted would block it anyway. The charts are ~60 lines of
 * canvas.
 */

const COLORS = ['#4da3ff', '#3fbf7f', '#e0a33e', '#e5534b', '#b57bff', '#48c9d4'];

const state = {
  runId: null,
  offset: 0,
  train: [],     // {step, loss, lr, grad_norm, step_s, mem_gb}
  evals: [],     // {step, ...printability/fidelity/diversity}
  timer: null,
};

const $ = (id) => document.getElementById(id);
const fmt = (v, d = 4) => (v === null || v === undefined || Number.isNaN(v)) ? '—' : (+v).toFixed(d);
const fmtDur = (s) => {
  if (s == null) return '—';
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  return h ? `${h}h ${m}m` : `${m}m ${Math.floor(s % 60)}s`;
};

/* ---------------- charts ---------------- */

function drawLines(canvas, series, opts = {}) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = +canvas.getAttribute('height');
  canvas.width = w * dpr; canvas.height = h * dpr;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  const css = getComputedStyle(document.documentElement);
  const line = css.getPropertyValue('--line').trim() || '#232b36';
  const muted = css.getPropertyValue('--muted').trim() || '#8b98a8';

  const live = series.filter(s => s.points && s.points.length);
  if (!live.length) {
    ctx.fillStyle = muted; ctx.font = '12px system-ui';
    ctx.fillText('waiting for data…', 8, h / 2);
    return;
  }

  const pad = { l: 46, r: 8, t: 8, b: 20 };
  const xs = live.flatMap(s => s.points.map(p => p[0]));
  const ys = live.flatMap(s => s.points.map(p => p[1])).filter(Number.isFinite);
  let x0 = Math.min(...xs), x1 = Math.max(...xs);
  let y0 = Math.min(...ys), y1 = Math.max(...ys);
  if (x1 === x0) x1 = x0 + 1;
  // Pad the value axis so a flat series doesn't render as a line glued to an edge.
  if (y1 === y0) { y1 = y0 + Math.abs(y0 || 1) * 0.1; y0 -= Math.abs(y0 || 1) * 0.1; }
  const padY = (y1 - y0) * 0.08; y0 -= padY; y1 += padY;

  const X = (v) => pad.l + (v - x0) / (x1 - x0) * (w - pad.l - pad.r);
  const Y = (v) => h - pad.b - (v - y0) / (y1 - y0) * (h - pad.t - pad.b);

  ctx.strokeStyle = line; ctx.lineWidth = 1; ctx.fillStyle = muted;
  ctx.font = '10px ui-monospace, monospace';
  for (let i = 0; i <= 3; i++) {
    const v = y0 + (y1 - y0) * i / 3, y = Y(v);
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
    ctx.fillText(opts.fmtY ? opts.fmtY(v) : v.toPrecision(3), 4, y + 3);
  }
  ctx.fillText(String(Math.round(x0)), pad.l, h - 6);
  const lastLabel = String(Math.round(x1));
  ctx.fillText(lastLabel, w - pad.r - ctx.measureText(lastLabel).width, h - 6);

  live.forEach((s, i) => {
    ctx.strokeStyle = s.color || COLORS[i % COLORS.length];
    ctx.lineWidth = s.width || 1.6;
    ctx.globalAlpha = s.alpha ?? 1;
    ctx.beginPath();
    s.points.forEach(([px, py], k) => {
      if (!Number.isFinite(py)) return;
      const cx = X(px), cy = Y(py);
      k === 0 ? ctx.moveTo(cx, cy) : ctx.lineTo(cx, cy);
    });
    ctx.stroke();
    ctx.globalAlpha = 1;
  });
}

function legend(el, series) {
  el.innerHTML = series.filter(s => s.points && s.points.length)
    .map((s, i) => `<span><i style="background:${s.color || COLORS[i % COLORS.length]}"></i>${s.label}</span>`)
    .join('');
}

function ema(points, alpha = 0.08) {
  let acc = null;
  return points.map(([x, y]) => { acc = acc === null ? y : alpha * y + (1 - alpha) * acc; return [x, acc]; });
}

/* ---------------- data ---------------- */

async function loadRuns() {
  const r = await fetch('/api/runs').then(x => x.json());
  const sel = $('run-select');
  const prev = state.runId;
  sel.innerHTML = r.runs.map(x =>
    `<option value="${x.run_id}">${x.run_id}${x.alive ? ' ●' : ''}</option>`).join('');
  if (!r.runs.length) { sel.innerHTML = '<option>no runs</option>'; return; }
  state.runId = (prev && r.runs.some(x => x.run_id === prev)) ? prev : r.runs[0].run_id;
  sel.value = state.runId;
}

async function poll() {
  if (!state.runId) return;
  try {
    const m = await fetch(`/api/runs/${state.runId}/metrics?offset=${state.offset}`).then(x => x.json());
    state.offset = m.offset;
    for (const rec of m.records) {
      if (rec.kind === 'train') state.train.push(rec);
      else if (rec.kind === 'eval') state.evals.push(rec);
    }
    // Bound memory on a multi-day run: keep full-resolution recent history and
    // thin the old tail rather than letting the arrays grow without limit.
    if (state.train.length > 6000) state.train = state.train.filter((_, i) => i % 2 === 0);

    const st = await fetch(`/api/runs/${state.runId}/status`).then(x => x.json());
    render(st);
    renderGallery();
    renderLog();
  } catch (e) {
    $('alive').textContent = 'offline';
    $('alive').className = 'pill dead';
  }
}

/* ---------------- render ---------------- */

function render(st) {
  const status = st.status || {};
  const pill = $('alive');
  if (st.alive) { pill.textContent = status.state || 'live'; pill.className = 'pill live'; }
  else if (status.heartbeat) { pill.textContent = `stale ${fmtDur(st.stale_s)}`; pill.className = 'pill stale'; }
  else { pill.textContent = 'no heartbeat'; pill.className = 'pill dead'; }

  const last = state.train[state.train.length - 1] || {};
  const step = status.step ?? last.step;
  $('step').textContent = `step ${step ?? '—'}`;
  $('s-step').textContent = step ?? '—';
  $('s-eta').textContent = status.eta_s ? `eta ${fmtDur(status.eta_s)}` : (status.total_steps ? `of ${status.total_steps}` : '');

  $('s-loss').textContent = fmt(last.loss);
  $('s-elapsed').textContent = fmtDur(status.elapsed_s);
  $('s-rate').textContent = last.step_s ? `${last.step_s.toFixed(1)}s/step` : '';
  $('s-steptime').textContent = last.grad_norm != null ? `|g| ${fmt(last.grad_norm, 2)}` : '';
  $('s-mem').textContent = last.mem_gb != null ? `${last.mem_gb.toFixed(1)} GB` : '—';

  const ev = state.evals[state.evals.length - 1] || {};
  $('s-heldout').textContent = fmt(ev.heldout_loss);
  $('s-wt').textContent = ev.watertight_rate != null ? `${(ev.watertight_rate * 100).toFixed(0)}%` : '—';

  // Deltas against the previous eval, so drift is visible without reading charts.
  const prev = state.evals[state.evals.length - 2];
  if (prev && ev.heldout_loss != null && prev.heldout_loss != null) {
    const d = ev.heldout_loss - prev.heldout_loss;
    const el = $('s-heldout-d');
    el.textContent = `${d >= 0 ? '+' : ''}${d.toFixed(4)}`;
    el.className = d > 0 ? 'up' : 'down';
  }

  const lossPts = state.train.map(r => [r.step, r.loss]).filter(p => Number.isFinite(p[1]));
  const lossSeries = [
    { label: 'train (raw)', points: lossPts, color: COLORS[0], alpha: 0.28, width: 1 },
    { label: 'train (ema)', points: ema(lossPts), color: COLORS[0] },
    { label: 'held-out', points: state.evals.map(r => [r.step, r.heldout_loss]).filter(p => Number.isFinite(p[1])), color: COLORS[1] },
  ];
  drawLines($('c-loss'), lossSeries); legend($('l-loss'), lossSeries);

  const P = (k) => state.evals.map(r => [r.step, r[k]]).filter(p => Number.isFinite(p[1]));
  const printSeries = [
    { label: 'L_OH', points: P('L_OH'), color: COLORS[0] },
    { label: 'L_Th', points: P('L_Th'), color: COLORS[1] },
    { label: 'L_Topo', points: P('L_Topo'), color: COLORS[2] },
    { label: 'R_Detail', points: P('R_Detail'), color: COLORS[3] },
    { label: 'watertight rate', points: P('watertight_rate'), color: COLORS[4] },
  ];
  drawLines($('c-print'), printSeries); legend($('l-print'), printSeries);

  const fidSeries = [
    { label: 'image similarity', points: P('image_similarity'), color: COLORS[0] },
    { label: 'chamfer vs base', points: P('chamfer_vs_base'), color: COLORS[2] },
  ];
  drawLines($('c-fid'), fidSeries); legend($('l-fid'), fidSeries);

  const divSeries = [
    { label: 'mode entropy', points: P('mode_entropy'), color: COLORS[1] },
    { label: 'pairwise dist', points: P('sample_dispersion'), color: COLORS[4] },
  ];
  drawLines($('c-div'), divSeries); legend($('l-div'), divSeries);

  const c = st.control || {};
  $('control-state').textContent =
    `v${c.version ?? 0}  lr=${c.lr ?? 'default'}  eval_every=${c.eval_every ?? '—'}  ` +
    `ckpt_every=${c.checkpoint_every ?? '—'}${c.pause ? '  PAUSED' : ''}${c.stop ? '  STOPPING' : ''}` +
    (c.note ? `\n${c.note}` : '');
}

async function renderGallery() {
  const r = await fetch(`/api/runs/${state.runId}/samples`).then(x => x.json()).catch(() => null);
  if (!r || !r.checkpoints.length) return;
  const latest = r.checkpoints[0];
  if (!latest.images.length) return;
  $('gallery').innerHTML = latest.images.slice(0, 12).map(f =>
    `<figure><img loading="lazy" src="/api/runs/${state.runId}/samples/${latest.dir}/${f}">
     <figcaption>${f.replace(/\.(png|jpg|webp)$/i, '')}</figcaption></figure>`).join('');
}

async function renderLog() {
  const r = await fetch(`/api/runs/${state.runId}/log?tail=200`).then(x => x.json()).catch(() => null);
  if (!r) return;
  const el = $('log');
  const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
  el.textContent = r.lines.join('\n');
  if (atBottom) el.scrollTop = el.scrollHeight;
}

/* ---------------- control ---------------- */

async function sendControl(patch) {
  const r = await fetch(`/api/runs/${state.runId}/control`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(patch),
  });
  if (!r.ok) { alert('control failed: ' + r.status); return; }
  poll();
}

function wire() {
  $('run-select').onchange = (e) => {
    state.runId = e.target.value;
    state.offset = 0; state.train = []; state.evals = [];
    poll();
  };
  $('refresh').onclick = poll;
  $('b-apply').onclick = () => {
    const patch = {};
    const lr = $('k-lr').value.trim();
    if (lr) patch.lr = parseFloat(lr);
    const ev = $('k-eval').value.trim();
    if (ev) patch.eval_every = parseInt(ev, 10);
    const ck = $('k-ckpt').value.trim();
    if (ck) patch.checkpoint_every = parseInt(ck, 10);
    const note = $('k-note').value.trim();
    if (note) patch.note = note;
    if (!Object.keys(patch).length) return;
    sendControl(patch);
  };
  $('b-eval').onclick = () => sendControl({ eval_now: true });
  $('b-pause').onclick = () => sendControl({ pause: true });
  $('b-resume').onclick = () => sendControl({ pause: false });
  $('b-stop').onclick = () => {
    if (confirm('Stop after the next checkpoint? The run can be resumed later.')) sendControl({ stop: true });
  };
  // Redraw on rotate/resize: the canvases are sized from clientWidth.
  window.addEventListener('resize', () => { if (state.runId) poll(); });
}

async function boot() {
  wire();
  await loadRuns();
  const h = await fetch('/api/host').then(x => x.json()).catch(() => ({}));
  $('hosts').textContent =
    [h.tailscale_ip ? `tailscale  http://${h.tailscale_ip}:${location.port}` : 'tailscale  not detected',
     h.lan_ip ? `lan        http://${h.lan_ip}:${location.port}` : ''].filter(Boolean).join('\n');
  await poll();
  state.timer = setInterval(poll, 5000);
  // Phones suspend timers in background tabs; refresh immediately on return
  // so the dashboard is never showing hours-old numbers after a screen unlock.
  document.addEventListener('visibilitychange', () => { if (!document.hidden) poll(); });
}

boot();

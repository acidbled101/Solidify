/* Training dashboard client.
 *
 * Incremental by design: the server hands back a byte offset into
 * metrics.jsonl and we only ever ask for what is past it. A dashboard left
 * open on a phone all weekend therefore transfers kilobytes per poll rather
 * than re-downloading a metrics file that has grown to tens of megabytes.
 *
 * No chart library: a phone on a tailnet cannot reach a CDN, and the CSP on
 * anything self-hosted would block it anyway. The charts are ~80 lines of
 * canvas.
 *
 * SMOOTHNESS
 * ----------
 * The first version did every piece of work on every 5s tick: four fetches
 * (one of them a 200KB disk read), a full re-render of the log element, a
 * canvas reallocation per chart, and a replot of the entire history. On a
 * phone, against a Mac already saturated by a dataset build, that stutters.
 * Four changes fix it, in rough order of effect:
 *
 *   1. Split cadences. Metrics and status are the only things that need 5s.
 *      Logs and the sample gallery change on the scale of minutes.
 *   2. Downsample before drawing. A weekend run reaches tens of thousands of
 *      points; a phone chart is ~800 CSS px wide. Plotting more points than
 *      the canvas has columns is pure waste, so buckets are reduced to their
 *      min and max, which preserves the visual envelope (spikes stay spikes)
 *      at a bounded cost.
 *   3. Only touch the DOM and the canvas backing store when something
 *      changed. Assigning canvas.width reallocates and clears even when the
 *      value is identical, so it is guarded on a real size change.
 *   4. Never overlap polls, and never let a resize trigger a network request.
 */

const COLORS = ['#4da3ff', '#3fbf7f', '#e0a33e', '#e5534b', '#b57bff', '#48c9d4'];

const CADENCE = { metrics: 5000, log: 20000, gallery: 30000 };
const MAX_DRAW_POINTS = 900;   // ~1 point per CSS pixel of chart width

const state = {
  runId: null,
  offset: 0,
  train: [],
  evals: [],
  status: null,
  lastLog: 0,
  lastGallery: 0,
  lastGalleryDir: null,
  lastLogText: null,
  inFlight: false,
  dirty: false,
  drawQueued: false,
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

/** Reduce to at most `max` points, keeping each bucket's min and max so
 *  spikes survive. Plotting 20k points into an 800px canvas is invisible
 *  work that still costs a path segment each. */
function downsample(points, max = MAX_DRAW_POINTS) {
  if (points.length <= max) return points;
  const bucket = Math.ceil(points.length / (max / 2));
  const out = [];
  for (let i = 0; i < points.length; i += bucket) {
    let mn = null, mx = null;
    for (let k = i; k < Math.min(i + bucket, points.length); k++) {
      const p = points[k];
      if (!Number.isFinite(p[1])) continue;
      if (!mn || p[1] < mn[1]) mn = p;
      if (!mx || p[1] > mx[1]) mx = p;
    }
    if (!mn) continue;
    // Emit in x-order so the polyline never doubles back on itself.
    if (mn[0] <= mx[0]) { out.push(mn); if (mx !== mn) out.push(mx); }
    else { out.push(mx); out.push(mn); }
  }
  return out;
}

/** Size the backing store only when it actually changed. */
function fitCanvas(cv) {
  // Cap DPR at 2: many phones report 3, which triples fill cost for no
  // perceptible gain on a line chart.
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = cv.clientWidth || 300;
  const h = +cv.getAttribute('height');
  const W = Math.round(w * dpr), H = Math.round(h * dpr);
  if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}

function drawLines(canvas, series, opts = {}) {
  const { ctx, w, h } = fitCanvas(canvas);

  const css = getComputedStyle(document.documentElement);
  const line = css.getPropertyValue('--line').trim() || '#232b36';
  const muted = css.getPropertyValue('--muted').trim() || '#8b98a8';

  const live = series
    .map(s => ({ ...s, points: downsample((s.points || []).filter(p => Number.isFinite(p[1]))) }))
    .filter(s => s.points.length);

  if (!live.length) {
    ctx.fillStyle = muted; ctx.font = '12px system-ui';
    ctx.fillText('waiting for data…', 8, h / 2);
    return;
  }

  const pad = { l: 46, r: 8, t: 8, b: 20 };
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const s of live) for (const [x, y] of s.points) {
    if (x < x0) x0 = x; if (x > x1) x1 = x;
    if (y < y0) y0 = y; if (y > y1) y1 = y;
  }
  if (x1 === x0) x1 = x0 + 1;
  if (y1 === y0) { const e = Math.abs(y0 || 1) * 0.1; y1 = y0 + e; y0 -= e; }
  const padY = (y1 - y0) * 0.08; y0 -= padY; y1 += padY;

  const X = (v) => pad.l + (v - x0) / (x1 - x0) * (w - pad.l - pad.r);
  const Y = (v) => h - pad.b - (v - y0) / (y1 - y0) * (h - pad.t - pad.b);

  ctx.strokeStyle = line; ctx.lineWidth = 1; ctx.fillStyle = muted;
  ctx.font = '10px ui-monospace, monospace';
  ctx.beginPath();
  for (let i = 0; i <= 3; i++) {
    const y = Y(y0 + (y1 - y0) * i / 3);
    ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y);
  }
  ctx.stroke();
  for (let i = 0; i <= 3; i++) {
    const v = y0 + (y1 - y0) * i / 3;
    ctx.fillText(opts.fmtY ? opts.fmtY(v) : v.toPrecision(3), 4, Y(v) + 3);
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
      const cx = X(px), cy = Y(py);
      k === 0 ? ctx.moveTo(cx, cy) : ctx.lineTo(cx, cy);
    });
    ctx.stroke();
    ctx.globalAlpha = 1;
  });
}

function legend(el, series) {
  const html = series.filter(s => s.points && s.points.length)
    .map((s, i) => `<span><i style="background:${s.color || COLORS[i % COLORS.length]}"></i>${s.label}</span>`)
    .join('');
  if (el.innerHTML !== html) el.innerHTML = html;   // avoid needless reflow
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
  if (!r.runs.length) { sel.innerHTML = '<option>no runs</option>'; return; }
  const html = r.runs.map(x => `<option value="${x.run_id}">${x.run_id}${x.alive ? ' ●' : ''}</option>`).join('');
  if (sel.innerHTML !== html) sel.innerHTML = html;
  state.runId = (prev && r.runs.some(x => x.run_id === prev)) ? prev : r.runs[0].run_id;
  sel.value = state.runId;
}

async function tick() {
  if (!state.runId || state.inFlight || document.hidden) return;
  state.inFlight = true;
  try {
    const [m, st] = await Promise.all([
      fetch(`/api/runs/${state.runId}/metrics?offset=${state.offset}`).then(x => x.json()),
      fetch(`/api/runs/${state.runId}/status`).then(x => x.json()),
    ]);

    if (m.records.length) {
      state.offset = m.offset;
      for (const rec of m.records) {
        if (rec.kind === 'train') state.train.push(rec);
        else if (rec.kind === 'eval') state.evals.push(rec);
      }
      state.dirty = true;
    }
    state.status = st;
    scheduleDraw();

    const now = Date.now();
    if (now - state.lastLog > CADENCE.log) { state.lastLog = now; refreshLog(); }
    if (now - state.lastGallery > CADENCE.gallery) { state.lastGallery = now; refreshGallery(); }

    $('alive').classList.remove('offline');
  } catch (e) {
    $('alive').textContent = 'offline';
    $('alive').className = 'pill dead';
  } finally {
    state.inFlight = false;
  }
}

function scheduleDraw() {
  if (state.drawQueued) return;
  state.drawQueued = true;
  requestAnimationFrame(() => { state.drawQueued = false; render(); });
}

/* ---------------- render ---------------- */

function setText(el, text) { if (el.textContent !== text) el.textContent = text; }

function render() {
  const st = state.status || {};
  const status = st.status || {};
  const pill = $('alive');
  if (st.alive) { setText(pill, status.state || 'live'); pill.className = 'pill live'; }
  else if (status.heartbeat) { setText(pill, `stale ${fmtDur(st.stale_s)}`); pill.className = 'pill stale'; }
  else { setText(pill, 'no heartbeat'); pill.className = 'pill dead'; }

  const last = state.train[state.train.length - 1] || {};
  const step = status.step ?? last.step;
  setText($('step'), `step ${step ?? '—'}`);
  setText($('s-step'), String(step ?? '—'));
  setText($('s-eta'), status.eta_s ? `eta ${fmtDur(status.eta_s)}` : (status.total_steps ? `of ${status.total_steps}` : ''));
  setText($('s-loss'), fmt(last.loss));
  setText($('s-elapsed'), fmtDur(status.elapsed_s));
  setText($('s-rate'), last.step_s ? `${last.step_s.toFixed(1)}s/step` : '');
  setText($('s-steptime'), last.grad_norm != null ? `|g| ${fmt(last.grad_norm, 2)}` : '');
  setText($('s-mem'), last.mem_gb != null ? `${last.mem_gb.toFixed(1)} GB` : '—');

  const ev = state.evals[state.evals.length - 1] || {};
  setText($('s-heldout'), fmt(ev.heldout_loss));
  setText($('s-wt'), ev.watertight_rate != null ? `${(ev.watertight_rate * 100).toFixed(0)}%` : '—');

  const prev = state.evals[state.evals.length - 2];
  if (prev && ev.heldout_loss != null && prev.heldout_loss != null) {
    const d = ev.heldout_loss - prev.heldout_loss;
    const el = $('s-heldout-d');
    setText(el, `${d >= 0 ? '+' : ''}${d.toFixed(4)}`);
    el.className = d > 0 ? 'up' : 'down';
  }

  // Charts are the expensive part; skip entirely when no new records arrived.
  if (state.dirty) {
    state.dirty = false;
    const lossPts = state.train.map(r => [r.step, r.loss]);
    const lossSeries = [
      { label: 'train (raw)', points: lossPts, color: COLORS[0], alpha: 0.28, width: 1 },
      { label: 'train (ema)', points: ema(lossPts.filter(p => Number.isFinite(p[1]))), color: COLORS[0] },
      { label: 'held-out', points: state.evals.map(r => [r.step, r.heldout_loss]), color: COLORS[1] },
    ];
    drawLines($('c-loss'), lossSeries); legend($('l-loss'), lossSeries);

    const P = (k) => state.evals.map(r => [r.step, r[k]]);
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
  }

  const c = st.control || {};
  setText($('control-state'),
    `v${c.version ?? 0}  lr=${c.lr ?? 'default'}  eval_every=${c.eval_every ?? '—'}  ` +
    `ckpt_every=${c.checkpoint_every ?? '—'}${c.pause ? '  PAUSED' : ''}${c.stop ? '  STOPPING' : ''}` +
    (c.note ? `\n${c.note}` : ''));
}

/** Redraw from cached data only — no network. Used on resize/rotate. */
function redrawOnly() { state.dirty = true; scheduleDraw(); }

async function refreshGallery() {
  const r = await fetch(`/api/runs/${state.runId}/samples`).then(x => x.json()).catch(() => null);
  if (!r || !r.checkpoints.length) return;
  const latest = r.checkpoints[0];
  if (!latest.images.length || latest.dir === state.lastGalleryDir) return;
  state.lastGalleryDir = latest.dir;
  $('gallery').innerHTML = latest.images.slice(0, 12).map(f =>
    `<figure><img loading="lazy" decoding="async" src="/api/runs/${state.runId}/samples/${latest.dir}/${f}">
     <figcaption>${f.replace(/\.(png|jpg|webp)$/i, '')}</figcaption></figure>`).join('');
}

async function refreshLog() {
  const r = await fetch(`/api/runs/${state.runId}/log?tail=200`).then(x => x.json()).catch(() => null);
  if (!r) return;
  const text = r.lines.join('\n');
  if (text === state.lastLogText) return;   // re-rendering a <pre> forces layout
  state.lastLogText = text;
  const el = $('log');
  const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
  el.textContent = text;
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
  tick();
}

function wire() {
  $('run-select').onchange = (e) => {
    state.runId = e.target.value;
    Object.assign(state, {
      offset: 0, train: [], evals: [], lastGalleryDir: null,
      lastLogText: null, lastLog: 0, lastGallery: 0, dirty: true,
    });
    tick();
  };
  $('refresh').onclick = () => { state.lastLog = 0; state.lastGallery = 0; tick(); };
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

  // Resize must never hit the network: rotating a phone fires this many times.
  let rt = null;
  window.addEventListener('resize', () => {
    clearTimeout(rt);
    rt = setTimeout(redrawOnly, 150);
  });

  // Phones suspend timers in background tabs. Refresh on return so the
  // dashboard is never showing hours-old numbers after a screen unlock.
  document.addEventListener('visibilitychange', () => { if (!document.hidden) tick(); });
}

async function boot() {
  wire();
  await loadRuns();
  const h = await fetch('/api/host').then(x => x.json()).catch(() => ({}));
  $('hosts').textContent =
    [h.tailscale_ip ? `tailscale  http://${h.tailscale_ip}:${location.port}` : 'tailscale  not detected',
     h.lan_ip ? `lan        http://${h.lan_ip}:${location.port}` : ''].filter(Boolean).join('\n');
  state.dirty = true;
  await tick();
  state.timer = setInterval(tick, CADENCE.metrics);
  // Pick up newly created runs without paying for it on every metrics tick.
  setInterval(loadRuns, 30000);
}

boot();

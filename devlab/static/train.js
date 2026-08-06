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

const CADENCE = { metrics: 5000, log: 20000, gallery: 30000, prog: 30000 };
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
  mode: null,
  baseline: null,      // step-0 eval with adapters disabled
  logScale: false,
  prog: null,          // {objects, steps, frames}
  progObject: null,
  progIdx: 0,
  progTimer: null,
  lastProg: 0,
  timer: null,
};

const $ = (id) => document.getElementById(id);
const fmt = (v, d = 4) => (v === null || v === undefined || Number.isNaN(v)) ? '—' : (+v).toFixed(d);
const median = (a) => { const b=[...a].sort((x,y)=>x-y); return b.length? (b.length%2? b[(b.length-1)/2] : (b[b.length/2-1]+b[b.length/2])/2) : null; };
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

/* Axis bounds are sticky. Recomputing exact min/max every poll makes the whole
 * chart shift each time a point arrives, so a curve appears to wobble even
 * when the numbers are settled. Instead: round out to "nice" bounds and keep
 * them until the data actually leaves, or until it occupies so little of the
 * range that the chart has gone flat. */
const AXIS = new Map();

function niceBounds(key, lo, hi) {
  if (!(hi > lo)) { const e = Math.abs(hi || 1) * 0.1 || 1; lo = hi - e; hi = hi + e; }
  const prev = AXIS.get(key);
  if (prev && lo >= prev.lo && hi <= prev.hi) {
    // Still inside. Keep unless the data now fills less than 40% of the
    // range, in which case the chart has visibly flattened and a rescale is
    // more honest than a stable-but-useless axis.
    if ((hi - lo) > 0.4 * (prev.hi - prev.lo)) return prev;
  }
  const span = hi - lo;
  const pad = span * 0.1;
  lo -= pad; hi += pad;
  const raw = (hi - lo) / 4;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].find(m => m * mag >= raw) * mag;
  const out = { lo: Math.floor(lo / step) * step, hi: Math.ceil(hi / step) * step, step };
  AXIS.set(key, out);
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

  let prepared = series.map(s => ({ ...s, points: (s.points || []).filter(p => Number.isFinite(p[1])) }));
  if (opts.log) {
    // Flow-matching loss spans orders of magnitude across noise levels; a
    // linear axis hides everything below the early spike.
    prepared = prepared.map(s => ({ ...s, points: s.points.filter(p => p[1] > 0).map(p => [p[0], Math.log10(p[1])]) }));
  }
  const live = prepared
    .map(s => ({ ...s, points: downsample(s.points) }))
    .filter(s => s.points.length);

  if (!live.length) {
    ctx.fillStyle = muted; ctx.font = '12px system-ui';
    ctx.fillText('waiting for data…', 8, h / 2);
    return;
  }

  const pad = { l: 46, r: 8, t: 8, b: 20 };
  let x0 = Infinity, x1 = -Infinity, dlo = Infinity, dhi = -Infinity;
  for (const s of live) for (const [x, y] of s.points) {
    if (x < x0) x0 = x; if (x > x1) x1 = x;
    if (y < dlo) dlo = y; if (y > dhi) dhi = y;
  }
  if (x1 === x0) x1 = x0 + 1;
  const b = niceBounds(opts.key || canvas.id, dlo, dhi);
  const y0 = b.lo, y1 = b.hi;

  const X = (v) => pad.l + (v - x0) / (x1 - x0) * (w - pad.l - pad.r);
  const Y = (v) => h - pad.b - (v - y0) / (y1 - y0) * (h - pad.t - pad.b);

  ctx.strokeStyle = line; ctx.lineWidth = 1; ctx.fillStyle = muted;
  ctx.font = '10px ui-monospace, monospace';
  const ticks = [];
  for (let v = y0; v <= y1 + b.step * 1e-6; v += b.step) ticks.push(v);
  ctx.beginPath();
  for (const v of ticks) { const y = Y(v); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); }
  ctx.stroke();
  const fy = opts.fmtY || (opts.log
    ? ((v) => Math.pow(10, v).toPrecision(2))
    : ((v) => Math.abs(v) >= 1000 ? v.toLocaleString() : String(+v.toFixed(6))));
  for (const v of ticks) ctx.fillText(fy(v), 4, Y(v) + 3);
  ctx.fillText(String(Math.round(x0)), pad.l, h - 6);
  const lastLabel = String(Math.round(x1));
  ctx.fillText(lastLabel, w - pad.r - ctx.measureText(lastLabel).width, h - 6);

  // Baseline reference lines. A printability curve without the base model's
  // value on the same axes cannot tell you whether training helped.
  if (opts.refs) {
    ctx.save();
    ctx.setLineDash([4, 4]);
    ctx.lineWidth = 1.2;
    for (const r of opts.refs) {
      if (!Number.isFinite(r.value)) continue;
      ctx.strokeStyle = r.color;
      ctx.globalAlpha = 0.7;
      const y = Y(r.value);
      ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
    }
    ctx.restore();
  }

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
        else if (rec.kind === 'baseline') state.baseline = rec;
      }
      state.dirty = true;
    }
    state.status = st;
    scheduleDraw();

    const now = Date.now();
    if (now - state.lastLog > CADENCE.log) { state.lastLog = now; refreshLog(); }
    if (now - state.lastGallery > CADENCE.gallery) { state.lastGallery = now; refreshGallery(); }
    if (now - state.lastProg > CADENCE.prog) { state.lastProg = now; refreshProgression(); }

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

function setText(el, text) { if (el && el.textContent !== text) el.textContent = text; }

/** Which panels make sense for this run. A dataset build has no loss, no
 *  held-out set and no samples; showing it four permanently-empty charts and
 *  a training control panel is worse than showing nothing. */
function runKind() {
  const meta = (state.status && state.status.meta) || {};
  return ((meta.config || {}).kind === 'dataset_build') ? 'dataset' : 'train';
}

function applyMode(mode) {
  if (state.mode === mode) return;
  state.mode = mode;
  document.querySelectorAll('[data-mode]').forEach(el => {
    el.classList.toggle('hidden-mode', el.dataset.mode !== mode);
  });
  AXIS.clear();
}

function statCards(cards) {
  const html = cards.map(c =>
    `<div class="card stat"><label>${c.label}</label><b>${c.value}</b><small class="${c.cls || ''}">${c.sub || ''}</small></div>`
  ).join('');
  const el = $('headline');
  if (el.innerHTML !== html) el.innerHTML = html;
}

function render() {
  const st = state.status || {};
  const status = st.status || {};
  const pill = $('alive');
  if (st.alive) { setText(pill, status.state || 'live'); pill.className = 'pill live'; }
  else if (status.heartbeat) { setText(pill, `stale ${fmtDur(st.stale_s)}`); pill.className = 'pill stale'; }
  else { setText(pill, 'no heartbeat'); pill.className = 'pill dead'; }

  const mode = runKind();
  applyMode(mode);

  const last = state.train[state.train.length - 1] || {};
  const step = status.step ?? last.step;
  setText($('step'), (mode === 'dataset' ? 'model ' : 'step ') + (step ?? '—'));

  // Progress bar whenever the run has a known end.
  const total = status.total_steps;
  const wrap = $('progress-wrap');
  if (total && step != null) {
    wrap.hidden = false;
    const pct = Math.max(0, Math.min(100, 100 * step / total));
    $('progress-bar').style.width = pct.toFixed(1) + '%';
    setText($('progress-text'),
      `${step} / ${total}  (${pct.toFixed(1)}%)` +
      (status.eta_s ? `   eta ${fmtDur(status.eta_s)}` : '') +
      (status.failures ? `   ${status.failures} failed` : ''));
  } else {
    wrap.hidden = true;
  }

  if (mode === 'dataset') {
    const toks = state.train.map(r => r.n_tokens ?? r.loss).filter(Number.isFinite);
    const secs = state.train.map(r => r.seconds ?? r.step_s).filter(Number.isFinite);
    const mean = (a) => a.length ? a.reduce((x, y) => x + y, 0) / a.length : null;
    statCards([
      { label: 'Built', value: String(step ?? '—'), sub: total ? `of ${total}` : '' },
      { label: 'Failed', value: String(status.failures ?? 0), cls: status.failures ? 'up' : '' },
      { label: 'Median voxels', value: toks.length ? String(Math.round(median(toks))) : '—', sub: 'per latent' },
      { label: 'Sec / model', value: secs.length ? mean(secs).toFixed(1) : '—', sub: 'mean' },
      { label: 'Elapsed', value: fmtDur(status.elapsed_s), sub: status.state || '' },
      { label: 'ETA', value: status.eta_s ? fmtDur(status.eta_s) : '—', sub: 'remaining' },
    ]);
    if (state.dirty) {
      state.dirty = false;
      const tokSeries = [{ label: 'voxels per model', points: state.train.map(r => [r.step, r.n_tokens ?? r.loss]), color: COLORS[4] }];
      drawLines($('c-tokens'), tokSeries, { key: 'tokens' }); legend($('l-tokens'), tokSeries);
      const secSeries = [{ label: 'seconds per model', points: state.train.map(r => [r.step, r.seconds ?? r.step_s]), color: COLORS[2] }];
      drawLines($('c-secs'), secSeries, { key: 'secs' }); legend($('l-secs'), secSeries);
    }
    setText($('control-state'), '');
    return;
  }

  const ev = state.evals[state.evals.length - 1] || {};
  const prev = state.evals[state.evals.length - 2];
  const base = state.baseline || {};
  const delta = (k) => (ev[k] != null && base[k] != null) ? ev[k] - base[k] : null;

  let heldoutSub = '', heldoutCls = '';
  if (prev && ev.heldout_loss != null && prev.heldout_loss != null) {
    const d = ev.heldout_loss - prev.heldout_loss;
    heldoutSub = `${d >= 0 ? '+' : ''}${d.toFixed(4)}`;
    heldoutCls = d > 0 ? 'up' : 'down';
  }
  const thru = last.step_s ? (1 / last.step_s).toFixed(3) : null;
  statCards([
    { label: 'Step', value: String(step ?? '—'), sub: total ? `of ${total}` : '' },
    { label: 'Epoch', value: last.epoch != null ? last.epoch.toFixed(2) : '—',
      sub: last.samples_seen != null ? `${last.samples_seen} samples` : '' },
    { label: 'Train loss', value: fmt(last.loss_ema ?? last.loss), sub: 'ema' },
    { label: 'Held-out', value: fmt(ev.heldout_loss), sub: heldoutSub, cls: heldoutCls },
    { label: 'LR', value: last.lr != null ? last.lr.toExponential(1) : '—',
      sub: last.grad_norm != null ? `|g| ${fmt(last.grad_norm, 2)}` : '' },
    { label: 'Throughput', value: last.step_s ? `${last.step_s.toFixed(1)}s` : '—',
      sub: thru ? `${thru} steps/s` : '' },
    { label: 'Memory', value: last.mem_gb != null ? `${last.mem_gb.toFixed(1)} GB` : '—',
      sub: last.mem_peak_gb != null ? `peak ${last.mem_peak_gb.toFixed(1)}` : '' },
    { label: 'ETA', value: status.eta_s ? fmtDur(status.eta_s) : '—', sub: fmtDur(status.elapsed_s) + ' elapsed' },
  ]);

  // Baseline deltas. Lower is better for the three penalties; for watertight
  // rate higher is better. R_Detail is a fidelity term, not a printability
  // one, so it is reported without a better/worse verdict.
  const chips = [
    ['L_OH', -1], ['L_Th', -1], ['L_Topo', -1], ['watertight_rate', +1], ['R_Detail', 0],
  ].map(([k, dir]) => {
    const d = delta(k);
    if (d == null) return '';
    const cls = dir === 0 ? '' : ((d * dir > 0) ? 'better' : 'worse');
    return `<span class="${cls}">${k} ${d >= 0 ? '+' : ''}${d.toFixed(4)} vs base</span>`;
  }).filter(Boolean).join('');
  const dEl = $('print-delta');
  if (dEl && dEl.innerHTML !== chips) dEl.innerHTML = chips;

  if (state.dirty) {
    state.dirty = false;
    const lossPts = state.train.map(r => [r.step, r.loss]);
    const lossSeries = [
      { label: 'train (raw)', points: lossPts, color: COLORS[0], alpha: 0.25, width: 1 },
      { label: 'train (ema)', points: state.train.map(r => [r.step, r.loss_ema]), color: COLORS[0] },
      { label: 'held-out', points: state.evals.map(r => [r.step, r.heldout_loss]), color: COLORS[1] },
    ];
    drawLines($('c-loss'), lossSeries, { key: 'loss', log: state.logScale });
    legend($('l-loss'), lossSeries);

    const optimSeries = [
      { label: 'grad norm', points: state.train.map(r => [r.step, r.grad_norm]), color: COLORS[2] },
    ];
    drawLines($('c-optim'), optimSeries, { key: 'optim' }); legend($('l-optim'), optimSeries);

    // Loss by noise level, latest step that carried bins.
    let binRec = null;
    for (let i = state.train.length - 1; i >= 0 && !binRec; i--) {
      if (Object.keys(state.train[i]).some(k => k.startsWith('t_bin_'))) binRec = state.train[i];
    }
    const binSeries = binRec ? [{
      label: 'loss vs t (0=data, 1=noise)',
      points: Array.from({ length: 10 }, (_, i) => [i / 10, binRec[`t_bin_${i}`]])
                   .filter(p => Number.isFinite(p[1])),
      color: COLORS[5],
    }] : [{ label: '', points: [] }];
    drawLines($('c-bins'), binSeries, { key: 'bins' }); legend($('l-bins'), binSeries);

    const P = (k) => state.evals.map(r => [r.step, r[k]]);
    const printSeries = [
      { label: 'L_OH', points: P('L_OH'), color: COLORS[0] },
      { label: 'L_Th', points: P('L_Th'), color: COLORS[1] },
      { label: 'L_Topo', points: P('L_Topo'), color: COLORS[2] },
      { label: 'R_Detail', points: P('R_Detail'), color: COLORS[3] },
      { label: 'watertight', points: P('watertight_rate'), color: COLORS[4] },
    ];
    const refs = printSeries.map(x => ({ value: base[x.label === 'watertight' ? 'watertight_rate' : x.label], color: x.color }));
    drawLines($('c-print'), printSeries, { key: 'print', refs });
    legend($('l-print'), printSeries);

    const fidSeries = [
      { label: 'image similarity', points: P('image_similarity'), color: COLORS[0] },
      { label: 'chamfer vs base', points: P('chamfer_vs_base'), color: COLORS[2] },
    ];
    drawLines($('c-fid'), fidSeries, { key: 'fid' }); legend($('l-fid'), fidSeries);

    const divSeries = [
      { label: 'sample dispersion', points: P('sample_dispersion'), color: COLORS[1] },
      { label: 'mode entropy', points: P('mode_entropy'), color: COLORS[4] },
    ];
    drawLines($('c-div'), divSeries, { key: 'div' }); legend($('l-div'), divSeries);
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

async function refreshProgression() {
  const r = await fetch(`/api/runs/${state.runId}/progression`).then(x => x.json()).catch(() => null);
  if (!r || !r.objects.length) return;
  const changed = !state.prog || state.prog.objects.length !== r.objects.length ||
                  state.prog.steps.length !== r.steps.length;
  state.prog = r;
  if (!state.progObject || !r.objects.includes(state.progObject)) state.progObject = r.objects[0];

  const sel = $('prog-object');
  const html = r.objects.map(o => `<option value="${o}">${o}</option>`).join('');
  if (sel.innerHTML !== html) { sel.innerHTML = html; sel.value = state.progObject; }

  const frames = framesFor(state.progObject);
  const slider = $('prog-slider');
  const wasAtEnd = +slider.value >= +slider.max;
  slider.max = String(Math.max(0, frames.length - 1));
  // Follow the newest checkpoint only if the user was already there; don't
  // yank the slider out from under someone inspecting an earlier step.
  if (changed && wasAtEnd) state.progIdx = frames.length - 1;
  state.progIdx = Math.min(state.progIdx, Math.max(0, frames.length - 1));
  slider.value = String(state.progIdx);
  showProgFrame();
}

function framesFor(obj) {
  if (!state.prog || !obj) return [];
  const m = state.prog.frames[obj] || {};
  return Object.keys(m).map(Number).sort((a, b) => a - b).map(step => ({ step, path: m[step] }));
}

function showProgFrame() {
  const frames = framesFor(state.progObject);
  if (!frames.length) return;
  const f = frames[Math.min(state.progIdx, frames.length - 1)];
  const url = `/api/runs/${state.runId}/samples/${f.path}`;
  const img = $('prog-img');
  if (img.getAttribute('src') !== url) img.src = url;
  setText($('prog-step'), `step ${f.step}`);
  // Ghost = step 0 (the base model), when we have it.
  const ghost = $('prog-ghost');
  const zero = frames.find(x => x.step === 0);
  const gurl = zero ? `/api/runs/${state.runId}/samples/${zero.path}` : '';
  if (ghost.getAttribute('src') !== gurl) { if (gurl) ghost.src = gurl; else ghost.removeAttribute('src'); }
  ghost.style.display = (zero && f.step !== 0) ? '' : 'none';
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
      lastLogText: null, lastLog: 0, lastGallery: 0, lastProg: 0, dirty: true,
      mode: null, baseline: null, prog: null, progObject: null, progIdx: 0,
    });
    AXIS.clear();
    tick();
  };
  $('refresh').onclick = () => { state.lastLog = 0; state.lastGallery = 0; state.lastProg = 0; tick(); };
  $('loss-scale').onclick = (e) => {
    state.logScale = !state.logScale;
    e.target.classList.toggle('on', state.logScale);
    AXIS.delete('loss'); redrawOnly();
  };
  $('prog-object').onchange = (e) => { state.progObject = e.target.value; state.progIdx = 0; refreshProgression(); };
  $('prog-slider').oninput = (e) => { state.progIdx = +e.target.value; showProgFrame(); };
  $('prog-play').onclick = () => {
    if (state.progTimer) { clearInterval(state.progTimer); state.progTimer = null; $('prog-play').textContent = '▶'; return; }
    $('prog-play').textContent = '❚❚';
    state.progTimer = setInterval(() => {
      const n = framesFor(state.progObject).length;
      if (!n) return;
      state.progIdx = (state.progIdx + 1) % n;
      $('prog-slider').value = String(state.progIdx);
      showProgFrame();
    }, 700);
  };
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
  refreshProgression();
  state.timer = setInterval(tick, CADENCE.metrics);
  // Pick up newly created runs without paying for it on every metrics tick.
  setInterval(loadRuns, 30000);
}

boot();

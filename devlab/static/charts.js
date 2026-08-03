// charts.js -- hand-rolled SVG chart primitives for DPO Inspector.
// No charting library: every event batch is tiny (a few dozen events over a
// ~20min run), so a full re-render per update is simplest and cheap. Every
// function takes a container element + data and replaces its contents.
"use strict";

const NS = "http://www.w3.org/2000/svg";
const COLOR = {
  reference: "#3ED6C4",   // teal -- the unperturbed branch, held constant across every panel
  delta: "#FF9F45",       // amber -- the steered branch
  deltaFinal: "#FF6B4A",  // deeper amber -- post-gradient-steps candidate
  grid: "#25313B",
  axis: "#4A5B66",
  text: "#B9C7CF",
  textDim: "#6C7C86",
  winner: "#7CE38B",
  bad: "#FF6B6B",
};

function el(tag, attrs, parent) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}
function fmt(n, d = 3) {
  if (n === null || n === undefined || Number.isNaN(n)) return "–";
  if (Math.abs(n) >= 1000) return n.toLocaleString(undefined, { maximumFractionDigits: 0 });
  return Number(n).toFixed(d);
}
function clear(container) {
  container.innerHTML = "";
}

// ---------------------------------------------------------------------------
// Line chart: N series over a shared x-index, e.g. loss/proxy/rms per
// gradient step. Draws a light grid, axis ticks, an optional horizontal
// reference line (e.g. the trust region ceiling or the reference proxy).
// ---------------------------------------------------------------------------
function lineChart(container, opts) {
  const { series, width = 380, height = 160, refLines = [], yLabel = "", title = "" } = opts;
  clear(container);
  const pad = { l: 46, r: 14, t: title ? 26 : 10, b: 24 };
  const svg = el("svg", { width, height, viewBox: `0 0 ${width} ${height}`, class: "chart" }, container);

  if (title) {
    el("text", { x: pad.l, y: 16, class: "chart-title" }, svg).textContent = title;
  }

  const allY = series.flatMap((s) => s.values).concat(refLines.map((r) => r.value));
  const allX = series.length ? series[0].values.map((_, i) => i) : [0];
  if (!allY.length || series.every((s) => s.values.length === 0)) {
    el("text", { x: width / 2, y: height / 2, class: "chart-empty", "text-anchor": "middle" }, svg).textContent = "no data yet";
    return;
  }
  let yMin = Math.min(...allY), yMax = Math.max(...allY);
  if (yMin === yMax) { yMin -= 1; yMax += 1; }
  const yPad = (yMax - yMin) * 0.12;
  yMin -= yPad; yMax += yPad;
  const xMax = Math.max(1, allX.length - 1);

  const xs = (i) => pad.l + (i / xMax) * (width - pad.l - pad.r);
  const ys = (v) => height - pad.b - ((v - yMin) / (yMax - yMin)) * (height - pad.t - pad.b);

  // grid
  const nTicksY = 4;
  for (let i = 0; i <= nTicksY; i++) {
    const v = yMin + (i / nTicksY) * (yMax - yMin);
    const y = ys(v);
    el("line", { x1: pad.l, x2: width - pad.r, y1: y, y2: y, class: "chart-grid" }, svg);
    el("text", { x: pad.l - 6, y: y + 3, class: "chart-tick", "text-anchor": "end" }, svg).textContent = fmt(v, Math.abs(v) < 1 ? 4 : 1);
  }
  el("line", { x1: pad.l, x2: pad.l, y1: pad.t, y2: height - pad.b, class: "chart-axis" }, svg);
  el("line", { x1: pad.l, x2: width - pad.r, y1: height - pad.b, y2: height - pad.b, class: "chart-axis" }, svg);

  refLines.forEach((r) => {
    const y = ys(r.value);
    el("line", { x1: pad.l, x2: width - pad.r, y1: y, y2: y, class: "chart-refline", stroke: r.color || COLOR.textDim }, svg);
    el("text", { x: width - pad.r, y: y - 3, class: "chart-reflabel", "text-anchor": "end", fill: r.color || COLOR.textDim }, svg).textContent = r.label;
  });

  series.forEach((s) => {
    if (!s.values.length) return;
    const pts = s.values.map((v, i) => `${xs(i)},${ys(v)}`).join(" ");
    el("polyline", { points: pts, fill: "none", stroke: s.color, "stroke-width": 2, class: "chart-line" }, svg);
    s.values.forEach((v, i) => {
      el("circle", { cx: xs(i), cy: ys(v), r: 3, fill: s.color, class: "chart-point" }, svg);
    });
  });

  if (yLabel) {
    el("text", { x: 12, y: (pad.t + height - pad.b) / 2, class: "chart-ylabel", transform: `rotate(-90 12 ${(pad.t + height - pad.b) / 2})` }, svg).textContent = yLabel;
  }

  // x ticks (step indices)
  allX.forEach((i) => {
    el("text", { x: xs(i), y: height - pad.b + 15, class: "chart-tick", "text-anchor": "middle" }, svg).textContent = i;
  });
}

// ---------------------------------------------------------------------------
// Histogram: {n, counts, edges, min, max, mean} from devlab.trace._histogram,
// with optional vertical reference line(s) (e.g. theta_crit, d_min).
// ---------------------------------------------------------------------------
function histogram(container, hist, opts = {}) {
  const { width = 300, height = 130, color = COLOR.reference, title = "", refLines = [], xFmt = (v) => fmt(v, 2) } = opts;
  clear(container);
  const pad = { l: 8, r: 8, t: title ? 20 : 6, b: 20 };
  const svg = el("svg", { width, height, viewBox: `0 0 ${width} ${height}`, class: "chart" }, container);
  if (title) el("text", { x: pad.l, y: 13, class: "chart-title-sm" }, svg).textContent = title;

  if (!hist || !hist.n) {
    el("text", { x: width / 2, y: height / 2, class: "chart-empty", "text-anchor": "middle" }, svg).textContent = "no data";
    return;
  }
  const counts = hist.counts, edges = hist.edges;
  const maxCount = Math.max(...counts, 1);
  const n = counts.length;
  const barW = (width - pad.l - pad.r) / n;
  const xs = (i) => pad.l + i * barW;
  const ys = (c) => height - pad.b - (c / maxCount) * (height - pad.t - pad.b);

  counts.forEach((c, i) => {
    const h = height - pad.b - ys(c);
    el("rect", {
      x: xs(i) + 0.5, y: ys(c), width: Math.max(0.5, barW - 1), height: h,
      fill: color, opacity: 0.85, class: "chart-bar",
    }, svg).appendChild(document.createComment(`count=${c}`));
  });
  el("line", { x1: pad.l, x2: width - pad.r, y1: height - pad.b, y2: height - pad.b, class: "chart-axis" }, svg);

  refLines.forEach((r) => {
    const t = (r.value - edges[0]) / (edges[edges.length - 1] - edges[0] || 1);
    if (t < 0 || t > 1) return;
    const x = pad.l + t * (width - pad.l - pad.r);
    el("line", { x1: x, x2: x, y1: pad.t, y2: height - pad.b, class: "chart-refline-v", stroke: r.color || COLOR.bad }, svg);
  });

  el("text", { x: pad.l, y: height - 6, class: "chart-tick", "text-anchor": "start" }, svg).textContent = xFmt(edges[0]);
  el("text", { x: width - pad.r, y: height - 6, class: "chart-tick", "text-anchor": "end" }, svg).textContent = xFmt(edges[edges.length - 1]);
  el("title", {}, svg).textContent = `n=${hist.n}  mean=${fmt(hist.mean)}  [${fmt(hist.min)}, ${fmt(hist.max)}]`;
}

// ---------------------------------------------------------------------------
// Grouped bars: score breakdown -- one group per weighted term, one bar per
// candidate within the group. Values may be negative (penalties, once
// weighted with their minus sign) so the baseline sits mid-chart.
// ---------------------------------------------------------------------------
function groupedBars(container, opts) {
  const { groups, series, width = 420, height = 200, title = "" } = opts;
  // groups: [{label, values: {seriesKey: number}}]; series: [{key,label,color}]
  clear(container);
  const pad = { l: 44, r: 10, t: title ? 24 : 10, b: 34 };
  const svg = el("svg", { width, height, viewBox: `0 0 ${width} ${height}`, class: "chart" }, container);
  if (title) el("text", { x: pad.l, y: 16, class: "chart-title" }, svg).textContent = title;

  const allV = groups.flatMap((g) => series.map((s) => g.values[s.key]).filter((v) => v !== undefined && v !== null));
  if (!allV.length) {
    el("text", { x: width / 2, y: height / 2, class: "chart-empty", "text-anchor": "middle" }, svg).textContent = "no data yet";
    return;
  }
  let vMax = Math.max(0, ...allV), vMin = Math.min(0, ...allV);
  const vPad = (vMax - vMin) * 0.15 || 1;
  vMax += vPad; vMin -= vPad;

  const innerW = width - pad.l - pad.r, innerH = height - pad.t - pad.b;
  const y0 = pad.t + innerH * (vMax / (vMax - vMin));
  const ys = (v) => pad.t + innerH * ((vMax - v) / (vMax - vMin));
  const groupW = innerW / groups.length;
  const barW = (groupW * 0.66) / series.length;

  [vMin, 0, vMax].forEach((v) => {
    const y = ys(v);
    el("line", { x1: pad.l, x2: width - pad.r, y1: y, y2: y, class: v === 0 ? "chart-axis" : "chart-grid" }, svg);
    el("text", { x: pad.l - 6, y: y + 3, class: "chart-tick", "text-anchor": "end" }, svg).textContent = fmt(v, 2);
  });

  groups.forEach((g, gi) => {
    const gx = pad.l + gi * groupW + groupW * 0.17;
    series.forEach((s, si) => {
      const v = g.values[s.key];
      if (v === undefined || v === null) return;
      const x = gx + si * barW;
      const y = ys(Math.max(v, 0));
      const h = Math.abs(ys(v) - y0);
      el("rect", {
        x, y: v >= 0 ? ys(v) : y0, width: barW - 2, height: Math.max(1, h),
        fill: s.color, opacity: 0.88, class: "chart-bar",
      }, svg);
    });
    el("text", {
      x: pad.l + gi * groupW + groupW / 2, y: height - pad.b + 15,
      class: "chart-tick", "text-anchor": "middle",
    }, svg).textContent = g.label;
  });
}

// ---------------------------------------------------------------------------
// Trajectory rail: the t=1->0 schedule, with a fork into two candidate paths
// (reference / delta) at every branch point the run has -- one for the
// original single-branch pipeline, more if DPOBranchConfig.num_branches > 1.
// The "where does the split happen" view. Ticks light up as sampler_step
// events arrive; each fork's post-segment is colored by ITS OWN re-rank gate
// decision, so a multi-branch run reads left-to-right as a chain of
// decisions, not just one.
// ---------------------------------------------------------------------------
function trajectoryRail(container, state) {
  const { tPairs = [], branches = [], stepsDone = 0, phase = "idle" } = state;
  const width = container.clientWidth || 760, height = 108;
  clear(container);
  const svg = el("svg", { width, height, viewBox: `0 0 ${width} ${height}`, class: "chart" }, container);

  if (!tPairs.length) {
    el("text", { x: width / 2, y: height / 2, class: "chart-empty", "text-anchor": "middle" }, svg).textContent =
      "waiting for schedule…";
    return;
  }
  const pad = { l: 20, r: 20 };
  const railY = 40;
  const n = tPairs.length;
  const xs = (i) => pad.l + (i / n) * (width - pad.l - pad.r);

  // base rail (full schedule, undifferentiated)
  el("line", { x1: xs(0), x2: xs(n), y1: railY, y2: railY, class: "rail-base" }, svg);

  const known = branches.filter((b) => b.iBranch !== null);

  // progress before the first fork (or the whole rail, if no fork yet)
  const firstForkStart = known.length ? known[0].iBranch : n;
  el("line", { x1: xs(0), x2: xs(Math.min(stepsDone, firstForkStart)), y1: railY, y2: railY, class: "rail-done" }, svg);

  const compact = known.length > 2; // fewer inline labels once forks get crowded

  known.forEach((br, idx) => {
    const forkEndAbs = br.iBranch + br.k;
    const nextForkStart = idx + 1 < known.length ? known[idx + 1].iBranch : n;
    const forkEndX = xs(forkEndAbs);

    // branch marker + t label
    el("circle", { cx: xs(br.iBranch), cy: railY, r: 5, class: "rail-branch-dot" }, svg);
    el("text", { x: xs(br.iBranch), y: railY - 12, class: "chart-tick", "text-anchor": "middle" }, svg).textContent =
      known.length > 1 ? `#${idx + 1} t=${fmt(br.branchT, 2)}` : `branch t=${fmt(br.branchT, 2)}`;

    // fork: reference (straight, teal) vs delta (arced up, amber)
    el("line", {
      x1: xs(br.iBranch), x2: forkEndX, y1: railY, y2: railY, stroke: COLOR.reference,
      "stroke-width": 3, class: "rail-fork",
    }, svg);
    const arcY = railY - 22;
    el("path", {
      d: `M ${xs(br.iBranch)} ${railY} Q ${(xs(br.iBranch) + forkEndX) / 2} ${arcY} ${forkEndX} ${railY}`,
      fill: "none", stroke: COLOR.delta, "stroke-width": 3, class: "rail-fork",
    }, svg);
    if (!compact) {
      el("text", { x: (xs(br.iBranch) + forkEndX) / 2, y: arcY - 6, class: "chart-tick", "text-anchor": "middle", fill: COLOR.delta }, svg).textContent =
        `${br.k} steer step${br.k === 1 ? "" : "s"}`;
    }

    // re-rank gate + resume
    const gateColor = br.resumedFrom === "delta" ? COLOR.delta : br.resumedFrom === "reference" ? COLOR.reference : COLOR.textDim;
    el("circle", { cx: forkEndX, cy: railY, r: 5, fill: gateColor, class: "rail-gate-dot" }, svg);
    if (br.resumedFrom && !compact) {
      el("text", { x: forkEndX, y: railY - 12, class: "chart-tick", "text-anchor": "middle", fill: gateColor }, svg).textContent =
        `resumed: ${br.resumedFrom}`;
    }

    // post-fork progress up to the next fork (or the end), colored by THIS
    // fork's own resumedFrom -- a later fork's segment can carry a different
    // color if a different branch won there.
    el("line", {
      x1: forkEndX, x2: xs(Math.min(stepsDone, nextForkStart)), y1: railY, y2: railY,
      stroke: gateColor, "stroke-width": 3, class: "rail-done",
    }, svg);
  });

  el("text", { x: xs(0), y: railY + 22, class: "chart-tick", "text-anchor": "start" }, svg).textContent = "t=1.0";
  el("text", { x: xs(n), y: railY + 22, class: "chart-tick", "text-anchor": "end" }, svg).textContent = "t=0";

  const phaseLabel = {
    idle: "idle", pre_branch: "sampling (pre-branch)", branch: "at branch point",
    steering: "gradient steering", post_branch: "sampling (post-branch)", done: "done", error: "error",
  }[phase] || phase;
  el("text", { x: width - pad.r, y: 16, class: "chart-tick", "text-anchor": "end" }, svg).textContent = phaseLabel;
}

window.DPOCharts = { lineChart, histogram, groupedBars, trajectoryRail, COLOR, fmt, clear };

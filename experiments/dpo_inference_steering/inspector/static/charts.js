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
  // Precision from the RANGE, not the magnitude: a band like [1.20, 1.40] at
  // 1 decimal renders as "1.4, 1.4, 1.3, 1.2, 1.2" -- duplicate labels that
  // look like a rendering bug and hide the actual scale.
  const span = (yMax - yMin) / nTicksY;
  const tickDp = span >= 1 ? 1 : Math.min(6, Math.max(2, Math.ceil(-Math.log10(span)) + 1));
  for (let i = 0; i <= nTicksY; i++) {
    const v = yMin + (i / nTicksY) * (yMax - yMin);
    const y = ys(v);
    el("line", { x1: pad.l, x2: width - pad.r, y1: y, y2: y, class: "chart-grid" }, svg);
    el("text", { x: pad.l - 6, y: y + 3, class: "chart-tick", "text-anchor": "end" }, svg).textContent = fmt(v, tickDp);
  }
  el("line", { x1: pad.l, x2: pad.l, y1: pad.t, y2: height - pad.b, class: "chart-axis" }, svg);
  el("line", { x1: pad.l, x2: width - pad.r, y1: height - pad.b, y2: height - pad.b, class: "chart-axis" }, svg);

  refLines.forEach((r) => {
    const y = ys(r.value);
    el("line", { x1: pad.l, x2: width - pad.r, y1: y, y2: y, class: "chart-refline", stroke: r.color || COLOR.textDim }, svg);
    el("text", { x: width - pad.r, y: y - 3, class: "chart-reflabel", "text-anchor": "end", fill: r.color || COLOR.textDim }, svg).textContent = r.label;
  });

  // Vertical markers over the x index -- used for the fork positions and the
  // step-slider cursor, so panel B lines up with the trajectory above it.
  (opts.vLines || []).forEach((v) => {
    if (v.index === null || v.index === undefined || v.index < 0) return;
    const x = xs(Math.min(v.index, xMax));
    el("line", { x1: x, x2: x, y1: pad.t, y2: height - pad.b,
                 stroke: v.color || COLOR.winner, "stroke-width": v.solid ? 1.6 : 1.2,
                 "stroke-dasharray": v.solid ? null : "3 3", opacity: v.solid ? 0.95 : 0.7 }, svg);
    if (v.label) {
      // Flip the label to the left of its line when it would run off the
      // right edge, otherwise a marker near the last step renders as "fo".
      const nearEdge = x > width - pad.r - 40;
      el("text", { x: nearEdge ? x - 4 : x + 4, y: pad.t + 10, class: "chart-reflabel",
                   "text-anchor": nearEdge ? "end" : "start", fill: v.color || COLOR.winner },
         svg).textContent = v.label;
    }
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
    // x=9, not 12: tick labels are right-aligned to pad.l-6 and can be 6+
    // characters at high precision, which collided with the rotated label.
    const cy = (pad.t + height - pad.b) / 2;
    el("text", { x: 9, y: cy, class: "chart-ylabel", "text-anchor": "middle",
                 transform: `rotate(-90 9 ${cy})` }, svg).textContent = yLabel;
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

// ---------------------------------------------------------------------------
// Below: primitives added for the Concept Proof page (concept.html). They
// follow the same contract as everything above -- take a container + data,
// replace its contents, no external dependency -- and reuse el()/fmt()/clear()
// and the shared COLOR map so the two pages stay visually consistent.
// ---------------------------------------------------------------------------

// Scalar field as a colour raster. `grid` is a res x res array indexed
// [i][j] = f(z0_i, z1_j); drawn as one <rect> per cell (res is ~40, so ~1600
// rects -- well inside what SVG handles without tiling to canvas).
function heatmap(container, opts) {
  const { grid, axis, width = 300, height = 300, title = "", colorFn, marks = [] } = opts;
  clear(container);
  const pad = { l: 34, r: 8, t: title ? 22 : 8, b: 26 };
  const svg = el("svg", { width, height, viewBox: `0 0 ${width} ${height}`, class: "chart" }, container);
  if (title) el("text", { x: pad.l, y: 14, class: "chart-title-sm" }, svg).textContent = title;
  if (!grid || !grid.length) {
    el("text", { x: width / 2, y: height / 2, class: "chart-empty", "text-anchor": "middle" }, svg).textContent = "no data";
    return;
  }
  const res = grid.length;
  const flat = grid.flat();
  const lo = Math.min(...flat), hi = Math.max(...flat);
  const innerW = width - pad.l - pad.r, innerH = height - pad.t - pad.b;
  const cw = innerW / res, ch = innerH / res;
  const paint = colorFn || ((v) => {
    // dark teal -> warm amber, through the page's own accent pair
    const u = (v - lo) / (hi - lo || 1);
    const r = Math.round(18 + u * 237), g = Math.round(90 + u * 69), b = Math.round(110 - u * 41);
    return `rgb(${r},${g},${b})`;
  });
  for (let i = 0; i < res; i++) {
    for (let j = 0; j < res; j++) {
      el("rect", {
        // j indexes z1 (vertical, flipped so +z1 is up), i indexes z0
        x: pad.l + i * cw, y: pad.t + (res - 1 - j) * ch,
        width: Math.ceil(cw) + 0.5, height: Math.ceil(ch) + 0.5,
        fill: paint(grid[i][j]), "shape-rendering": "crispEdges",
      }, svg);
    }
  }
  const sx = (z) => pad.l + ((z + 1) / 2) * innerW;
  const sy = (z) => pad.t + (1 - (z + 1) / 2) * innerH;
  marks.forEach((m) => {
    el("circle", { cx: sx(m[0]), cy: sy(m[1]), r: 3.5, fill: "none",
                   stroke: "#0A0E12", "stroke-width": 2.5 }, svg);
    el("circle", { cx: sx(m[0]), cy: sy(m[1]), r: 3.5, fill: "none",
                   stroke: COLOR.text, "stroke-width": 1.2 }, svg);
  });
  el("rect", { x: pad.l, y: pad.t, width: innerW, height: innerH, fill: "none", class: "chart-axis" }, svg);
  if (axis) {
    el("text", { x: pad.l, y: height - 8, class: "chart-tick" }, svg).textContent = axis.x0;
    el("text", { x: width - pad.r, y: height - 8, class: "chart-tick", "text-anchor": "end" }, svg).textContent = axis.x1;
    el("text", { x: 10, y: pad.t + 8, class: "chart-tick" }, svg).textContent = axis.y1;
    el("text", { x: 10, y: height - pad.b, class: "chart-tick" }, svg).textContent = axis.y0;
  }
  el("title", {}, svg).textContent = `min ${fmt(lo, 2)}  max ${fmt(hi, 2)}`;
}

// Quiver plot of a 2D velocity field, optionally over a heatmap background.
// Arrow length is normalised against the field's own max so two fields drawn
// side by side are comparable in direction; magnitude goes to opacity.
function vectorField(container, opts) {
  const { field, width = 300, height = 300, title = "", color = COLOR.reference,
          background = null, backgroundColorFn = null, marks = [] } = opts;
  clear(container);
  const pad = { l: 34, r: 8, t: title ? 22 : 8, b: 26 };
  if (background) {
    heatmap(container, { grid: background, width, height, title, colorFn: backgroundColorFn, marks });
  }
  const svg = background
    ? container.querySelector("svg")
    : el("svg", { width, height, viewBox: `0 0 ${width} ${height}`, class: "chart" }, container);
  if (!background && title) el("text", { x: pad.l, y: 14, class: "chart-title-sm" }, svg).textContent = title;
  if (!field || !field.x) return;

  const innerW = width - pad.l - pad.r, innerH = height - pad.t - pad.b;
  const lim = 1.6;  // the field is sampled on [-1.6, 1.6]^2
  const sx = (z) => pad.l + ((z + lim) / (2 * lim)) * innerW;
  const sy = (z) => pad.t + (1 - (z + lim) / (2 * lim)) * innerH;

  const mags = field.u.map((u, i) => Math.hypot(u, field.v[i]));
  const maxMag = Math.max(...mags, 1e-9);
  const cell = innerW / field.res;
  const defs = el("defs", {}, svg);
  // Marker ids are DOCUMENT-global, so two vectorField() calls on one page must
  // not collide. Deriving the id from the colour via Math.abs() silently yielded
  // NaN for every hex string ("3ED6C4" is not a number), so both panels defined
  // "arrow-NaN", url(#arrow-NaN) resolved to whichever was defined first, and the
  // second field rendered its own line colour with the FIRST field's arrowheads.
  vectorField._seq = (vectorField._seq || 0) + 1;
  const mid = `arrow-${color.replace(/[^a-zA-Z0-9]/g, "")}-${vectorField._seq}`;
  const marker = el("marker", { id: mid, markerWidth: 4, markerHeight: 4, refX: 2.4,
                                refY: 2, orient: "auto" }, defs);
  el("path", { d: "M0,0 L4,2 L0,4 z", fill: color }, marker);

  field.x.forEach((x, i) => {
    const m = mags[i] / maxMag;
    const len = Math.min(cell * 0.92, cell * 0.35 + cell * 0.6 * m);
    const ux = field.u[i] / (mags[i] || 1), uy = field.v[i] / (mags[i] || 1);
    const x0 = sx(x), y0 = sy(field.y[i]);
    el("line", {
      x1: x0 - (ux * len) / 2, y1: y0 + (uy * len) / 2,
      x2: x0 + (ux * len) / 2, y2: y0 - (uy * len) / 2,
      stroke: color, "stroke-width": 1.3, opacity: 0.35 + 0.65 * m,
      "marker-end": `url(#${mid})`,
    }, svg);
  });
}

// Sampled ODE trajectories in 2D: where each path actually went. Start points
// hollow, endpoints solid, so direction of travel reads without a legend.
function trajectoryPaths(container, opts) {
  const { paths, width = 300, height = 300, title = "", color = COLOR.reference, lim = 1.6 } = opts;
  clear(container);
  const pad = { l: 34, r: 8, t: title ? 22 : 8, b: 26 };
  const svg = el("svg", { width, height, viewBox: `0 0 ${width} ${height}`, class: "chart" }, container);
  if (title) el("text", { x: pad.l, y: 14, class: "chart-title-sm" }, svg).textContent = title;
  if (!paths || !paths.length) {
    el("text", { x: width / 2, y: height / 2, class: "chart-empty", "text-anchor": "middle" }, svg).textContent = "no data";
    return;
  }
  const innerW = width - pad.l - pad.r, innerH = height - pad.t - pad.b;
  const sx = (z) => pad.l + ((Math.max(-lim, Math.min(lim, z)) + lim) / (2 * lim)) * innerW;
  const sy = (z) => pad.t + (1 - (Math.max(-lim, Math.min(lim, z)) + lim) / (2 * lim)) * innerH;

  el("rect", { x: pad.l, y: pad.t, width: innerW, height: innerH, fill: "none", class: "chart-axis" }, svg);
  paths.forEach((p) => {
    el("polyline", {
      points: p.map((q) => `${sx(q[0])},${sy(q[1])}`).join(" "),
      fill: "none", stroke: color, "stroke-width": 1.1, opacity: 0.5,
    }, svg);
    el("circle", { cx: sx(p[0][0]), cy: sy(p[0][1]), r: 1.8, fill: "none",
                   stroke: COLOR.textDim, "stroke-width": 0.9 }, svg);
    const last = p[p.length - 1];
    el("circle", { cx: sx(last[0]), cy: sy(last[1]), r: 2.2, fill: color, opacity: 0.95 }, svg);
  });
}

// Two histograms on one shared x-axis. Used for the pre/post reward shift,
// where the whole point is that the two distributions must be directly
// comparable -- drawing them as separate charts with independent axes would
// make any shift unreadable.
function overlayHistogram(container, opts) {
  const { series, width = 420, height = 190, title = "", refLines = [], xLabel = "" } = opts;
  clear(container);
  const pad = { l: 40, r: 12, t: title ? 24 : 10, b: 30 };
  const svg = el("svg", { width, height, viewBox: `0 0 ${width} ${height}`, class: "chart" }, container);
  if (title) el("text", { x: pad.l, y: 15, class: "chart-title" }, svg).textContent = title;

  const valid = series.filter((s) => s.hist && s.hist.n);
  if (!valid.length) {
    el("text", { x: width / 2, y: height / 2, class: "chart-empty", "text-anchor": "middle" }, svg).textContent = "no data";
    return;
  }
  let lo = Math.min(...valid.map((s) => s.hist.edges[0]));
  let hi = Math.max(...valid.map((s) => s.hist.edges[s.hist.edges.length - 1]));
  // Pad the range by 3%: a collapsed distribution piles into the last bin and
  // would otherwise be drawn flush against the frame, reading as clipped data.
  const padX = (hi - lo) * 0.03 || 1;
  lo -= padX; hi += padX;
  const innerW = width - pad.l - pad.r, innerH = height - pad.t - pad.b;
  // Normalise to density so unequal n does not fake a shift in height.
  const peak = Math.max(...valid.map((s) => Math.max(...s.hist.counts) / s.hist.n));
  const sx = (v) => pad.l + ((v - lo) / (hi - lo || 1)) * innerW;
  const sy = (d) => pad.t + innerH - (d / (peak || 1)) * innerH;

  [0, 0.5, 1].forEach((f) => {
    const y = pad.t + innerH * f;
    el("line", { x1: pad.l, x2: width - pad.r, y1: y, y2: y, class: f === 1 ? "chart-axis" : "chart-grid" }, svg);
  });

  valid.forEach((s) => {
    const e = s.hist.edges, c = s.hist.counts;
    const pts = [];
    for (let i = 0; i < c.length; i++) {
      pts.push(`${sx(e[i])},${sy(c[i] / s.hist.n)}`, `${sx(e[i + 1])},${sy(c[i] / s.hist.n)}`);
    }
    el("polygon", {
      points: `${sx(e[0])},${sy(0)} ${pts.join(" ")} ${sx(e[e.length - 1])},${sy(0)}`,
      fill: s.color, opacity: 0.28,
    }, svg);
    el("polyline", { points: pts.join(" "), fill: "none", stroke: s.color, "stroke-width": 1.8 }, svg);
    const mean = s.hist.mean;
    el("line", { x1: sx(mean), x2: sx(mean), y1: pad.t, y2: pad.t + innerH,
                 stroke: s.color, "stroke-width": 1.4, "stroke-dasharray": "4 3" }, svg);
  });

  refLines.forEach((r) => {
    el("line", { x1: sx(r.value), x2: sx(r.value), y1: pad.t, y2: pad.t + innerH,
                 class: "chart-refline-v", stroke: r.color || COLOR.bad }, svg);
  });

  el("text", { x: pad.l, y: height - 8, class: "chart-tick" }, svg).textContent = fmt(lo, 2);
  el("text", { x: width - pad.r, y: height - 8, class: "chart-tick", "text-anchor": "end" }, svg).textContent = fmt(hi, 2);
  if (xLabel) el("text", { x: pad.l + innerW / 2, y: height - 8, class: "chart-tick", "text-anchor": "middle" }, svg).textContent = xLabel;
}

// Two y-axes over a shared x. Exists for exactly one job on the concept page:
// showing reward and mode-entropy against DPO step together, because the
// finding IS their divergence -- reward climbing while coverage collapses.
// Plotting them on separate charts hides the crossover.
function dualAxisChart(container, opts) {
  const { x, left, right, width = 440, height = 210, title = "", xLabel = "" } = opts;
  clear(container);
  const pad = { l: 46, r: 46, t: title ? 26 : 12, b: 30 };
  const svg = el("svg", { width, height, viewBox: `0 0 ${width} ${height}`, class: "chart" }, container);
  if (title) el("text", { x: pad.l, y: 16, class: "chart-title" }, svg).textContent = title;
  if (!x || !x.length) {
    el("text", { x: width / 2, y: height / 2, class: "chart-empty", "text-anchor": "middle" }, svg).textContent = "no data";
    return;
  }
  const innerW = width - pad.l - pad.r, innerH = height - pad.t - pad.b;
  const sx = (i) => pad.l + (i / Math.max(1, x.length - 1)) * innerW;

  const mk = (axis) => {
    const all = axis.series.flatMap((s) => s.values);
    let lo = Math.min(...all), hi = Math.max(...all);
    if (lo === hi) { lo -= 1; hi += 1; }
    const p = (hi - lo) * 0.12; lo -= p; hi += p;
    return { lo, hi, y: (v) => pad.t + innerH * (1 - (v - lo) / (hi - lo)) };
  };
  const L = mk(left), R = mk(right);

  for (let i = 0; i <= 4; i++) {
    const y = pad.t + (i / 4) * innerH;
    el("line", { x1: pad.l, x2: width - pad.r, y1: y, y2: y, class: i === 4 ? "chart-axis" : "chart-grid" }, svg);
    el("text", { x: pad.l - 6, y: y + 3, class: "chart-tick", "text-anchor": "end", fill: left.color }, svg)
      .textContent = fmt(L.hi - (i / 4) * (L.hi - L.lo), 2);
    el("text", { x: width - pad.r + 6, y: y + 3, class: "chart-tick", fill: right.color }, svg)
      .textContent = fmt(R.hi - (i / 4) * (R.hi - R.lo), 2);
  }

  [[left, L], [right, R]].forEach(([axis, sc]) => {
    axis.series.forEach((s) => {
      el("polyline", {
        points: s.values.map((v, i) => `${sx(i)},${sc.y(v)}`).join(" "),
        fill: "none", stroke: s.color, "stroke-width": 2,
        "stroke-dasharray": s.dashed ? "5 3" : null,
      }, svg);
      s.values.forEach((v, i) => el("circle", { cx: sx(i), cy: sc.y(v), r: 2.4, fill: s.color }, svg));
    });
  });

  // Optional vertical annotation, e.g. "best real printability was here".
  (opts.vlines || []).forEach((vl) => {
    const i = x.indexOf(vl.x);
    if (i < 0) return;
    el("line", { x1: sx(i), x2: sx(i), y1: pad.t, y2: pad.t + innerH,
                 stroke: vl.color || COLOR.winner, "stroke-width": 1.4, "stroke-dasharray": "3 3" }, svg);
    el("text", { x: sx(i) + 4, y: pad.t + 10, class: "chart-reflabel", fill: vl.color || COLOR.winner }, svg)
      .textContent = vl.label;
  });

  [0, Math.floor(x.length / 2), x.length - 1].forEach((i) => {
    el("text", { x: sx(i), y: height - 10, class: "chart-tick", "text-anchor": "middle" }, svg).textContent = x[i];
  });
  if (xLabel) el("text", { x: pad.l + innerW / 2, y: height - 1, class: "chart-tick", "text-anchor": "middle" }, svg).textContent = xLabel;
}

// ---------------------------------------------------------------------------
// Latent trajectory in the run's own 2-D random projection (LatentProbe in
// trellis_core/dpo_branch.py). Draws the pre-branch path, the fork, the
// candidate branches diverging, and the resumed tail -- revealed up to
// `upTo` so a slider can scrub through the run.
//
// The projection basis is fixed for the whole run and its columns are unit
// vectors, so projected distance <= sqrt(2) * true distance: visible separation
// implies real separation. The converse fails at 2 dimensions -- branches can
// look coincident while genuinely differing -- so read distance here as a lower
// bound. (dpo_branch_test.test_latent_probe_is_deterministic_and_cannot_fake_separation)
// ---------------------------------------------------------------------------
function latentPath(container, opts) {
  const { frames, upTo, width = 700, height = 380, title = "", colors = {} } = opts;
  clear(container);
  const pad = { l: 40, r: 14, t: title ? 26 : 12, b: 26 };
  const svg = el("svg", { width, height, viewBox: `0 0 ${width} ${height}`, class: "chart" }, container);
  if (title) el("text", { x: pad.l, y: 16, class: "chart-title" }, svg).textContent = title;

  const pts = [];
  frames.forEach((f) => {
    if (f.kind === "fork") Object.values(f.points).forEach((p) => p && pts.push(p));
    else if (f.proj) pts.push(f.proj);
  });
  if (pts.length < 2) {
    el("text", { x: width / 2, y: height / 2, class: "chart-empty", "text-anchor": "middle" }, svg)
      .textContent = "no latent telemetry in this run";
    return;
  }

  const xsAll = pts.map((p) => p[0]), ysAll = pts.map((p) => p[1]);
  let x0 = Math.min(...xsAll), x1 = Math.max(...xsAll);
  let y0 = Math.min(...ysAll), y1 = Math.max(...ysAll);
  const padX = (x1 - x0) * 0.08 || 1, padY = (y1 - y0) * 0.08 || 1;
  x0 -= padX; x1 += padX; y0 -= padY; y1 += padY;
  const innerW = width - pad.l - pad.r, innerH = height - pad.t - pad.b;
  const SX = (v) => pad.l + ((v - x0) / (x1 - x0)) * innerW;
  const SY = (v) => pad.t + innerH - ((v - y0) / (y1 - y0)) * innerH;

  el("rect", { x: pad.l, y: pad.t, width: innerW, height: innerH, fill: "none", class: "chart-axis" }, svg);

  const C = {
    pre: colors.pre || COLOR.textDim,
    reference: colors.reference || COLOR.reference,
    delta_initial: colors.delta_initial || COLOR.delta,
    delta_final: colors.delta_final || COLOR.deltaFinal,
    post: colors.post || COLOR.winner,
    future: "#2A3540",
  };

  // Ghost of the whole trajectory, so scrubbing reads as revealing a known
  // path rather than drawing an unknown one.
  const ghost = (key) => {
    const seq = [];
    frames.forEach((f) => {
      const p = f.kind === "fork" ? f.points[key] : (key === "main" ? f.proj : null);
      if (p) seq.push(p);
    });
    if (seq.length > 1) {
      el("polyline", { points: seq.map((p) => `${SX(p[0])},${SY(p[1])}`).join(" "),
                       fill: "none", stroke: C.future, "stroke-width": 1 }, svg);
    }
  };
  ghost("main"); ghost("reference"); ghost("delta_initial"); ghost("delta_final");

  // Revealed segments.
  const drawn = { main: [], reference: [], delta_initial: [], delta_final: [] };
  const lastMainBefore = [];
  frames.forEach((f, i) => {
    if (i > upTo) return;
    if (f.kind === "fork") {
      ["reference", "delta_initial", "delta_final"].forEach((k) => {
        if (f.points[k]) drawn[k].push(f.points[k]);
      });
    } else if (f.proj) {
      drawn.main.push(f.proj);
      if (f.phase === "pre_branch") lastMainBefore.push(f.proj);
    }
  });

  const seg = (arr, color, wgt) => {
    if (arr.length > 1) {
      el("polyline", { points: arr.map((p) => `${SX(p[0])},${SY(p[1])}`).join(" "),
                       fill: "none", stroke: color, "stroke-width": wgt, "stroke-linejoin": "round" }, svg);
    }
    arr.forEach((p, i) => el("circle", { cx: SX(p[0]), cy: SY(p[1]), r: i === arr.length - 1 ? 3.4 : 1.9,
                                         fill: color }, svg));
  };
  seg(drawn.main, C.pre, 1.8);
  // Anchor each branch to the fork it left from.
  frames.forEach((f) => {
    if (f.kind !== "fork" || !f.forkOrigin) return;
    ["reference", "delta_initial", "delta_final"].forEach((k) => {
      const first = drawn[k][0];
      if (first && f.points[k]) {
        el("line", { x1: SX(f.forkOrigin[0]), y1: SY(f.forkOrigin[1]), x2: SX(first[0]), y2: SY(first[1]),
                     stroke: C[k], "stroke-width": 1.4, opacity: 0.75 }, svg);
      }
    });
  });
  seg(drawn.reference, C.reference, 2);
  seg(drawn.delta_initial, C.delta_initial, 2);
  seg(drawn.delta_final, C.delta_final, 2);

  // Fork markers.
  frames.forEach((f, i) => {
    if (f.kind !== "fork" || !f.forkOrigin || f.forkIndex !== 0) return;
    const revealed = i <= upTo;
    el("circle", { cx: SX(f.forkOrigin[0]), cy: SY(f.forkOrigin[1]), r: 6, fill: "none",
                   stroke: revealed ? COLOR.winner : C.future, "stroke-width": 2 }, svg);
    el("text", { x: SX(f.forkOrigin[0]) + 9, y: SY(f.forkOrigin[1]) - 6, class: "chart-reflabel",
                 fill: revealed ? COLOR.winner : C.future },
       svg).textContent = `fork ${f.branch + 1} · t=${fmt(f.t, 3)}`;
  });

  // Direction is not inferable from a static polyline, and the path can run in
  // any direction depending on the projection basis -- so mark the start.
  const firstMain = frames.find((f) => f.kind === "step" && f.proj);
  if (firstMain) {
    el("circle", { cx: SX(firstMain.proj[0]), cy: SY(firstMain.proj[1]), r: 5, fill: "none",
                   stroke: COLOR.text, "stroke-width": 1.4, "stroke-dasharray": "2 2" }, svg);
    el("text", { x: SX(firstMain.proj[0]) + 8, y: SY(firstMain.proj[1]) + 4, class: "chart-reflabel",
                 fill: COLOR.textDim }, svg).textContent = `start  t=${fmt(firstMain.t, 2)}`;
  }
  el("text", { x: pad.l, y: height - 8, class: "chart-tick" }, svg).textContent = "random projection axis 1";
  el("text", { x: 10, y: pad.t + 8, class: "chart-tick" }, svg).textContent = "axis 2";
}

// One explicit export surface, matching the window.DPOCharts assignment above
// (which app.js destructures). The concept-page functions were originally left
// as bare globals, which works but makes a stale-cache mismatch surface as
// "Can't find variable: latentPath" -- indistinguishable from a scoping bug.
// Exporting them here lets consumers assert what they need up front.
Object.assign(window.DPOCharts, {
  heatmap, vectorField, trajectoryPaths, overlayHistogram, dualAxisChart, latentPath,
  BUILD: "concept-v2",
});

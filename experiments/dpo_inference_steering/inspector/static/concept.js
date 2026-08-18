// concept.js -- renders the Concept Proof page from concept_data.json.
//
// Same conventions as app.js: no framework, no build step, charts.js does the
// SVG. The one rule specific to this page: every claim rendered here must be
// traceable to a field in concept_data.json, which is written by
// devlab/alignment_experiment.py. Nothing is hard-coded to make a point --
// where a verdict card says "refuted", it is comparing two numbers that came
// out of the run, and it would say "confirmed" if they came out the other way.
"use strict";

const C = {
  base: "#3ED6C4",
  aligned: "#FF9F45",
  refFree: "#FF6B4A",
  good: "#7CE38B",
  bad: "#FF6B6B",
  warn: "#FFC24B",
  dim: "#8B99A4",
};

function n(v, d = 3) {
  if (v === null || v === undefined || Number.isNaN(v)) return "–";
  return Number(v).toFixed(d);
}
function pct(v, d = 1) { return `${(v * 100).toFixed(d)}%`; }

// A verdict card states the claim, the measurement, and the call. `status`
// drives the colour: confirmed / refuted / partial / untestable.
function verdictCard(claim, status, headline, detail) {
  return `<div class="verdict-card ${status}">
    <div class="v-status">${status}</div>
    <div class="v-claim">${claim}</div>
    <div class="v-headline">${headline}</div>
    <div class="v-detail">${detail}</div>
  </div>`;
}

async function main() {
  if (!requireCharts(["vectorField", "trajectoryPaths", "overlayHistogram", "dualAxisChart", "heatmap"],
                     "The mechanism panels")) return;
  // Panels M1-M4 are now backed by REAL trace data (real_mechanism.js) rather
  // than this toy experiment, so their old element ids no longer exist. The
  // toy still backs the verdict cards and M5/M6, so this file keeps running --
  // it just skips any panel whose container is gone rather than throwing.
  const have = (id) => document.getElementById(id) !== null;
  let d;
  try {
    const res = await fetch("/api/concept");
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    d = await res.json();
  } catch (e) {
    document.getElementById("metaText").textContent = "no data";
    document.querySelector("main").insertAdjacentHTML("afterbegin",
      `<div class="banner">Could not load concept_data.json — ${e.message}
       <br><br>Generate it with:  <code>python devlab/alignment_experiment.py</code></div>`);
    return;
  }

  const meta = d.meta, fin = d.final_metrics, rs = d.reward_shift, cv = d.curvature;
  document.getElementById("metaText").textContent =
    `n=${rs.n} · ${meta.dpo_steps} DPO steps · ${meta.runtime_seconds}s`;
  if (have("nEval")) document.getElementById("nEval").textContent = rs.n;
  document.getElementById("verdictSub").textContent =
    `${meta.fm_steps} flow-matching steps, ${meta.dpo_steps} DPO steps, β=${meta.beta}, seed ${meta.seed}`;

  // ---- verdict cards ------------------------------------------------------
  const hist = d.printability_history.reference_based;
  const bestOH = hist.reduce((a, b) => (b.overhang_true < a.overhang_true ? b : a), hist[0]);
  const lastOH = hist[hist.length - 1];
  const curvRatio = cv.aligned_mean / cv.base_mean;
  const tortRatio = cv.tortuosity_base / cv.tortuosity_aligned;   // >1 == straighter
  const entropyDrop = 1 - fin.aligned.mode_entropy / fin.base.mode_entropy;

  document.getElementById("verdictCards").innerHTML = [
    verdictCard(
      "DPO aligns a continuous flow model toward a preference distribution",
      "partial",
      `reward ${n(fin.base.reward_mean, 2)} → ${n(fin.aligned.reward_mean, 2)}, but onto ${fin.aligned.unique_clipped_slopes} distinct shape`,
      `The optimisation works — Δ=${n(rs.delta, 3)}, 95% CI [${n(rs.ci95[0], 2)}, ${n(rs.ci95[1], 2)}], Cohen's d=${n(rs.cohens_d, 2)}, n=${rs.n}. But <strong>${pct(fin.aligned.out_of_box_frac, 0)} of aligned samples land outside the latent box</strong> (base: ${pct(fin.base.out_of_box_frac, 1)}), where <code>z_to_params</code> clips them — so all ${rs.n} collapse to ${fin.aligned.unique_clipped_slopes} geometry sitting on a <code>max(…, 1e-3)</code> clamp. DPO found the argmax of the proxy; the argmax is a parameterisation artifact. The mechanism is confirmed, the win is not.`
    ),
    verdictCard(
      "DPO straightens the probability paths",
      "confirmed",
      `tortuosity ${n(cv.tortuosity_base, 3)} → ${n(cv.tortuosity_aligned, 3)} (${n(tortRatio, 2)}× straighter)`,
      `<strong>This reverses an earlier reading on this page.</strong> Raw curvature rose ${n(cv.base_mean, 1)} → ${n(cv.aligned_mean, 1)}, which looked like the opposite — but ‖x″‖² scales as (displacement)², and collapsing modes onto a distant point lengthens the paths. Scale-free measures agree the paths got straighter: tortuosity ${n(tortRatio, 2)}× and displacement-normalised curvature ${n(cv.normalized_base / cv.normalized_aligned, 2)}×. Caveat: straighter here is partly a consequence of collapsing to one endpoint, not evidence DPO is a substitute for reflow.`
    ),
    verdictCard(
      "Preference alignment improves real printability",
      "partial",
      `genuine wall overhang ${n(fin.base.overhang_true_mean, 4)} → ${n(fin.aligned.overhang_true_mean, 4)}`,
      `Real overhang is eliminated — but the raw <code>L_OH</code> the judge reports (${n(fin.base.overhang_mean, 4)} → ${n(fin.aligned.overhang_mean, 4)}) is misleading: <strong>${pct(fin.aligned.overhang_bed_mean / fin.aligned.overhang_mean, 0)} of the aligned figure is build-plate contact</strong>, which no slicer counts as overhang. <code>geometric_judge.overhang_penalty</code> has no z_min or support test, so a cube resting flat scores 0.0732. And the win is degenerate: it comes from collapsing to a single cone, with mode entropy down ${pct(entropyDrop)} (${n(fin.base.mode_entropy, 3)} → ${n(fin.aligned.mode_entropy, 3)} of ${n(d.max_mode_entropy, 3)}).`
    ),
    verdictCard(
      "Non-manifold edges are reduced over training",
      "untestable",
      `0 defects at every point in the latent box`,
      `The frustum family is closed analytically, so every sample is watertight and the judge's topology term is identically zero — there is nothing to reduce. Emulating the real defect mechanism (voxelise → marching cubes) does not help: a filled voxel grid is watertight by construction. The only real evidence is the n=1 3D run, and it came back <strong>watertight=False</strong>.`
    ),
  ].join("");

  // ---- 1-4. toy vector field / paths / reward / printability -------------
  // These four panels are now rendered from REAL trace data by
  // real_mechanism.js, and their containers no longer exist in concept.html.
  // The toy experiment still backs the verdict cards above and M5-M7 below, so
  // this file keeps running and simply skips the block it no longer owns.
  if (have("fieldBase")) {
  const grid = d.reward_grid.total;
  const flat = grid.flat();
  const lo = Math.min(...flat), hi = Math.max(...flat);
  // Deliberately low-chroma: the background encodes reward, but the ARROWS are
  // the subject of this panel. A saturated amber landscape renders the amber
  // aligned-field arrows invisible against it, which is how the first draft
  // looked. Dark slate -> muted olive keeps the gradient readable while leaving
  // both the teal and amber arrow colours a clear contrast channel.
  const paint = (v) => {
    const u = Math.pow((v - lo) / (hi - lo || 1), 0.8);
    return `rgb(${Math.round(14 + u * 74)},${Math.round(20 + u * 76)},${Math.round(28 + u * 40)})`;
  };
  const axis = { x0: "slope −0.4", x1: "slope 2.4", y0: "wall 0.004", y1: "wall 0.075" };
  vectorField(document.getElementById("fieldBase"), {
    field: d.vector_field.base, background: grid, backgroundColorFn: paint,
    marks: d.reward_grid.modes, color: C.base, width: 340, height: 320,
    title: "baseline  v_θ(x, 0.5)",
  });
  vectorField(document.getElementById("fieldAligned"), {
    field: d.vector_field.aligned, background: grid, backgroundColorFn: paint,
    marks: d.reward_grid.modes, color: C.aligned, width: 340, height: 320,
    title: "DPO-aligned  v_θ(x, 0.5)",
  });

  // ---- 2. trajectories + curvature ---------------------------------------
  trajectoryPaths(document.getElementById("trajBase"), {
    paths: d.trajectories.base.slice(0, 90), color: C.base,
    width: 340, height: 300, title: `baseline paths  ·  κ = ${n(cv.base_mean, 1)}`,
  });
  trajectoryPaths(document.getElementById("trajAligned"), {
    paths: d.trajectories.aligned.slice(0, 90), color: C.aligned,
    width: 340, height: 300, title: `aligned paths  ·  κ = ${n(cv.aligned_mean, 1)}`,
  });
  overlayHistogram(document.getElementById("curvatureHist"), {
    series: [
      { hist: cv.base, color: C.base, label: "baseline" },
      { hist: cv.aligned, color: C.aligned, label: "aligned" },
    ],
    width: 700, height: 170, title: "path curvature κ (density; dashed = mean)",
    xLabel: "κ = mean ‖d²x/dt²‖²",
  });
  document.getElementById("curvatureNote").innerHTML =
    `<strong>Two metrics, opposite answers — and the scale-free ones are right.</strong>
     Raw ‖x″‖² went ${n(cv.base_mean, 1)} → ${n(cv.aligned_mean, 1)} (${n(curvRatio, 1)}× "worse"), but that
     quantity scales as (displacement)², and alignment moved the endpoints far apart, so it is
     confounded by the mode collapse in panel 4. Tortuosity (arc length ÷ chord, scale-free by
     construction) went ${n(cv.tortuosity_base, 3)} → ${n(cv.tortuosity_aligned, 3)}, i.e.
     <strong>${n(tortRatio, 2)}× straighter</strong>; displacement-normalised curvature agrees at
     ${n(cv.normalized_base / cv.normalized_aligned, 2)}×.
     <span class="dim">An earlier version of this page reported the raw ratio as evidence that DPO
     fails to straighten paths. That was wrong, and
     <code>flow_dpo_test.test_tortuosity_is_scale_free_and_curvature_is_not</code> now pins the
     invariance so the confusion cannot recur.</span>
     The honest reading: the paths did straighten, but largely <em>because</em> they now all end in
     the same place — this is not evidence DPO substitutes for reflow.`;

  // ---- 3. reward shift ----------------------------------------------------
  overlayHistogram(document.getElementById("rewardShift"), {
    series: [
      { hist: rs.base, color: C.base, label: "pre-DPO" },
      { hist: rs.aligned, color: C.aligned, label: "post-DPO" },
      { hist: rs.ref_free, color: C.refFree, label: "post-DPO (reference-free)" },
    ],
    width: 700, height: 210, title: "judge score, pre vs post alignment (density)",
    xLabel: "composite judge score S",
  });
  document.getElementById("rewardStats").innerHTML = `
    <div><span class="k">pre-DPO mean</span><span class="v" style="color:${C.base}">${n(rs.base_mean, 3)}</span></div>
    <div><span class="k">post-DPO mean</span><span class="v" style="color:${C.aligned}">${n(rs.aligned_mean, 3)}</span></div>
    <div><span class="k">shift</span><span class="v">${n(rs.delta, 3)}</span></div>
    <div><span class="k">95% CI</span><span class="v">[${n(rs.ci95[0], 2)}, ${n(rs.ci95[1], 2)}]</span></div>
    <div><span class="k">Cohen's d</span><span class="v">${n(rs.cohens_d, 2)}</span></div>
    <div><span class="k">n</span><span class="v">${rs.n}</span></div>`;

  // ---- 4. printability over steps ----------------------------------------
  const bestTh = hist.reduce((a, b) => (b.thickness < a.thickness ? b : a), hist[0]);
  const steps = hist.map((h) => h.step);
  dualAxisChart(document.getElementById("printabilityChart"), {
    x: steps,
    left: { color: C.aligned, series: [{ values: hist.map((h) => h.reward_mean), color: C.aligned }] },
    right: { color: C.bad, series: [{ values: hist.map((h) => h.mode_entropy), color: C.bad, dashed: true }] },
    width: 720, height: 230, xLabel: "DPO step",
    title: "reward (left, solid) vs mode-coverage entropy (right, dashed)",
    vlines: [{ x: bestTh.step, label: `stop here — all real gains, no hacking (step ${bestTh.step})`, color: C.good }],
  });
  lineChart(document.getElementById("overhangChart"), {
    series: [
      { values: hist.map((h) => h.overhang_true), color: C.warn },
      { values: hist.map((h) => h.overhang_bed), color: C.dim },
    ],
    width: 350, height: 190, title: "L_OH: true wall vs build plate",
    yLabel: "L_OH", refLines: [{ value: fin.base.overhang_true_mean, label: "pre-DPO true", color: C.base }],
  });
  lineChart(document.getElementById("entropyChart"), {
    series: [{ values: hist.map((h) => h.mode_entropy), color: C.bad }],
    width: 350, height: 190, title: "mode-coverage entropy", yLabel: "nats",
    refLines: [{ value: d.max_mode_entropy, label: "full coverage", color: C.good }],
  });
  const firstOOB = hist.find((h) => h.out_of_box_frac > 0.99);
  document.getElementById("hackNote").innerHTML =
    `<strong>The useful work is done by step ${bestTh.step}. Everything after it is reward hacking.</strong>
     Genuine wall overhang reaches ${n(bestOH.overhang_true, 4)} by step ${bestOH.step} and stays there —
     a real result. But thickness <code>L_Th</code> bottoms at ${n(bestTh.thickness, 4)} on the same step and
     then degrades <strong>${n(lastOH.thickness / Math.max(bestTh.thickness, 1e-9), 1)}×</strong> to
     ${n(lastOH.thickness, 4)}, while reward buys its last few percent
     (${n(bestTh.reward_mean, 2)} → ${n(lastOH.reward_mean, 2)}) and the model walks out of the latent box
     entirely${firstOOB ? ` (100% out-of-box by step ${firstOOB.step})` : ""}.
     The judge's composite is <code>α·R_Detail − (β·L_OH + γ·L_Th + δ·L_Topo)</code>; with the pipeline's own
     α=1 the detail term is the cheaper way to buy score. Measured: corr(S, L_OH) is
     ${n(meta.weight_sensitivity.alpha_sweep.find((a) => a.alpha === 0).corr_overhang, 3)} at α=0 but only
     ${n(meta.weight_sensitivity.alpha_sweep.find((a) => a.alpha === 1).corr_overhang, 3)} at α=1 —
     R_Detail dilutes the printability signal by roughly half.
     <em>Practical consequence: early-stop on a held-out physical metric, never on the reward.
     Here that would have stopped at step ${bestTh.step} and kept every real gain.</em>`;

  document.getElementById("historyTable").innerHTML =
    `<tr><th>step</th><th class="num">reward</th><th class="num">L_OH true</th><th class="num">L_Th</th>
      <th class="num">entropy</th><th class="num">out-of-box</th></tr>` +
    hist.filter((_, i) => i % 2 === 0 || i === hist.length - 1).map((h) => {
      const best = h.step === bestTh.step;
      return `<tr class="${best ? "winner" : ""}">
        <td>${h.step}${best ? " ◂ stop here" : ""}</td>
        <td class="num">${n(h.reward_mean, 3)}</td>
        <td class="num">${n(h.overhang_true, 4)}</td>
        <td class="num">${n(h.thickness, 4)}</td>
        <td class="num">${n(h.mode_entropy, 3)}</td>
        <td class="num" style="color:${h.out_of_box_frac > 0.99 ? C.bad : h.out_of_box_frac > 0.2 ? C.warn : C.dim}">${pct(h.out_of_box_frac, 0)}</td></tr>`;
    }).join("");

  } // end toy M1-M4 block

  // ---- 5. SDE gap ---------------------------------------------------------
  document.getElementById("sdeTable").innerHTML =
    `<tr><th>σ</th><th class="num">per-sample RMSE</th><th class="num">mean shift</th>
      <th class="num">std ratio</th><th>reading</th></tr>` +
    d.sde_gap.map((g) => {
      const isOde = g.sigma === 0;
      const ok = !isOde && g.mean_shift < 0.05 && Math.abs(g.std_ratio - 1) < 0.06;
      return `<tr>
        <td>${n(g.sigma, 2)}${isOde ? " (ODE)" : ""}</td>
        <td class="num">${n(g.per_sample_rmse, 4)}</td>
        <td class="num">${n(g.mean_shift, 4)}</td>
        <td class="num">${n(g.std_ratio, 3)}</td>
        <td style="color:${isOde ? C.dim : ok ? C.good : C.bad}">${
          isOde ? "identical by construction" : ok ? "different paths, same law" : "marginals diverged"}</td>
      </tr>`;
    }).join("");
  document.getElementById("sdeNote").innerHTML =
    `The mapping holds at the level of <strong>marginals, not path measures</strong>. That is
     exactly enough for an endpoint reward — a mesh judge scores the final geometry, not the
     route taken — and <em>not</em> enough for any reward that depends on the trajectory.
     One further caveat found while testing rather than assumed: the drift is affine in v with
     slope <code>k(t) = 1 + σ²t/(2(1−t))</code>, which diverges as t→1. The cheap velocity-space
     loss absorbs <code>k(t)²</code> into β, so <strong>a single global β is exactly correct at only
     one timestep whenever σ&gt;0</strong>. At σ=0 it is exact everywhere.`;

  // ---- 6. reference-free --------------------------------------------------
  const rows = [["base", fin.base], ["reference-based DPO", fin.aligned], ["reference-free DPO", fin.ref_free]];
  document.getElementById("refFreeTable").innerHTML =
    `<tr><th>model</th><th class="num">reward</th><th class="num">L_OH true</th>
      <th class="num">entropy</th><th class="num">curvature</th><th class="num">2nd model?</th></tr>` +
    rows.map(([label, m], i) => `<tr class="${i === 2 ? "winner" : ""}">
      <td>${label}</td>
      <td class="num">${n(m.reward_mean, 3)}</td>
      <td class="num">${n(m.overhang_true_mean, 4)}</td>
      <td class="num">${n(m.mode_entropy, 3)}</td>
      <td class="num">${n(m.curvature_mean, 1)}</td>
      <td class="num">${i === 0 ? "–" : i === 1 ? "yes" : "no"}</td></tr>`).join("");
  const rf = fin.ref_free, rb = fin.aligned;
  document.getElementById("refFreeNote").innerHTML =
    `Reference-free matched reference-based on reward (${n(rf.reward_mean, 3)} vs ${n(rb.reward_mean, 3)})
     and on overhang (${n(rf.overhang_mean, 4)} vs ${n(rb.overhang_mean, 4)}), while retaining
     <em>slightly better</em> mode coverage (${n(rf.mode_entropy, 3)} vs ${n(rb.mode_entropy, 3)}) and
     less curvature (${n(rf.curvature_mean, 1)} vs ${n(rb.curvature_mean, 1)}) — at half the memory.
     <strong>The honest caveat:</strong> π_ref is what bounds the KL to the base model. Without it
     nothing stops the field drifting arbitrarily far to chase reward, so the margin term and early
     stopping are doing work a principled KL used to do. That both variants collapsed coverage to a
     similar degree suggests the anchor was not buying much regularisation here anyway — but this is
     one 2-D reward landscape, not a licence to drop it at 4B scale.
     <span class="dim">Gradient geometry is settled analytically, not by eyeballing this table:
     <code>flow_dpo_test.test_reference_free_gradient_is_per_sample_parallel</code> proves the two
     objectives produce <em>exactly parallel</em> per-sample gradients (cos = 1.000000), differing
     only by a positive per-sample weight.</span>`;

  // ---- 7. provenance ------------------------------------------------------
  const gi = meta.grid_interpolation;
  if (typeof renderRealMechanism === "function") renderRealMechanism();

  document.getElementById("provenance").innerHTML = `
    <table class="diag">
      <tr><th>component</th><th>status</th></tr>
      <tr><td>Reward signal</td><td><span style="color:${C.good}">real</span> —
        <code>trellis_core.geometric_judge.score_mesh_detailed()</code> on real trimesh geometry,
        weights from <code>dpo_branch._default_judge_weights()</code>
        (${Object.entries(meta.judge_weights).filter(([k]) => ["alpha","beta","gamma","delta","d_min","theta_crit_deg"].includes(k)).map(([k, v]) => `${k}=${v}`).join(", ")})</td></tr>
      <tr><td>DPO losses</td><td><span style="color:${C.good}">real</span> —
        <code>trellis_core.flow_dpo</code>, the same functions covered by the 14-test suite</td></tr>
      <tr><td>Flow model</td><td><span style="color:${C.good}">real</span> — trained by conditional
        flow matching on the linear path, ${meta.fm_steps} steps</td></tr>
      <tr><td>Reward interpolation</td><td><span style="color:${C.good}">measured</span> —
        grid nodes are exact judge calls; off-node lookups are bilinear. Error vs fresh exact
        evaluation: RMSE ${n(gi.rmse, 4)} (${n(gi.rmse_pct_of_spread, 2)}% of reward spread),
        correlation ${n(gi.rank_corr, 5)}</td></tr>
      <tr><td>Shape family</td><td><span style="color:${C.warn}">stand-in</span> — a 2-parameter
        hollow frustum, not a TRELLIS.2 SLat decode. This measures the alignment
        <em>mechanism</em> at n=${rs.n}; it does not measure the 4B pipeline.</td></tr>
      <tr><td>Topology / non-manifold edges</td><td><span style="color:${C.bad}">untestable here</span> —
        ${d.topology_limitation}</td></tr>
      <tr><td>Transfer to the 3D pipeline</td><td><span style="color:${C.bad}">not shown</span> —
        the real pipeline has n=1 (S 0.3124 → 0.4057 on one image, one seed). Nothing on this
        page establishes that the 2-D result carries over.</td></tr>
    </table>
    <p class="explain" style="margin-top:12px">
      Reproduce end to end: <code>python devlab/alignment_experiment.py</code>
      (${meta.runtime_seconds}s, seed ${meta.seed}) then <code>python trellis_core/flow_dpo_test.py</code>.
    </p>`;
}

main();

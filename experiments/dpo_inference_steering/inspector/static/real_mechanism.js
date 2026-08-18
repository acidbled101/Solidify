// real_mechanism.js -- panels M1-M4 rendered from REAL TRELLIS.2 traces.
//
// Replaces the TinyFlow toy data that previously backed these panels. Every
// number here comes from devlab/real_mechanism.py, which reads only real
// trace.jsonl files produced by actual 4B-parameter generations.
//
// The honesty rules this file follows, because the real data has sharp edges
// the toy did not:
//   * n is stated on every panel, and it differs BETWEEN panels (scored
//     branches vs branches with latent telemetry). Never implied.
//   * Nothing is pooled across incommensurable configs -- branch_noise_scale
//     spans 0.02-1.0 on disk, so M4 draws one line per branch rather than a
//     mean that would be a config-composition artifact.
//   * Where the real pipeline has no analogue of a toy panel (M1's "base vs
//     aligned field"), the panel says so instead of inventing one.
"use strict";

const RM = {
  reference: "#3ED6C4",
  delta_initial: "#FF9F45",
  delta_final: "#FF6B4A",
  good: "#7CE38B",
  bad: "#FF6B6B",
  warn: "#FFC24B",
  dim: "#8B99A4",
};

function rmN(v, d = 4) {
  if (v === null || v === undefined || Number.isNaN(v)) return "–";
  return Number(v).toFixed(d);
}
function rmStat(k, v, color) {
  return `<div><span class="k">${k}</span><span class="v"${color ? ` style="color:${color}"` : ""}>${v}</span></div>`;
}
function ciText(b) {
  if (!b || b.ci95[0] === null) return `n=${b ? b.n : 0} (too few to bootstrap)`;
  const excludesZero = (b.ci95[0] > 0 && b.ci95[1] > 0) || (b.ci95[0] < 0 && b.ci95[1] < 0);
  return `${rmN(b.mean)} , 95% CI [${rmN(b.ci95[0], 3)}, ${rmN(b.ci95[1], 3)}]` +
         (excludesZero ? " — excludes zero" : " — includes zero");
}

async function renderRealMechanism() {
  let d;
  try {
    const res = await fetch("/api/real-mechanism");
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    d = await res.json();
  } catch (e) {
    const host = document.getElementById("realMechNotice");
    if (host) host.innerHTML = `<div class="banner">could not load /api/real-mechanism — ${e.message}</div>`;
    return;
  }

  const m = d.meta, rs = d.reward_shift, cv = d.curvature, gh = d.grad_step_history;

  document.getElementById("realMechSub").textContent =
    `${m.n_runs_ok}/${m.n_runs_total} real runs · ${m.n_branches_scored} branches scored · ` +
    `${m.n_branches_with_latent} with latent telemetry`;

  // ---- M1: velocity along reference vs delta continuations ---------------
  const perB = cv.per_branch || [];
  const vGroups = perB
    .filter((r) => r.v_rms_reference !== undefined)
    .map((r) => ({
      label: `${r.run_id.slice(9, 15)}·${r.branch_index}`,
      values: {
        reference: r.v_rms_reference,
        delta_initial: r.v_rms_delta_initial,
        delta_final: r.v_rms_delta_final,
      },
    }));
  if (vGroups.length) {
    groupedBars(document.getElementById("m1Chart"), {
      groups: vGroups,
      series: [
        { key: "reference", label: "reference", color: RM.reference },
        { key: "delta_initial", label: "delta (random)", color: RM.delta_initial },
        { key: "delta_final", label: "delta (steered)", color: RM.delta_final },
      ],
      width: 880, height: 240,
      title: "‖v_θ‖ RMS along each continuation, per real branch",
    });
  } else {
    document.getElementById("m1Chart").innerHTML =
      `<div class="callout" style="border-left-color:${RM.warn}">No branch in any real run carries
       velocity telemetry yet.</div>`;
  }
  const vMeans = ["reference", "delta_initial", "delta_final"].map((w) => {
    const vals = perB.map((r) => r[`v_rms_${w}`]).filter((v) => v !== undefined);
    return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
  });
  document.getElementById("m1Note").innerHTML =
    `<strong>This is not the toy's "base field vs aligned field" comparison, because the real
     pipeline has no such pair.</strong> There is exactly one frozen <code>v_θ</code>
     (<code>frozen_parameters()</code> asserts it) and DPO here steers the <em>input latent</em>,
     not the weights. The real analogue is the same field evaluated along the two trajectories the
     pipeline actually explored: the unperturbed reference continuation vs the perturbed-then-steered
     one, from the same fork.
     Mean ‖v_θ‖: reference ${rmN(vMeans[0], 3)}, random ${rmN(vMeans[1], 3)},
     steered ${rmN(vMeans[2], 3)} across ${perB.length} branches.`;

  // ---- M2: path straightness --------------------------------------------
  // Deliberately a per-branch bar chart rather than a histogram: at n=7 a
  // density plot buries the reference (a narrow spike at ~1.0) underneath the
  // much wider delta distributions, and invents smoothness the sample size
  // does not support. Showing all 7 observations directly is both honest and
  // more legible.
  const tGroups = perB
    .filter((r) => r.tortuosity_reference !== undefined)
    .map((r) => ({
      label: `${r.run_id.slice(9, 15)}·${r.branch_index}`,
      values: {
        reference: r.tortuosity_reference,
        delta_initial: r.tortuosity_delta_initial,
        delta_final: r.tortuosity_delta_final,
      },
    }));
  if (tGroups.length) {
    groupedBars(document.getElementById("m2Chart"), {
      groups: tGroups,
      series: [
        { key: "reference", label: "reference", color: RM.reference },
        { key: "delta_initial", label: "delta (random)", color: RM.delta_initial },
        { key: "delta_final", label: "delta (steered)", color: RM.delta_final },
      ],
      width: 880, height: 240,
      title: `path tortuosity per real branch (1.0 = perfectly straight, n=${cv.n_branches})`,
    });
  } else {
    document.getElementById("m2Chart").innerHTML =
      `<div class="chart-empty">no branch has enough trajectory points to measure</div>`;
  }
  const tRef = cv.tortuosity_reference_mean, tRand = cv.tortuosity_delta_initial_mean,
        tSteer = cv.tortuosity_delta_final_mean;
  document.getElementById("m2Note").innerHTML =
    `<strong>On real data the perturbed paths are measurably less straight than the reference,
     and steering makes them less straight still.</strong>
     Mean tortuosity: reference <code>${rmN(tRef, 4)}</code> (essentially a straight line),
     random perturbation <code>${rmN(tRand, 4)}</code>, steered <code>${rmN(tSteer, 4)}</code>.
     The reference continuation is nearly perfectly straight because it is only
     <code>continuation_steps</code> (2-3) Euler steps of an already-converging trajectory —
     there is very little room to bend. The delta branches bend because the perturbation displaces
     the latent and the field then pulls back toward the data manifold.
     <span class="dim">n=${cv.n_branches} branches, from the runs carrying LatentProbe telemetry.
     Tortuosity is scale-free and basis-independent, so it pools across runs even though the raw 2-D
     projections do not (different voxel counts reseed the projection basis).</span>`;

  // ---- M3: reward distribution shift ------------------------------------
  overlayHistogram(document.getElementById("m3Chart"), {
    series: [
      { hist: rs.reference, color: RM.reference, label: "reference" },
      { hist: rs.delta_initial, color: RM.delta_initial, label: "delta (random)" },
      { hist: rs.delta_final, color: RM.delta_final, label: "delta (steered)" },
    ],
    width: 880, height: 210,
    title: `real judge score across every scored fork (n=${rs.n})`,
    xLabel: "composite judge score S",
  });
  document.getElementById("m3Stats").innerHTML = [
    rmStat("reference mean", rmN(rs.reference_mean), RM.reference),
    rmStat("random mean", rmN(rs.delta_initial_mean), RM.delta_initial),
    rmStat("steered mean", rmN(rs.delta_final_mean), RM.delta_final),
    rmStat("n forks", rs.n),
  ].join("");
  const steerWorse = rs.delta_final_mean < rs.reference_mean;
  document.getElementById("m3Note").innerHTML =
    `<strong>The real reward distribution shifts the wrong way.</strong>
     Random perturbation vs reference: ${ciText(rs.delta_initial_vs_reference)}.
     Steered vs reference: ${ciText(rs.delta_final_vs_reference)}.
     ${steerWorse
       ? `Both the random perturbation and the steered result score <em>below</em> the unperturbed
          reference on average — the opposite of the toy, where alignment produced a large rightward
          shift (Cohen's d = 2.28 at n=3000).`
       : `The steered candidates score above the reference on average.`}
     <span class="dim">This is the honest headline of the whole exercise: the mechanism demonstrably
     works on a toy where the reward is a clean function of two parameters, and does not
     demonstrably work on the real pipeline at the n available. The re-rank gate means this costs
     compute rather than output quality — the pipeline resumes from the reference whenever the
     steered candidate loses, which on these traces is most of the time.</span>`;

  // ---- M4: gradient steps ------------------------------------------------
  const traces = (gh.per_branch || []).filter((b) => b.proxy_gap_relative && b.proxy_gap_relative.length > 1);
  if (traces.length) {
    // Pad to the longest trace so every series shares an x axis; lineChart
    // ignores trailing nulls rather than drawing them as zeros.
    const maxLen = Math.max(...traces.map((t) => t.proxy_gap_relative.length));
    lineChart(document.getElementById("m4Chart"), {
      series: traces.map((t) => ({
        values: t.proxy_gap_relative.concat(Array(maxLen - t.proxy_gap_relative.length).fill(null)),
        color: t.branch_noise_scale >= 0.5 ? RM.bad
             : t.branch_noise_scale >= 0.1 ? RM.delta_final : RM.delta_initial,
      })),
      width: 880, height: 240, yLabel: "proxy gap ÷ own step-0 gap",
      title: "detail-proxy gap over gradient steps — one line per real branch",
      refLines: [{ value: 1.0, label: "unchanged from step 0", color: RM.dim },
                 { value: 0.0, label: "gap fully closed", color: RM.good }],
    });
  } else {
    document.getElementById("m4Chart").innerHTML = `<div class="chart-empty">no gradient traces</div>`;
  }
  const cs = gh.case_study, rsum = gh.rms_summary;
  document.getElementById("m4Note").innerHTML =
    `<strong>Each line is one real branch, normalised to its own starting gap — deliberately not a
     pooled mean.</strong> <code>branch_noise_scale</code> spans 0.02–1.0 across these runs, and the
     raw proxy gap at step 0 ranges from −0.06 to +1563 as a direct result. An earlier version of
     this panel averaged them and produced a dramatic "drop" that was entirely the large-noise
     branches leaving the cohort, not a trend. Colour encodes noise scale
     (<span style="color:${RM.delta_initial}">low</span>,
      <span style="color:${RM.delta_final}">mid</span>,
      <span style="color:${RM.bad}">high</span>).
     ${cs ? `Deepest real case study: <code>${cs.run_id}</code> branch ${cs.branch_index},
       ${cs.n_steps} gradient steps at noise scale ${cs.branch_noise_scale}.` : ""}
     ${rsum ? `<br><br><strong>The trust region never binds.</strong> Across
       ${rsum.n_observations} recorded gradient steps, delta's RMS sits at
       <code>${rmN(rsum.mean_frac_of_trust_region, 4)}</code> of its cap
       (range ${rmN(rsum.min, 3)}–${rmN(rsum.max, 3)}), never above
       ${rmN(rsum.max, 3)}. <code>trust_region = branch_noise_scale × delta_max_norm_ratio</code>
       with the ratio at 3.0, and the steered delta stays at essentially its initial random
       magnitude — so the gradient steps <em>rotate</em> delta far more than they grow it, and
       <code>project_delta_</code>'s hard cap has never once been the binding constraint on any
       real run.` : ""}`;

  document.getElementById("realMechNotice").innerHTML = "";
}

# Inference-time preference steering for TRELLIS.2 — experiment retrospective

**Branch:** `physics-aware-pipeline` · **Period:** 2026-07-31 → 2026-08-05
**Status:** concluded. The approach was retired in favour of the weight-space
LoRA fine-tune that ships; see [`../../training/README.md`](../../training/README.md).

This is the closing report for the DreamDPO-style inference-time steering
experiment. It records what was built, how it was measured, what the
measurements said, and — the part worth keeping — which of our own claims did
not survive contact with data. Written before the worktree is repurposed, so
that the next approach starts from what was actually learned rather than from
what we hoped.

---

## 1. The hypothesis

From `experiments/dpo_inference_steering/DESIGN_NOTE.md` (2026-07-31): a single-stage latent flow
model has a differentiable path from final geometry back to the generative
latent, which multi-stage image→multiview→mesh pipelines do not. If so, we
should be able to intercept the SLat sampling ODE mid-flight, fork two
candidate trajectories, rank their decoded meshes by a physical printability
judge, and use that preference to steer the trajectory toward printable
geometry — at inference time, with no training, no weight updates, and no
preference dataset.

The intended payoff: printability enforced *during* generation rather than
patched afterward by `printable.py`'s heuristic repair.

## 2. What was built

| Component | File | Size |
|---|---|---|
| Geometric judge (S = αR_Detail − βL_OH − γL_Th − δL_Topo) | `evaluation/geometric_judge.py` | 8 tests |
| Fork / steer / re-rank / resume sampler | `experiments/dpo_inference_steering/dpo_branch.py` | 1785 lines, 23 tests |
| Flow-matching DPO theory + 2-D validation | `experiments/dpo_inference_steering/flow_dpo.py` | 15 tests |
| Differentiable printability objectives | `experiments/dpo_inference_steering/printability_proxy.py` | 10 tests |
| Live inspector dashboard + run harness | `experiments/dpo_inference_steering/inspector/` | 4 tests |
| Pooled-run analysis | `experiments/dpo_inference_steering/inspector/metrics.py` | 378 lines |

60 tests total. 22 recorded runs in `experiments/dpo_inference_steering/inspector/traces/`.

### How the steering step works

Run the ordinary Euler sampler to a branch timestep `t_branch ∈ [0.3, 0.7]`.
Fork two candidates: branch A is the unperturbed continuation (ε=0, detached,
standing in for `P_ref`); branch B carries a learnable per-voxel perturbation
`delta ~ N(0, 0.02²)`. Continue both differentiably for `k=2` Euler steps,
decode both to meshes, and rank them with the judge. Convert that discrete
verdict into a signed margin on a differentiable latent proxy:

```
sign = +1 if the judge preferred the delta branch else −1
L    = −log σ( β · sign · (s(x_B) − s(x_A).detach()) )
```

Take 3 SGD steps on `delta` under a hard RMS trust region, re-decode, re-judge
against the reference, and resume the original schedule from whichever branch
the judge actually prefers.

`s` is `slat_detail_proxy` — the umbrella-Laplacian energy of the SLat features
over the voxel adjacency graph. It is the latent-space analogue of the judge's
`R_Detail` term, and it was the *only* differentiable scalar available, because
overhang, thickness and topology only exist after mesh extraction.

### Engineering that was genuinely required

- **Gradient access.** `FlowEulerSampler.sample_once` is `@torch.no_grad()`.
  `differentiable_continuation()` re-implements its three lines against the
  undecorated helpers under `enable_grad`, which is the only reason any
  gradient exists at all.
- **Memory.** The differentiable segment retained ~38 GB of activations and
  reliably OOMed a 36 GB unified-memory ceiling. `checkpointed_blocks()` trades
  recompute for that; a reference-based objective needing a second frozen model
  copy produced a measured 36.27 GiB OOM and is why the design went
  reference-free.
- **Backend.** `mtlgemm`'s Metal sparse-conv kernel has no reliable registered
  backward; `SPARSE_CONV_BACKEND=none` is forced around — and only around — the
  single call that runs `.backward()`.
- **Fairness.** The judge scores candidate pairs under common random numbers.
  Dropping CRN biased a controlled comparison by 0.0043, the same order as the
  effects being measured.

## 3. How it was measured

Three independent layers, deliberately not one:

1. **Unit tests** against dummy models on CPU, so the math is checkable without
   a GPU or a checkpoint.
2. **A 2-D toy** (`experiments/dpo_inference_steering/inspector/alignment_experiment.py`, ~104 s, n=3000/model) with a
   closed-form geometry family, where the *alignment mechanism* can be isolated
   from the 4B-parameter pipeline.
3. **Pooled real runs** (`experiments/dpo_inference_steering/inspector/metrics.py`) flattening every trace on disk
   into per-fork rows. The decisive column is `verdict_flipped`: `dpo_branch`
   records the judge's verdict on the random perturbation and again after the
   gradient steps. If those never disagree, the gradient adds nothing the
   random draw did not.

Finally, an adversarial pass: two independent agents, one tasked with proving
the theory and one with breaking it, with every load-bearing claim from both
re-verified by direct execution before being accepted.

## 4. Results

### 4.1 The alignment mechanism works — in 2-D

n=3000 per model, reward from the real judge:

| model | reward | L_OH (raw) | L_OH true wall | mode entropy | out-of-box |
|---|---|---|---|---|---|
| base | 7.367 | 0.0838 | 0.0686 | 1.790 | 1.6% |
| reference-based DPO | 19.460 | 0.0606 | **0.0000** | 0.317 | **100%** |
| reference-free DPO | 19.468 | 0.0579 | **0.0000** | 0.417 | **100%** |

Reward shift Δ=12.09, 95% CI [11.91, 12.29], Cohen's d=2.28. True wall overhang
goes to zero. The preference machinery does what it says.

But: **the optimum is a parameterisation artifact.** All 3000 aligned samples
land outside the latent box and collapse to *one* geometry, sitting on a clamp
inside the toy's own mesh constructor. DPO found the argmax of the proxy, and
the argmax was an artifact of how the proxy was parameterised. Critically this
happened under the **reference-based** objective — the KL anchor did not
prevent it, because the velocity-space surrogate contains no term penalising
distance from the reference.

And reward hacking appeared in thickness: `L_Th` bottoms at 0.0279 at step 120
then degrades **7.4×** to 0.2069 by step 499, while reward buys its last 9% and
the loss curve looks healthy throughout.

### 4.2 The steering does not work in 3-D

Pooled across every real fork on disk (`experiments/dpo_inference_steering/inspector/traces/metrics.json`, n=17 scored
branches from 10 completed runs):

| quantity | value |
|---|---|
| mean gain from gradient steering | **−0.0027 ± 0.0078** (SE) |
| steering helped / hurt | 8 / 9 forks |
| verdict flipped by steering | 5/17 (rate 0.29) |
| distinguishable from zero? | **no** |

An earlier, stricter subset (6 runs / 10 forks, `THEORY.md` §3) gave
−0.0049 ± 0.0115 with the *perturbation* gain positive at +0.0067, where the
current pooled table has it at −0.025. **The two disagree on the sign of the
perturbation term.** That disagreement is itself a finding: at n≈17, filter
choices move conclusions as much as effects do.

What this does *not* say: it does not say the output got worse. The re-rank gate
only resumes from the steered candidate when the judge prefers it, so a negative
steering gain costs compute, not quality.

### 4.3 Why: the gradient optimises something else

The optimiser works perfectly — the proxy moves as instructed 15/17 times. The
objective is the problem:

- within-fork top-1 agreement (proxy argmax == judge argmax): **1/17 = 5.9%**,
  against a 33% chance line (binomial p = 0.018)
- mean within-fork Spearman(proxy, S) = **−0.265**
- `sd(ΔR_Detail) = 0.035` vs `sd(ΔL_OH) = 0.00145` — a 24× gap, so the one term
  the proxy models is the one the judge barely varies on

We were ascending a hill that is, if anything, mildly anti-correlated with the
goal.

### 4.4 The judge is its own noise floor

Within-fork `ΔL_Th` has test-retest reliability **0.208**, which mathematically
caps *any* proxy's correlation with the judge at **|r| ≤ 0.456**. Two agents
reached this independently by different routes. On near-identical candidates
the judge's preference label is close to a coin flip — and near-identical
candidates are exactly the modal case inside a fork.

The root cause is a single stochastic ray-cast sampler in `thickness_penalty`.
The other three terms are exact.

### 4.5 The judge measures the wrong thing

`overhang_penalty` is purely normal-based — no z_min, no support test. Measured
on real meshes:

- only **~5%** of measured `L_OH` is geometry that genuinely needs support
- ~67% is build-plate contact, which every slicer scores zero
- `corr(L_OH_judge, L_OH_truly_unsupported)` = +0.049 Pearson, **−0.432 Spearman**

Replicating the judge faithfully would give a proxy that is *rank-wise opposed*
to real support cost. A cube resting flat on the plate scores L_OH = 0.0732,
entirely its bottom face.

### 4.6 The differentiability wall was not there

The premise that justified using a latent proxy at all — "mesh extraction blocks
the gradient" — is false. TRELLIS.2's shape decoder is dual-grid
(FlexiCubes-family) and **directly regresses vertex positions**:

```python
vertices = (1 + 2*margin) * F.sigmoid(h.feats[..., 0:3]) - margin   # smooth
intersected = h.feats[..., 3:6] > 0                                  # boolean
```

Positions are differentiable; only *connectivity* is not. Holding connectivity
fixed and differentiating through positions is the standard DMTet/FlexiCubes
recipe. `printability_proxy.py` implements R_Detail, L_OH and L_Th as exactly
differentiable functions using the judge's own formulas, and
`test_gradient_flows_through_extraction` verifies it by calling the *real*
extractor, not a mock.

Against a de-noised judge this reaches r = 0.793; against true physical
printability, 0.760. Against the judge **as deployed**, no proxy can exceed
0.456 — the ceiling in §4.4.

## 5. What was learned

**1. A preference method's weakest link is wherever the gradient comes from,
not wherever the preference comes from.** DreamDPO tolerates a weak reward model
because its gradient comes from a frozen diffusion prior that is known-good. We
kept the weak-ranker tolerance and replaced the known-good gradient source with
a proxy. That single substitution accounts for the entire negative result.

**2. Validate the reward before building anything on it.** Three of this
project's four measured failures are properties of `geometric_judge`, not of
DPO: the thickness sampler's 0.208 reliability, the overhang term's −0.432 rank
correlation with real support cost, and R_Detail diluting the printability
signal by roughly half. All three were measurable on day one from meshes already
on disk, with zero GPU time.

**3. "Not differentiable" is a claim about an implementation, not about
mathematics — check it.** Six weeks of design followed from an unverified
assertion. The check took one `sed` of two source files.

**4. Pooled metrics lie when variance is hierarchical.** 96.8% of S variance is
*between* forks. On a pooled metric, vertex count scores r=0.959 and a
*perfect* within-fork ranker scores 0.180. Any evaluation here must use
within-fork paired deltas; both agents independently converged on this.

**5. Build the fail-safe.** The re-rank gate is why a null result cost compute
instead of quality. Every speculative optimisation step should be gated on an
independent check that it actually helped.

**6. Instrument for the null hypothesis.** `delta_branch_won_initial` vs
`delta_branch_won_final` was written into the report struct *before* any run,
specifically so the question "is the gradient doing anything?" would be
answerable from data already collected. It is what ended the project cleanly
rather than ambiguously.

**7. Adversarial review found things careful solo work did not.** The Prover
overturned the project's founding premise; the Skeptic overturned three of our
own published statistics. Neither was reachable by re-reading our own reasoning.

## 6. Claims we published and later had to retract

Kept deliberately, because the pattern is the lesson.

| Claim | Correction |
|---|---|
| "Mesh extraction blocks the gradient" | False. Vertex positions are differentiable; only connectivity isn't. |
| "r = +0.20, so the gradient optimises the wrong thing" | Statistic reproduces but is leverage-driven: Spearman +0.054, CI [−0.31, +0.62], and dropping one point flips the sign. Conclusion was right; the evidence cited for it was not. |
| "Best-of-5 beat steering" | True on `S`, false on printability. |
| "Overhang improved 27%" | Was measuring build-plate footprint. The real result (true wall overhang → 0) was *better* than the one reported. |
| "DPO did not straighten paths" | Metric error — `path_curvature` scales as displacement². Tortuosity says 1.66× straighter. |
| "R_Detail:L_OH dominance is 93:1" | On real meshes it is 3.4:1. A wrong figure given to both review agents. |

Six retractions in five days, every one caught by measurement rather than by
review. The methodological takeaway is that the rate of self-correction was
roughly proportional to the amount of instrumentation, not to the amount of
care.

## 7. What survives

Reusable as-is by the next approach:

- **`geometric_judge.py`** — after the two repairs in §4.4/§4.5. Its ranking
  interface and CRN pairing are sound.
- **`printability_proxy.py`** — 10 passing tests; the differentiable
  formulation is correct and independent of how preferences are used.
- **`flow_dpo.py`** — the SDE/ODE equivalence, the exact score identity on the
  linear path, and the reference-free gradient analysis are general results
  about flow-matching DPO, verified to <1e-5.
- **`experiments/dpo_inference_steering/inspector/`** — the run harness, trace format, and pooled-metrics analysis
  transfer directly to evaluating a trained model.
- **`checkpointed_blocks`, `sparse_conv_backend`, `frozen_parameters`** — the
  MPS memory and autograd plumbing is the same for training.
- **22 trace directories** — every real run's telemetry. Irreplaceable; each
  cost 9–60 minutes of GPU time.

Retired:

- `dpo_branch.py`'s steering path, `slat_detail_proxy`, `steer_delta`,
  `preference_loss`, `project_delta_` — superseded.
- `dpo_branch.py`'s **best-of-N path** is worth keeping separately: it is the
  compute-matched baseline any future method must beat.

## 8. Preservation

All work after 2026-07-31 is **uncommitted**. `origin/physics-aware-pipeline`
is level with local `HEAD`, so nothing built in the last five days exists
anywhere but this directory. Before repurposing:

```bash
cd dpo-worktree
git add -A && git commit -m "Archive inference-time DPO steering experiment"
git push origin physics-aware-pipeline
```

The trace directories are the part that cannot be regenerated cheaply.

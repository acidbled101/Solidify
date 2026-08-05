# Project plan — weight-space DPO for TRELLIS.2 shape generation

**Supersedes:** `3D_PRINTING_AWARE_PIPELINE.md` and the inference-time steering
approach retired in `experiment_retrospective.md`.
**Objective function:** already written and tested —
`trellis_core/flow_dpo_objective.py`.

---

## 1. Goal

Fine-tune TRELLIS.2's shape flow model so that its *default* output is more
printable, using a preference dataset labelled by the geometric judge. Not
inference-time steering: real weight updates, a frozen reference model, and an
offline dataset.

**Success is defined once, up front:** on held-out conditioning images, the
tuned model beats best-of-N sampling from the base model **at matched inference
compute**, measured by a held-out physical printability metric that was never
part of the training reward. Anything less is a null result, and the plan below
is built so a null result arrives early and cheaply.

## 2. Why this is a different bet

The retired approach failed for one structural reason: the gradient came from a
differentiable proxy for the judge, and that proxy's within-fork agreement with
the judge was 5.9% against a 33% chance line. Weight-space DPO removes the proxy
entirely — the judge runs offline, produces labels, and never needs a gradient.
The gradient comes from the model's own flow-matching objective, differenced
against a frozen reference.

| | Retired approach | This plan |
|---|---|---|
| Optimised | a latent perturbation, discarded per fork | model weights, persisted |
| Gradient source | `slat_detail_proxy` (agreement 5.9%) | flow-matching error vs. frozen reference |
| Judge must be | differentiable-adjacent | offline and non-differentiable |
| Judge noise | corrupts every gradient step | filtered out at dataset-build time |
| Optimisation horizon | 3 SGD steps per fork | thousands of steps over a dataset |
| Cost | at inference, every generation | once at training; inference unchanged |

## 3. Phases and gates

Each phase ends in a gate with a number. **If a gate fails, stop and re-plan —
do not proceed on hope.** The retired project's central mistake was building
three layers on an unvalidated reward.

---

### Phase 0 — Preserve and reset (½ day)

1. Commit and push everything on `physics-aware-pipeline` (see
   `experiment_retrospective.md` §8). Nothing since 2026-07-31 exists anywhere
   else, including all 22 trace directories.
2. Branch `flow-dpo-training` from it.
3. Keep: `geometric_judge.py`, `printability_proxy.py`, `flow_dpo.py`,
   `flow_dpo_objective.py`, `devlab/`, and `dpo_branch.py`'s **best-of-N path**
   (it is the baseline this project must beat).
4. Delete the steering path: `slat_detail_proxy`, `steer_delta`,
   `preference_loss`, `project_delta_`, and the fork/resume machinery.

**Gate 0:** `origin/flow-dpo-training` contains every trace directory.

---

### Phase 1 — Repair the judge (2–3 days, no GPU)

The judge is the dataset's only source of truth. Two measured defects make it
unfit as-is.

1. **Determinism.** `thickness_penalty`'s ray-cast sampler gives within-fork
   `ΔL_Th` a test-retest reliability of **0.208**. Seed it from a mesh hash, or
   switch to deterministic every-k-th-face sampling. The other three terms are
   already exact — this one sampler is the entire noise floor.
2. **Support awareness.** `overhang_penalty` is normal-only: ~5% of what it
   measures genuinely needs support, ~67% is build-plate contact, and its rank
   correlation with true unsupported area is **−0.432**. Add a downward raycast
   ("is there material beneath?"); bed-exclusion alone is not enough (r=+0.31).
3. Re-run the existing 8 judge tests plus new determinism tests.

**Gate 1**, on the ~70 real meshes already on disk — no new GPU time:
- `ΔL_Th` test-retest reliability **≥ 0.90** (from 0.208)
- Spearman(`L_OH_judge`, `L_OH_truly_unsupported`) **≥ +0.70** (from −0.432)

Failing this, every later phase trains on coin flips. This phase is not optional
and it is cheap.

---

### Phase 2 — Variance pre-flight (1 day, ~4 h GPU)

**The decisive question: is the shape-SLat flow model even the right model to
train?**

TRELLIS.2 samples voxel coordinates from a *separate* sparse-structure flow
model, then the SLat model fills features onto those coords. DPO requires the
winner and loser to share a coordinate set (`assert_pairable`), so the dataset
must fix coords per condition — which means **any printability variance carried
by the coarse voxel structure is outside the reach of SLat training.**

Overhang and wall thickness are substantially properties of gross shape. If most
of S's variance lives between coordinate sets, we would be training the wrong
model, exactly as the retired project ascended the wrong proxy.

Protocol:
1. Pick 6 conditioning images.
2. For each: sample **1** structure → 8 SLat candidates on those fixed coords →
   decode → judge. Gives `std_within`.
3. For each: sample **8** independent structures → 1 SLat each → decode → judge.
   Gives `std_between`.
4. Report the variance decomposition, per condition and pooled. Use within-group
   paired deltas, never a pooled correlation — the retired project measured a
   *perfect* within-fork ranker scoring 0.180 on a pooled metric while raw vertex
   count scored 0.959.

**Gate 2:** `std_within² / (std_within² + std_between²) ≥ 0.40`.

- **Pass** → train the shape-SLat flow model as planned.
- **Fail** → the target is the **sparse-structure flow model** instead. That is
  a *better* problem: it operates on a dense 3D tensor, so there is no shared-
  support constraint at all, the model is smaller, and the objective in
  `flow_dpo_objective.py` applies unchanged with `sample_index=None`.

Either outcome is a good outcome. This is the cheapest decision-relevant
measurement in the project — do it before building any dataset.

---

### Phase 3 — Dataset builder (3–4 days + generation time)

`devlab/dataset_builder.py`. For each conditioning image:

1. Sample the structure stage **once**; freeze `coords`.
2. Sample N=8 SLat candidates on those coords, varying only the SLat noise.
3. Decode and judge all N under common random numbers.
4. Emit pairs, **storing latents in normalized space** (`normalize_slat` — the
   pipeline de-normalizes after sampling; getting this wrong silently rescales
   every error term).
5. **Filter by score gap.** This is DreamDPO's τ, relocated from the loss to the
   dataset. Drop pairs whose |ΔS| is inside the judge's own repaired noise band;
   pass the surviving gap through as `pair_weight`.
6. Cache reference velocities: fix K=4 `(t, ε)` draws per pair, run the frozen
   model once, store `v_ref`. This removes an entire model copy and two forward
   passes from every training step — the single biggest memory win available.

**Compute is the binding constraint.** Measured on this Mac from
`devlab/traces/metrics.json`: a vanilla 512/12-step run takes **921–1290 s**
(n=2), branched runs 518–3640 s (median 1071 s, n=10). At 15 min/candidate, 200
conditions × 8 candidates is **~400 GPU-hours**. Mitigations, in order:

- Per-candidate cost on shared coords is only SLat-sample + decode + judge, not
  a full pipeline run. **Measure this split first** — it sets the real budget.
- **Batch candidates.** Shared coords means identical token count, so N
  candidates may batch into one forward. Potentially an N× win; prototype it
  before committing to a serial generation run.
- Develop and validate the whole pipeline at 20 conditions × 4 candidates on the
  Mac, then generate the real dataset on rented CUDA. A 400-hour serial job on
  a machine that also has a GPU watchdog is not a good plan.

**Gate 3:**
- ≥ 1000 surviving pairs (be realistic: Diffusion-DPO used ~10⁵ Pick-a-Pic
  pairs — we are firmly in the small-data regime, which is why Phase 4 uses
  LoRA and why the success criterion in §1 is a measurable shift, not a
  transformation)
- **Label reproducibility ≥ 0.90**: re-judge a held-out 10% with fresh seeds;
  the same candidate must win. This is Gate 1 re-tested end-to-end on the
  actual data.

---

### Phase 4 — Training (3–4 days)

`trellis_core/flow_dpo_train.py`, using `flow_dpo_objective.flow_dpo_loss`.

- **LoRA on the flow model**, not a full fine-tune. Small dataset, 36 GB
  ceiling, and it makes the reference model free (base weights *are* the
  reference — drop the LoRA adapter to get `v_ref`).
- Batch size 1 + gradient accumulation; reuse `checkpointed_blocks` and force
  `SPARSE_CONV_BACKEND=none` around the backward.
- `sft_weight > 0`. The DPO objective is invariant to degrading both branches
  as long as the loser degrades faster; this repo has already measured that
  failure (reward rising while L_Th degraded 7.4×, all 3000 samples collapsing
  to one geometry — **under the reference-based objective**, so the KL anchor
  does not save you).
- Sweep `β` first (O(2e3–5e3) for mean-reduced errors). It is the one
  hyperparameter that matters.
- **Never early-stop on the reward.** Held-out physical metric only.
- Log every step: `reward_acc`, `margin`, and both raw flow-matching errors.
  If `err_theta_w` rises while `margin` widens, the model is hacking the
  difference — stop and raise `sft_weight`.

**Gate 4:** on held-out pairs, implicit-reward accuracy **> 0.65**, with the
winner's raw flow-matching error no worse than the reference's.

---

### Phase 5 — Evaluation (2 days)

The bar from §1, adapted from `devlab/metrics.py`:

1. Held-out conditioning images, never used in training.
2. Three arms at **matched inference compute**: base model 1 sample, base model
   best-of-N (N chosen for parity), tuned model 1 sample.
3. Score with the repaired judge **and** with a physical metric held out of the
   training reward entirely (slicer support volume is the honest choice).
4. Paired comparisons per condition, not pooled — §4 of the retrospective.
5. Report distributional health too: mode entropy and out-of-box rate. The 2-D
   experiment collapsed to a single geometry while its reward tripled.

**Gate 5:** tuned model beats base best-of-N on the held-out physical metric,
with distributional diversity not collapsed.

---

### Phase 6 — Iterate on-policy (optional)

DPO degrades off-policy. If Phase 5 passes, regenerate candidates from the
*tuned* model and repeat Phases 3–5. Two or three rounds; stop when the
held-out metric plateaus.

## 4. Risks

| Risk | Why it might happen | Mitigation |
|---|---|---|
| **Gate 2 fails** | Printability may live mostly in the voxel structure | Pivot to the sparse-structure flow model — cheaper and better-posed. Plan explicitly accommodates this. |
| **Too few pairs** | Generation is ~400 GPU-hours at full scale | LoRA + low rank; batch candidates on shared coords; rent CUDA for the dataset |
| **Reward hacking** | Documented in this repo's own 2-D run | `sft_weight`, early stop on held-out physical metric, log both raw errors |
| **CFG mismatch** | Preferences collected under CFG-tilted sampling; loss is over the raw conditional | Collect pairs at cfg=1, or measure the gap and state it |
| **Cascade ambiguity** | Default `1024_cascade` has two shape flow models; training one lets the other undo it | Develop on the 512 non-cascade path; decide LR vs HR explicitly before scaling |
| **MPS backward gaps** | `mtlgemm` has no reliable registered backward | `conv_none` around backward — already solved and tested |
| **Judge repair changes the target** | Gate 1 alters what "printable" means | Re-baseline all historical numbers after Phase 1; do not compare across the repair |

## 5. Explicitly not doing

- Inference-time steering, latent perturbation, mid-ODE forking.
- Any differentiable proxy for the judge. `printability_proxy.py` stays in the
  tree as a *direct physics loss* option — the honest alternative if DPO fails —
  but it is not part of this plan's critical path.
- Training the SLat VAE, the decoder, or the texture pipeline. All frozen.
- Texture/appearance preferences. Shape only.

## 6. Where this could still be wrong

Stated up front, since the retired project's failure was an unexamined premise:

**The core assumption is that printability preferences are learnable from ~10³
pairs by a LoRA on one stage of a 4B-parameter pipeline.** Nothing here
establishes that. Diffusion-DPO used two orders of magnitude more data for a
much broader target. Gate 4 is the first place this assumption gets tested, and
it is placed before any large compute commitment for that reason.

The fallback if it fails is not "try harder": it is the direct differentiable
physics loss from `printability_proxy.py`, which reaches r=0.760 against true
physical printability and needs no preference data at all. `flow_dpo_theory.md`
§2.2 already argues that three of the judge's four terms never needed preference
machinery — only topology genuinely does. If DPO does not clear Gate 4, that
argument wins by default and should be followed.

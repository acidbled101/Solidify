# Weight-space DPO — what to train, and the proof of concept

**Objective function:** `trellis_core/flow_dpo_objective.py` (written, smoke-tested).
**Purpose of this document:** decide, using only this Mac, whether renting a
bigger machine is justified.

---

## 1. What to train

**Target:** `slat_flow_img2shape_dit_1_3B_512` — the 512 non-cascade shape SLat
flow model. 1.3B parameters, 2.41 GB in bf16.

**Method:** LoRA, not a full fine-tune.

**Everything else stays frozen**: the SLat VAE, both decoders, the texture
models, and the sparse-structure model. This matters more than it sounds — the
**shape decoder is not needed during a training step at all**. It only runs
offline while building the dataset. A training step is two forward/backward
passes through a 2.41 GB model, nothing else.

Why the 512 non-cascade path: the default `1024_cascade` has two shape flow
models with an upsample between them, so training one lets the other undo it.
Settle that later; it is not a proof-of-concept question.

**Memory on a 36 GB Mac:**

| | |
|---|---|
| base weights (bf16, frozen) | 2.41 GB |
| LoRA adapter + Adam state (rank 16) | < 0.5 GB |
| reference model | **0 GB** — precompute `v_ref` offline and pass it in |
| activations, 2 checkpointed fwd/bwd | the binding term; `checkpointed_blocks` already handles it |

Backward through this exact model on MPS is already demonstrated — the retired
steering loop ran it with `SPARSE_CONV_BACKEND=none` plus checkpointing. The
one untested difference is that gradients now flow to *weights* rather than to
a latent, which adds optimizer state but no new activation memory. With LoRA
that state is under half a gigabyte.

## 2. What is actually in doubt

Not the plumbing. Two things:

1. **Is there any headroom?** DPO can only push the model toward the better tail
   of what it already samples. Its ceiling is the **within-coords spread** of
   the target metric. The retired experiment measured `sd(ΔL_OH) = 0.00145`
   within a fork against a population `std(L_OH) = 0.0602` across real meshes —
   if that ratio holds, a *perfect* ranker moves overhang by ~2% of its
   population spread, and no amount of GPU fixes that.
2. **Can training capture the headroom that exists, at a data scale we can
   afford?** Diffusion-DPO used ~10⁵ pairs. We will have ~10².

The PoC answers both, in that order, cheapest first.

## 3. The design: measure headroom, then use a positive control

The mistake to avoid is testing the method and the hardest target at the same
time. If DPO-on-printability fails, you cannot tell whether the method doesn't
work or whether printability was unlearnable — and that ambiguity is exactly
what wasted the last experiment.

So: **train on the easiest real axis first.** `R_Detail` is the positive
control. It is one of the judge's own four terms, it is exact and deterministic
(no repair needed), and it has **24× more within-fork variance** than overhang
(`sd(ΔR_Detail) = 0.035` vs `0.00145`). If DPO cannot move `R_Detail` on this
pipeline, it certainly cannot move overhang, and the answer is no.

If it *can*, you get a measured extraction efficiency, multiply by the measured
headroom for the real axis, and you have a number to make the rental decision
with instead of an intuition.

**The bar, stated as one sentence:** a single sample from the tuned model should
recover a meaningful fraction of the gap between one base sample and
best-of-6 from the base model. That is what "the model learned the preference"
means operationally — the selection got distilled into the weights.

## 4. Steps

### Step 0 — Wire it up (½ day, minimal GPU)

LoRA onto the shape flow model; one training step on 2 synthetic pairs; confirm
loss decreases and `reward_acc` moves. Purely a plumbing check.

**Pass:** a gradient step completes on MPS without OOM, and `margin` widens on a
fixed draw.

### Step 1 — Headroom (4–6 GPU-hours) — **this step alone can end the project**

No training. For 6 conditioning images:

1. Sample the structure stage once; freeze `coords`.
2. Sample 8 SLat candidates on those coords; decode; score all four judge terms.
3. Report, per axis, the within-coords spread against the population spread from
   the ~70 real meshes already on disk.
4. Separately: 8 independent structure samples, 1 SLat each — the between-coords
   spread.

**Also record the per-candidate wall-clock on shared coords.** A full 512/12-step
run measured 921–1290 s, but candidates sharing coords skip structure sampling
and post-processing. This number sets every budget below, so measure it before
planning around it.

**Gate 1a — is there anything to learn?**
`sd_within(R_Detail) ≥ 0.15 × std_population(R_Detail)`.
Below that, preference learning on this stage has no room, full stop.

**Gate 1b — is the SLat model the right stage?**
`sd_within² / (sd_within² + sd_between²) ≥ 0.40`.
If it fails, the printability signal lives in the voxel structure, and the
target becomes the **sparse-structure flow model** (`ss_flow_img_dit_1_3B_64`,
also 1.3B) — a *better* problem: dense 64³ tensor, no shared-support
constraint, and `flow_dpo_objective.py` works unchanged with
`sample_index=None`. Re-run Step 1 against that model and continue.

### Step 2 — Small dataset (~1–2 nights of GPU)

30 conditions × 6 candidates on fixed coords ≈ 180 candidates. From each
condition take the best/worst pair on `R_Detail` plus one mid pair → ~60–90
pairs. Hold out 8 conditions entirely.

- Store latents **normalized** (`normalize_slat`) — the pipeline de-normalizes
  after sampling, and getting this wrong silently rescales every error term.
- Fix 4 `(t, ε)` draws per pair and cache `v_ref`. This is what keeps the
  reference model off the device.

**Pass:** ≥ 60 pairs, and re-scoring a held-out 10% with fresh seeds picks the
same winner ≥ 90% of the time. (`R_Detail` is deterministic, so this should be
100%; if it isn't, something upstream is nondeterministic and must be found
before training.)

### Step 3 — Train (3–8 GPU-hours)

LoRA rank 16, batch 1 + accumulation, `sft_weight > 0`, sweep `β` over
{1e3, 5e3, 2e4}.

Run the **overfit check first**: train on 10 pairs only and confirm train
`reward_acc → 1.0`. If the model cannot overfit 10 pairs, stop — that is a bug,
not a data-scale problem, and no larger machine fixes it.

Log `reward_acc`, `margin`, and both raw flow-matching errors every step. If
`err_theta_w` rises while `margin` widens, the model is exploiting the
difference rather than learning the preference — raise `sft_weight`.

**Pass:** held-out `reward_acc > 0.65`, winner's raw flow-matching error no
worse than the reference's.

### Step 4 — The decision measurement (~1 night of GPU)

On the 8 held-out conditions, three arms:

- base model, 1 sample
- base model, best-of-6 (the baseline the whole approach must beat)
- tuned model, 1 sample

Score `R_Detail`, paired per condition — never pooled. The retired experiment
measured a *perfect* within-fork ranker scoring 0.180 on a pooled metric while
raw vertex count scored 0.959; pooled numbers here are worthless.

Also record mode diversity and out-of-box rate: the 2-D experiment tripled its
reward while collapsing all 3000 samples onto a single geometry.

---

## 5. The gate that justifies renting a machine

```
capture = (tuned_1sample − base_1sample) / (base_bestof6 − base_1sample)
```

**Go** if all three hold on held-out conditions:

- `capture ≥ 0.50` — one tuned sample recovers at least half of what best-of-6
  selection buys
- held-out `reward_acc > 0.65`
- diversity not collapsed (mode entropy within 20% of base)

Then extrapolate before spending anything:

```
expected gain on the real axis ≈ capture × sd_within(axis)     [both measured]
```

Run that for `L_OH` and `L_Th` using Step 1's numbers. If the predicted gain is
smaller than the judge's own measurement noise, **the honest answer is still no**
— even with a passing positive control. Write the number down before you look at
it.

**No-go** if `capture ≈ 0`. That means ~80 pairs is far below what this model
needs, and the extrapolation to a rentable dataset (10³–10⁴ pairs, hundreds of
GPU-hours) is a guess rather than an inference. Don't buy a guess.

## 6. Cost, honestly

Roughly **3–4 days wall-clock**, most of it overnight generation on this Mac.
Step 1 is 4–6 hours and is the highest-information-per-hour measurement in the
project — do it first even if you do nothing else.

## 7. If it fails

The fallback is not "try harder with more compute." It is the **direct
differentiable physics loss** in `printability_proxy.py`, which reaches r=0.760
against true physical printability, needs no preference data, no reference
model, and no dataset generation at all. `flow_dpo_theory.md` §2.2 argues that
three of the judge's four terms never needed preference machinery — only
topology genuinely does. A failed PoC here is that argument winning, and it
saves the rental.

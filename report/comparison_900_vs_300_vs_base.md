# Fine-tuned vs untrained: mesh quality comparison

**Date:** 17 August 2026
**Output:** `runs/comparison_900/` — 20 GLB + 20 STL, raw, no repair applied

## The four models compared

| # | Name | What it is |
|---|---|---|
| 1 | **base** | The untrained TRELLIS.2-4B shape-SLat flow model, exactly as downloaded. No adapter. This is the control. |
| 2 | **sft-1200 @ 1050** | LoRA rank 16 fine-tune on **300** curated Thingi10K objects, 8.7 epochs (2,349 object-passes). Checkpoint at step 1050 of 1200. The previous best model. |
| 3 | **sft-900 @ 5000** | LoRA rank 16 fine-tune on **900** curated Thingi10K objects, 5.75 epochs (5,000 object-passes). Best checkpoint by held-out loss. |
| 4 | **sft-900 @ 5220** | Same run as #3 at its final step, 6.00 epochs (5,220 object-passes). |

Models 2, 3 and 4 use an identical recipe — rank-16 LoRA on the attention and
MLP projections of the flow model's transformer blocks, base weights frozen,
18.68 M trainable parameters, AdamW, cosine schedule, conditioning dropout 0.1.
Only the dataset size and the number of passes differ.

## How this was measured

Five input images: the 3DBenchy reference photo and four real photographs from
the service's own trace log.

Every number is computed **on the mesh the model produces directly** — before
manifold3d, before the trimesh repair path, before any decimation. This matters:
the same mesh measured after a simplification step reports a defect rate roughly
fifty times higher, because simplification itself creates non-manifold edges.

The comparison is **paired**. All four models were asked for the same five
objects from the same starting noise, so the only difference between runs is the
model. Without this the numbers are unusable — two runs of the same base model
on different random draws disagreed by more than a third on some metrics, which
is larger than the effect being measured. As a check that the pairing held, base
reproduced its values from the previous comparison exactly on all five images.

**Non-manifold edge rate** — fraction of edges shared by three or more triangles;
geometry folding back on itself in a way a slicer cannot read as a surface.
**Open edge rate** — fraction of edges belonging to only one triangle; a hole or
border. **Components** — how many disconnected pieces the mesh is in; a clean
object is one piece, a large count means loose fragments. **Detail** — surface
energy, high for sharp features and low for smooth ones; reported so that a
printability gain bought by smoothing everything away would be visible.

## Results, averaged over the five images

| Model | Non-manifold % | vs base | Open % | vs base | Components | vs base | Detail |
|---|---|---|---|---|---|---|---|
| base (no training) | 0.923 | — | 0.984 | — | 6301 | — | 0.772 |
| **sft-1200 @1050** (300 obj) | **0.515** | **−44%** | 0.836 | −15% | **3800** | **−40%** | 0.784 |
| sft-900 @5000 (900 obj) | 0.606 | −34% | 0.802 | −18% | 4298 | −32% | 0.761 |
| sft-900 @5220 (900 obj) | 0.610 | −34% | **0.793** | **−19%** | 4362 | −31% | 0.782 |

## What this says

**Every fine-tuned model beats the untrained model on every printability metric.**
Non-manifold edges fall by a third to nearly a half, loose components by about a
third, open edges by 15–19%.

**Detail does not degrade.** Base scores 0.772; the tuned models score 0.761 to
0.784. The improvement is not the model quietly producing smoother, emptier
shapes — which is the failure mode this metric exists to catch.

**More data did not win.** The 900-object model does not beat the 300-object
model on non-manifold rate: 0.606 against 0.515, and it loses on 3 of the 5
images. This is despite more than twice the total training (5,220 object-passes
against 2,349).

**But the 900-object model is better on open edges** — 0.793 against 0.836,
the best of any model tested. So the two runs are not simply better and worse;
they are better at different things.

### Per-image non-manifold rate

| Image | base | sft-1200 @1050 | sft-900 @5000 | sft-900 @5220 | Best |
|---|---|---|---|---|---|
| 3DBenchy | 0.417 | 0.346 | 0.345 | **0.332** | sft-900 @5220 |
| trace 20260731 | 1.360 | **0.644** | 0.741 | 0.761 | sft-1200 |
| trace 20260803-134 | 1.041 | 0.617 | 0.618 | **0.612** | sft-900 @5220 |
| trace 20260804-121 | 1.030 | **0.459** | 0.739 | 0.742 | sft-1200 |
| trace 20260803-163 | 0.766 | **0.508** | 0.589 | 0.604 | sft-1200 |

## What this does not settle

The two fine-tunes differ in **both** dataset size (300 vs 900) and epochs
(8.7 vs 6.0), so this experiment cannot separate the two. The earlier hypothesis
was that the 300-object model won only because it made more passes over its
data; six passes over 900 objects was meant to test that, and it did not
reproduce the result. That weakens the epochs explanation but does not kill it,
because 6.0 is still fewer passes than 8.7.

Isolating it needs one more run: either 900 objects for 8.7 epochs, or 300
objects for 6.0. On the lab machine each is days of compute; on a rented A100
either is a few hours.

A second possibility worth naming is that the 300-object set was simply a
better or luckier subset, and that the curation quality matters more than
the count. Nothing measured so far distinguishes that from the epoch
explanation.

## A note on what fine-tuning cannot fix

None of these models produce watertight output, and none will. Encoding a
perfectly watertight mesh and decoding it straight back — with no generation at
all — already yields ~0.086% open edges and 953 components. The decoder
introduces its own cracks regardless of how clean the input was, so
watertightness is downstream of everything being trained here and belongs to the
post-processing layer permanently.

What fine-tuning is doing is reducing how much work that repair step has to do,
which is what preserves detail: less repair means less of the model's own
geometry is destroyed on the way to a printable file.

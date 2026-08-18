# Fine-tuning the shape-SLat flow model

How the model that ships in `adapters/` was made, and how to remake it.

## The result

LoRA rank 16 / alpha 32 on `shape_slat_flow_model_512`: **18.68M trainable
parameters across 210 modules**, against a frozen 1.29B-parameter base.
Supervised flow matching on 300 clean Thingi10K solids, 8.7 epochs
(2,349 object-passes). Best checkpoint: step 1050 of 1200.

| | non-manifold edges | loose components | detail |
|---|---:|---:|---:|
| stock TRELLIS.2 | 0.923% | 6,301 | 0.772 |
| **sft-1200 @1050** | **0.515%** (−44%) | **3,800** (−40%) | 0.784 |

Raw output, no repair. Full method: [`../report/comparison_900_vs_300_vs_base.md`](../report/comparison_900_vs_300_vs_base.md).

### More data did not win

A later run on **900** objects for 6 epochs — 5,220 object-passes, more than
twice the training exposure — scored only **−34%**, and lost to the 300-object
run on three of the five test inputs. It is better on open edges (0.793 vs
0.836) and nothing else.

Recorded because it is the most useful thing the experiment produced: the win
came from the *curation* of the 300 (clean, watertight, deduplicated by
provenance), not from volume.

## The dataset

`data/thingi10k_sft/` — 300 objects. Only the manifest is in git:

| part | size | what | recompute |
|---|---:|---|---|
| `manifest.jsonl` | 84 KB | **in git** — the index that defines the set | — |
| `latents/` | 45 MB | encoded SLat, the expensive artefact | ~25 s each (~2 h) |
| `images/` | 17 MB | conditioning renders | ~0.1 s each |
| `cond/` | 604 MB | DINOv3 embeddings | ~1.6 s each (~8 min) |

Note the asymmetry: 91% of the bulk is the *cheapest* part. `cond/` is derived
from `images/` and regenerates in minutes, so it is never worth shipping.

### Rebuilding it

`dataset_build.py` seeds `np.random.default_rng(seed)` and its defaults are
already this set, so one command reproduces it:

```bash
python -m training.dataset_build --name thingi10k_sft --limit 300 --seed 0
```

Every manifest record carries a `latent_sha256`, so a rebuild can be **verified
identical** rather than assumed:

```bash
python - <<'EOF'
import json, hashlib, pathlib
d = pathlib.Path("data/thingi10k_sft")
bad = [r["file_id"] for r in map(json.loads, open(d/"manifest.jsonl"))
       if hashlib.sha256((d/"latents"/r["latent_file"]).read_bytes()).hexdigest()
          != r["latent_sha256"]]
print("mismatched:", bad or "none")
EOF
```

### Why the models themselves are not here

The 300 come from Thingi10K under **per-object** Creative Commons licences, and
they are not uniform: **11 are No-Derivatives** — a render or an encoded latent
is a derivative work, so those cannot be republished in derived form at all —
**94 are Non-Commercial**, and **142 are Share-Alike**, which asks derivatives
to carry the same licence and sits badly beside this repository's MIT.

Shipping the index rather than the content sidesteps all of it, and is what
ImageNet and LAION do for the same reason. Full breakdown and per-model
sources: [`../data/thingi10k_sft/ATTRIBUTION.md`](../data/thingi10k_sft/ATTRIBUTION.md).

`dataset_build.py` has an `--exclude-noncommercial` flag that was *not* used
when this set was built. It is the right default for the next run.

## Training

```bash
python -m training.sft_train --data data/thingi10k_sft --steps 1200 \
       --rank 16 --alpha 32 --lr 1e-4 --accum 2 --holdout 30
```

On an M4 Pro this is roughly 116 hours. On one A100, an estimated 3–6 —
see [`cuda/`](cuda/README.md) for the packaged notebook.

`train_run.py` writes a run directory with a metrics stream, live control
channel and checkpoint manager; [`dashboard/`](dashboard/) serves it as a live
page while training runs.

### Two failure modes worth knowing about

**NaN gradients pass a threshold check.** The spike-skip guard read
`gn > threshold * median`, and every comparison against NaN is False — so NaN
gradients sailed through into AdamW and an 18-hour run trained on poisoned
weights, silently. Finiteness is now checked *first*:

```python
nonfinite = not (math.isfinite(loss_v) and math.isfinite(gn))
```

It fired 16 times in the successful rerun.

**Mini-batching corrupts the loss on non-flash-attention backends.** Padding
mixed-length objects into a batch without an attention mask lets padding
participate in attention. Measured error: **6.5%**. `cuda/check_batching.py`
is the gate that catches it, and `sft_cuda.py` refuses `--max-batch > 1`
unless the backend is `flash_attn`. Run the gate before any long training job.

## Layout

| file | what |
|---|---|
| `sft_train.py` | the trainer |
| `flow_matching.py` | path parameterisation, timestep sampling, checkpointing |
| `train_run.py` | run directory, metrics stream, checkpoints |
| `dataset_build.py` | Thingi10K → conditioning image + SLat latent |
| `vae_roundtrip.py` | does the shape VAE preserve printability? |
| `render_mesh.py` | offscreen renderer for conditioning images |
| `export_samples.py` | regenerate held-out samples from a checkpoint |
| `model_summary.py` / `profile_step.py` | layer dump; where a step's time goes |
| `dashboard/` | live training dashboard |
| `cuda/` | the A100 package |

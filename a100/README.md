# A100 training package — shape-SLat flow model, supervised fine-tune

Everything needed to run the Thingi10K fine-tune on a rented CUDA box from a
Jupyter notebook, and to bring the trained adapters back to the Mac.

## What to upload

| file | size | what it is |
|---|---|---|
| `train_a100.ipynb` | 20 KB | the notebook — run its cells in order |
| `sft_cuda.py` | 26 KB | the trainer |
| `lora_cuda.py` | 6 KB | the LoRA adapter |
| `check_batching.py` | 4 KB | correctness gate, run once before training |
| `requirements-cuda.txt` | 1 KB | dependency list |
| `thingi10k_5k_core.tar` | 725 MB | latents + manifest — **required** |
| `thingi10k_5k_cond.tar` | 9.9 GB | precomputed DINOv3 conditioning — **required** |

`thingi10k_5k_images.tar` (300 MB) is *not* needed: it feeds the image-fidelity
check in the Mac eval, which does not run here.

Build the tarballs with `./pack_dataset.sh` from the repo root. It also prints
the Hugging Face upload command, which is the practical way to move 10 GB —
resumable, and the GPU box pulls it at line rate instead of you uploading twice.

## What does NOT need installing

The flow model is `SparseLinear` + attention and contains no `SparseConv3d` and
no rasteriser, so training it needs only torch, safetensors, huggingface_hub,
einops and numpy — plus flash-attn. **No spconv, no o_voxel, no nvdiffrast, no
diso, no utils3d, no DINOv3, no rembg.** Those all belong to the decode/eval
path, which stays on the Mac.

This is why `sft_cuda.py` loads the flow model directly rather than
constructing the full `Trellis2ImageTo3DPipeline`: building the pipeline would
pull in the decoder and the image encoder and fail on their native extensions,
for models training never touches.

## The one thing that will silently ruin a run

**Mini-batching is only correct with flash-attn.**

trellis2's sparse attention has two paths. `flash_attn` uses
`flash_attn_varlen_*` with `cu_seqlens` — objects are packed with no padding
and each attends only to itself. The `sdpa` fallback pads every object out to
the longest in the batch with zeros and calls `scaled_dot_product_attention`
**with no attention mask**, so padded key positions enter the softmax of every
real query (`trellis2/modules/sparse/attention/full_attn.py`, the
`config.ATTN in ('sdpa','naive')` branch).

Measured on this dataset with three mixed-size objects: **the loss differs by
6.5%**. No exception, no NaN — just a corrupted gradient for the whole run.

With one object per batch there is no padding and the result is exact, which is
why the Mac runs (`--max-batch 1`) were unaffected.

`sft_cuda.py` refuses `--max-batch > 1` unless the backend is flash_attn.
`check_batching.py` verifies the invariant directly; cell 6 of the notebook runs
it. **PASS → use `--max-batch 8`. FAIL → fix flash-attn or use `--max-batch 1`.**

## How long, and how many GPUs

**One A100 is enough for 6 epochs in 10 hours, with real margin.** Expect
roughly **3–6 hours**; 40 GB and 80 GB cards both work, since peak usage is
about 20 GB with activation checkpointing on.

The arithmetic, so you can check it rather than trust it:

- 6 epochs over 4,990 training objects = **75.9 M token-passes**
  (12.65 M tokens per epoch; token counts come from the manifest).
- Measured on this Mac's GPU with the identical code path: **181 tokens/s**
  → 116 hours. That is the number that made a Mac run impractical.
- The work is ~13 GFLOP per token (forward + backward + activation-checkpoint
  recompute, including attention), so the Mac is achieving ~2.4 TFLOPS.
- An A100 at 25–35% MFU of its 312 TFLOPS bf16 peak gives 6,000–8,400 tokens/s
  → **2.5–3.5 h**. At a pessimistic 12% MFU it is still ~7 h.

A large part of the speedup beyond raw FLOPS is that flash-attn makes batching
safe: ~5.6 objects and ~14,200 tokens per step instead of one object, which
amortises the per-launch and Python overhead that dominates small steps.

**This is an estimate — I have no CUDA hardware to measure on.** Cell 7 of the
notebook replaces it with a measurement: it runs 60 real steps and projects the
6-epoch wall clock from your GPU's actual tokens/s, in about two minutes. Decide
there, not from this table. If it reports more than ~9 h, either use 2 GPUs, cut
to 4 epochs, or accept `--time-budget-h` stopping you partway with a resumable
checkpoint.

There is no multi-GPU path in this package. Adding one would mean untested
distributed code for a run that fits comfortably on one card.

## Autosave

The one thing that must never happen is finishing without weights on disk.
Saves are written:

- every `--save-every` steps → `ckpt/last/` (with optimizer state, resumable)
- at every eval → `ckpt/best_step<N>/`, pruned to the `--keep-best` lowest
  held-out losses
- on SIGINT / SIGTERM, at the next step boundary
- on any exception, before it propagates
- on normal completion → `ckpt/final/`

Best-K rather than last-K because on the Mac the final step was routinely not
the best: step 1050 of 1200 beat step 1200, and the checkpoint worth shipping is
whichever measured best.

Training is launched detached (`start_new_session=True`), so a dead kernel, a
closed tab or a dropped SSH session cannot interrupt it. Every channel is a file
under the run directory; the monitor cell only reads them.

Resume with `--resume <run>/ckpt/last`. Data order is deterministic, so it picks
up the identical sequence of batches.

## What comes back to the Mac

`ckpt/best_step*/adapter.pt` — ~75 MB each. Parameter names and shapes are
byte-compatible with the checkpoints already in `runs/`, verified key-by-key
(420 tensors, identical key sets), so they load straight into
`trellis_core/compare_models.py` with no conversion.

Send `metrics.jsonl` and `meta.json` too, plus the measured tokens/s from cell 7
and the PASS/FAIL from cell 6.

## What is deliberately not in here

The geometric eval — decode → mesh → printability judge → render → non-manifold
rate. It needs the decoder's native stack, it is the slowest part of the Mac
loop, and the comparison harness for it already exists and works. Held-out
flow-matching loss is what ranks checkpoints during the run; it is a proxy, and
the ranking may well shift when the real mesh comparison runs on the Mac.

## Settings that differ from the Mac runs, and why

| | Mac 5k run | here | why |
|---|---|---|---|
| `--max-batch` | 1 | 8 | safe once flash-attn is present |
| `--token-budget` | 9,000 | 16,384 | 80 GB instead of 36 GB shared with the OS |
| `--max-tokens` | 8,192 | 20,000 | keeps all 5,020 objects; 71 were being dropped |
| epoch accounting | 4× overcounted | exact | the Mac trainer added `--accum` (4) per step while consuming one object, so its "3.12 epochs" was really 0.78 |
| eval | full decode + judge | held-out loss | decoder stack stays on the Mac |

Learning rate, schedule, warmup, weight decay, LoRA rank/alpha/targets,
conditioning dropout, gradient-spike skipping and `sigma_min` are all unchanged,
so this is the same recipe run longer — which is exactly the open question the
5k comparison left: whether the 300-object run only won because it made 8.7
passes over its data against the 5k run's 0.78.

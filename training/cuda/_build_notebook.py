"""Generates train_a100.ipynb. Kept as a script so the notebook is diffable
and so its JSON is always valid -- hand-edited .ipynb files rot quickly."""

import json
import os

C = []


def md(text):
    C.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").split("\n")})


def code(text):
    C.append({"cell_type": "code", "execution_count": None, "metadata": {},
              "outputs": [], "source": text.strip("\n").split("\n")})


md(r"""
# TRELLIS.2 shape-SLat flow model — supervised fine-tune on A100

Trains a rank-16 LoRA on the `shape_slat_flow_model_512` flow model against
5,020 watertight Thingi10K meshes, and writes checkpoints that load straight
back into the Mac inference and comparison harness.

**Run the cells in order.** Cells 6 and 7 are gates: cell 6 proves mini-batching
is numerically safe on this box, cell 7 measures throughput and tells you how
long 6 epochs will actually take *before* you commit to it.

| | |
|---|---|
| GPU | 1× A100 (40 GB or 80 GB both fine — peak usage is ~20 GB) |
| Expected | 6 epochs in roughly 3–6 h |
| Trains | 18.68 M LoRA parameters over 210 modules; base model frozen |
| Does not need | spconv, o_voxel, nvdiffrast, DINOv3, rembg |
| Produces | `adapter.pt` files ~75 MB each — send these back |
""")

md("## 1 · Check the GPU")

code(r"""
!nvidia-smi
import torch
print(torch.__version__, "| cuda", torch.version.cuda, "| available", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"{p.name}  {p.total_memory/1e9:.0f} GB  sm_{p.major}{p.minor}  {torch.cuda.device_count()} device(s)")
""")

md(r"""
## 2 · Working directory

Everything lives under `WORK`. Point it at persistent storage — on most rented
boxes `/workspace` survives a restart and the home directory may not.
""")

code(r"""
import os, subprocess, sys, json, time, pathlib

WORK = "/workspace/trellis-sft"          # <-- change if your box differs
PKG  = f"{WORK}/a100"                    # where sft_cuda.py / lora_cuda.py / check_batching.py live
DATA = f"{WORK}/data/thingi10k_5k"       # dataset root (must end up containing manifest.jsonl)
TRELLIS = f"{WORK}/TRELLIS.2"
RUNS = f"{WORK}/runs"

for d in (WORK, DATA, RUNS):
    os.makedirs(d, exist_ok=True)
os.chdir(WORK)
print("cwd:", os.getcwd())
print("upload sft_cuda.py, lora_cuda.py and check_batching.py into:", PKG)
""")

md(r"""
## 3 · Dependencies and the TRELLIS.2 source

The commit is pinned to the one the Mac work was done against. Do not float it:
the objective here reads its noise schedule off this checkpoint's config, and a
changed sampler convention would be invisible in the loss curve and wrong at
inference.
""")

code(r"""
!pip install -q safetensors "huggingface_hub>=0.23" einops "numpy>=1.24" tqdm

TRELLIS_COMMIT = "baf9632abe2a053612b84c984293110c7d8d0ced"
if not os.path.isdir(f"{TRELLIS}/trellis2"):
    !git clone -q https://github.com/microsoft/TRELLIS.2.git {TRELLIS}
!cd {TRELLIS} && git fetch -q origin && git checkout -q {TRELLIS_COMMIT} && git rev-parse --short HEAD
""")

md(r"""
## 4 · flash-attn

**This is the one dependency that matters for speed and correctness.**

trellis2's sparse attention has two paths. `flash_attn` uses
`flash_attn_varlen_*` with `cu_seqlens`, which packs variable-length objects
with no padding — correct for any batch size. The `sdpa` fallback pads every
object out to the longest in the batch **with zeros and no attention mask**, so
padded keys enter the softmax of every real query. Measured on a batch of 3
mixed-size objects, that shifts the loss by **6.5%** — silently. No error, no
NaN, just a corrupted gradient.

So: with flash-attn you train ~6 objects per step. Without it you must use
`--max-batch 1` and the epoch takes about five times as long.

Prefer a prebuilt wheel from the
[flash-attention releases](https://github.com/Dao-AILab/flash-attention/releases)
matching your torch/CUDA/python — the source build below takes 20–40 minutes.
""")

code(r"""
try:
    import flash_attn
    print("flash_attn already present:", flash_attn.__version__)
except ImportError:
    print("installing flash-attn (this compiles; expect 20-40 min) ...")
    !pip install flash-attn --no-build-isolation
    import flash_attn; print("installed", flash_attn.__version__)
""")

md(r"""
## 5 · Dataset

Two tarballs, ~10.6 GB total. `cond/` is 93% of it: one DINOv3 feature grid per
object, `[1, 1029, 1024]` fp16. It is shipped precomputed rather than rebuilt
here because recomputing it would drag in DINOv3, background removal and the
renderer — most of the dependency surface this package avoids.

Fill in the repo you uploaded to with `pack_dataset.sh`, or skip the download
if you copied the tarballs across some other way.
""")

code(r"""
HF_DATASET = ""      # e.g. "acid101/thingi10k-slat-5k"; leave "" if copying manually

if HF_DATASET:
    !hf download {HF_DATASET} --repo-type=dataset --local-dir {WORK}/data_tar
    for tarball in ("core", "cond"):        # images/ is not used by training
        !tar -xf {WORK}/data_tar/thingi10k_5k_{tarball}.tar -C {DATA}

import json
n = sum(1 for _ in open(f"{DATA}/manifest.jsonl"))
latents = len(os.listdir(f"{DATA}/latents"))
conds   = len(os.listdir(f"{DATA}/cond"))
toks = [json.loads(l)["n_tokens"] for l in open(f"{DATA}/manifest.jsonl")]
print(f"manifest {n} | latents {latents} | cond {conds}")
print(f"tokens: total {sum(toks):,}  mean {sum(toks)//len(toks)}  max {max(toks)}")
assert n == latents == conds, "dataset incomplete -- re-extract before training"
print("\ndataset OK")
""")

md(r"""
## 6 · Gate: is mini-batching numerically safe here?

Runs three objects of deliberately different sizes as one batch and as three
batches of one, with identical timesteps and identical noise, and compares.

**PASS** → mini-batching is exact, use `--max-batch 8`.
**FAIL** → you are on the padded sdpa path. Either fix flash-attn or drop to
`--max-batch 1`. Do not train through a FAIL.
""")

code(r"""
!cd {PKG} && python check_batching.py --data {DATA} --trellis-path {TRELLIS} --device cuda
""")

md(r"""
## 7 · Gate: measure throughput, then project 6 epochs

Runs 60 real training steps and extrapolates. The estimate in the header of
this notebook (3–6 h) is derived from measurements on Apple silicon scaled by
FLOPs — this cell replaces it with a number measured on *your* GPU. Takes about
two minutes.
""")

code(r"""
CAL = f"{RUNS}/_calib"
!rm -rf {CAL}
!cd {PKG} && python sft_cuda.py --data {DATA} --trellis-path {TRELLIS} --device cuda \
    --out {RUNS} --run-id _calib --smoke 60 --log-every 5 \
    --eval-every 0 --save-every 0 --max-batch 8 --token-budget 16384 2>&1 | tail -12

import json
rows = [json.loads(l) for l in open(f"{CAL}/metrics.jsonl") if '"train"' in l]
warm = rows[3:]                      # drop the first few: allocator + kernel autotune
tok_s = sum(r["tokens"] for r in warm) / sum(r["step_s"] for r in warm)
meta  = json.load(open(f"{CAL}/meta.json"))

toks = [json.loads(l)["n_tokens"] for l in open(f"{DATA}/manifest.jsonl")]
EPOCHS = 6
total_tokens = sum(t for t in toks if t <= 20000) * EPOCHS
hours = total_tokens / tok_s / 3600

print(f"\nmeasured        {tok_s:,.0f} tokens/s   ({sum(r['step_s'] for r in warm)/len(warm):.2f} s/step)")
print(f"peak memory     {max(r['mem_gb'] for r in warm):.1f} GB")
print(f"steps/epoch     {meta['steps_per_epoch']}   ({meta['total_steps']} for {EPOCHS} epochs)")
print(f"\n{EPOCHS} epochs = {total_tokens/1e6:,.0f}M tokens -> {hours:.1f} h on this GPU")
print(f"within a 10 h budget: {'YES, 1 GPU is enough' if hours < 9 else f'NO -- you need {int(hours//9)+1} GPUs or fewer epochs'}")
""")

md(r"""
## 8 · Train

Launched **detached** with `nohup`, deliberately. A notebook kernel that dies,
a browser tab that closes or an SSH session that drops must not cost hours of
GPU time, so the trainer holds no socket and talks to no server — every channel
is a file under the run directory. Re-run the monitor cell as often as you like;
it cannot disturb the run.

Autosave covers every exit path: every `--save-every` steps, on the best
held-out losses, on SIGINT/SIGTERM, on an exception, and on normal completion.

`--time-budget-h 9.5` makes the run stop *cleanly* — saving — inside a 10 h
window even if throughput is worse than cell 7 projected. Resume from
`ckpt/last` if that happens.
""")

code(r"""
RUN_ID = time.strftime("sft-a100-%Y%m%d-%H%M")
LOG = f"{RUNS}/{RUN_ID}.log"
MAX_BATCH = 8          # set to 1 if cell 6 said FAIL

cmd = [sys.executable, "sft_cuda.py",
       "--data", DATA, "--trellis-path", TRELLIS, "--device", "cuda",
       "--out", RUNS, "--run-id", RUN_ID,
       "--epochs", "6",
       "--max-batch", str(MAX_BATCH), "--token-budget", "16384",
       "--lr", "1e-4", "--rank", "16", "--alpha", "32",
       "--warmup", "300", "--p-uncond", "0.1",
       "--eval-every", "250", "--save-every", "250", "--keep-best", "5",
       "--workers", "8", "--time-budget-h", "9.5"]

with open(LOG, "w") as f:
    proc = subprocess.Popen(cmd, cwd=PKG, stdout=f, stderr=subprocess.STDOUT,
                            start_new_session=True)      # survives this kernel
print("pid", proc.pid, "\nrun", f"{RUNS}/{RUN_ID}", "\nlog", LOG)
print(" ".join(cmd))
""")

md("## 9 · Monitor — safe to re-run at any time, including after a kernel restart")

code(r"""
import json, os, time

def progress(run_id=None):
    run_id = run_id or sorted(d for d in os.listdir(RUNS)
                              if d.startswith("sft-a100"))[-1]
    d = f"{RUNS}/{run_id}"
    st = json.load(open(f"{d}/status.json"))
    print(f"{run_id}: {st['state']}  step {st['step']}/{st['total_steps']}  "
          f"epoch {st.get('epoch')}  loss_ema {st.get('loss_ema', float('nan')):.4f}")
    if st.get("eta_s"):
        print(f"  elapsed {st['elapsed_s']/3600:.2f} h   eta {st['eta_s']/3600:.2f} h   "
              f"{st.get('step_s')} s/step   {st.get('mem_gb')} GB   spikes {st.get('n_spikes')}")
    ev = [json.loads(l) for l in open(f"{d}/metrics.jsonl") if '"eval"' in l]
    for e in ev[-6:]:
        print(f"  eval @ {e['step']:6d}  heldout_loss {e['heldout_loss']:.5f}")
    return d, run_id

d, run_id = progress()
!tail -5 {RUNS}/{run_id}.log
""")

code(r"""
# Loss curve. Matplotlib only -- nothing here needs a live connection.
import json
import matplotlib.pyplot as plt

tr = [json.loads(l) for l in open(f"{d}/metrics.jsonl") if '"train"' in l]
ev = [json.loads(l) for l in open(f"{d}/metrics.jsonl") if '"eval"' in l]

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].plot([r["epoch"] for r in tr], [r["loss"] for r in tr], lw=.4, alpha=.3, label="loss")
ax[0].plot([r["epoch"] for r in tr], [r["loss_ema"] for r in tr], lw=1.6, label="EMA")
if ev:
    ax[0].plot([e["epoch"] for e in ev], [e["heldout_loss"] for e in ev],
               "o-", ms=4, label="held-out")
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("flow-matching loss")
ax[0].set_yscale("log"); ax[0].legend(); ax[0].set_title("loss")

ax[1].plot([r["epoch"] for r in tr], [r["grad_norm"] for r in tr], lw=.5)
ax[1].set_yscale("log"); ax[1].set_xlabel("epoch"); ax[1].set_ylabel("grad norm")
ax[1].set_title(f"gradient norm ({tr[-1]['n_spikes']} batches skipped as spikes)")
plt.tight_layout(); plt.show()
""")

md(r"""
## 10 · Collect the checkpoints to send back

Packs the five best-by-held-out-loss adapters plus `final`, with the metrics
stream and config. Adapters are ~75 MB each; without optimizer state the whole
zip is a few hundred MB.

The parameter names and shapes are byte-compatible with the checkpoints already
on the Mac, so these load directly into the existing comparison harness — no
conversion step.
""")

code(r"""
import shutil, json, os

BUNDLE = f"{WORK}/deliver_{run_id}"
os.makedirs(f"{BUNDLE}/ckpt", exist_ok=True)
for name in sorted(os.listdir(f"{d}/ckpt")):
    if name.startswith("best_step") or name == "final":
        shutil.copytree(f"{d}/ckpt/{name}", f"{BUNDLE}/ckpt/{name}", dirs_exist_ok=True)
for f in ("meta.json", "status.json", "metrics.jsonl"):
    shutil.copy(f"{d}/{f}", BUNDLE)
shutil.copy(f"{RUNS}/{run_id}.log", BUNDLE)

archive = shutil.make_archive(BUNDLE, "zip", BUNDLE)
print(archive, f"{os.path.getsize(archive)/1e6:.0f} MB")
print("\nranking (lower is better):")
st = json.load(open(f"{d}/status.json"))
for step, loss in sorted(st.get("best", {}).items(), key=lambda kv: kv[1]):
    print(f"  step {step:>6}  heldout_loss {loss:.5f}")
""")

md(r"""
### Send back

The zip from cell 10 — or just `ckpt/best_step*/adapter.pt`, `metrics.jsonl`
and `meta.json`. Also worth reporting: the measured tokens/s from cell 7 and
the PASS/FAIL from cell 6, since both change how the results should be read.

The held-out loss ranks checkpoints during training, but it is **not** the
metric that decided anything so far — non-manifold edge rate and component
count on real decoded meshes are, and those are measured on the Mac. Expect the
ranking to shift when the real comparison runs.
""")

nb = {"cells": C,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.10"}},
      "nbformat": 4, "nbformat_minor": 5}

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_a100.ipynb")
with open(out, "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", out, f"({len(C)} cells)")

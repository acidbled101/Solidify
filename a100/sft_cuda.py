"""Supervised flow-matching fine-tune of TRELLIS.2's shape-SLat flow model,
packaged to run standalone on a CUDA box (A100) from a Jupyter notebook.

WHAT THIS IS
------------
A port of trellis_core/sft_train.py with three changes and nothing else moved:

  1. It loads ONLY the shape-SLat flow model, not the whole pipeline. That
     model uses no sparse convolution and no rasteriser -- just SparseLinear
     and attention -- so the install is torch + safetensors + einops + the
     trellis2 source. No spconv, no nvdiffrast, no o_voxel, no rembg, no
     DINOv3. (See `load_flow_model`.)
  2. Real mini-batching, gated on a correctness check (see MINI-BATCHING
     below).
  3. Epoch-exact data ordering, autosave on every exit path, and resume.

The heavy geometric eval (decode -> mesh -> printability judge -> render) is
deliberately NOT here. It needs the decoder and its rasteriser stack, it is the
slowest part of the Mac loop, and the comparison harness for it already exists
and works on the Mac. Ship the adapters back and evaluate there. What runs here
is the held-out flow-matching loss, which is cheap and is what ranks
checkpoints during the run.

THE OBJECTIVE
-------------
In TRELLIS.2's own path convention (read off flow_euler.py -- t runs 1 -> 0):

    x_t      = (1-t) x_0 + (sigma_min + (1-sigma_min) t) eps
    v_target = (1-sigma_min) eps - x_0
    L        = || v_theta(x_t, t, cond) - v_target ||^2

with t ~ Uniform(0,1), sigma_min and weight decay taken from TRELLIS.2's own
training config for this exact checkpoint, not from defaults invented here.

MINI-BATCHING IS GATED ON flash_attn, ON PURPOSE
------------------------------------------------
trellis2's sparse attention has two paths. `flash_attn` uses
flash_attn_varlen_* with cu_seqlens, which packs variable-length objects with
no padding and is correct for any batch. The `sdpa` fallback pads every
object out to the longest in the batch with ZEROS and then calls
scaled_dot_product_attention WITHOUT an attention mask
(trellis2/modules/sparse/attention/full_attn.py, the `config.ATTN in ('sdpa',
'naive')` branch). Padded key positions therefore take part in the softmax of
every real query. With one object per batch there is no padding and the result
is exact -- which is why the Mac runs, which used batch 1, were fine. With
batch > 1 on sdpa the loss is silently wrong: no error, no NaN, just a
corrupted gradient.

So `--max-batch > 1` raises unless the attention backend is flash_attn.
Override with --i-know-sdpa-batching-is-wrong if you want to measure it.

EPOCH ACCOUNTING
----------------
The Mac trainer incremented `samples_seen` by `--accum` (default 4) while
actually consuming one object per step, so its logged `epoch` was 4x the truth
-- the 5k run reported 3.12 epochs having made 0.78 passes over the data. Here
an epoch is a real permutation of the dataset consumed exactly once, batches
are built from it up front, and `total_steps` is therefore known before step 1
(which the cosine schedule needs anyway).
"""

import argparse
import json
import math
import os
import random
import shutil
import signal
import statistics
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Environment. MUST run before trellis2 is imported.
# ---------------------------------------------------------------------------


def configure_backend(device: str, prefer: Optional[str] = None) -> str:
    """Pick and pin the attention backend, returning what was chosen.

    trellis2 reads ATTN_BACKEND at import time
    (trellis2/modules/attention/config.py) and defaults to flash_attn, which
    raises an ImportError deep inside the first forward if it is not
    installed. Probing here turns that into a clear message at startup.
    """
    if prefer:
        chosen = prefer
    elif device.startswith("cuda"):
        try:
            import flash_attn  # noqa: F401
            chosen = "flash_attn"
        except Exception:
            chosen = "sdpa"
    else:
        chosen = "sdpa"

    os.environ["ATTN_BACKEND"] = chosen
    os.environ["SPARSE_ATTN_BACKEND"] = chosen
    # The flow model contains no SparseConv3d, so no conv backend is needed.
    os.environ.setdefault("SPARSE_CONV_BACKEND", "none")
    if device == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    return chosen


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

FLOW_MODEL_PATH = "microsoft/TRELLIS.2-4B/ckpts/slat_flow_img2shape_dit_1_3B_512_bf16"
PIPELINE_REPO = "microsoft/TRELLIS.2-4B"


def load_flow_model(device: str, model_path: str = FLOW_MODEL_PATH):
    """Load the shape-SLat flow model alone, and its sampler's sigma_min.

    sigma_min is read from the pipeline config rather than hardcoded, because
    the noise schedule the objective assumes has to be the one the checkpoint
    was trained under; a silent drift there is invisible in the loss curve and
    fatal at inference.
    """
    from trellis2 import models

    model = models.from_pretrained(model_path)
    model.to(device)

    sigma_min = 1e-5
    try:
        from huggingface_hub import hf_hub_download
        cfg = json.load(open(hf_hub_download(PIPELINE_REPO, "pipeline.json")))
        sigma_min = cfg["args"]["shape_slat_sampler"]["args"]["sigma_min"]
    except Exception as e:
        print(f"[warn] could not read pipeline.json ({e}); using sigma_min={sigma_min}")
    return model, float(sigma_min)


class checkpointed_blocks:
    """Turn on activation checkpointing for every block that supports it.

    trellis2's ModulatedSparseTransformerCrossBlock.forward already dispatches
    to torch.utils.checkpoint(..., use_reentrant=False) when self.use_checkpoint
    is set; the pretrained config just leaves it False. Trading ~33% extra
    compute for roughly an order of magnitude less activation memory is what
    lets a 16k-token batch fit, and on an A100 the run is not compute-bound at
    these sizes anyway.
    """

    def __init__(self, model, enabled: bool = True):
        self.model, self.enabled, self.saved = model, enabled, []

    def __enter__(self):
        if not self.enabled:
            return self.model
        for m in self.model.modules():
            if hasattr(m, "use_checkpoint"):
                self.saved.append((m, m.use_checkpoint))
                m.use_checkpoint = True
        return self.model

    def __exit__(self, *exc):
        for m, old in self.saved:
            m.use_checkpoint = old
        self.saved = []
        return False


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------


def noise_latent(x_0, eps, t, sigma_min):
    """x_t = (1-t) x_0 + (sigma_min + (1-sigma_min) t) eps."""
    return (1.0 - t) * x_0 + (sigma_min + (1.0 - sigma_min) * t) * eps


def velocity_target(x_0, eps, sigma_min):
    """v = (1-sigma_min) eps - x_0. Independent of t on the linear path."""
    return (1.0 - sigma_min) * eps - x_0


def flow_matching_loss(model, samples, sigma_min, *, p_uncond=0.0, rng=None,
                       t_override=None, eps_override=None):
    """One forward over a mini-batch of sparse latents.

    SparseTensor carries a batch index in coords[:, 0], so several objects
    share one forward instead of being run separately and summed by gradient
    accumulation.

    The loss is the mean of PER-OBJECT means, not the mean over all tokens.
    Token-mean would weight a 5,000-token object seven times as heavily as a
    700-token one purely for being bigger, which is a silent reweighting of the
    dataset toward large meshes.
    """
    from trellis2.modules import sparse as sp

    device = samples[0][1].device
    B = len(samples)
    rng = rng or random
    t = (torch.rand(B, device=device) if t_override is None
         else t_override.to(device)).clamp(1e-3, 1.0)

    feats_all, coords_all, tgt_all, idx_all, conds = [], [], [], [], []
    for b, (coords, feats, cond) in enumerate(samples):
        eps = (torch.randn(feats.shape, device=device, dtype=feats.dtype)
               if eps_override is None else eps_override[b].to(device))
        feats_all.append(noise_latent(feats, eps, t[b], sigma_min))
        tgt_all.append(velocity_target(feats, eps, sigma_min))
        c = coords.clone()
        c[:, 0] = b                                   # batch index for this object
        coords_all.append(c)
        idx_all.append(torch.full((coords.shape[0],), b, dtype=torch.long, device=device))
        # Conditioning dropout is per-object, as in the official CFG trainer.
        # Without it the unconditional branch drifts away from the base model
        # while inference still samples with CFG against it, so guidance
        # quietly degrades even as the conditional loss improves.
        drop = p_uncond > 0 and rng.random() < p_uncond
        conds.append(torch.zeros_like(cond) if drop else cond)

    x_t = torch.cat(feats_all, 0)
    v_tgt = torch.cat(tgt_all, 0)
    sample_index = torch.cat(idx_all, 0)

    # feats stay FLOAT32. The model is mixed precision -- input_layer, out_layer
    # and t_embedder are fp32 while the 30 transformer blocks are bf16, and
    # `manual_cast` handles the transition internally. The pipeline's own
    # sampler builds its noise with a plain torch.randn for the same reason.
    st = sp.SparseTensor(feats=x_t.float(), coords=torch.cat(coords_all, 0))
    pred = model(st, (1000.0 * t).to(torch.float32), torch.cat(conds, 0))
    v_pred = pred.feats if hasattr(pred, "feats") else pred

    se = (v_pred.float() - v_tgt.float()).pow(2).mean(dim=1)          # per token
    per_obj = torch.zeros(B, device=device, dtype=se.dtype).index_add(0, sample_index, se)
    counts = torch.zeros(B, device=device, dtype=se.dtype).index_add(
        0, sample_index, torch.ones_like(se))
    return (per_obj / counts.clamp_min(1)).mean(), [float(x) for x in t]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


class SlatDataset(torch.utils.data.Dataset):
    """(latent, conditioning) pairs written by trellis_core/dataset_build.py.

        <root>/latents/<id>.npz   coords int32 [N,3], feats float16 [N,32]
        <root>/cond/<id>.npy      float16 [1, 1029, 1024]  (DINOv3 features)
        <root>/manifest.jsonl     one record per object, carries n_tokens

    Returned as CPU tensors so DataLoader workers can do the file I/O off the
    training thread; ~2.1 MB of conditioning per object means a 6-epoch run
    reads ~64 GB, which is worth overlapping.
    """

    def __init__(self, root: str, ids: List[str]):
        self.root, self.ids = root, ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        fid = self.ids[i]
        z = np.load(os.path.join(self.root, "latents", f"{fid}.npz"))
        coords = torch.from_numpy(z["coords"].astype(np.int32))
        feats = torch.from_numpy(z["feats"].astype(np.float32))
        cond = torch.from_numpy(
            np.load(os.path.join(self.root, "cond", f"{fid}.npy")).astype(np.float32))
        # SparseTensor coords carry a leading batch column.
        coords = torch.cat([torch.zeros(coords.shape[0], 1, dtype=torch.int32), coords], 1)
        return coords, feats, cond


def read_manifest(root: str) -> List[Tuple[str, int]]:
    out = []
    with open(os.path.join(root, "manifest.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            out.append((r["file_id"], int(r.get("n_tokens", 0))))
    return out


def build_epoch_batches(tokens: List[int], *, budget: int, max_batch: int,
                        max_tokens: int, epochs: int, seed: int) -> List[List[int]]:
    """One flat list of batches covering `epochs` full passes over the data.

    Objects vary from 136 to 19,767 tokens, so batches are formed by a TOKEN
    budget rather than a fixed count: a fixed count would either overflow
    memory on a batch of large meshes or waste it on a batch of small ones.

    Building every epoch up front costs nothing and buys two things the online
    version could not give: `total_steps` is exact before step 1, which the
    cosine schedule needs, and resume can seek to a batch index and continue
    the identical data order.
    """
    idx = [i for i, n in enumerate(tokens) if n <= max_tokens]
    dropped = len(tokens) - len(idx)
    batches: List[List[int]] = []
    for e in range(epochs):
        order = list(idx)
        random.Random(seed * 1000 + e).shuffle(order)
        cur, cur_tok = [], 0
        for i in order:
            n = tokens[i]
            if cur and (len(cur) >= max_batch or cur_tok + n > budget):
                batches.append(cur)
                cur, cur_tok = [], 0
            cur.append(i)
            cur_tok += n
        if cur:
            batches.append(cur)
    if dropped:
        print(f"[data] {dropped} object(s) over --max-tokens {max_tokens} excluded")
    return batches


def identity_collate(batch):
    """Keep the objects as a list of (coords, feats, cond) triples.

    Module level, not a lambda: DataLoader workers are spawned (the default on
    macOS and on any CUDA process that has already initialised a context), and
    spawn pickles the collate function.
    """
    return batch


def to_device(sample, device):
    coords, feats, cond = sample
    return (coords.to(device, non_blocking=True),
            feats.to(device, non_blocking=True),
            cond.to(device, non_blocking=True))


# ---------------------------------------------------------------------------
# Eval: held-out flow-matching loss
# ---------------------------------------------------------------------------


@torch.no_grad()
def heldout_loss(model, ds: SlatDataset, sigma_min: float, device: str,
                 n: int = 30, seed: int = 1234) -> float:
    """Held-out loss on a FIXED draw of noise and timesteps.

    Common random numbers across evals make successive measurements a paired
    comparison of the same trajectory rather than independent draws. Two evals
    of the same model on independent noise differ by more than any training
    effect of plausible size, which would bury the signal entirely.
    """
    model.eval()
    losses = []
    for i in range(min(n, len(ds))):
        coords, feats, cond = to_device(ds[i], device)
        g = torch.Generator(device="cpu").manual_seed(seed + i)
        eps = torch.randn(feats.shape, generator=g).to(device)
        t = torch.full((1,), 0.5, device=device)
        loss, _ = flow_matching_loss(model, [(coords, feats, cond)], sigma_min,
                                     p_uncond=0.0, t_override=t, eps_override=[eps])
        losses.append(float(loss))
        del coords, feats, cond
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


# ---------------------------------------------------------------------------
# Run directory
# ---------------------------------------------------------------------------


def _json_safe(obj):
    """Replace NaN/Infinity with null, recursively.

    Python's json module happily writes bare `NaN`, which is NOT valid JSON.
    Nothing downstream accepts it: FastAPI's serializer raises, so a single NaN
    in one run's status.json returned HTTP 500 for the dashboard's entire run
    list, and browser JSON.parse rejects it too, so the metrics stream stops
    loading. A run going bad must not also take out the instrument you would
    use to notice.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


class RunDir:
    """Every channel is a file, so nothing supervising the run can break it.

    A notebook kernel dying, a browser tab closing or an SSH session dropping
    must not cost training progress -- so the trainer never holds a socket and
    never talks to a server. Readers tail metrics.jsonl; that is the whole
    protocol.
    """

    def __init__(self, root: str, run_id: str):
        self.dir = os.path.join(root, run_id)
        os.makedirs(os.path.join(self.dir, "ckpt"), exist_ok=True)
        self.metrics = os.path.join(self.dir, "metrics.jsonl")

    def log(self, kind: str, step: int, **kw):
        rec = _json_safe({"kind": kind, "step": step, "t": time.time(), **kw})
        with open(self.metrics, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def write_json(self, name: str, obj):
        tmp = os.path.join(self.dir, name + ".tmp")
        with open(tmp, "w") as f:
            json.dump(_json_safe(obj), f, indent=1)
        os.replace(tmp, os.path.join(self.dir, name))   # atomic: never half-read

    def save(self, model_lora_sd, opt, step: int, batch_idx: int, *,
             tag: str, with_optimizer: bool, extra: Dict):
        d = os.path.join(self.dir, "ckpt", tag)
        os.makedirs(d, exist_ok=True)
        payload = {"lora": model_lora_sd, "step": step, "batch_idx": batch_idx, **extra}
        if with_optimizer:
            payload["optimizer"] = opt.state_dict()
        tmp = os.path.join(d, "adapter.pt.tmp")
        torch.save(payload, tmp)
        os.replace(tmp, os.path.join(d, "adapter.pt"))
        return d


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="dataset root (has manifest.jsonl)")
    ap.add_argument("--out", default="runs", help="where run directories are written")
    ap.add_argument("--run-id", default=time.strftime("sft-a100-%Y%m%d-%H%M%S"))
    ap.add_argument("--epochs", type=float, default=6.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--attn-backend", default=None,
                    help="flash_attn | sdpa | xformers (default: probe)")
    ap.add_argument("--trellis-path", default=os.environ.get("TRELLIS2_PATH", ""),
                    help="path to the TRELLIS.2 checkout (the dir containing trellis2/)")

    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--lr-min-ratio", type=float, default=0.02)
    ap.add_argument("--p-uncond", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--grad-spike", type=float, default=5.0,
                    help="skip a batch whose grad norm exceeds this x running median")

    ap.add_argument("--token-budget", type=int, default=16384)
    ap.add_argument("--max-batch", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=20000,
                    help="exclude latents larger than this (20000 keeps all 5020)")
    ap.add_argument("--holdout", type=int, default=30)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-grad-checkpointing", dest="grad_checkpointing",
                    action="store_false")

    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-n", type=int, default=30)
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--keep-best", type=int, default=5)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--time-budget-h", type=float, default=0.0,
                    help="stop cleanly (saving) after this many hours; 0 = no limit")
    ap.add_argument("--check-finite-every", type=int, default=50,
                    help="verify parameters are still finite every N steps")
    ap.add_argument("--resume", default="", help="checkpoint dir to resume from")
    ap.add_argument("--smoke", type=int, default=0, help="run N steps and exit")
    ap.add_argument("--i-know-sdpa-batching-is-wrong", action="store_true")
    args = ap.parse_args(argv)

    if args.trellis_path and args.trellis_path not in sys.path:
        sys.path.insert(0, args.trellis_path)

    backend = configure_backend(args.device, args.attn_backend)
    print(f"[env] device={args.device} attention backend={backend}")
    if args.max_batch > 1 and backend != "flash_attn" and not args.i_know_sdpa_batching_is_wrong:
        print(
            f"\nERROR: --max-batch {args.max_batch} with the '{backend}' attention backend.\n"
            "trellis2's sdpa path pads sequences to the batch maximum with zeros and\n"
            "calls scaled_dot_product_attention with NO attention mask, so padded keys\n"
            "enter the softmax of every real query. The loss would be silently wrong.\n"
            "Either install flash-attn, or run with --max-batch 1.\n", file=sys.stderr)
        return 2

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            print("ERROR: --device cuda but torch.cuda.is_available() is False", file=sys.stderr)
            return 2
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"[env] {torch.cuda.get_device_name(0)}  "
              f"{torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB  "
              f"torch {torch.__version__}  cuda {torch.version.cuda}")

    # ---- data ----------------------------------------------------------
    manifest = read_manifest(args.data)
    rng = random.Random(args.seed)
    order = list(range(len(manifest)))
    rng.shuffle(order)
    hold_idx, train_idx = order[:args.holdout], order[args.holdout:]
    train_ds = SlatDataset(args.data, [manifest[i][0] for i in train_idx])
    eval_ds = SlatDataset(args.data, [manifest[i][0] for i in hold_idx])
    train_tokens = [manifest[i][1] for i in train_idx]

    whole = int(args.epochs) + (1 if args.epochs % 1 else 0)
    batches = build_epoch_batches(train_tokens, budget=args.token_budget,
                                  max_batch=args.max_batch, max_tokens=args.max_tokens,
                                  epochs=max(whole, 1), seed=args.seed + 1)
    per_epoch = max(1, len(batches) // max(whole, 1))
    # `--smoke N` caps how many steps are RUN, but must not shorten the
    # schedule: truncating `batches` here would make the cosine decay finish in
    # N steps, so a smoke test would not exercise the same learning rates a
    # real run does -- and it would collide with --resume, whose start offset
    # indexes into this same list.
    total_steps = int(round(per_epoch * args.epochs))
    batches = batches[:total_steps]
    print(f"[data] train {len(train_ds)}  holdout {len(eval_ds)}  "
          f"{per_epoch} steps/epoch  {total_steps} steps for {args.epochs} epochs  "
          f"mean batch {np.mean([len(b) for b in batches]):.2f} objects")

    # ---- model ---------------------------------------------------------
    print("loading flow model ...", flush=True)
    t0 = time.time()
    model, sigma_min = load_flow_model(args.device)
    print(f"loaded in {time.time()-t0:.0f}s  sigma_min={sigma_min}", flush=True)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import lora_cuda as lora_mod

    adapted = lora_mod.apply_lora(model, rank=args.rank, alpha=args.alpha)
    n_train_p = lora_mod.count_trainable(model)
    print(f"LoRA on {len(adapted)} modules, {n_train_p/1e6:.2f}M trainable", flush=True)
    params = list(lora_mod.lora_parameters(model))
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay,
                            betas=(0.9, 0.95), eps=1e-8)
    model.train()

    run = RunDir(args.out, args.run_id)
    run.write_json("meta.json", {"config": vars(args), "sigma_min": sigma_min,
                                 "total_steps": total_steps, "steps_per_epoch": per_epoch,
                                 "trainable_params": n_train_p, "backend": backend,
                                 "n_train": len(train_ds), "started_at": time.time()})
    # Write status before the first step so a monitor started immediately does
    # not trip over a missing file while the model is still loading.
    run.write_json("status.json", {"step": 0, "total_steps": total_steps,
                                   "state": "starting", "epoch": 0.0})

    start_batch = 0
    if args.resume:
        state = torch.load(os.path.join(args.resume, "adapter.pt"), map_location="cpu")
        lora_mod.load_lora_state_dict(model, state["lora"])
        if "optimizer" in state:
            opt.load_state_dict(state["optimizer"])
        start_batch = int(state.get("batch_idx", state.get("step", 0)))
        print(f"resumed from {args.resume} at batch {start_batch}", flush=True)

    def lr_at(step):
        """Linear warmup then cosine decay to `lr_min_ratio` of peak.

        A constant 1e-4 is what destroyed the first Mac run at ~step 1100, its
        gradient norm going from 0.05 to 332. Decay is the standard guard;
        warmup keeps the first updates small while AdamW's moments are cold.
        """
        if step <= args.warmup:
            return args.lr * step / max(args.warmup, 1)
        p = min(max((step - args.warmup) / max(total_steps - args.warmup, 1), 0.0), 1.0)
        return args.lr * (args.lr_min_ratio +
                          (1 - args.lr_min_ratio) * 0.5 * (1 + math.cos(math.pi * p)))

    # ---- autosave on every exit path -----------------------------------
    # A 6-epoch run is hours of GPU time; the one thing that must never happen
    # is finishing without the weights on disk. Signals set a flag rather than
    # saving inline, so the save happens at a step boundary with consistent
    # optimizer state instead of halfway through a backward pass.
    stop_flag = {"stop": False, "why": ""}

    def _sig(signum, _frame):
        stop_flag["stop"] = True
        stop_flag["why"] = f"signal {signum}"
        print(f"\n[signal {signum}] finishing this step, then saving ...", flush=True)

    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(s, _sig)
        except Exception:
            pass

    loader = torch.utils.data.DataLoader(
        train_ds, batch_sampler=batches[start_batch:], num_workers=args.workers,
        collate_fn=identity_collate, pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.workers > 0, prefetch_factor=2 if args.workers else None)

    ctrl_path = os.path.join(run.dir, "control.json")
    lr_override = None
    ema, gn_hist, n_spikes, n_nonfinite, best = None, [], 0, 0, {}
    step = start_batch
    t_start = time.time()
    deadline = t_start + args.time_budget_h * 3600 if args.time_budget_h else None
    exit_reason = "completed"

    def do_save(tag, with_optimizer, metric=None):
        d = run.save(lora_mod.lora_state_dict(model), opt, step, step, tag=tag,
                     with_optimizer=with_optimizer,
                     extra={"rank": args.rank, "alpha": args.alpha,
                            "targets": list(lora_mod.DEFAULT_TARGETS),
                            "heldout_loss": metric, "run_id": args.run_id})
        return d

    def prune_best():
        """Keep the K checkpoints with the lowest held-out loss.

        Best-K rather than last-K because the Mac runs showed the final step is
        routinely not the best one: step 1050 of 1200 beat step 1200, and the
        thing you actually want to ship is whichever checkpoint measured best.
        """
        keep = sorted(best.items(), key=lambda kv: kv[1])[:args.keep_best]
        keep_tags = {f"best_step{s}" for s, _ in keep}
        root = os.path.join(run.dir, "ckpt")
        for name in os.listdir(root):
            if name.startswith("best_step") and name not in keep_tags:
                shutil.rmtree(os.path.join(root, name), ignore_errors=True)
        return {s: v for s, v in keep}

    try:
        for batch in loader:
            step += 1

            # Control channel: the dashboard writes control.json, the trainer
            # reads it. Honouring `stop` matters more than the other knobs --
            # a Stop button that silently does nothing is worse than no button,
            # because you believe the GPU is free when it is not. Every failure
            # here degrades to "keep going with the current settings"; a
            # half-written file during a weekend run must never be a traceback
            # inside the training loop.
            try:
                if os.path.exists(ctrl_path):
                    ctrl = json.load(open(ctrl_path))
                    if ctrl.get("stop"):
                        stop_flag["stop"], stop_flag["why"] = True, "stop via control.json"
                    while ctrl.get("pause") and not stop_flag["stop"]:
                        run.write_json("status.json", {"run_id": args.run_id,
                                                       "heartbeat": time.time(),
                                                       "step": step - 1, "state": "paused",
                                                       "total_steps": total_steps})
                        time.sleep(5)
                        ctrl = json.load(open(ctrl_path))
                    lr_override = ctrl.get("lr")
            except Exception:
                pass

            lr_now = lr_override or lr_at(step)
            for g in opt.param_groups:
                g["lr"] = lr_now

            t_step = time.time()
            samples = [to_device(s, args.device) for s in batch]
            n_tok = sum(int(c.shape[0]) for c, _, _ in samples)

            opt.zero_grad(set_to_none=True)
            with checkpointed_blocks(model, args.grad_checkpointing):
                loss, t_vals = flow_matching_loss(model, samples, sigma_min,
                                                  p_uncond=args.p_uncond)
                loss.backward()
            loss_v = float(loss.detach())
            del samples, loss

            gn = float(torch.nn.utils.clip_grad_norm_(params, args.grad_clip))
            # Drop, do not merely clip, a batch whose gradient is a wild
            # outlier. Clipping bounds one update but cannot stop AdamW's
            # moments filling with garbage across a sustained blow-up, which is
            # how the first Mac run destroyed itself. The median is over recent
            # healthy steps, so the threshold adapts as training settles.
            # NON-FINITE COMES FIRST, and is checked with isfinite rather than
            # by comparison. Every comparison against NaN is False, so the
            # spike test below silently passes a NaN gradient straight into
            # opt.step(), and AdamW then writes NaN into the parameters --
            # permanently, because NaN propagates through every later forward.
            # That is exactly how the first 900-object run destroyed itself:
            # healthy at step 130 (loss 0.15, grad norm 0.056), NaN at 135, and
            # 18 more hours of training on dead weights with no error raised.
            # One transient bad batch must cost one batch, not the run.
            nonfinite = not (math.isfinite(loss_v) and math.isfinite(gn))
            gn_hist.append(gn) if not nonfinite else None
            if len(gn_hist) > 100:
                gn_hist.pop(0)
            med = statistics.median(gn_hist) if len(gn_hist) >= 20 else None
            spiked = (not nonfinite) and med is not None and gn > args.grad_spike * max(med, 1e-8)
            if nonfinite:
                n_nonfinite += 1
                run.log("nonfinite", step, loss=loss_v, grad_norm=gn, tokens=n_tok,
                        n_nonfinite=n_nonfinite)
                print(f"[step {step}] NON-FINITE loss/grad -- batch skipped "
                      f"({n_nonfinite} so far)", flush=True)
            elif spiked:
                n_spikes += 1
            else:
                opt.step()

            # Backstop: verify the parameters themselves are still finite.
            # The skip above stops the known route to NaN weights, but any
            # route at all costs the whole run, so this asserts the invariant
            # directly rather than trusting that the guards are exhaustive.
            # ~19M parameters is a few milliseconds, amortised over 50 steps.
            if step % args.check_finite_every == 0:
                if not all(torch.isfinite(p).all() for p in params):
                    run.write_json("status.json", {"step": step, "state": "aborted",
                                                   "reason": "non-finite parameters"})
                    print(f"\nABORT at step {step}: parameters contain NaN/Inf.\n"
                          f"Resume from a clean checkpoint:\n"
                          f"  --resume {run.dir}/ckpt/last\n", file=sys.stderr, flush=True)
                    return 3

            # A skipped batch must not poison the EMA either: one NaN folded in
            # makes the headline loss NaN for the rest of the run, which is the
            # number you would be watching to decide whether it is healthy.
            if not nonfinite:
                ema = loss_v if ema is None else 0.92 * ema + 0.08 * loss_v
            dt = time.time() - t_step
            elapsed = time.time() - t_start
            mem = (torch.cuda.max_memory_allocated() / 1e9
                   if args.device.startswith("cuda") else 0.0)

            if step % args.log_every == 0 or args.smoke:
                run.log("train", step, loss=loss_v, loss_ema=ema, lr=lr_now,
                        grad_norm=gn, spiked=int(spiked), n_spikes=n_spikes,
                        batch=len(t_vals), tokens=n_tok, step_s=dt, mem_gb=mem,
                        epoch=step / per_epoch)
                run.write_json("status.json", {
                    # run_id/pid/heartbeat are what devlab/train_server.py reads
                    # to decide a run is alive; without heartbeat the dashboard
                    # shows a perfectly healthy run as dead.
                    "run_id": args.run_id, "pid": os.getpid(), "heartbeat": time.time(),
                    "step": step, "total_steps": total_steps, "state": "training",
                    "loss_ema": ema, "epoch": round(step / per_epoch, 3),
                    "elapsed_s": round(elapsed), "step_s": round(dt, 3),
                    "eta_s": round((total_steps - step) * elapsed / max(step - start_batch, 1)),
                    "mem_gb": round(mem, 2), "n_spikes": n_spikes,
                    "n_nonfinite": n_nonfinite})
                print(f"step {step}/{total_steps} ({step/per_epoch:.2f} ep)  "
                      f"loss {loss_v:.4f}  ema {ema:.4f}  gn {gn:.3f}  "
                      f"{len(t_vals)}obj/{n_tok}tok  {dt:.2f}s  {mem:.1f}GB", flush=True)

            if args.smoke and step - start_batch >= args.smoke:
                exit_reason = "smoke"
                break

            if args.eval_every and step % args.eval_every == 0:
                hl = heldout_loss(model, eval_ds, sigma_min, args.device, n=args.eval_n)
                run.log("eval", step, heldout_loss=hl, epoch=step / per_epoch)
                print(f"[eval @ {step}] heldout_loss={hl:.5f}", flush=True)
                best[step] = hl
                do_save(f"best_step{step}", with_optimizer=False, metric=hl)
                kept = prune_best()
                run.log("event", step, kept=sorted(kept))

            if args.save_every and step % args.save_every == 0:
                do_save("last", with_optimizer=True)

            if stop_flag["stop"]:
                exit_reason = stop_flag["why"]
                break
            if deadline and time.time() > deadline:
                exit_reason = "time budget reached"
                break

    except KeyboardInterrupt:
        exit_reason = "KeyboardInterrupt"
    except Exception as e:
        exit_reason = f"{type(e).__name__}: {e}"
        do_save("last", with_optimizer=True)
        run.write_json("status.json", {"step": step, "state": "crashed",
                                       "reason": exit_reason})
        print(f"\n[crash] saved to ckpt/last before re-raising: {exit_reason}",
              file=sys.stderr, flush=True)
        raise
    finally:
        if not args.smoke or step > start_batch:
            d = do_save("final", with_optimizer=True)
            print(f"\nsaved final adapter -> {d}/adapter.pt", flush=True)

    if args.smoke:
        print(f"\nsmoke: {step - start_batch} steps in "
              f"{time.time()-t_start:.1f}s -- no final eval, nothing pruned")
        return 0

    hl = heldout_loss(model, eval_ds, sigma_min, args.device, n=args.eval_n)
    best[step] = hl
    run.log("eval", step, heldout_loss=hl, epoch=step / per_epoch)
    do_save(f"best_step{step}", with_optimizer=False, metric=hl)
    kept = prune_best()
    run.write_json("status.json", {
        "step": step, "total_steps": total_steps, "state": "finished",
        "reason": exit_reason, "elapsed_s": round(time.time() - t_start),
        "final_heldout_loss": hl, "best": {str(k): v for k, v in sorted(kept.items())}})
    print(f"\ndone ({exit_reason}) after {step} steps / "
          f"{step/per_epoch:.2f} epochs in {(time.time()-t_start)/3600:.2f}h")
    print(f"final held-out loss {hl:.5f}")
    print(f"best checkpoints: {sorted(kept.items(), key=lambda kv: kv[1])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

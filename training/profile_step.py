"""Where does a training step actually go?

Measured 464s for one step (accum=2) on an M4, against an estimate of 60-90s
derived from the old DPO gradient steps. A 5x gap is not a tuning detail -- it
is the difference between an overnight run and a four-day one -- so this
isolates the cost before any long run is launched.

Three candidates, in order of suspicion:

  1. CPU fallback. `bootstrap` sets PYTORCH_ENABLE_MPS_FALLBACK=1 because a few
     ops (segment_reduce among them) crash on MPS otherwise. A hot op falling
     back to CPU means every block round-trips 2233x1536 tensors across the
     memory boundary with a sync each way, which would dominate everything
     else. Silent by construction: the whole point of the flag is that it does
     not raise.
  2. LoRA overhead. 210 adapters, each casting both factors to the base dtype
     on every call -- 420 extra allocations plus 420 small matmuls per forward.
  3. Genuine model cost: 30 blocks, self-attention over ~2233 tokens plus
     cross-attention over 1029 conditioning tokens, mlp_ratio 5.33.

Usage:
    PYTHONPATH=. python -m trellis_core.profile_step
"""

import os
import sys
import time

from trellis_core import bootstrap  # noqa: F401

import numpy as np
import torch


def timed(fn, n=3, warmup=1):
    for _ in range(warmup):
        fn()
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    t0 = time.time()
    for _ in range(n):
        fn()
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    return (time.time() - t0) / n


def main() -> int:
    from trellis2.modules import sparse as sp
    from trellis_core import pipeline as pipe_mod
    from trellis_core import lora as lora_mod
    from training.sft_train import split_dataset, flow_matching_loss
    from .flow_matching import FlowMatchConfig

    device = "mps"
    print(f"MPS fallback: {os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')}")
    print(f"conv backend: {os.environ.get('SPARSE_CONV_BACKEND')}   "
          f"attn backend: {os.environ.get('ATTN_BACKEND')}")

    p = pipe_mod.load_pipeline("microsoft/TRELLIS.2-4B", device)
    p.low_vram = False
    model = p.models["shape_slat_flow_model_512"]
    model.to(device)

    train_ds, _ = split_dataset(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "data", "thingi10k_sft"), 30)
    # A median-sized latent, so the numbers describe the typical step.
    coords, feats, cond = train_ds.load(train_ds.ids[0], device)
    print(f"latent: {coords.shape[0]} tokens, cond {tuple(cond.shape)}\n")

    cfg = FlowMatchConfig(sigma_min=1e-5, t_sampling="uniform", t_max=1.0)
    st = sp.SparseTensor(feats=feats.float(), coords=coords)
    t = torch.full((1,), 500.0, device=device)

    print("--- BASE MODEL (no LoRA) ---")
    with torch.no_grad():
        fwd_base = timed(lambda: model(st, t, cond))
    print(f"forward, no grad          {fwd_base:7.2f}s")

    def fwd_bwd():
        loss, _ = flow_matching_loss(model, coords, feats, cond, cfg)
        loss.backward()
        model.zero_grad(set_to_none=True)

    # Nothing requires grad yet, so this measures forward + graph construction.
    print("\n--- WITH LoRA ---")
    lora_mod.apply_lora(model, rank=16)
    with torch.no_grad():
        fwd_lora = timed(lambda: model(st, t, cond))
    print(f"forward, no grad          {fwd_lora:7.2f}s   "
          f"({fwd_lora / max(fwd_base, 1e-9):.2f}x base)")

    params = list(lora_mod.lora_parameters(model))
    t_fb = timed(fwd_bwd, n=2)
    print(f"forward + backward        {t_fb:7.2f}s")
    print(f"  -> backward alone      ~{t_fb - fwd_lora:7.2f}s")

    with lora_mod.disabled_lora(model):
        with torch.no_grad():
            fwd_disabled = timed(lambda: model(st, t, cond))
    print(f"forward, adapters off     {fwd_disabled:7.2f}s")

    print("\n--- WHERE THE FORWARD GOES (top modules by self time) ---")
    from torch.profiler import profile, ProfilerActivity
    with torch.no_grad():
        with profile(activities=[ProfilerActivity.CPU], record_shapes=False) as prof:
            model(st, t, cond)
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=14))
    return 0


if __name__ == "__main__":
    sys.exit(main())

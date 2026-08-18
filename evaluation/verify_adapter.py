"""Verify the served adapter: it loads, it changes the model, and "base" is exact.

The claim the website now makes to a FabLab user is that picking "Original
TRELLIS.2" gives them the stock model. That is only true if disabling the
adapter restores the base weights EXACTLY, not approximately. This checks it
rather than trusting it:

  1. capture a forward pass on fixed noise before any LoRA exists
  2. wrap + load the adapter, disable it, forward again -> must be bit-identical
  3. enable it, forward again -> must DIFFER, or the adapter is doing nothing

Step 3 matters as much as step 2. A silently-not-loaded adapter would pass
step 2 perfectly while serving the base model under a "Fine-tuned" label.

    python verify_adapter.py
"""

import sys

import torch

sys.path.insert(0, ".")
from trellis_core import bootstrap  # noqa: F401  (env before torch/trellis)
from trellis_core import lora
from trellis_core.pipeline import load_pipeline
from server import config


def forward(flow, st, t, cond):
    with torch.no_grad():
        return flow(st, t, cond).feats.float().cpu().clone()


def main() -> int:
    print(f"adapter: {config.ADAPTER_PATH}")
    if not config.ADAPTER_PATH.exists():
        print("MISSING -- the server would serve base only")
        return 1

    print("loading pipeline ...", flush=True)
    pipeline = load_pipeline(config.MODEL_ID, device=config.DEVICE)
    flow = pipeline.models["shape_slat_flow_model_512"]
    flow.to(config.DEVICE).eval()

    from trellis2.modules import sparse as sp

    # A small deterministic input. Coords must be a plausible sparse grid, so
    # take a compact cube rather than random positions.
    g = torch.Generator().manual_seed(0)
    n = 256
    xyz = torch.stack(torch.meshgrid(
        torch.arange(8), torch.arange(8), torch.arange(4), indexing="ij"), -1).reshape(-1, 3)
    coords = torch.cat([torch.zeros(n, 1, dtype=torch.int32), xyz.int()], 1).to(config.DEVICE)
    feats = torch.randn(n, flow.in_channels, generator=g).to(config.DEVICE)
    st = sp.SparseTensor(feats=feats, coords=coords)
    t = torch.full((1,), 500.0, device=config.DEVICE)
    cond = torch.randn(1, 1029, flow.cond_channels, generator=g).to(config.DEVICE)

    before = forward(flow, st, t, cond)
    print(f"baseline forward: {tuple(before.shape)}  mean {before.mean():+.6f}")

    adapted = lora.apply_lora(flow, rank=config.ADAPTER_RANK, alpha=config.ADAPTER_ALPHA)
    state = torch.load(config.ADAPTER_PATH, map_location="cpu")
    loaded = lora.load_lora_state_dict(flow, state["lora"])
    print(f"wrapped {len(adapted)} modules, loaded {loaded}/{len(state['lora'])} tensors, "
          f"{lora.count_trainable(flow)/1e6:.2f}M adapter params")

    ok = True
    if loaded != len(state["lora"]):
        print("FAIL: not every tensor matched a wrapped module")
        ok = False

    lora.set_lora_enabled(flow, False)
    off = forward(flow, st, t, cond)
    same = torch.equal(before, off)
    print(f"adapter OFF vs baseline: {'identical' if same else 'DIFFERENT'} "
          f"(max abs diff {(off - before).abs().max():.3e})")
    if not same:
        print("FAIL: 'base' is not the stock model -- do not offer it as one")
        ok = False

    lora.set_lora_enabled(flow, True)
    on = forward(flow, st, t, cond)
    delta = (on - before).abs().max()
    print(f"adapter ON  vs baseline: max abs diff {delta:.3e}")
    if delta == 0:
        print("FAIL: enabling the adapter changed nothing -- weights did not load")
        ok = False

    print("\nPASS" if ok else "\nFAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

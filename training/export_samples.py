"""Regenerate held-out samples from saved LoRA checkpoints and export real meshes.

WHY THIS EXISTS
---------------
`sft_train.evaluate` renders a PNG per held-out object at each checkpoint but
throws the decoded mesh away -- so a finished run leaves pictures of geometry
and no geometry. This reconstructs it: load an adapter, re-sample the held-out
set, decode, and write GLB/STL.

Reproduces the eval exactly rather than approximately. The sampling noise is
seeded per held-out object with the same formula eval uses (9000 + index), and
the sampler gets the pipeline's own CFG parameters, so a mesh exported here is
the same object the corresponding PNG shows -- not a fresh draw that happens to
come from the same checkpoint.

Step 0 means the base model: the adapter is disabled rather than absent, which
is the same thing the baseline eval measured.

Run it AFTER training finishes. It loads a second full pipeline (~10GB) and on
a 32GB machine that will fight the trainer for memory -- the exact contention
that turned a 32s step into 464s of swap thrashing earlier in this project.

    PYTHONPATH=. python -m trellis_core.export_samples --run-id <run>
    PYTHONPATH=. python -m trellis_core.export_samples --run-id <run> --steps 0,900
"""

import argparse
import os
import re
import sys
import time
from typing import List, Optional

from trellis_core import bootstrap  # noqa: F401

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def available_checkpoints(run_dir: str) -> List[int]:
    root = os.path.join(run_dir, "ckpt")
    if not os.path.isdir(root):
        return []
    out = []
    for d in sorted(os.listdir(root)):
        if d.startswith("step_") and os.path.exists(os.path.join(root, d, "DONE")):
            out.append(int(re.sub(r"\D", "", d)))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--runs-dir", default=os.path.join(REPO, "runs"))
    ap.add_argument("--data", default=os.path.join(REPO, "data", "thingi10k_sft"))
    ap.add_argument("--steps", default="", help="comma list; default = base + every checkpoint")
    ap.add_argument("--n", type=int, default=6, help="held-out objects to export")
    ap.add_argument("--sample-steps", type=int, default=12)
    ap.add_argument("--decode-resolution", type=int, default=512)
    ap.add_argument("--holdout", type=int, default=30)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--model-id", default="microsoft/TRELLIS.2-4B")
    args = ap.parse_args(argv)

    import trimesh
    from trellis2.modules import sparse as sp
    from trellis_core import pipeline as pipe_mod
    from trellis_core import lora as lora_mod
    from training.sft_train import split_dataset

    run_dir = os.path.join(args.runs_dir, args.run_id)
    if not os.path.isdir(run_dir):
        print(f"no such run: {run_dir}", file=sys.stderr)
        return 1

    ckpts = available_checkpoints(run_dir)
    if args.steps:
        want = [int(x) for x in args.steps.split(",") if x.strip()]
    else:
        want = [0] + ckpts     # 0 = base model
    print(f"checkpoints on disk: {ckpts or 'none'}")
    print(f"exporting steps    : {want}")

    out_root = os.path.join(run_dir, "meshes")
    os.makedirs(out_root, exist_ok=True)

    _, eval_ds = split_dataset(args.data, args.holdout, seed=args.seed)
    print("loading pipeline ...", flush=True)
    pipeline = pipe_mod.load_pipeline(args.model_id, args.device)
    model = pipeline.models["shape_slat_flow_model_512"]
    model.to(args.device)
    model.eval()
    lora_mod.apply_lora(model, rank=args.rank, alpha=args.alpha)

    sampler = pipeline.shape_slat_sampler
    norm = pipeline.shape_slat_normalization
    mean = torch.as_tensor(norm["mean"], device=args.device)
    std = torch.as_tensor(norm["std"], device=args.device)
    sp_params = {**pipeline.shape_slat_sampler_params}
    sp_params["steps"] = args.sample_steps

    n_written = 0
    for step in want:
        if step == 0:
            lora_mod.set_lora_enabled(model, False)
            label = "base model (adapters disabled)"
        else:
            ck = os.path.join(run_dir, "ckpt", f"step_{step:08d}", "adapter.pt")
            if not os.path.exists(ck):
                print(f"step {step}: no adapter on disk, skipping")
                continue
            state = torch.load(ck, map_location="cpu")
            lora_mod.load_lora_state_dict(model, state["lora"])
            lora_mod.set_lora_enabled(model, True)
            label = f"checkpoint step {step}"

        d = os.path.join(out_root, f"step_{step:08d}")
        os.makedirs(d, exist_ok=True)
        print(f"\n--- {label} -> {d}", flush=True)

        for i, fid in enumerate(eval_ds.ids[:args.n]):
            t0 = time.time()
            coords, feats, cond = eval_ds.load(fid, args.device)
            # Same seeding as evaluate(), so this mesh IS the object in the PNG.
            g = torch.Generator(device="cpu").manual_seed(9000 + i)
            noise = sp.SparseTensor(
                feats=torch.randn(coords.shape[0], model.in_channels, generator=g).to(args.device),
                coords=coords)
            try:
                with torch.no_grad():
                    slat = sampler.sample(model, noise, cond=cond,
                                          neg_cond=torch.zeros_like(cond),
                                          verbose=False, **sp_params).samples
                    slat = slat * std + mean
                    meshes, _ = pipeline.decode_shape_slat(slat, args.decode_resolution)
                m = meshes[0]
                tm = trimesh.Trimesh(
                    vertices=m.vertices.detach().float().cpu().numpy(),
                    faces=m.faces.detach().cpu().numpy(), process=False)
                base = os.path.join(d, f"{fid}_step{step:04d}")
                tm.export(base + ".glb")
                tm.export(base + ".stl")
                n_written += 2
                print(f"  {fid}  {len(tm.faces):>8} faces  watertight={tm.is_watertight}  "
                      f"{time.time()-t0:.0f}s", flush=True)
            except Exception as e:
                print(f"  {fid}  FAILED {type(e).__name__}: {e}", flush=True)

    print(f"\nwrote {n_written} files under {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

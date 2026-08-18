"""Generate the same photo through several model checkpoints and export raw meshes.

This is the REAL inference path, not the eval shortcut. Eval reuses a held-out
object's stored coordinates; here a photograph goes in and the whole pipeline
runs: image conditioning, sparse-structure sampling, shape-SLat sampling,
decode. That is what a user actually gets.

Paired by construction. The sparse structure is sampled ONCE per photo and
reused by every model, and the SLat noise is seeded per photo and reused too.
The structure model is not something we train, so letting it re-sample per
model would inject variance that has nothing to do with the checkpoint being
compared. With both fixed, any difference between the exported meshes is the
shape-SLat weights and nothing else.

Meshes are exported raw -- no repair, no decimation, no solidify -- because the
question is what the model produces before post-processing, not after.

    PYTHONPATH=. python -m trellis_core.compare_models \\
        --images a.png b.png \\
        --models base sft-1200-20260806-0119:1050 sft-5k-...:1000
"""

import argparse
import os
import sys
import time
from typing import List, Optional, Tuple

from trellis_core import bootstrap  # noqa: F401

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_model(spec: str, runs_dir: str) -> Tuple[str, Optional[str]]:
    """'base' -> no adapter; 'run_id:step' -> that checkpoint's adapter."""
    if spec == "base":
        return "base", None
    run_id, _, step = spec.partition(":")
    step = int(step)
    # Two naming schemes, because two trainers wrote these: sft_train.py used
    # ckpt/step_00005000/, the portable training/cuda/sft_cuda.py uses ckpt/best_step5000/
    # (and 'final'/'last'). Both produce byte-identical adapter payloads, so the
    # only thing that differs is where to look.
    for sub in ("ckpt_keep", "ckpt"):
        for name in (f"step_{step:08d}", f"best_step{step}"):
            p = os.path.join(runs_dir, run_id, sub, name, "adapter.pt")
            if os.path.exists(p):
                return f"{run_id}@{step}", p
    raise FileNotFoundError(f"no adapter for {spec}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", default=os.path.join(REPO, "runs", "comparison"))
    ap.add_argument("--runs-dir", default=os.path.join(REPO, "runs"))
    ap.add_argument("--cond-resolution", type=int, default=512,
                    help="image conditioning resolution; run() uses 512 for the '512' path")
    ap.add_argument("--ss-resolution", type=int, default=32,
                    help="sparse-structure grid. run() maps '512'->32; passing 512 here makes "
                         "sample_sparse_structure compute a pooling ratio of 0 and abort")
    ap.add_argument("--decode-resolution", type=int, default=512)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--model-id", default="microsoft/TRELLIS.2-4B")
    args = ap.parse_args(argv)

    import trimesh
    from PIL import Image
    from trellis2.modules import sparse as sp
    from trellis_core import pipeline as pipe_mod
    from trellis_core import lora as lora_mod
    from . import geometric_judge
    from evaluation.topology_test import mesh_topology

    os.makedirs(args.out, exist_ok=True)
    models = [parse_model(m, args.runs_dir) for m in args.models]
    print(f"{len(args.images)} images x {len(models)} models "
          f"= {len(args.images)*len(models)} generations\n", flush=True)

    print("loading pipeline ...", flush=True)
    pipeline = pipe_mod.load_pipeline(args.model_id, args.device)
    flow = pipeline.models["shape_slat_flow_model_512"]
    flow.to(args.device); flow.eval()
    lora_mod.apply_lora(flow, rank=args.rank, alpha=args.alpha)

    sampler = pipeline.shape_slat_sampler
    norm = pipeline.shape_slat_normalization
    mean = torch.as_tensor(norm["mean"], device=args.device)
    std = torch.as_tensor(norm["std"], device=args.device)
    sp_params = {**pipeline.shape_slat_sampler_params}
    sp_params["steps"] = args.steps

    rows = []
    for img_i, img_path in enumerate(args.images):
        name = os.path.splitext(os.path.basename(img_path))[0]
        stem = f"{img_i:02d}_{name}"
        print(f"\n=== {img_path}", flush=True)
        img = Image.open(img_path).convert("RGBA")

        # Conditioning and structure are computed ONCE and shared by every
        # model, so the comparison isolates the shape-SLat weights.
        with torch.no_grad():
            # get_cond already returns {'cond', 'neg_cond'}, which is exactly what
            # sample_sparse_structure splats into its sampler.
            cond_d = pipeline.get_cond([img], args.cond_resolution)
            torch.manual_seed(args.seed + img_i)
            coords = pipeline.sample_sparse_structure(cond_d, args.ss_resolution, 1)
        cond = cond_d["cond"]
        print(f"    structure: {coords.shape[0]} voxels", flush=True)

        g = torch.Generator(device="cpu").manual_seed(9000 + img_i)
        noise_feats = torch.randn(coords.shape[0], flow.in_channels, generator=g).to(args.device)

        for label, adapter in models:
            t0 = time.time()
            if adapter is None:
                lora_mod.set_lora_enabled(flow, False)
            else:
                lora_mod.load_lora_state_dict(flow, torch.load(adapter, map_location="cpu")["lora"])
                lora_mod.set_lora_enabled(flow, True)
            try:
                with torch.no_grad():
                    noise = sp.SparseTensor(feats=noise_feats.clone(), coords=coords)
                    slat = sampler.sample(flow, noise, cond=cond,
                                          neg_cond=cond_d["neg_cond"],
                                          verbose=False, **sp_params).samples
                    slat = slat * std + mean
                    meshes, _ = pipeline.decode_shape_slat(slat, args.decode_resolution)
                m = meshes[0]
                tm = trimesh.Trimesh(
                    vertices=m.vertices.detach().float().cpu().numpy(),
                    faces=m.faces.detach().cpu().numpy(), process=False)
                safe = label.replace("@", "_at_").replace(":", "_")
                base = os.path.join(args.out, f"{stem}__{safe}")
                tm.export(base + ".glb")
                tm.export(base + ".stl")
                topo = mesh_topology(tm)
                s, _ = geometric_judge.score_mesh_detailed(
                    tm, geometric_judge.JudgeWeights(), rng=np.random.default_rng(7))
                rows.append({"image": name, "model": label, **topo,
                             "detail": s.detail_reward})
                print(f"    {label:<28} nm {100*topo['nonmanifold_rate']:.3f}%  "
                      f"open {100*topo['open_rate']:.3f}%  comps {topo['components']:>5}  "
                      f"detail {s.detail_reward:7.3f}  {time.time()-t0:.0f}s", flush=True)
            except Exception as e:
                print(f"    {label:<28} FAILED {type(e).__name__}: {e}", flush=True)
            if args.device == "mps":
                torch.mps.empty_cache()

    import json
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(rows, f, indent=1)

    # paired summary
    print("\n" + "=" * 78)
    print(f"{'model':<30}{'non-manifold %':>16}{'open %':>10}{'comps':>8}{'detail':>10}")
    for label, _ in models:
        r = [x for x in rows if x["model"] == label]
        if not r:
            continue
        print(f"{label:<30}{100*np.mean([x['nonmanifold_rate'] for x in r]):>16.4f}"
              f"{100*np.mean([x['open_rate'] for x in r]):>10.4f}"
              f"{np.mean([x['components'] for x in r]):>8.0f}"
              f"{np.mean([x['detail'] for x in r]):>10.3f}")
    print(f"\nraw meshes (no post-processing) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

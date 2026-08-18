"""Does a checkpoint consistently produce cleaner topology than the base model?

THE QUESTION
------------
The exported meshes suggested step 1050 has fewer non-manifold edges than base,
with 750 and 900 in between. One sample per object cannot establish "consistently"
-- it establishes "on this draw". This runs a paired test across many objects and
several sampling seeds.

TWO MEASUREMENT DECISIONS THAT MATTER
-------------------------------------
1. Counted on the FULL mesh, never the decimated one. `geometric_judge` decimates
   before scoring, and that decimation is itself a large source of non-manifold
   edges: measured 0.086% open edges on a decoded mesh versus 4.25% after the
   judge's preprocessing, a ~50x inflation. Scoring L_Topo would therefore mostly
   measure the decimator, not the model.

2. Paired, with common random numbers. Every model sees the same (object, seed)
   combinations, so each comparison is like-for-like and the test is a paired
   one -- far more sensitive than comparing two independent means, which is what
   matters when the effect is real but the between-object variance is large.

Reports per-model rates plus, for each tuned checkpoint against base, the number
of pairs won and a sign test. Rates are per-edge, so meshes of different sizes
are comparable.

    PYTHONPATH=. python -m trellis_core.topology_test --run-id <run> \
        --steps 0,750,900,1050 --objects 10 --seeds 3
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List

from . import bootstrap  # noqa: F401

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def mesh_topology(tm) -> Dict:
    """Edge-level topology of the FULL mesh. No decimation, no repair."""
    edges = tm.edges_sorted
    _uniq, edge_counts = np.unique(edges, axis=0, return_counts=True)
    n_edges = len(edge_counts)
    # Component counting was capped at 1.2M faces for speed, but every mesh
    # from the real inference path exceeds that, so the metric silently
    # reported -1 for the entire comparison. Counting via the face-adjacency
    # graph instead of trimesh.split avoids building N submeshes, which is
    # what was actually slow -- we only need the count and the largest share.
    comps = None
    try:
        import scipy.sparse as _sp
        from scipy.sparse.csgraph import connected_components as _cc
        f = len(tm.faces)
        adj = tm.face_adjacency
        if f:
            g = _sp.coo_matrix(
                (np.ones(len(adj)), (adj[:, 0], adj[:, 1])), shape=(f, f))
            n_comp, labels = _cc(g, directed=False)
            comp_sizes = np.bincount(labels)
            comps, big = int(n_comp), int(comp_sizes.max())
    except Exception:
        comps = None
    if comps is None:                       # fall back to the old path
        _c = tm.split(only_watertight=False) if len(tm.faces) < 1_200_000 else []
        comps = len(_c) if _c else -1
        big = max((len(c.faces) for c in _c), default=len(tm.faces))

    return {
        "faces": int(len(tm.faces)),
        "edges": int(n_edges),
        "n_open": int((edge_counts == 1).sum()),
        "n_nonmanifold": int((edge_counts > 2).sum()),
        "open_rate": float((edge_counts == 1).mean()),
        "nonmanifold_rate": float((edge_counts > 2).mean()),
        "components": int(comps),
        "largest_frac": float(big / max(len(tm.faces), 1)),
        "watertight": bool(tm.is_watertight),
        "euler": int(tm.euler_number),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--runs-dir", default=os.path.join(REPO, "runs"))
    ap.add_argument("--data", default=os.path.join(REPO, "data", "thingi10k_sft"))
    ap.add_argument("--steps", default="0,750,900,1050")
    ap.add_argument("--objects", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--sample-steps", type=int, default=12)
    ap.add_argument("--decode-resolution", type=int, default=512)
    ap.add_argument("--holdout", type=int, default=30)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--model-id", default="microsoft/TRELLIS.2-4B")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    import trimesh
    from trellis2.modules import sparse as sp
    from . import pipeline as pipe_mod, lora as lora_mod
    from .sft_train import split_dataset

    run_dir = os.path.join(args.runs_dir, args.run_id)
    out_path = args.out or os.path.join(run_dir, "topology_test.jsonl")
    want = [int(x) for x in args.steps.split(",") if x.strip()]

    # Resume: this is a multi-hour job and must survive an interruption.
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            try:
                r = json.loads(line)
                done.add((r["step"], r["fid"], r["seed"]))
            except Exception:
                pass
    print(f"{len(done)} results already on disk")

    _, eval_ds = split_dataset(args.data, args.holdout, seed=0)
    fids = eval_ds.ids[:args.objects]

    print("loading pipeline ...", flush=True)
    pipeline = pipe_mod.load_pipeline(args.model_id, args.device)
    model = pipeline.models["shape_slat_flow_model_512"]
    model.to(args.device); model.eval()
    lora_mod.apply_lora(model, rank=args.rank, alpha=args.alpha)

    sampler = pipeline.shape_slat_sampler
    norm = pipeline.shape_slat_normalization
    mean = torch.as_tensor(norm["mean"], device=args.device)
    std = torch.as_tensor(norm["std"], device=args.device)
    sp_params = {**pipeline.shape_slat_sampler_params}
    sp_params["steps"] = args.sample_steps

    fh = open(out_path, "a", buffering=1)
    total = len(want) * len(fids) * args.seeds
    n = 0
    t_start = time.time()

    for step in want:
        if step == 0:
            lora_mod.set_lora_enabled(model, False)
        else:
            ck = os.path.join(run_dir, "ckpt_keep", f"step_{step:08d}", "adapter.pt")
            if not os.path.exists(ck):
                ck = os.path.join(run_dir, "ckpt", f"step_{step:08d}", "adapter.pt")
            if not os.path.exists(ck):
                print(f"step {step}: adapter missing, skipping")
                continue
            lora_mod.load_lora_state_dict(model, torch.load(ck, map_location="cpu")["lora"])
            lora_mod.set_lora_enabled(model, True)
        print(f"\n=== step {step} ===", flush=True)

        for i, fid in enumerate(fids):
            coords, feats, cond = eval_ds.load(fid, args.device)
            for seed in range(args.seeds):
                n += 1
                if (step, fid, seed) in done:
                    continue
                # Identical seed across every model: this is what makes the
                # comparison paired rather than two independent samples.
                g = torch.Generator(device="cpu").manual_seed(50_000 + 991 * i + seed)
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
                    rec = {"step": step, "fid": fid, "seed": seed, **mesh_topology(tm)}
                    fh.write(json.dumps(rec) + "\n")
                    el = time.time() - t_start
                    print(f"  [{n}/{total}] {fid} s{seed}  nm={rec['nonmanifold_rate']*100:.3f}% "
                          f"open={rec['open_rate']*100:.3f}%  comps={rec['components']}  "
                          f"eta {(total-n)*el/max(n,1)/60:.0f}m", flush=True)
                except Exception as e:
                    print(f"  [{n}/{total}] {fid} s{seed}  FAILED {type(e).__name__}: {e}", flush=True)
                del noise
                if args.device == "mps":
                    torch.mps.empty_cache()
            del coords, feats, cond

    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

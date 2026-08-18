"""Does TRELLIS.2's shape VAE preserve printability?

THE QUESTION THIS ANSWERS
-------------------------
Any plan that fine-tunes the shape flow model toward printable geometry --
supervised on real printable meshes, or DPO against them -- assumes the target
is expressible. The flow model produces a SLat latent; the decoder turns that
latent into a mesh. If the decoder cannot emit a watertight mesh even when
handed the encoding of a *perfectly watertight input*, then no amount of flow
model training can produce one, because the defect is downstream of everything
being trained.

So: encode a known-clean mesh, decode it straight back, and judge both. This
measures the ceiling of the whole approach and needs no training at all.

Two outcomes, both decisive:
  * round-trip preserves watertightness -> the target is reachable, and the
    flow model is the right component to train.
  * round-trip destroys it -> the decoder is the binding constraint. Training
    the flow model is tuning the wrong component, and the honest response is
    either to fine-tune the decoder as well or to keep repairing in
    post-processing.

Note this is a strict *upper* bound on what training can achieve. The encoder
sees the real mesh; the flow model at inference never does. Anything lost here
is lost for good.

Usage:
    PYTHONPATH=. python trellis_core/vae_roundtrip.py <mesh> [<mesh> ...]
"""

import argparse
import os
import sys
import time
from typing import Optional

from trellis_core import bootstrap  # noqa: F401  (env before torch)

import numpy as np
import torch
import trimesh

from trellis_core import dual_grid_cpu
from evaluation import geometric_judge


# Encoder resolution / latent resolution. The encoder has 4 downsample blocks
# (2^4 = 16), so a dual grid built at 512 yields a latent on a 32^3 grid --
# which is what the '512' pipeline's SLat actually lives on (observed median
# 2818 active voxels across this repo's recorded runs).
DUAL_GRID_RESOLUTION = 512
DECODE_RESOLUTION = 512


def load_encoder(device: str = "mps"):
    """Load the shape VAE encoder.

    It is NOT in pipeline.json's `models` dict -- inference only ever needs the
    decoder -- but the checkpoint and its config ship in the same repo, so it
    loads through the ordinary `models.from_pretrained` path.
    """
    from trellis2 import models

    enc = models.from_pretrained("microsoft/TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16")
    return enc.eval().to(device)


def encode_mesh(mesh: trimesh.Trimesh, encoder, device: str = "mps", resolution: int = DUAL_GRID_RESOLUTION):
    """Watertight mesh -> SLat SparseTensor (de-normalized, i.e. VAE space)."""
    from trellis2.modules import sparse as sp

    mesh = dual_grid_cpu.normalize_mesh(mesh.copy())
    v = torch.from_numpy(np.asarray(mesh.vertices)).float()
    f = torch.from_numpy(np.asarray(mesh.faces)).long()

    voxel_indices, local, intersected = dual_grid_cpu.mesh_to_dual_grid(v, f, resolution=resolution)

    # SparseTensor coords carry a leading batch column.
    coords = torch.cat([
        torch.zeros(voxel_indices.shape[0], 1, dtype=torch.int32),
        voxel_indices.int(),
    ], dim=1).to(device)

    vertices_st = sp.SparseTensor(feats=local.float().to(device), coords=coords)
    intersected_st = sp.SparseTensor(feats=intersected.to(device), coords=coords)

    with torch.no_grad():
        latent = encoder(vertices_st, intersected_st)
    return latent


def decode_latent(pipeline, latent, resolution: int = DECODE_RESOLUTION) -> Optional[trimesh.Trimesh]:
    with torch.no_grad():
        meshes, _subs = pipeline.decode_shape_slat(latent, resolution)
    m = meshes[0]
    verts = m.vertices.detach().float().cpu().numpy()
    faces = m.faces.detach().cpu().numpy()
    if len(faces) == 0:
        return None
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def describe(mesh: Optional[trimesh.Trimesh], weights) -> dict:
    if mesh is None or len(mesh.faces) == 0:
        return {"faces": 0, "watertight": False, "failed": True}
    rng = np.random.default_rng(0)  # fixed: this is a paired before/after comparison
    s, _details = geometric_judge.score_mesh_detailed(mesh, weights, rng=rng)
    return {
        "faces": len(mesh.faces),
        "watertight": bool(mesh.is_watertight),
        "components": len(mesh.split(only_watertight=False)) if len(mesh.faces) < 400000 else -1,
        "R_Detail": float(s.detail_reward),
        "L_OH": float(s.overhang_penalty),
        "L_Th": float(s.thickness_penalty),
        "L_Th_valid": bool(s.thickness_valid),
        "L_Topo": float(s.topology_penalty),
        "S": float(s.total),
        "failed": False,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("meshes", nargs="+")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--model-id", dest="model_id", default="microsoft/TRELLIS.2-4B")
    ap.add_argument("--resolution", type=int, default=DUAL_GRID_RESOLUTION)
    args = ap.parse_args(argv)

    from trellis_core import pipeline as pipe_mod

    print("loading pipeline ...", flush=True)
    t0 = time.time()
    pipeline = pipe_mod.load_pipeline(args.model_id, args.device)
    encoder = load_encoder(args.device)
    print(f"loaded in {time.time() - t0:.0f}s\n", flush=True)

    weights = geometric_judge.JudgeWeights()
    rows = []
    for path in args.meshes:
        name = os.path.basename(path)
        try:
            src = trimesh.load(path, force="mesh")
            if not hasattr(src, "faces") or len(src.faces) == 0:
                print(f"{name}: no faces, skipping")
                continue
            before = describe(dual_grid_cpu.normalize_mesh(src.copy()), weights)

            t = time.time()
            latent = encode_mesh(src, encoder, args.device, args.resolution)
            n_tok = latent.coords.shape[0]
            out = decode_latent(pipeline, latent)
            after = describe(out, weights)
            dt = time.time() - t

            rows.append((name, n_tok, before, after))
            print(
                f"{name:<16} tokens {n_tok:>6}  {dt:>5.0f}s\n"
                f"   watertight  {before['watertight']!s:>5} -> {after['watertight']!s:<5}"
                f"   faces {before['faces']:>7} -> {after['faces']:<7}\n"
                f"   L_Th        {before.get('L_Th', 0):.4f} -> {after.get('L_Th', 0):.4f}"
                f"   L_OH {before.get('L_OH', 0):.4f} -> {after.get('L_OH', 0):.4f}"
                f"   L_Topo {before.get('L_Topo', 0):.4f} -> {after.get('L_Topo', 0):.4f}",
                flush=True,
            )
        except Exception as e:
            print(f"{name}: FAILED {type(e).__name__}: {e}", flush=True)
            import traceback; traceback.print_exc()

    if rows:
        kept = sum(1 for _, _, b, a in rows if b["watertight"] and a["watertight"])
        wt_in = sum(1 for _, _, b, _ in rows if b["watertight"])
        print(f"\nwatertight preserved: {kept}/{wt_in} of the watertight inputs")
    return 0


if __name__ == "__main__":
    sys.exit(main())

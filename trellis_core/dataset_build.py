"""Build the Thingi10K SFT dataset: clean mesh -> (conditioning image, SLat latent).

WHAT THIS PRODUCES
------------------
    data/<name>/manifest.jsonl      one line per completed model (append-only)
    data/<name>/images/<id>.png     rendered conditioning view, RGBA
    data/<name>/latents/<id>.npz    coords [N,3] int32 + feats [N,32] fp16
    data/<name>/cond/<id>.npy       precomputed image conditioning, fp16

Latents are stored in the flow model's own NORMALISED space. The pipeline
de-normalises (`slat = slat * std + mean`) after sampling, so a latent captured
from pipeline output is in a different space than the one the model regresses
in; storing the wrong one silently rescales every loss term at training time
and looks like a badly chosen learning rate.

SOURCE AND FILTERING
--------------------
Thingi10K is explicitly a dataset of real-world 3D-printing *messiness*, not a
clean corpus -- its own headline figures are 50% non-solid, 45% with
self-intersections, 26% multi-component, 22% non-manifold. Training on the raw
dump would teach the model to reproduce exactly the defects we want removed.

We therefore filter on the dataset's own precomputed attributes, which is both
more reliable and vastly cheaper than loading 10,000 meshes to test them:

    Closed, Edge manifold, Vertex manifold, Single Component, PWN,
    No degenerate faces, and num_self_intersections == 0

That leaves 3,935 of 10,000 models. Licences are carried into the manifest --
the corpus is mostly CC-BY and CC-BY-SA, but ~1,100 of the clean models are
non-commercial variants, which matters if this ever leaves personal research.

UNATTENDED OPERATION
--------------------
Resumable and append-only: re-running skips anything already in the manifest,
so an interrupted overnight build continues rather than restarting. Progress is
written through `TrainRun`, so the dataset build shows up live on the same
dashboard as training.
"""

import argparse
import csv
import json
import os
import sys
import time
from typing import Dict, List, Optional

from . import bootstrap  # noqa: F401  (env before torch)

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF_REPO = "Thingi10K/Thingi10K"


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def _truthy(v) -> bool:
    return str(v).strip().upper() == "TRUE"


def select_clean_ids(
    limit: Optional[int] = None,
    min_faces: int = 200,
    max_faces: int = 400_000,
    exclude_noncommercial: bool = False,
    per_thing_cap: int = 12,
    dedup_geometry: bool = True,
    seed: int = 0,
) -> List[Dict]:
    """Return metadata rows for models passing every cleanliness filter.

    Shuffled with a fixed seed before truncation: the file ids are ordered, and
    ordered ids correlate with upload date and therefore with object type, so
    taking the first N would hand back a temporally clustered and much less
    diverse sample than the corpus actually offers.
    """
    from huggingface_hub import hf_hub_download

    s_path = hf_hub_download(HF_REPO, "metadata/input_summary.csv", repo_type="dataset")
    g_path = hf_hub_download(HF_REPO, "metadata/geometry_data.csv", repo_type="dataset")
    summary = {r["ID"]: r for r in csv.DictReader(open(s_path))}
    geometry = {r["file_id"]: r for r in csv.DictReader(open(g_path))}

    out = []
    for fid, r in summary.items():
        g = geometry.get(fid)
        if not g:
            continue
        if not all(_truthy(r[k]) for k in
                   ("Closed", "Edge manifold", "Vertex manifold",
                    "Single Component", "PWN", "No degenerate faces")):
            continue
        try:
            if float(g["num_self_intersections"]) != 0:
                continue
            nf = float(g["num_faces"])
        except (TypeError, ValueError):
            continue
        if not (min_faces <= nf <= max_faces):
            continue
        lic = r.get("License", "")
        if exclude_noncommercial and "Non-Commercial" in lic:
            continue
        out.append({"file_id": fid, "license": lic, "thing_id": r.get("Thing ID"),
                    "num_faces": int(nf),
                    "num_vertices": int(float(g.get("num_vertices") or 0)),
                    "_area": round(float(g.get("total_area") or 0), 3),
                    "_euler": int(float(g.get("euler_characteristic") or 0))})

    rng = np.random.default_rng(seed)
    rng.shuffle(out)

    # --- curation beyond per-file cleanliness -------------------------------
    # Thingi10K is organised as ~2,000 "things" split into ~10,000 files, and
    # the split is wildly uneven: within the clean pool, 3,886 files come from
    # just 1,003 things, and a single thing contributes 184 of them. Uncapped,
    # that one thing would be 4.7% of the dataset and the model would spend a
    # meaningful share of its updates on one designer's parts library.
    #
    # A "thing" is not always duplicates -- it is often a multi-part assembly,
    # where the parts are genuinely different objects. So this caps rather than
    # collapses, keeping several files per thing while preventing any one from
    # dominating.
    if dedup_geometry:
        seen, deduped = set(), []
        for r in out:
            # Identical vertex count, face count, area and Euler characteristic
            # is a near-certain duplicate; 118 such files exist in the pool.
            key = (r["num_vertices"], r["num_faces"], r["_area"], r["_euler"])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(r)
        out = deduped

    if per_thing_cap and per_thing_cap > 0:
        counts: Dict[str, int] = {}
        capped = []
        for r in out:                      # already shuffled, so the kept
            t = r.get("thing_id") or ""    # files per thing are a random subset
            if counts.get(t, 0) >= per_thing_cap:
                continue
            counts[t] = counts.get(t, 0) + 1
            capped.append(r)
        out = capped

    for r in out:
        r.pop("_area", None); r.pop("_euler", None)
    return out[:limit] if limit else out


# ---------------------------------------------------------------------------
# Per-model build
# ---------------------------------------------------------------------------


def build_one(fid: str, pipeline, encoder, out_dir: str, *,
              resolution: int, image_size: int, views: int,
              cond_resolution: int) -> Dict:
    """Download, render, encode one model. Returns a manifest row."""
    import torch
    import trimesh
    from huggingface_hub import hf_hub_download
    from backends import dual_grid_cpu
    from . import render_mesh, vae_roundtrip

    t0 = time.time()
    stl = hf_hub_download(HF_REPO, f"raw_meshes/{fid}.stl", repo_type="dataset")
    mesh = trimesh.load(stl, force="mesh")
    if not hasattr(mesh, "faces") or len(mesh.faces) == 0:
        raise ValueError("no faces")
    t_dl = time.time() - t0

    # -- conditioning image --------------------------------------------------
    t = time.time()
    imgs = render_mesh.render_views(mesh, n_views=views, size=image_size)
    cov = render_mesh.coverage(imgs[0])
    if cov < 0.02:
        # Framing failed (degenerate bounds, stray vertex). Such a pair would
        # train the model on an almost-empty image, so drop it rather than
        # quietly poison the dataset.
        raise ValueError(f"render coverage {cov:.3f} too low")
    img_path = os.path.join(out_dir, "images", f"{fid}.png")
    imgs[0].save(img_path)
    t_render = time.time() - t

    # -- latent --------------------------------------------------------------
    t = time.time()
    latent = vae_roundtrip.encode_mesh(mesh, encoder, resolution=resolution)
    feats = latent.feats.detach().float().cpu()
    norm = pipeline.shape_slat_normalization
    mean = torch.as_tensor(norm["mean"], dtype=feats.dtype)
    std = torch.as_tensor(norm["std"], dtype=feats.dtype)
    feats_n = (feats - mean[None]) / std[None]          # flow-model space
    coords = latent.coords.detach().cpu().numpy()[:, 1:].astype(np.int32)  # drop batch col
    np.savez_compressed(os.path.join(out_dir, "latents", f"{fid}.npz"),
                        coords=coords, feats=feats_n.numpy().astype(np.float16))
    t_encode = time.time() - t

    # -- conditioning embedding ---------------------------------------------
    t = time.time()
    cond = pipeline.get_cond([imgs[0]], cond_resolution, include_neg_cond=False)["cond"]
    np.save(os.path.join(out_dir, "cond", f"{fid}.npy"),
            cond.detach().cpu().numpy().astype(np.float16))
    t_cond = time.time() - t

    return {
        "file_id": fid,
        "n_tokens": int(coords.shape[0]),
        "latent_channels": int(feats_n.shape[1]),
        "coverage": round(cov, 4),
        "cond_shape": list(cond.shape),
        "faces": int(len(mesh.faces)),
        "seconds": {"download": round(t_dl, 2), "render": round(t_render, 2),
                    "encode": round(t_encode, 2), "cond": round(t_cond, 2)},
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", default="thingi10k_sft")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--resolution", type=int, default=512, help="dual-grid resolution")
    ap.add_argument("--image-size", type=int, default=518)
    ap.add_argument("--cond-resolution", type=int, default=518)
    ap.add_argument("--views", type=int, default=1)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--model-id", default="microsoft/TRELLIS.2-4B")
    ap.add_argument("--exclude-noncommercial", action="store_true")
    ap.add_argument("--per-thing-cap", type=int, default=12,
                    help="max files from one Thingiverse thing (0 = uncapped)")
    ap.add_argument("--no-dedup", dest="dedup_geometry", action="store_false")
    ap.add_argument("--max-faces", type=int, default=400_000)
    ap.add_argument("--data-root", default=os.path.join(REPO, "data"))
    ap.add_argument("--runs-dir", default=os.path.join(REPO, "runs"))
    ap.add_argument("--dry-run", action="store_true", help="select and report, build nothing")
    args = ap.parse_args(argv)

    out_dir = os.path.join(args.data_root, args.name)
    for sub in ("images", "latents", "cond"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    print("selecting clean models ...", flush=True)
    rows = select_clean_ids(limit=args.limit, exclude_noncommercial=args.exclude_noncommercial,
                            per_thing_cap=args.per_thing_cap, dedup_geometry=args.dedup_geometry,
                            max_faces=args.max_faces)
    print(f"selected {len(rows)} models", flush=True)
    if args.dry_run:
        import collections
        lic = collections.Counter(r["license"] for r in rows)
        for k, v in lic.most_common():
            print(f"  {k[:60]:<60} {v}")
        import collections as _c
        th = _c.Counter(r["thing_id"] for r in rows)
        print(f"  median faces      : {int(np.median([r['num_faces'] for r in rows]))}")
        print(f"  distinct things   : {len(th)}")
        print(f"  max files / thing : {max(th.values())} ({100*max(th.values())/len(rows):.2f}% of set)")
        return 0

    # Resume: anything already recorded is skipped.
    manifest_path = os.path.join(out_dir, "manifest.jsonl")
    done = set()
    if os.path.exists(manifest_path):
        for line in open(manifest_path):
            try:
                done.add(json.loads(line)["file_id"])
            except Exception:
                pass
    todo = [r for r in rows if r["file_id"] not in done]
    print(f"{len(done)} already built, {len(todo)} to go", flush=True)
    if not todo:
        return 0

    from .train_run import TrainRun
    run = TrainRun(f"dataset-{args.name}", args.runs_dir)
    run.write_meta({"kind": "dataset_build", "target": len(rows), "args": vars(args)})
    # Heartbeat before the pipeline load, not after: the load takes ~92s, and
    # without this the dashboard shows the run as dead for that whole window --
    # which on a phone, mid-build, is indistinguishable from a crash.
    run.status(step=len(done), total_steps=len(rows), state="loading pipeline")

    from . import pipeline as pipe_mod, vae_roundtrip
    print("loading pipeline ...", flush=True)
    t0 = time.time()
    pipeline = pipe_mod.load_pipeline(args.model_id, args.device)
    encoder = vae_roundtrip.load_encoder(args.device)
    print(f"loaded in {time.time() - t0:.0f}s", flush=True)

    mf = open(manifest_path, "a", buffering=1)
    n_ok, n_fail, t_start = len(done), 0, time.time()
    for i, r in enumerate(todo, 1):
        fid = r["file_id"]
        try:
            info = build_one(fid, pipeline, encoder, out_dir,
                             resolution=args.resolution, image_size=args.image_size,
                             views=args.views, cond_resolution=args.cond_resolution)
            info.update({k: r[k] for k in ("license", "thing_id")})
            mf.write(json.dumps(info) + "\n")
            n_ok += 1
            per = (time.time() - t_start) / i
            # Named fields, not a `loss` flavoured with token counts. An
            # earlier version logged n_tokens as `loss` so the build would
            # appear on the training dashboard at all; the chart then plotted
            # voxel counts jumping between 1258 and 4326 under a "Loss" axis,
            # which is not a rough edge but a dashboard telling you something
            # false. Run kind now drives the view instead.
            run.log("progress", n_ok,
                    n_tokens=int(info["n_tokens"]),
                    seconds=sum(info["seconds"].values()),
                    faces=int(info["faces"]),
                    coverage=info["coverage"])
            run.status(step=n_ok, total_steps=len(rows), state="building",
                       eta_s=(len(todo) - i) * per, failures=n_fail)
            print(f"[{i}/{len(todo)}] {fid}  tokens {info['n_tokens']:>5}  "
                  f"{sum(info['seconds'].values()):.1f}s  eta {(len(todo)-i)*per/3600:.1f}h", flush=True)
        except Exception as e:
            n_fail += 1
            print(f"[{i}/{len(todo)}] {fid}  FAILED {type(e).__name__}: {str(e)[:90]}", flush=True)
            run.log("event", n_ok, error=f"{fid}: {type(e).__name__}")

    run.status(step=n_ok, total_steps=len(rows), state="finished", failures=n_fail)
    print(f"\ndone: {n_ok} built, {n_fail} failed -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

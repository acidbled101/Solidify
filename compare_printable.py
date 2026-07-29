"""
Run every print-prep pipeline on the same .glb and report a side-by-side diff.

  v1 = trellis_core/printable.py    (trimesh repair + voxel flood-fill)
  v2 = trellis_core/printable_v2.py (PyMeshLab clean/Taubin + Manifold3D)
  v3 = the same v2 stages with repair_backend="meshlib" (MeshLib ray-parity
       orientation repair + MeshLib SDF rebuild)

All are measured with the same ruler: fidelity is recomputed here against one
shared reference mesh (the input's largest component, pre-repair), and the
printability diagnostics come from the same printable.diagnostics() for each.

Usage:
    python compare_printable.py test/output_3d.glb
    python compare_printable.py in.glb --outdir /tmp/cmp --quiet
    python compare_printable.py in.glb --only v3 --taubin-steps 20

Outputs land in <outdir>/v{1,2,3}.{glb,stl} so they can be opened side by side
in a slicer.
"""

import argparse
import json
import os
import sys
import time
from contextlib import redirect_stdout
from io import StringIO

import trimesh

import trellis_core.printable as printable
import trellis_core.printable_v2 as printable_v2
import mesh_io


def _file_kb(path):
    try:
        return os.path.getsize(path) / 1024.0
    except OSError:
        return None


def _run(label, fn, quiet):
    """Run one pipeline, returning (result, error, log, seconds)."""
    buf = StringIO()
    t0 = time.time()
    try:
        if quiet:
            with redirect_stdout(buf):
                result = fn()
        else:
            print(f"\n{'=' * 70}\n{label} pipeline\n{'=' * 70}")
            result = fn()
        return result, None, buf.getvalue(), time.time() - t0
    except Exception as e:
        import traceback
        return None, f"{type(e).__name__}: {e}", buf.getvalue() + traceback.format_exc(), time.time() - t0


def _path_of(result, kind):
    """glb_path/stl_path are only set in single mode; files[] always is."""
    return next((f["path"] for f in result.files if f["kind"] == kind), None)


def _measure(result, reference, overhang_angle):
    """Metrics for one pipeline's output, all computed against `reference`."""
    if result is None:
        return None

    glb_path = _path_of(result, "printable_glb")
    stl_path = _path_of(result, "printable_stl")

    # Geometry is measured on the GLB, not the STL: STL is a triangle soup with
    # no shared vertices, so an unprocessed load of it always looks
    # non-watertight. The STL is still checked below, with vertex merging on --
    # that is exactly what a slicer does when it ingests the file.
    mesh = mesh_io.load_glb(glb_path)
    fid = printable.fidelity(reference, mesh)
    diag = printable.diagnostics(mesh, overhang_angle)
    stl_mesh = trimesh.load(stl_path, force="mesh")

    return {
        "seconds": result.seconds,
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
        "watertight": bool(mesh.is_watertight),
        "stl_watertight": bool(stl_mesh.is_watertight),
        "volume": float(mesh.volume) if mesh.is_watertight else None,
        "bodies": len(mesh.split(only_watertight=False)),
        "euler_number": int(mesh.euler_number),
        "chamfer_pct": fid["chamfer_pct"] if fid else None,
        "hausdorff_pct": fid["hausdorff_pct"] if fid else None,
        "vol_change_pct": fid["vol_change_pct"] if fid else None,
        "overhang_pct": diag["overhang_pct"],
        "thin_wall_warnings": diag["thin_wall_warnings"],
        "stl_kb": _file_kb(stl_path),
        "glb_kb": _file_kb(glb_path),
        "stl_path": stl_path,
        "glb_path": glb_path,
        "stages": dict(getattr(result, "stages", {}) or {}),
        "notes": list(getattr(result, "notes", []) or []),
    }


# (key, label, format, lower_is_better or None if not a "score")
ROWS = [
    ("seconds", "Wall time (s)", "{:.1f}", True),
    ("vertices", "Vertices", "{:,}", None),
    ("faces", "Faces", "{:,}", None),
    ("watertight", "Watertight (GLB)", "{}", None),
    ("stl_watertight", "Watertight (STL, merged)", "{}", None),
    ("bodies", "Disconnected bodies", "{:,}", True),
    ("euler_number", "Euler number", "{:,}", None),
    ("volume", "Volume", "{:.5f}", None),
    ("chamfer_pct", "Chamfer vs original (% size)", "{:.3f}", True),
    ("hausdorff_pct", "Hausdorff vs original (% size)", "{:.3f}", True),
    ("vol_change_pct", "Volume change vs original (%)", "{:+.2f}", None),
    ("overhang_pct", "Overhang area (%)", "{:.1f}", True),
    ("thin_wall_warnings", "Thin-wall warnings", "{:,}", True),
    ("stl_kb", "STL size (KB)", "{:,.0f}", None),
    ("glb_kb", "GLB size (KB)", "{:,.0f}", None),
]


def _fmt(value, spec):
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "NO"
    return spec.format(value)


VARIANTS = {
    "v1": "v1 voxel",
    "v2": "v2 pymeshlab",
    "v3": "v3 meshlib",
}


def print_table(metrics, baseline="v1"):
    """Side-by-side table; deltas are measured against `baseline`."""
    cols = [k for k in VARIANTS if metrics.get(k) is not None]
    label_w = max(len(r[1]) for r in ROWS) + 2
    col_w = 15
    print(f"\n{'=' * 70}")
    print("COMPARISON")
    print("=" * 70)
    header = f"{'Metric':<{label_w}}" + "".join(f"{VARIANTS[k]:>{col_w}}" for k in cols)
    print(header)
    print("-" * len(header))

    for key, label, spec, lower_better in ROWS:
        row = f"{label:<{label_w}}"
        base = metrics[baseline].get(key) if metrics.get(baseline) else None
        for k in cols:
            value = metrics[k].get(key)
            cell = _fmt(value, spec)
            # Annotate every non-baseline column with its ratio to the baseline.
            if (k != baseline and lower_better is not None
                    and isinstance(value, (int, float)) and not isinstance(value, bool)
                    and isinstance(base, (int, float)) and not isinstance(base, bool)
                    and base and value):
                factor = base / value if lower_better else value / base
                if factor >= 1.5:
                    cell += f" {factor:.0f}x"
            row += f"{cell:>{col_w}}"
        print(row)

    for k in cols:
        if metrics[k].get("stages"):
            print(f"\n{VARIANTS[k]} stage breakdown (s): " +
                  ", ".join(f"{n}={s:.1f}" for n, s in metrics[k]["stages"].items()))
    for k in cols:
        for note in metrics[k].get("notes", []):
            print(f"  [{k} note] {note}")


def main():
    p = argparse.ArgumentParser(description="Compare the v1 and v2 print-prep pipelines")
    p.add_argument("input", help="Path to input .glb")
    p.add_argument("--outdir", default=None,
                   help="Output directory (default: model_output/compare_<stem>_<timestamp>)")
    p.add_argument("--only", choices=["v1", "v2", "v3", "all", "both"], default="all",
                   help="Which pipelines to run ('both' = v1+v2, kept for compatibility)")
    p.add_argument("--quiet", action="store_true", help="Suppress each pipeline's own logs")
    p.add_argument("--json", dest="json_path", default=None, help="Also write metrics to this JSON file")

    p.add_argument("--target-faces", type=int, default=1000000)
    p.add_argument("--overhang-angle", type=float, default=45.0)
    p.add_argument("--no-solid-infill", dest="solid_infill", action="store_false", default=True)
    p.add_argument("--multi-object", action="store_true",
                   help="Process every large object (v1 concatenates, v2 boolean-unions)")
    p.add_argument("--max-objects", type=int, default=10)
    p.add_argument("--min-object-ratio", type=float, default=0.05)

    # v2-only knobs
    p.add_argument("--taubin-steps", type=int, default=10)
    p.add_argument("--taubin-lambda", type=float, default=0.5)
    p.add_argument("--taubin-mu", type=float, default=-0.53)
    p.add_argument("--crease-angle", type=float, default=60.0,
                   help="Lock vertices on edges sharper than this out of smoothing (0 = smooth all)")
    p.add_argument("--min-fragment-ratio", type=float, default=0.005,
                   help="Drop shells whose bbox diagonal is below this fraction of the largest")
    p.add_argument("--hollow", type=float, default=0.0,
                   help="Hollow to this wall thickness (model units, v2 only; slow)")
    args = p.parse_args()

    if not os.path.exists(args.input):
        sys.exit(f"Input not found: {args.input}")

    stem = os.path.splitext(os.path.basename(args.input))[0]
    outdir = args.outdir or os.path.join("model_output", f"compare_{stem}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(outdir, exist_ok=True)

    # Shared reference: the same largest-component mesh both pipelines start
    # from, captured BEFORE either of them touches it.
    print(f"Loading reference: {args.input}")
    reference = printable.significant_components(
        mesh_io.load_glb(args.input),
        min_ratio=args.min_object_ratio,
        max_objects=args.max_objects,
    )[0]
    print(f"Reference: {len(reference.vertices):,} vertices, {len(reference.faces):,} faces, "
          f"watertight={reference.is_watertight}")

    common = dict(
        target_faces=args.target_faces,
        overhang_angle=args.overhang_angle,
        solid_infill=args.solid_infill,
        multi_object=args.multi_object,
        max_objects=args.max_objects,
        min_object_ratio=args.min_object_ratio,
    )

    v2_kwargs = dict(
        taubin_steps=args.taubin_steps, taubin_lambda=args.taubin_lambda,
        taubin_mu=args.taubin_mu, crease_angle=args.crease_angle,
        hollow_thickness=args.hollow, min_fragment_ratio=args.min_fragment_ratio, **common,
    )
    wanted = {"all": ["v1", "v2", "v3"], "both": ["v1", "v2"]}.get(args.only, [args.only])

    results = {}
    logs = {}
    if "v1" in wanted:
        r, err, log, _s = _run("v1 (trimesh + voxel fill)", lambda: printable.run_make_printable(
            args.input, output_prefix=os.path.join(outdir, "v1"), **common), args.quiet)
        results["v1"], logs["v1"] = r, log
        if err:
            print(f"\nv1 FAILED: {err}")
            if args.quiet:
                print(log)

    if "v2" in wanted:
        r, err, log, _s = _run("v2 (PyMeshLab + Manifold3D)", lambda: printable_v2.run_make_printable_v2(
            args.input, output_prefix=os.path.join(outdir, "v2"), **v2_kwargs), args.quiet)
        results["v2"], logs["v2"] = r, log
        if err:
            print(f"\nv2 FAILED: {err}")
            if args.quiet:
                print(log)

    if "v3" in wanted:
        r, err, log, _s = _run("v3 (MeshLib repair + Manifold3D)", lambda: printable_v2.run_make_printable_v2(
            args.input, output_prefix=os.path.join(outdir, "v3"),
            repair_backend="meshlib", **v2_kwargs), args.quiet)
        results["v3"], logs["v3"] = r, log
        if err:
            print(f"\nv3 FAILED: {err}")
            if args.quiet:
                print(log)

    print("\nMeasuring all outputs against the same reference mesh...")
    metrics = {k: _measure(v, reference, args.overhang_angle) for k, v in results.items()}
    print_table(metrics)

    print(f"\nFiles written to: {outdir}")
    for k, m in metrics.items():
        if m is not None:
            print(f"  {k}: {m['stl_path']}  |  {m['glb_path']}")

    if args.json_path:
        with open(args.json_path, "w") as fh:
            json.dump({"input": args.input, "outdir": outdir, "metrics": metrics}, fh, indent=2)
        print(f"  metrics JSON: {args.json_path}")


if __name__ == "__main__":
    main()

"""Run any print-prep pipeline (v1/v2/v3) on a .glb from the command line.

The stock make_printable.py CLI only reaches v1 -- v2/v3 are selected by the
web server, not by a flag -- so testing what the website actually does meant
starting the server. This mirrors server/worker.py's _run_printable() exactly:
same entry points, same argument names, same defaults as server/config.py, so
a run here is the same computation a web job performs.

  python run_printprep.py mesh.glb --pipeline v3

Writes <prefix>.glb and <prefix>.stl next to model_output/ unless --output is
given, and prints the same diagnostics/fidelity report the web job logs.
"""

import argparse
import os
import sys
import time


def run(pipeline, *, glb_path, output_prefix, target_faces, overhang_angle,
        solid_infill, voxel_pitch):
    """Dispatch to the requested pipeline. Unlike the server, an ImportError is
    NOT swallowed into a v1 fallback: when you asked for v3 you want to know it
    ran v3, not silently get v1's result labelled as v3."""
    if pipeline in ("v2", "v3"):
        from trellis_core.printprep.printable_v2 import run_make_printable_v2

        return run_make_printable_v2(
            glb_path=glb_path,
            output_prefix=output_prefix,
            target_faces=target_faces,
            overhang_angle=overhang_angle,
            solid_infill=solid_infill,
            voxel_pitch=voxel_pitch,
            repair_backend="meshlib" if pipeline == "v3" else "pymeshlab",
        )

    from trellis_core.printprep.printable import run_make_printable

    return run_make_printable(
        glb_path=glb_path,
        output_prefix=output_prefix,
        target_faces=target_faces,
        overhang_angle=overhang_angle,
        solid_infill=solid_infill,
        voxel_pitch=voxel_pitch,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="Path to input .glb")
    p.add_argument("--pipeline", choices=("v1", "v2", "v3"), default="v3",
                   help="v1 trimesh+voxel fill | v2 PyMeshLab+Manifold3D | "
                        "v3 v2 with the MeshLib backend (default, same as the website)")
    p.add_argument("--output", default=None,
                   help="Output prefix without extension (default: "
                        "model_output/<stem>_<pipeline>_<timestamp>_printable)")
    p.add_argument("--target-faces", type=int, default=1000000)
    p.add_argument("--overhang-angle", type=float, default=45.0)
    p.add_argument("--solid-infill", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--voxel-pitch", type=float, default=None,
                   help="Default: bbox_diagonal / 1024")
    args = p.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found")
        sys.exit(1)

    prefix = args.output
    if prefix is None:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_output")
        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(args.input))[0]
        prefix = os.path.join(
            out_dir, f"{stem}_{args.pipeline}_{time.strftime('%Y%m%d_%H%M%S')}_printable")

    print(f"Pipeline: {args.pipeline}  ->  {prefix}.glb / .stl")
    t0 = time.time()
    result = run(args.pipeline, glb_path=args.input, output_prefix=prefix,
                 target_faces=args.target_faces, overhang_angle=args.overhang_angle,
                 solid_infill=args.solid_infill, voxel_pitch=args.voxel_pitch)
    print(f"\nDone in {time.time() - t0:.1f}s")
    for note in getattr(result, "notes", []) or []:
        print(f"  note: {note}")


if __name__ == "__main__":
    main()

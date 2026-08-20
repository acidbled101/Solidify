"""
Make a .glb mesh more 3D-print-friendly:
  1. Repair to watertight/manifold geometry (best-effort; falls back to a
     voxel remesh that guarantees watertightness, at the cost of some
     appearance/topology fidelity).
  2. Report (not auto-fix) minimum-wall-thickness and overhang diagnostics,
     since those two rules can't be safely auto-corrected without risking
     visible shape distortion.
  3. Fill the object solid instead of leaving it a hollow shell, by voxelizing
     and flood-filling from the exterior -- eliminates any internal cavities/air
     pockets. On by default; disable with --no-solid-infill.
  4. (Optional, --multi-object / --explode) Handle models made of several
     disconnected objects: --multi-object repairs and keeps every object in one
     combined file; --explode writes each object to its own file for printing
     one at a time. Splitting keeps only large objects -- floating-dot noise
     below --min-object-ratio of the biggest is removed and the count is capped
     at --max-objects. Floating-dot noise is cleaned in every mode (the default
     single path keeps the largest object).

Objects keep their original orientation (no auto-reorientation).

This is a thin CLI wrapper: the print-prep pipeline lives in
trellis_core/printable.py so it can be reused by the web server. run_make_printable
keeps every progress/diagnostics/fidelity print, so CLI stdout is unchanged.
"""

import argparse
import os
import sys
import time

import trellis_core.printable as printable


def main():
    parser = argparse.ArgumentParser(description="Make a .glb mesh more 3D-print-friendly")
    parser.add_argument("input", help="Path to input .glb")
    parser.add_argument("--output", default=None,
                         help="Output path prefix, without extension (default: "
                              "model_output/<input_stem>_<timestamp>_printable). "
                              "Both .glb and .stl are written using this prefix. "
                              "With --explode, each object is written as <prefix>_partNN.")
    parser.add_argument("--output-dir", default=None,
                         help="Directory for default output naming (default: model_output/ "
                              "next to this script). Ignored if --output is given.")
    parser.add_argument("--voxel-pitch", type=float, default=None,
                         help="Voxel pitch for watertight fallback (default: bbox_diagonal / 1024)")
    parser.add_argument("--target-faces", type=int, default=1000000,
                         help="Re-simplify to this many faces (per object) if voxel remeshing inflates "
                              "face count. Applied to each object independently when splitting "
                              "(--multi-object/--explode). (default: 1000000)")
    parser.add_argument("--overhang-angle", type=float, default=45.0,
                         help="Overhang rule angle in degrees, from vertical (default: 45.0)")
    parser.add_argument("--solid-infill", action=argparse.BooleanOptionalAction, default=True,
                         help="Fill objects solid instead of leaving them hollow shells, by voxelizing "
                              "and flood-filling from the exterior (eliminates internal cavities/air "
                              "pockets). ON BY DEFAULT -- pass --no-solid-infill to keep best-effort "
                              "repair only. The solid path always voxel-remeshes (even already-watertight "
                              "meshes), discarding original topology/UVs and baking vertex colors.")
    parser.add_argument("--multi-object", action="store_true",
                         help="Treat the input as multiple disconnected objects: repair and keep every "
                              "object (instead of merging/dropping all but the largest) in one combined "
                              "output file.")
    parser.add_argument("--explode", action="store_true",
                         help="Split disconnected objects apart and write each one to its own "
                              "<prefix>_partNN.glb/.stl, auto-oriented independently. Implies --multi-object.")
    parser.add_argument("--max-objects", type=int, default=10,
                         help="Hard cap on the number of separate objects kept when splitting; the "
                              "smaller overflow is treated as noise and dropped (default: 10).")
    parser.add_argument("--min-object-ratio", type=float, default=0.05,
                         help="Objects whose bounding-box diagonal is under this fraction of the largest "
                              "object's are removed as floating-dot noise. Applied in every mode, so stray "
                              "specks get cleaned even without --explode (default: 0.05).")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found")
        sys.exit(1)

    if args.output:
        output_prefix = args.output
    else:
        output_dir = args.output_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_output")
        os.makedirs(output_dir, exist_ok=True)
        input_stem = os.path.splitext(os.path.basename(args.input))[0]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_prefix = os.path.join(output_dir, f"{input_stem}_{timestamp}_printable")

    printable.run_make_printable(
        args.input,
        output_prefix=output_prefix,
        target_faces=args.target_faces,
        overhang_angle=args.overhang_angle,
        solid_infill=args.solid_infill,
        voxel_pitch=args.voxel_pitch,
        multi_object=args.multi_object,
        explode=args.explode,
        max_objects=args.max_objects,
        min_object_ratio=args.min_object_ratio,
    )


if __name__ == "__main__":
    main()

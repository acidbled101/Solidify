"""
Simplify (decimate) an existing .glb mesh, preserving texture/vertex-color
data where present. Uses the same fast_simplification method as generate.py,
but as a standalone tool that operates on an already-exported .glb.
"""

import argparse
import os
import sys
import time

import numpy as np

from trellis_core import mesh_io


def main():
    parser = argparse.ArgumentParser(description="Simplify a .glb mesh, preserving texture/vertex colors")
    parser.add_argument("input", help="Path to input .glb")
    parser.add_argument("--target-faces", type=int, default=200000, help="Target face count (default: 200000)")
    parser.add_argument("--output", default=None, help="Output .glb path (default: <input>_simplified.glb)")
    parser.add_argument("--agg", type=float, default=7.0, help="fast_simplification aggressiveness, 0-10 (default: 7.0)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found")
        sys.exit(1)

    output_path = args.output or (os.path.splitext(args.input)[0] + "_simplified.glb")

    print(f"Loading: {args.input}")
    mesh = mesh_io.load_glb(args.input)
    verts = mesh.vertices
    faces = mesh.faces
    print(f"Mesh: {len(verts):,} vertices, {len(faces):,} faces")

    visual_info = mesh_io.extract_visual_info(mesh)
    print(f"Detected visual data: {visual_info['kind']}")

    if len(faces) <= args.target_faces:
        print(f"Faces already <= {args.target_faces:,}; skipping simplification.")
        new_verts, new_faces, indice_mapping = verts, faces, None
    else:
        print(f"Simplifying: {len(faces):,} -> ~{args.target_faces:,} faces (agg={args.agg})...")
        t0 = time.time()
        new_verts, new_faces, indice_mapping = mesh_io.simplify_with_attrs(
            verts, faces, args.target_faces, agg=args.agg,
        )
        print(f"Simplified in {time.time() - t0:.1f}s: {len(faces):,} -> {len(new_faces):,} faces, "
              f"{len(verts):,} -> {len(new_verts):,} vertices")

    new_uv = None
    new_vertex_colors = None
    if indice_mapping is not None:
        if visual_info["kind"] == "texture":
            new_uv = mesh_io.resample_per_vertex(visual_info["uv"], indice_mapping, len(new_verts))
        elif visual_info["kind"] == "vertex_color":
            new_vertex_colors = mesh_io.resample_per_vertex(
                visual_info["vertex_colors"], indice_mapping, len(new_verts),
            )
    else:
        new_uv = visual_info["uv"]
        new_vertex_colors = visual_info["vertex_colors"]

    print(f"Exporting: {output_path}")
    mesh_io.export_mesh(new_verts, new_faces, visual_info, output_path, new_uv=new_uv, new_vertex_colors=new_vertex_colors)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()

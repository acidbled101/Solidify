"""
Make a .glb mesh more 3D-print-friendly:
  1. Repair to watertight/manifold geometry (best-effort; falls back to a
     voxel remesh that guarantees watertightness, at the cost of some
     appearance/topology fidelity).
  2. Auto-orient the mesh onto its best resting pose: flat base contact,
     low center of gravity, minimized overhang area (45-degree rule).
  3. Report (not auto-fix) minimum-wall-thickness and overhang diagnostics,
     since those two rules can't be safely auto-corrected without risking
     visible shape distortion.
  4. (Optional, --solid-infill) Fill the object solid instead of leaving it
     a hollow shell, by voxelizing and flood-filling from the exterior --
     eliminates any internal cavities/air pockets.
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import trimesh
from scipy.spatial import ConvexHull, Delaunay

import mesh_io


def repair_watertight(mesh, voxel_pitch, force_solid=False):
    """Best-effort repair; falls back to a guaranteed-watertight voxel remesh.

    If force_solid is True, always takes the voxel remesh path (even if the
    mesh is already watertight), since voxel fill() floods from the exterior
    and so also eliminates any internal cavities/hollow chambers.

    Returns (repaired_mesh, used_voxel_fallback: bool).
    """
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    trimesh.repair.fill_holes(mesh)
    trimesh.repair.fix_normals(mesh)

    if mesh.is_watertight and not force_solid:
        return mesh, False

    if force_solid:
        print("  --solid-infill requested; voxelizing to eliminate internal cavities...")
    else:
        print("  Best-effort repair did not achieve watertightness; falling back to voxel remesh...")
    print("  Warning: original topology and any UV/texture atlas will be discarded; "
          "appearance will be resampled as vertex colors (best-effort).")
    # Solidify with method='orthographic', NOT the default 'holes'. The
    # 'holes' method (scipy binary_fill_holes) only fills fully-enclosed
    # voids -- any tiny gap in the voxelized shell lets the flood fill leak,
    # so the interior stays hollow and you get a thin shell (floating islands
    # in the slicer). 'orthographic' fills every voxel that sits between the
    # first and last surface voxel along all three axes, robustly filling the
    # interior into one solid mass even when the shell has small leaks.
    vox = trimesh.voxel.creation.voxelize(mesh, pitch=voxel_pitch).fill(method="orthographic")
    remeshed = vox.marching_cubes
    # vox.marching_cubes is in raw grid-index space; vox.transform maps it
    # back to the original mesh's world coordinates (scale + origin).
    remeshed.apply_transform(vox.transform)

    # Marching cubes can still shed a few disconnected single-voxel specks
    # (they show up as floating fragments in a slicer). Keep only the largest
    # connected body so the result is a single printable blob.
    bodies = remeshed.split(only_watertight=False)
    if len(bodies) > 1:
        largest = max(bodies, key=lambda b: b.volume)
        print(f"  Dropping {len(bodies) - 1} floating fragment(s); keeping largest solid body.")
        remeshed = largest
    return remeshed, True


def bake_colors_from_original(original_mesh, visual_info, new_vertices):
    """Sample per-vertex colors on new_vertices from the original mesh's
    appearance (vertex colors or texture), via closest-point + barycentric
    interpolation. Returns an (N,4) uint8 array, or None if no color data."""
    if visual_info["kind"] == "none":
        return None

    closest_pts, _dist, tri_id = trimesh.proximity.closest_point(original_mesh, new_vertices)
    tri_verts_idx = original_mesh.faces[tri_id]
    bary = trimesh.triangles.points_to_barycentric(
        original_mesh.triangles[tri_id], closest_pts,
    )

    if visual_info["kind"] == "vertex_color":
        corner_colors = visual_info["vertex_colors"][tri_verts_idx].astype(np.float64)  # (n,3,4)
        colors = np.einsum("ij,ijk->ik", bary, corner_colors)
        return np.clip(colors, 0, 255).astype(np.uint8)

    if visual_info["kind"] == "texture":
        corner_uv = visual_info["uv"][tri_verts_idx]  # (n,3,2)
        interp_uv = np.einsum("ij,ijk->ik", bary, corner_uv)
        material = original_mesh.visual.material
        return material.to_color(interp_uv)

    return None


def candidate_hull_normals(mesh):
    hull = mesh.convex_hull
    if len(hull.facets) > 0:
        normals = hull.facets_normal
    else:
        # Smooth/curved hull (no coplanar triangle groups) -- fall back to
        # per-triangle hull normals, deduped by rounding.
        normals = hull.face_normals
    rounded = np.round(normals, decimals=3)
    _, unique_idx = np.unique(rounded, axis=0, return_index=True)
    return normals[unique_idx]


def score_orientation(mesh, normal, overhang_angle):
    R = trimesh.geometry.align_vectors(normal, [0, 0, -1])
    m = mesh.copy()
    m.apply_transform(R)
    m.apply_translation([0, 0, -m.bounds[0][2]])

    diag = np.linalg.norm(m.extents)
    eps = max(diag * 1e-4, 1e-6)
    base_mask = m.vertices[:, 2] < eps
    base_xy = m.vertices[base_mask][:, :2]

    if len(base_xy) < 3:
        base_area, stable = 0.0, False
    else:
        try:
            hull2d = ConvexHull(base_xy)
            base_area = hull2d.volume
            tri = Delaunay(base_xy)
            stable = tri.find_simplex(m.center_mass[:2]) >= 0
        except Exception:
            base_area, stable = 0.0, False

    normal_z = m.face_normals[:, 2]
    overhang_mask = normal_z < -math.cos(math.radians(overhang_angle))
    overhang_area = m.area_faces[overhang_mask].sum()
    cog_height = m.center_mass[2]

    if not stable:
        score = -np.inf
    else:
        score = (base_area * 10.0) - (overhang_area * 1.0) - (cog_height * 0.5)

    return score, R, base_area, overhang_area, cog_height


def auto_orient(mesh, overhang_angle):
    best_score = -np.inf
    best_R = np.eye(4)
    best_stats = (0.0, 0.0, 0.0)
    for normal in candidate_hull_normals(mesh):
        score, R, base_area, overhang_area, cog_height = score_orientation(mesh, normal, overhang_angle)
        if score > best_score:
            best_score, best_R, best_stats = score, R, (base_area, overhang_area, cog_height)

    mesh.apply_transform(best_R)
    mesh.apply_translation([0, 0, -mesh.bounds[0][2]])
    return mesh, best_stats


def print_diagnostics(mesh, overhang_angle, n_ray_samples=500):
    total_area = mesh.area_faces.sum()
    normal_z = mesh.face_normals[:, 2]
    overhang_mask = normal_z < -math.cos(math.radians(overhang_angle))
    overhang_pct = 100.0 * mesh.area_faces[overhang_mask].sum() / max(total_area, 1e-9)

    n_faces = len(mesh.faces)
    sample_idx = np.random.choice(n_faces, size=min(n_ray_samples, n_faces), replace=False)
    origins = mesh.triangles_center[sample_idx] - mesh.face_normals[sample_idx] * 1e-4
    directions = -mesh.face_normals[sample_idx]

    thin_count = 0
    min_wall_mm = 1.0  # ~2.5x a typical 0.4mm nozzle
    try:
        locations, index_ray, _index_tri = mesh.ray.intersects_location(origins, directions)
        if len(locations) > 0:
            dists = np.linalg.norm(locations - origins[index_ray], axis=1)
            first_hit = {}
            for d, ray_i in zip(dists, index_ray):
                if ray_i not in first_hit or d < first_hit[ray_i]:
                    first_hit[ray_i] = d
            thin_count = sum(1 for d in first_hit.values() if d < min_wall_mm)
    except Exception as e:
        print(f"  (thin-wall ray sampling skipped: {e})")

    print("\nPrintability report (heuristic, not exhaustive):")
    print(f"  Overhang area (> {overhang_angle:.0f} deg from vertical): {overhang_pct:.1f}% of surface")
    print(f"  Thin-wall warnings (< {min_wall_mm}mm, sampled {len(sample_idx)} rays): {thin_count}")


def main():
    parser = argparse.ArgumentParser(description="Make a .glb mesh more 3D-print-friendly")
    parser.add_argument("input", help="Path to input .glb")
    parser.add_argument("--output", default=None,
                         help="Output path prefix, without extension (default: "
                              "model_output/<input_stem>_<timestamp>_printable). "
                              "Both .glb and .stl are written using this prefix.")
    parser.add_argument("--output-dir", default=None,
                         help="Directory for default output naming (default: model_output/ "
                              "next to this script). Ignored if --output is given.")
    parser.add_argument("--voxel-pitch", type=float, default=None,
                         help="Voxel pitch for watertight fallback (default: bbox_diagonal / 256)")
    parser.add_argument("--target-faces", type=int, default=200000,
                         help="Re-simplify to this many faces if voxel remeshing inflates face count (default: 200000)")
    parser.add_argument("--overhang-angle", type=float, default=45.0,
                         help="Overhang rule angle in degrees, from vertical (default: 45.0)")
    parser.add_argument("--solid-infill", action="store_true",
                         help="Fill the object solid instead of leaving it a hollow shell -- use this "
                              "if the mesh has an empty/hollow interior and you want it fully filled "
                              "with geometry for printing. Eliminates internal cavities/air pockets by "
                              "forcing a voxel remesh even if the mesh is already watertight; discards "
                              "original topology/UVs like the voxel fallback does.")
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

    output_glb_path = output_prefix + ".glb"
    output_stl_path = output_prefix + ".stl"

    print(f"Loading: {args.input}")
    mesh = mesh_io.load_glb(args.input)
    print(f"Mesh: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces, watertight={mesh.is_watertight}")

    visual_info = mesh_io.extract_visual_info(mesh)
    original_mesh = mesh.copy()

    voxel_pitch = args.voxel_pitch or (np.linalg.norm(mesh.extents) / 256.0)

    print("\nRepairing geometry...")
    t0 = time.time()
    mesh, used_voxel_fallback = repair_watertight(mesh, voxel_pitch, force_solid=args.solid_infill)
    print(f"Repaired in {time.time() - t0:.1f}s: {len(mesh.vertices):,} vertices, "
          f"{len(mesh.faces):,} faces, watertight={mesh.is_watertight}")

    new_vertex_colors = None
    if used_voxel_fallback:
        print("  Baking colors from original mesh onto new topology...")
        new_vertex_colors = bake_colors_from_original(original_mesh, visual_info, mesh.vertices)
        visual_info = {"kind": "vertex_color" if new_vertex_colors is not None else "none",
                        "uv": None, "base_color_img": None, "mr_img": None,
                        "vertex_colors": new_vertex_colors}

        if len(mesh.faces) > args.target_faces:
            print(f"  Voxel remesh inflated faces to {len(mesh.faces):,}; re-simplifying to ~{args.target_faces:,}...")
            new_verts, new_faces, indice_mapping = mesh_io.simplify_with_attrs(
                mesh.vertices, mesh.faces, args.target_faces,
            )
            if indice_mapping is not None and new_vertex_colors is not None:
                new_vertex_colors = mesh_io.resample_per_vertex(new_vertex_colors, indice_mapping, len(new_verts))
            mesh = trimesh.Trimesh(vertices=new_verts, faces=new_faces, process=False)

    print("\nAuto-orienting for best resting pose...")
    t0 = time.time()
    mesh, (base_area, overhang_area, cog_height) = auto_orient(mesh, args.overhang_angle)
    print(f"Oriented in {time.time() - t0:.1f}s: base area={base_area:.4f}, "
          f"overhang area={overhang_area:.4f}, CoG height={cog_height:.4f}")

    print_diagnostics(mesh, args.overhang_angle)

    print(f"\nExporting: {output_glb_path}")
    mesh_io.export_mesh(
        mesh.vertices, mesh.faces, visual_info, output_glb_path,
        new_uv=visual_info.get("uv"), new_vertex_colors=new_vertex_colors or visual_info.get("vertex_colors"),
    )
    print(f"Saved: {output_glb_path}")

    print(f"\nConverting to STL: {output_stl_path}")
    trimesh.load(output_glb_path, force="mesh", process=False).export(output_stl_path)
    print(f"Saved: {output_stl_path}")


if __name__ == "__main__":
    main()

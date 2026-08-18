"""
Print-prep logic behind make_printable.py (CLI) and server/worker.py (web).

Turns a generated .glb into watertight, optionally solid-filled, print-ready
geometry, and reports overhang/thin-wall diagnostics plus a fidelity score.

Torch-free on purpose: this only needs trimesh/numpy, so it stays fast to run
standalone without loading the whole TRELLIS stack. The diagnostics()/fidelity()
functions RETURN structured dicts (a server needs data); make_printable.py's CLI
output is preserved verbatim by run_make_printable printing those dicts through
the _print_*_text helpers at the exact points print_diagnostics /
print_fidelity_report used to print.
"""

import math
import time
from dataclasses import dataclass
from typing import Optional

from .. import bootstrap  # noqa: F401  (ensures env/path setup already ran)

import numpy as np
import trimesh

from .. import mesh_io


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
    # Use voxelize_subdivide with max_iter=None: the default voxelize() caps
    # subdivision at max_iter=10, which raises "max_iter exceeded!" whenever the
    # pitch is fine enough relative to the mesh's triangles to need more than 10
    # subdivision passes. max_iter=None auto-computes the exact number of passes
    # needed from the longest edge, so it never over- or under-shoots.
    vox = trimesh.voxel.creation.voxelize_subdivide(
        mesh, pitch=voxel_pitch, max_iter=None,
    ).fill(method="orthographic")
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


def diagnostics(mesh, overhang_angle, n_ray_samples=500):
    """Compute overhang + thin-wall printability diagnostics.

    Same computation as make_printable.py's former print_diagnostics(), but
    RETURNS the numbers instead of printing them:
      {"overhang_pct", "overhang_angle_deg", "thin_wall_warnings",
       "thin_wall_threshold_mm", "rays_sampled"}
    (The mid-computation "ray sampling skipped" notice is still printed as a
    side effect, exactly as before.)
    """
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

    return {
        "overhang_pct": overhang_pct,
        "overhang_angle_deg": overhang_angle,
        "thin_wall_warnings": thin_count,
        "thin_wall_threshold_mm": min_wall_mm,
        "rays_sampled": len(sample_idx),
    }


def fidelity(original_mesh, processed_mesh, n_samples=20000):
    """
    Quantify how much shape was lost/altered by post-processing (hole-filling,
    voxel remesh, decimation), by comparing the pre-repair mesh against the
    final processed mesh. Both share the same coordinate frame (processing does
    not move the object), so a point-to-surface comparison is meaningful.

    Uses symmetric point-to-surface distance (Chamfer + Hausdorff) rather than
    a vertex-to-vertex comparison, since repair/remeshing/decimation changes
    vertex count and topology -- this only requires both meshes occupy the
    same space, not matching structure.

    Returns a dict of metrics, or None if either mesh has no usable surface.
    """
    if len(original_mesh.faces) == 0 or len(processed_mesh.faces) == 0:
        return None

    diag = np.linalg.norm(original_mesh.extents)
    if diag < 1e-9:
        return None

    pts_a = trimesh.sample.sample_surface(original_mesh, n_samples)[0]
    pts_b = trimesh.sample.sample_surface(processed_mesh, n_samples)[0]

    _, dist_a_to_b, _ = trimesh.proximity.closest_point(processed_mesh, pts_a)
    _, dist_b_to_a, _ = trimesh.proximity.closest_point(original_mesh, pts_b)

    chamfer = float((dist_a_to_b.mean() + dist_b_to_a.mean()) / 2)
    hausdorff = float(max(dist_a_to_b.max(), dist_b_to_a.max()))

    vol_change_pct = None
    if original_mesh.is_watertight and processed_mesh.is_watertight:
        vol_orig = abs(original_mesh.volume)
        if vol_orig > 1e-12:
            vol_change_pct = 100.0 * (abs(processed_mesh.volume) - vol_orig) / vol_orig

    return {
        "chamfer": chamfer,
        "chamfer_pct": 100.0 * chamfer / diag,
        "hausdorff": hausdorff,
        "hausdorff_pct": 100.0 * hausdorff / diag,
        "vol_change_pct": vol_change_pct,
        "face_ratio_pct": 100.0 * len(processed_mesh.faces) / len(original_mesh.faces),
    }


def _print_diagnostics_text(d):
    """Print a diagnostics() dict in the exact format print_diagnostics used."""
    print("\nPrintability report (heuristic, not exhaustive):")
    print(f"  Overhang area (> {d['overhang_angle_deg']:.0f} deg from vertical): {d['overhang_pct']:.1f}% of surface")
    print(f"  Thin-wall warnings (< {d['thin_wall_threshold_mm']}mm, sampled {d['rays_sampled']} rays): {d['thin_wall_warnings']}")


def _print_fidelity_text(metrics):
    """Print a fidelity() dict in the exact format print_fidelity_report used."""
    if metrics is None:
        print("\nFidelity report: skipped (empty geometry).")
        return
    print("\nFidelity report (shape deviation from post-processing, vs. original):")
    print(f"  Chamfer distance (avg deviation):   {metrics['chamfer']:.5f}  ({metrics['chamfer_pct']:.2f}% of model size)")
    print(f"  Hausdorff distance (worst deviation): {metrics['hausdorff']:.5f}  ({metrics['hausdorff_pct']:.2f}% of model size)")
    if metrics["vol_change_pct"] is not None:
        print(f"  Volume change:                      {metrics['vol_change_pct']:+.2f}%")
    else:
        print("  Volume change:                      N/A (original mesh wasn't watertight pre-repair)")
    print(f"  Face count vs. original:             {metrics['face_ratio_pct']:.1f}%")


@dataclass
class PrintableOptions:
    target_faces: int = 1000000
    solid_infill: bool = True
    voxel_pitch: Optional[float] = None


def process_object(mesh, opts):
    """Run the full print-prep pipeline on a single object (mesh).

    Repairs to watertight (optionally solid-filled) and bakes colors /
    simplifies if the voxel path was taken. The object keeps its original
    orientation. Returns (processed_mesh, visual_info, new_vertex_colors).

    Fidelity is no longer measured here (it's computed by the caller against a
    pristine pre-repair copy, so the value can also be returned as data);
    everything else matches the former argparse-based process_object.
    """
    visual_info = mesh_io.extract_visual_info(mesh)
    original_mesh = mesh.copy()

    # Per-object voxel pitch from this body's own extents, so small parts get
    # a fine enough grid. opts.voxel_pitch overrides for every object.
    voxel_pitch = opts.voxel_pitch or (np.linalg.norm(mesh.extents) / 1024.0)

    print("\nRepairing geometry...")
    t0 = time.time()
    mesh, used_voxel_fallback = repair_watertight(mesh, voxel_pitch, force_solid=opts.solid_infill)
    print(f"Repaired in {time.time() - t0:.1f}s: {len(mesh.vertices):,} vertices, "
          f"{len(mesh.faces):,} faces, watertight={mesh.is_watertight}")

    new_vertex_colors = None
    if used_voxel_fallback:
        print("  Baking colors from original mesh onto new topology...")
        new_vertex_colors = bake_colors_from_original(original_mesh, visual_info, mesh.vertices)
        visual_info = {"kind": "vertex_color" if new_vertex_colors is not None else "none",
                        "uv": None, "base_color_img": None, "mr_img": None,
                        "vertex_colors": new_vertex_colors}

        if len(mesh.faces) > opts.target_faces:
            print(f"  Voxel remesh inflated faces to {len(mesh.faces):,}; re-simplifying to ~{opts.target_faces:,}...")
            new_verts, new_faces, indice_mapping = mesh_io.simplify_with_attrs(
                mesh.vertices, mesh.faces, opts.target_faces,
            )
            if indice_mapping is not None and new_vertex_colors is not None:
                new_vertex_colors = mesh_io.resample_per_vertex(new_vertex_colors, indice_mapping, len(new_verts))
            mesh = trimesh.Trimesh(vertices=new_verts, faces=new_faces, process=False)

    return mesh, visual_info, new_vertex_colors


def export_glb_and_stl(mesh, visual_info, new_vertex_colors, prefix):
    """Export a processed mesh to <prefix>.glb (with appearance) and <prefix>.stl.

    Returns (glb_path, stl_path).
    """
    glb_path = prefix + ".glb"
    stl_path = prefix + ".stl"

    print(f"\nExporting: {glb_path}")
    mesh_io.export_mesh(
        mesh.vertices, mesh.faces, visual_info, glb_path,
        new_uv=visual_info.get("uv"), new_vertex_colors=new_vertex_colors or visual_info.get("vertex_colors"),
    )
    print(f"Saved: {glb_path}")

    print(f"Converting to STL: {stl_path}")
    trimesh.load(glb_path, force="mesh", process=False).export(stl_path)
    print(f"Saved: {stl_path}")

    return glb_path, stl_path


def significant_components(mesh, min_ratio=0.05, max_objects=10):
    """Split a mesh into its disconnected objects and keep only the large ones.

    Size is measured per component by bounding-box diagonal. Components smaller
    than min_ratio * (largest component's size) are dropped as floating-dot
    noise, and at most max_objects (the largest) are kept. This prevents a
    voxel-remeshed mesh -- which can carry hundreds of tiny specks -- from
    exploding into hundreds of junk objects.

    Returns the kept components sorted largest-first, or [mesh] if the mesh
    isn't meaningfully splittable.
    """
    comps = mesh.split(only_watertight=False)
    if len(comps) <= 1:
        return [mesh]

    sizes = np.array([np.linalg.norm(c.extents) for c in comps])
    max_size = sizes.max()
    if max_size < 1e-12:
        return [mesh]

    order = np.argsort(sizes)[::-1]
    kept = [comps[i] for i in order if sizes[i] >= min_ratio * max_size]
    n_noise = len(comps) - len(kept)

    n_over_cap = max(0, len(kept) - max_objects)
    kept = kept[:max_objects]

    summary = (f"  Split: {len(comps)} raw components -> kept {len(kept)} large object(s) "
               f"(removed {n_noise} noise/small < {min_ratio:.0%} of largest")
    summary += f", {n_over_cap} over the {max_objects}-object cap)" if n_over_cap else ")"
    print(summary)

    # kept is never empty (the largest always clears the threshold). A single
    # survivor means one real object surrounded by noise -- return the cleaned
    # component, not the original noisy mesh.
    return kept


def split_objects(mesh, min_ratio=0.05, max_objects=10):
    """Thin wrapper: the large disconnected objects to process separately."""
    return significant_components(mesh, min_ratio=min_ratio, max_objects=max_objects)


@dataclass
class PrintableResult:
    """Outcome of run_make_printable.

    Several fields carry a per-mode shape (documented below) rather than one
    fixed type, matching make_printable.py's differing per-mode CLI call sites:

    - diagnostics:
        "single"       -> dict          (the one processed object)
        "multi_object" -> dict          (the combined mesh)
        "explode"      -> list[dict]    (one per object)
    - fidelity:
        "single"       -> Optional[dict]         (the one object; None if empty)
        "multi_object" -> Optional[dict]         (the primary/largest object)
        "explode"      -> list[Optional[dict]]   (one per object)
    - watertight:
        "single"       -> bool          (the processed object)
        "multi_object" -> bool          (the combined mesh)
        "explode"      -> list[bool]    (one per object)

    files: [{"kind": "printable_glb"|"printable_stl", "path": ...}, ...]; for
    "explode" each dict also carries "part": int.

    glb_path / stl_path are convenience shortcuts set only in "single" mode
    (the only mode the server uses today).
    """
    mode: str
    files: list
    diagnostics: object
    fidelity: object
    watertight: object
    seconds: float
    glb_path: Optional[str] = None
    stl_path: Optional[str] = None


def run_make_printable(
    glb_path,
    *,
    output_prefix,
    target_faces: int = 1000000,
    overhang_angle: float = 45.0,
    solid_infill: bool = True,
    voxel_pitch: Optional[float] = None,
    multi_object: bool = False,
    explode: bool = False,
    max_objects: int = 10,
    min_object_ratio: float = 0.05,
) -> PrintableResult:
    """Reproduce make_printable.py main()'s body, parameterized instead of
    reading argparse args: load glb_path -> split/significant-components ->
    process_object per body -> fidelity -> diagnostics -> export.

    Writes files at <output_prefix>.glb/.stl (or <output_prefix>_partNN.glb/.stl
    per object for explode), matching the CLI's naming exactly. Does NOT create
    the output directory or compute a default output_prefix -- callers pass an
    explicit prefix. Keeps all of the CLI's progress prints (a server ignores
    stdout and reads the returned PrintableResult).
    """
    t_start = time.time()
    opts = PrintableOptions(target_faces=target_faces, solid_infill=solid_infill, voxel_pitch=voxel_pitch)

    print(f"Loading: {glb_path}")
    mesh = mesh_io.load_glb(glb_path)
    print(f"Mesh: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces, watertight={mesh.is_watertight}")

    # Find the large objects (noise/floating dots removed, count capped).
    if explode or multi_object:
        bodies = split_objects(mesh, min_ratio=min_object_ratio, max_objects=max_objects)
        print(f"Objects to process: {len(bodies)}")
    else:
        # Default (single-object) path still cleans floating-dot noise: keep the
        # largest object and drop the rest. Warn if several large objects exist.
        comps = significant_components(mesh, min_ratio=min_object_ratio, max_objects=max_objects)
        if len(comps) > 1:
            print(f"  Note: {len(comps)} large objects detected; keeping the largest only. "
                  f"Use --multi-object or --explode to keep them all.")
        bodies = [comps[0]]

    if explode:
        # One file per object, each kept in its original orientation.
        width = max(2, len(str(len(bodies))))
        files = []
        diags = []
        fids = []
        waters = []
        for i, body in enumerate(bodies, start=1):
            print(f"\n===== Object {i}/{len(bodies)} =====")
            original = body.copy()
            proc, visual_info, new_vertex_colors = process_object(body, opts)
            fid = fidelity(original, proc)
            _print_fidelity_text(fid)
            d = diagnostics(proc, overhang_angle)
            _print_diagnostics_text(d)
            part_prefix = f"{output_prefix}_part{i:0{width}d}"
            glb, stl = export_glb_and_stl(proc, visual_info, new_vertex_colors, part_prefix)
            files.append({"kind": "printable_glb", "path": glb, "part": i})
            files.append({"kind": "printable_stl", "path": stl, "part": i})
            diags.append(d)
            fids.append(fid)
            waters.append(bool(proc.is_watertight))
        print(f"\nDone: exploded {len(bodies)} objects into {output_prefix}_part* .glb/.stl")
        return PrintableResult(
            mode="explode", files=files, diagnostics=diags, fidelity=fids,
            watertight=waters, seconds=time.time() - t_start,
        )

    if multi_object:
        # Repair each object in place, then combine into one file.
        processed = []
        fids = []
        for i, body in enumerate(bodies, start=1):
            print(f"\n===== Object {i}/{len(bodies)} =====")
            original = body.copy()
            proc, _visual_info, _colors = process_object(body, opts)
            fid = fidelity(original, proc)
            _print_fidelity_text(fid)
            processed.append(proc)
            fids.append(fid)

        print("\nCombining objects...")
        combined = trimesh.util.concatenate(processed)

        d = diagnostics(combined, overhang_angle)
        _print_diagnostics_text(d)
        combined_visual = mesh_io.extract_visual_info(combined)
        glb, stl = export_glb_and_stl(combined, combined_visual, combined_visual.get("vertex_colors"), output_prefix)
        return PrintableResult(
            mode="multi_object",
            files=[{"kind": "printable_glb", "path": glb}, {"kind": "printable_stl", "path": stl}],
            diagnostics=d,
            fidelity=fids[0] if fids else None,
            watertight=bool(combined.is_watertight),
            seconds=time.time() - t_start,
        )

    # Single-object (default) path -- bodies[0] is the largest object with
    # floating-dot noise already removed.
    body = bodies[0]
    original = body.copy()
    proc, visual_info, new_vertex_colors = process_object(body, opts)
    fid = fidelity(original, proc)
    _print_fidelity_text(fid)
    d = diagnostics(proc, overhang_angle)
    _print_diagnostics_text(d)
    glb, stl = export_glb_and_stl(proc, visual_info, new_vertex_colors, output_prefix)
    return PrintableResult(
        mode="single",
        files=[{"kind": "printable_glb", "path": glb}, {"kind": "printable_stl", "path": stl}],
        diagnostics=d,
        fidelity=fid,
        watertight=bool(proc.is_watertight),
        seconds=time.time() - t_start,
        glb_path=glb,
        stl_path=stl,
    )

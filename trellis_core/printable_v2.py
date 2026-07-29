"""
Experimental v2 print-prep pipeline (PyMeshLab + Manifold3D).

Same job as printable.py, different strategy. v1 guarantees watertightness by
voxelizing and flood-filling, which works but resamples the model onto a grid:
topology and UVs are destroyed and fine detail is lost. v2 keeps the original
triangles and instead does:

  1. Load + clean   (PyMeshLab)  -- zero-area faces, duplicate vertices/faces,
                                    unreferenced points.
  2. Denoise        (PyMeshLab)  -- Taubin (lambda/mu) smoothing, which is
                                    volume-preserving by construction; sharp
                                    edges are additionally locked out of the
                                    smoothing pass via a crease-angle mask.
  3. Refine/solidify (Manifold3D)-- build a guaranteed 2-manifold, boolean-union
                                    overlapping parts, drop enclosed air
                                    cavities, optionally hollow.
  4. Export         (Manifold3D -> trimesh) -- .stl for the slicer (+ .glb).

repair_backend="meshlib" swaps the repair strategy (not the stages): MeshLib's
ray-parity orientation fix runs right after cleaning, and its SDF rebuild
replaces v1's voxel remesh as the last resort. On real TRELLIS meshes that is
the configuration that wins -- see POSTPROC_V2.md.

The public entry point run_make_printable_v2() mirrors printable.run_make_printable
(same arguments, same PrintableResult shape) so the two can be swapped in the
server/CLI and compared directly -- see compare_printable.py.

Diagnostics/fidelity/component-splitting are deliberately imported from v1
rather than reimplemented: the comparison is only meaningful if both pipelines
are measured with the exact same ruler.

Requires the optional deps: pip install pymeshlab manifold3d
"""

import math
import time
from dataclasses import dataclass, field
from typing import Optional

from . import bootstrap  # noqa: F401  (ensures env/path setup already ran)

import numpy as np
import trimesh

import mesh_io

from . import meshlib_repair
from .printable import (
    PrintableResult,
    bake_colors_from_original,
    diagnostics,
    export_glb_and_stl,
    fidelity,
    repair_watertight,
    significant_components,
    split_objects,
    _print_diagnostics_text,
    _print_fidelity_text,
)


@dataclass
class PrintableV2Options:
    target_faces: int = 1000000
    # Taubin: lambda_ > 0 smooths, mu < -lambda_ pushes back out. The default
    # pair is MeshLab's and is the classic shrink-free (volume-preserving)
    # setting from Taubin '95 -- do not tune one without the other.
    taubin_lambda: float = 0.5
    taubin_mu: float = -0.53
    taubin_steps: int = 10
    # Vertices touching an edge whose dihedral angle exceeds this are excluded
    # from smoothing, so hard corners/creases stay crisp. 0 disables the lock
    # (smooth everything).
    crease_angle: float = 60.0
    # Max hole size (in edges) PyMeshLab is allowed to close during repair
    # escalation. Only used if the mesh fails to become a manifold.
    close_hole_size: int = 1000
    # Remove enclosed air cavities so the slicer sees one solid mass.
    solid_infill: bool = True
    # Positive shells whose bounding-box diagonal is below this fraction of the
    # largest body's are dropped as floating specks (extent, not volume -- see
    # fill_cavities).
    min_fragment_ratio: float = 0.005
    # > 0 hollows the solid to this wall thickness (model units) via a
    # Minkowski erosion. Expensive on dense meshes; off by default.
    hollow: float = 0.0
    # Bake appearance onto the new topology when the input had colors/texture.
    bake_colors: bool = True
    # "meshlib" adds MeshLib's ray-parity orientation repair before the manifold
    # stage and swaps the last-resort remesh for MeshLib's SDF rebuild.
    # "pymeshlab" keeps PyMeshLab repairs + v1's voxel remesh.
    repair_backend: str = "pymeshlab"
    # Voxel pitch for the last-resort v1-style remesh fallback.
    voxel_pitch: Optional[float] = None


@dataclass
class PrintableV2Result(PrintableResult):
    """PrintableResult + v2-only telemetry (per-stage seconds, repair notes)."""
    stages: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)


# --------------------------------------------------------------------------
# Stage 1 + 2: PyMeshLab (clean, denoise)
# --------------------------------------------------------------------------

def _new_meshset(vertices, faces, v_scalar=None):
    """Wrap raw arrays in a single-mesh PyMeshLab MeshSet."""
    import pymeshlab

    kwargs = {
        "vertex_matrix": np.ascontiguousarray(vertices, dtype=np.float64),
        "face_matrix": np.ascontiguousarray(faces, dtype=np.int32),
    }
    if v_scalar is not None:
        kwargs["v_scalar_array"] = np.ascontiguousarray(v_scalar, dtype=np.float64)

    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(**kwargs), "mesh")
    return ms


def _current_arrays(ms):
    m = ms.current_mesh()
    return np.asarray(m.vertex_matrix()), np.asarray(m.face_matrix())


def _apply_filters(ms, filters, notes):
    """Run (name, kwargs) filters in order, tolerating individual failures.

    A cleaning filter that errors on a pathological mesh must not kill the
    whole run -- the later Manifold3D stage is what actually guarantees the
    output, so a skipped clean is a quality issue, not a correctness one.
    """
    for name, kwargs in filters:
        try:
            getattr(ms, name)(**kwargs)
        except Exception as e:  # pymeshlab raises its own exception types
            msg = f"pymeshlab filter {name} skipped: {e}"
            print(f"  ({msg})")
            notes.append(msg)


def clean_mesh(vertices, faces, notes):
    """Stage 1: basic PyMeshLab cleaning.

    Removes zero-area (null) faces, duplicate faces, duplicate vertices and
    unreferenced (orphan) points. Returns (vertices, faces).
    """
    ms = _new_meshset(vertices, faces)
    n_v0, n_f0 = len(vertices), len(faces)

    _apply_filters(ms, [
        ("meshing_remove_null_faces", {}),
        ("meshing_remove_duplicate_faces", {}),
        ("meshing_remove_duplicate_vertices", {}),
        ("meshing_remove_unreferenced_vertices", {}),
    ], notes)

    v, f = _current_arrays(ms)
    print(f"  Cleaned: {n_v0:,} -> {len(v):,} vertices, {n_f0:,} -> {len(f):,} faces "
          f"(removed {n_v0 - len(v):,} verts, {n_f0 - len(f):,} faces)")
    return v, f


def _crease_locked_vertices(vertices, faces, crease_angle):
    """Boolean mask of vertices sitting on an edge sharper than crease_angle.

    face_adjacency_angles is the dihedral angle between neighbouring faces, so
    a large angle means a hard crease/corner that smoothing would round off.
    """
    tm = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    locked = np.zeros(len(vertices), dtype=bool)
    angles = tm.face_adjacency_angles
    if len(angles) == 0:
        return locked
    sharp_edges = tm.face_adjacency_edges[angles > math.radians(crease_angle)]
    if len(sharp_edges):
        locked[np.unique(sharp_edges.ravel())] = True
    return locked


def denoise_taubin(vertices, faces, opts, notes):
    """Stage 2: Taubin smoothing with volume preserved and creases locked.

    Taubin's alternating lambda/mu passes are shrink-free by construction (that
    is the whole point of the filter vs. plain Laplacian). Sharp edges are
    protected separately: crease vertices get scalar 0, everything else 1, and
    the smoothing runs on the `q > 0.5` selection only.
    """
    if opts.taubin_steps <= 0:
        print("  Taubin smoothing disabled (steps=0)")
        return vertices, faces

    locked = (_crease_locked_vertices(vertices, faces, opts.crease_angle)
              if opts.crease_angle > 0 else np.zeros(len(vertices), dtype=bool))
    n_locked = int(locked.sum())

    if n_locked == len(vertices):
        msg = ("every vertex is on a crease at "
               f"{opts.crease_angle:.0f} deg; skipping Taubin smoothing")
        print(f"  ({msg})")
        notes.append(msg)
        return vertices, faces

    ms = _new_meshset(vertices, faces, v_scalar=(~locked).astype(np.float64))
    selected = n_locked > 0
    if selected:
        _apply_filters(ms, [
            ("compute_selection_by_condition_per_vertex", {"condselect": "(q > 0.5)"}),
        ], notes)

    _apply_filters(ms, [
        ("apply_coord_taubin_smoothing", {
            "lambda_": opts.taubin_lambda,
            "mu": opts.taubin_mu,
            "stepsmoothnum": opts.taubin_steps,
            "selected": selected,
        }),
    ], notes)

    v, f = _current_arrays(ms)
    moved = np.linalg.norm(v - vertices, axis=1) if len(v) == len(vertices) else None
    detail = f", mean displacement {moved.mean():.5f}" if moved is not None else ""
    print(f"  Taubin: {opts.taubin_steps} steps (lambda={opts.taubin_lambda}, mu={opts.taubin_mu}), "
          f"{n_locked:,} crease vertices locked{detail}")
    return v, f


def close_boundary_loops(vertices, faces, notes, max_loop=64):
    """Fan-triangulate the small open boundary loops PyMeshLab refused to close.

    In practice a TRELLIS mesh comes out of the repair chain with a handful of
    3-5 vertex holes that meshing_close_holes leaves alone (it declines holes
    that touch non-manifold vertices), and those few holes are enough for
    Manifold3D to reject the whole mesh. Patching them is what keeps v2 on its
    original topology instead of dropping to the voxel fallback.

    Winding matters: a boundary half-edge a->b belongs to exactly one existing
    face, so the patch must traverse it as b->a. Fanning from the loop's first
    vertex in reverse boundary order gives that for free.
    """
    faces = np.asarray(faces)
    tm = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    _uniq, inverse, counts = np.unique(tm.edges_sorted, axis=0,
                                      return_inverse=True, return_counts=True)
    directed = tm.edges[counts[inverse.ravel()] == 1]
    if len(directed) == 0:
        return vertices, faces

    # Walk by consuming half-edges, not by marking vertices: two holes that meet
    # at a single pinch vertex give that vertex two outgoing boundary edges, and
    # a vertex-based walk would either merge the two holes or bail out. Consuming
    # edges splits them into the two cycles they actually are.
    successors = {}
    for a, b in directed.tolist():
        successors.setdefault(a, []).append(b)

    patches = []
    skipped = 0
    for start in list(successors):
        while successors.get(start):
            loop = [start]
            v = successors[start].pop()
            while v != start and successors.get(v):
                loop.append(v)
                v = successors[v].pop()
            if v != start or len(loop) < 3 or len(loop) > max_loop or len(set(loop)) != len(loop):
                skipped += 1
                continue
            apex = loop[0]
            patches.extend([apex, loop[i + 1], loop[i]] for i in range(1, len(loop) - 1))

    if not patches:
        return vertices, faces

    msg = f"patched {len(patches)} triangle(s) over small open boundary loops"
    if skipped:
        msg += f" ({skipped} loop(s) too large or non-manifold, left open)"
    print(f"  {msg}")
    notes.append(msg)
    return vertices, np.vstack([faces, np.array(patches, dtype=faces.dtype)])


def repair_for_manifold(vertices, faces, opts, notes):
    """Escalated PyMeshLab repair, used only when Manifold3D rejects a mesh."""
    ms = _new_meshset(vertices, faces)
    _apply_filters(ms, [
        ("meshing_repair_non_manifold_edges", {}),
        ("meshing_repair_non_manifold_vertices", {}),
        ("meshing_close_holes", {"maxholesize": opts.close_hole_size,
                                  "selfintersection": False}),
        ("meshing_remove_unreferenced_vertices", {}),
    ], notes)
    v, f = _current_arrays(ms)
    print(f"  Repaired for manifold: {len(v):,} vertices, {len(f):,} faces")
    return v, f


# --------------------------------------------------------------------------
# Stage 3: Manifold3D (refine + solidify)
# --------------------------------------------------------------------------

def _try_manifold(vertices, faces):
    """Build a manifold3d.Manifold from arrays. Returns (manifold_or_None, why).

    Manifold's constructor never raises for non-manifold input: it returns an
    empty Manifold and records the reason in status(), so both have to be
    checked.
    """
    import manifold3d

    if len(faces) == 0:
        return None, "empty mesh"

    mesh = manifold3d.Mesh(
        vert_properties=np.ascontiguousarray(vertices, dtype=np.float32),
        tri_verts=np.ascontiguousarray(faces, dtype=np.uint32),
    )
    # Stitches vertices that are coincident-but-unshared across open edges,
    # which is what most "almost manifold" exported meshes actually suffer from.
    mesh.merge()

    man = manifold3d.Manifold(mesh)
    status = man.status()
    if man.is_empty() or status != manifold3d.Error.NoError:
        return None, str(status)
    return man, None


def _ensure_positive(man, vertices, faces, notes):
    """Flip inward-facing shells so every object is a positive-volume solid.

    A shell that was an interior cavity in the source model is oriented inward
    and comes out with a negative volume. Booleans on such a manifold behave as
    operations on its complement, which is never what "merge these parts" means
    -- so reverse the winding and rebuild.
    """
    if man.volume() >= 0:
        return man, faces

    msg = "object was inward-facing (negative volume); reversing winding"
    print(f"  {msg}")
    notes.append(msg)
    flipped = np.ascontiguousarray(np.asarray(faces)[:, ::-1])
    fixed, _why = _try_manifold(vertices, flipped)
    return (fixed, flipped) if fixed is not None else (man, faces)


def _manifold_with_repair(vertices, faces, opts, notes):
    """Try to build a Manifold, escalating repairs that preserve the topology.

    Rungs: as-is -> PyMeshLab non-manifold repair + hole closing -> fan-patch of
    the small boundary loops that survived. Returns (manifold_or_None, v, f).
    """
    man, why = _try_manifold(vertices, faces)
    if man is not None:
        return man, vertices, faces

    msg = f"not a manifold ({why}); running PyMeshLab repair"
    print(f"  {msg}...")
    notes.append(msg)
    v, f = repair_for_manifold(vertices, faces, opts, notes)
    man, why = _try_manifold(v, f)
    if man is not None:
        return man, v, f

    v, f = close_boundary_loops(v, f, notes)
    man, why = _try_manifold(v, f)
    return man, v, f


def to_manifold(vertices, faces, opts, notes):
    """Get a valid Manifold, escalating through repair strategies.

    1. topology-preserving repairs (see _manifold_with_repair)
    2. v1's voxel remesh -- always watertight, but resamples the surface, so it
       is the last resort rather than the default the way it is in v1.

    Returns (manifold, vertices, faces, used_voxel_fallback).
    """
    man, v, f = _manifold_with_repair(vertices, faces, opts, notes)
    if man is not None:
        man, f = _ensure_positive(man, v, f, notes)
        return man, v, f, False

    tm = trimesh.Trimesh(vertices=v, faces=f, process=False)
    pitch = opts.voxel_pitch or (np.linalg.norm(tm.extents) / 1024.0)

    if opts.repair_backend == "meshlib":
        # MeshLib's SDF rebuild instead of trimesh's binary voxelization: the
        # distance field is sub-voxel accurate and comes back through dual
        # marching cubes, so it lands far closer to the original surface for a
        # fraction of the triangles. Orientation was already repaired above,
        # which is what makes the winding-number sign detection trustworthy.
        msg = "repairs did not produce a manifold; rebuilding through MeshLib's SDF"
        print(f"  {msg}...")
        notes.append(msg)
        v, f = meshlib_repair.rebuild(v, f, pitch, notes)
    else:
        msg = "repairs did not produce a manifold; falling back to v1 voxel remesh"
        print(f"  {msg}...")
        print("  Warning: this discards original topology/UVs, exactly like the v1 pipeline.")
        notes.append(msg)
        remeshed, _used = repair_watertight(tm, pitch, force_solid=True)
        v, f = np.asarray(remeshed.vertices), np.asarray(remeshed.faces)

    # The remesh runs after the pre-manifold decimation, so it can blow past the
    # face budget; re-apply it here. Decimation can break manifoldness (that is
    # exactly how v1 ends up with a non-watertight result), so a valid manifold
    # always wins over hitting the target face count.
    if len(f) > opts.target_faces:
        vd, fd = _decimate_if_needed(v, f, opts.target_faces)
        man, vd, fd = _manifold_with_repair(vd, fd, opts, notes)
        if man is not None:
            man, fd = _ensure_positive(man, vd, fd, notes)
            return man, vd, fd, True
        msg = (f"decimation to {opts.target_faces:,} faces broke manifoldness; "
               f"keeping the full {len(f):,}-face remesh")
        print(f"  {msg}")
        notes.append(msg)

    man, v, f = _manifold_with_repair(v, f, opts, notes)
    if man is None:
        raise RuntimeError("Manifold3D rejected even the remeshed geometry")
    man, f = _ensure_positive(man, v, f, notes)
    return man, v, f, True


def _extent_diagonal(man):
    """Bounding-box diagonal of a Manifold (bounding_box is min xyz + max xyz)."""
    box = man.bounding_box()
    return float(np.linalg.norm(np.array(box[3:]) - np.array(box[:3])))


def fill_cavities(man, notes, min_fragment_ratio=0.005):
    """Make the solid printable: fill enclosed voids and drop floating specks.

    In a valid manifold an enclosed cavity is a topologically separate,
    inward-oriented shell, so it shows up in decompose() with a NEGATIVE volume.
    Unioning only the positive shells therefore yields the same outer surface
    with the voids filled -- the exact-arithmetic equivalent of what v1 gets by
    voxel flood-filling, but without resampling the surface.

    Specks are judged by bounding-box diagonal, NOT by volume, the same way v1's
    significant_components() judges input components. A thin whisker has almost
    no volume but real spatial extent, and dropping it is losing part of the
    model: measured on a real mesh, a 1% volume threshold discarded whiskers and
    pushed Hausdorff error from 0.40% to 3.97%, while a 1% extent threshold kept
    them and cost nothing.
    """
    import manifold3d

    parts = man.decompose()
    if len(parts) <= 1:
        return man

    solids = [p for p in parts if p.volume() > 0]
    n_voids = len(parts) - len(solids)
    if not solids:
        return man

    largest = max(_extent_diagonal(p) for p in solids)
    kept = [p for p in solids if _extent_diagonal(p) >= min_fragment_ratio * largest]
    n_specks = len(solids) - len(kept)

    if n_voids == 0 and n_specks == 0:
        return man

    parts_msg = []
    if n_voids:
        parts_msg.append(f"filled {n_voids} enclosed cavity/cavities")
    if n_specks:
        parts_msg.append(f"dropped {n_specks} floating fragment(s) "
                         f"< {min_fragment_ratio:.2%} of the main body's size")
    msg = "; ".join(parts_msg)
    print(f"  {msg}")
    notes.append(msg)
    return manifold3d.Manifold.batch_boolean(kept, manifold3d.OpType.Add)


def union_parts(mans, notes):
    """Boolean-union several Manifolds into one solid (Manifold3D "merge parts")."""
    import manifold3d

    if len(mans) == 1:
        return mans[0]
    print(f"  Boolean-union of {len(mans)} parts...")
    u = manifold3d.Manifold.batch_boolean(mans, manifold3d.OpType.Add)
    if u.is_empty():
        msg = "boolean union produced an empty result; keeping the largest part"
        print(f"  Warning: {msg}")
        notes.append(msg)
        return max(mans, key=lambda m: m.volume())
    return u


def hollow(man, thickness, notes):
    """Hollow a solid to the given wall thickness via Minkowski erosion.

    inner = man (erode) sphere(thickness); result = man - inner. Cost scales
    with the product of face counts for non-convex inputs, so this is opt-in
    and warns loudly on dense meshes.
    """
    import manifold3d

    if thickness <= 0:
        return man
    if man.num_tri() > 200_000:
        msg = (f"hollowing a {man.num_tri():,}-triangle solid via Minkowski erosion "
               "is very slow; consider lowering --target-faces first")
        print(f"  Warning: {msg}")
        notes.append(msg)

    print(f"  Hollowing to {thickness} wall thickness (Minkowski erosion)...")
    tool = manifold3d.Manifold.sphere(thickness, circular_segments=16)
    inner = man.minkowski_difference(tool)
    if inner.is_empty():
        msg = f"wall thickness {thickness} erodes the whole solid; leaving it solid"
        print(f"  Warning: {msg}")
        notes.append(msg)
        return man
    return man - inner


def separate_coincident_vertices(mesh, notes=None):
    """Nudge apart distinct vertices that land on the same float32 position.

    Manifold3D can legitimately hold two separate vertices whose coordinates
    round to the same float32 value. Nothing is wrong in memory, but .stl stores
    bare float32 triangles, so every loader (including every slicer) re-welds
    those vertices on import and the model arrives with non-manifold edges. The
    fix is to make the collision impossible before writing: displace the extra
    vertices along their normal by ~1e-5 of the model size -- four orders of
    magnitude below a 0.4mm nozzle, and topology is untouched.
    """
    verts = np.asarray(mesh.vertices)
    _uniq, inverse, counts = np.unique(verts.astype(np.float32), axis=0,
                                      return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    if not (counts > 1).any():
        return mesh

    # Keep the first vertex of each colliding group where it is, move the rest.
    order = np.argsort(inverse, kind="stable")
    is_first = np.ones(len(order), dtype=bool)
    is_first[1:] = inverse[order][1:] != inverse[order][:-1]
    to_move = order[~is_first]

    directions = np.asarray(mesh.vertex_normals)[to_move]
    lengths = np.linalg.norm(directions, axis=1, keepdims=True)
    # A zero-length normal (isolated/degenerate vertex) gets an arbitrary axis.
    directions = np.where(lengths > 1e-12, directions / np.maximum(lengths, 1e-12),
                          np.array([0.0, 0.0, 1.0]))
    eps = max(float(np.linalg.norm(mesh.extents)) * 1e-5, 1e-9)

    verts = verts.copy()
    verts[to_move] += directions * eps
    msg = f"nudged {len(to_move)} vertex/vertices apart to survive the float32 STL round-trip"
    print(f"  {msg}")
    if notes is not None:
        notes.append(msg)
    return trimesh.Trimesh(vertices=verts, faces=np.asarray(mesh.faces), process=False)


def manifold_to_trimesh(man, notes=None):
    """Manifold -> trimesh.Trimesh (positions are the first 3 vertex properties)."""
    mesh = man.to_mesh()
    verts = np.asarray(mesh.vert_properties)[:, :3].astype(np.float64)
    faces = np.asarray(mesh.tri_verts).astype(np.int64)
    tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    return separate_coincident_vertices(tm, notes)


# --------------------------------------------------------------------------
# Per-object pipeline
# --------------------------------------------------------------------------

def _decimate_if_needed(vertices, faces, target_faces):
    """Decimate before the manifold stage, so Manifold3D still validates the result."""
    if len(faces) <= target_faces:
        return vertices, faces
    print(f"  Decimating {len(faces):,} -> ~{target_faces:,} faces...")
    v, f, _mapping = mesh_io.simplify_with_attrs(vertices, faces, target_faces)
    return np.asarray(v), np.asarray(f)


def process_object_v2(mesh, opts, stages, notes):
    """Run stages 1-3 on one object. Returns (processed_mesh, manifold, visual_info,
    new_vertex_colors)."""
    visual_info = mesh_io.extract_visual_info(mesh)
    original_mesh = mesh.copy()

    v = np.asarray(mesh.vertices)
    f = np.asarray(mesh.faces)

    print("\n[1/4] Cleaning (PyMeshLab)...")
    t0 = time.time()
    v, f = clean_mesh(v, f, notes)
    # Orientation first, before anything reads the winding. Every later stage --
    # Manifold3D, the SDF sign detection, even MeshLib's own importer -- assumes
    # coherent winding and quietly degrades without it.
    if opts.repair_backend == "meshlib":
        f, _n = meshlib_repair.fix_orientation(v, f, notes)
    stages["clean"] = stages.get("clean", 0.0) + time.time() - t0

    print("[2/4] Denoising (PyMeshLab Taubin)...")
    t0 = time.time()
    v, f = denoise_taubin(v, f, opts, notes)
    stages["denoise"] = stages.get("denoise", 0.0) + time.time() - t0

    print("[3/4] Solidifying (Manifold3D)...")
    t0 = time.time()
    v, f = _decimate_if_needed(v, f, opts.target_faces)
    man, v, f, used_voxel_fallback = to_manifold(v, f, opts, notes)

    if used_voxel_fallback and opts.taubin_steps > 0:
        # The remesh resamples onto a grid, so it arrives with marching-cubes
        # staircase artifacts that the earlier denoise pass never saw. Taubin is
        # purely positional -- topology (and therefore manifoldness) is
        # untouched -- but re-validate anyway rather than assume it.
        print("  Re-denoising the remesh (staircase artifacts)...")
        v_s, f_s = denoise_taubin(v, f, opts, notes)
        man_s, why = _try_manifold(v_s, f_s)
        if man_s is not None:
            man, v, f = man_s, v_s, f_s
        else:
            msg = f"post-remesh denoise broke manifoldness ({why}); keeping the un-denoised remesh"
            print(f"  {msg}")
            notes.append(msg)

    if opts.solid_infill:
        man = fill_cavities(man, notes, opts.min_fragment_ratio)
    proc = manifold_to_trimesh(man, notes)
    stages["solidify"] = stages.get("solidify", 0.0) + time.time() - t0
    print(f"  Manifold: {len(proc.vertices):,} vertices, {len(proc.faces):,} faces, "
          f"volume={man.volume():.5f}, genus={man.genus()}, watertight={proc.is_watertight}")

    new_vertex_colors = None
    if opts.bake_colors and visual_info["kind"] != "none":
        # Cleaning/smoothing/manifolding all renumber vertices, so the original
        # per-vertex arrays no longer line up -- resample by closest point, the
        # same way v1 does after its voxel remesh.
        print("  Baking colors from the original mesh onto the new topology...")
        new_vertex_colors = bake_colors_from_original(original_mesh, visual_info, proc.vertices)
        visual_info = {"kind": "vertex_color" if new_vertex_colors is not None else "none",
                       "uv": None, "base_color_img": None, "mr_img": None,
                       "vertex_colors": new_vertex_colors}
    if used_voxel_fallback:
        notes.append(f"{'meshlib SDF' if opts.repair_backend == 'meshlib' else 'voxel'} "
                     "remesh fallback was used for this object")

    return proc, man, visual_info, new_vertex_colors


def run_make_printable_v2(
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
    taubin_steps: int = 10,
    taubin_lambda: float = 0.5,
    taubin_mu: float = -0.53,
    crease_angle: float = 60.0,
    close_hole_size: int = 1000,
    hollow_thickness: float = 0.0,
    bake_colors: bool = True,
    repair_backend: str = "pymeshlab",
    min_fragment_ratio: float = 0.005,
) -> PrintableV2Result:
    """v2 counterpart of printable.run_make_printable -- same inputs/outputs.

    Writes <output_prefix>.glb/.stl (or <output_prefix>_partNN.* for explode),
    and returns a PrintableV2Result whose diagnostics/fidelity/watertight fields
    follow the same per-mode shapes documented on PrintableResult.

    Mode differences from v1:
      - multi_object boolean-UNIONS the objects into one solid (v1 concatenates
        them as separate shells in one file).
      - single/explode behave as in v1.
    """
    t_start = time.time()
    stages = {}
    notes = []

    if repair_backend == "meshlib" and not meshlib_repair.available():
        msg = "repair_backend='meshlib' requested but the meshlib SDK is not installed; using pymeshlab"
        print(f"  Warning: {msg}")
        notes.append(msg)
        repair_backend = "pymeshlab"

    opts = PrintableV2Options(
        target_faces=target_faces,
        taubin_lambda=taubin_lambda,
        taubin_mu=taubin_mu,
        taubin_steps=taubin_steps,
        crease_angle=crease_angle,
        close_hole_size=close_hole_size,
        solid_infill=solid_infill,
        hollow=hollow_thickness,
        bake_colors=bake_colors,
        voxel_pitch=voxel_pitch,
        repair_backend=repair_backend,
        min_fragment_ratio=min_fragment_ratio,
    )

    print(f"Loading: {glb_path}")
    t0 = time.time()
    mesh = mesh_io.load_glb(glb_path)
    stages["load"] = time.time() - t0
    print(f"Mesh: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces, "
          f"watertight={mesh.is_watertight}")

    if explode or multi_object:
        bodies = split_objects(mesh, min_ratio=min_object_ratio, max_objects=max_objects)
        print(f"Objects to process: {len(bodies)}")
    else:
        comps = significant_components(mesh, min_ratio=min_object_ratio, max_objects=max_objects)
        if len(comps) > 1:
            print(f"  Note: {len(comps)} large objects detected; keeping the largest only. "
                  f"Use --multi-object or --explode to keep them all.")
        bodies = [comps[0]]

    if explode:
        width = max(2, len(str(len(bodies))))
        files, diags, fids, waters = [], [], [], []
        for i, body in enumerate(bodies, start=1):
            print(f"\n===== Object {i}/{len(bodies)} =====")
            original = body.copy()
            proc, man, visual_info, colors = process_object_v2(body, opts, stages, notes)
            if opts.hollow > 0:
                proc = manifold_to_trimesh(hollow(man, opts.hollow, notes), notes)
            fid = fidelity(original, proc)
            _print_fidelity_text(fid)
            d = diagnostics(proc, overhang_angle)
            _print_diagnostics_text(d)
            print("\n[4/4] Exporting...")
            t0 = time.time()
            glb, stl = export_glb_and_stl(proc, visual_info, colors, f"{output_prefix}_part{i:0{width}d}")
            stages["export"] = stages.get("export", 0.0) + time.time() - t0
            files += [{"kind": "printable_glb", "path": glb, "part": i},
                      {"kind": "printable_stl", "path": stl, "part": i}]
            diags.append(d)
            fids.append(fid)
            waters.append(bool(proc.is_watertight))
        print(f"\nDone: exploded {len(bodies)} objects into {output_prefix}_part* .glb/.stl")
        return PrintableV2Result(
            mode="explode", files=files, diagnostics=diags, fidelity=fids,
            watertight=waters, seconds=time.time() - t_start, stages=stages, notes=notes,
        )

    if multi_object:
        mans, fids = [], []
        for i, body in enumerate(bodies, start=1):
            print(f"\n===== Object {i}/{len(bodies)} =====")
            original = body.copy()
            proc, man, _visual_info, _colors = process_object_v2(body, opts, stages, notes)
            fids.append(fidelity(original, proc))
            _print_fidelity_text(fids[-1])
            mans.append(man)

        print("\nMerging parts (Manifold3D boolean union)...")
        t0 = time.time()
        combined_man = union_parts(mans, notes)
        if opts.solid_infill:
            combined_man = fill_cavities(combined_man, notes, opts.min_fragment_ratio)
        if opts.hollow > 0:
            combined_man = hollow(combined_man, opts.hollow, notes)
        combined = manifold_to_trimesh(combined_man, notes)
        stages["merge"] = time.time() - t0
        print(f"  Combined: {len(combined.vertices):,} vertices, {len(combined.faces):,} faces, "
              f"watertight={combined.is_watertight}")

        d = diagnostics(combined, overhang_angle)
        _print_diagnostics_text(d)
        print("\n[4/4] Exporting...")
        t0 = time.time()
        combined_visual = mesh_io.extract_visual_info(combined)
        glb, stl = export_glb_and_stl(combined, combined_visual,
                                       combined_visual.get("vertex_colors"), output_prefix)
        stages["export"] = stages.get("export", 0.0) + time.time() - t0
        return PrintableV2Result(
            mode="multi_object",
            files=[{"kind": "printable_glb", "path": glb}, {"kind": "printable_stl", "path": stl}],
            diagnostics=d,
            fidelity=fids[0] if fids else None,
            watertight=bool(combined.is_watertight),
            seconds=time.time() - t_start,
            glb_path=glb, stl_path=stl, stages=stages, notes=notes,
        )

    body = bodies[0]
    original = body.copy()
    proc, man, visual_info, colors = process_object_v2(body, opts, stages, notes)
    if opts.hollow > 0:
        proc = manifold_to_trimesh(hollow(man, opts.hollow, notes), notes)
    fid = fidelity(original, proc)
    _print_fidelity_text(fid)
    d = diagnostics(proc, overhang_angle)
    _print_diagnostics_text(d)
    print("\n[4/4] Exporting...")
    t0 = time.time()
    glb, stl = export_glb_and_stl(proc, visual_info, colors, output_prefix)
    stages["export"] = time.time() - t0
    return PrintableV2Result(
        mode="single",
        files=[{"kind": "printable_glb", "path": glb}, {"kind": "printable_stl", "path": stl}],
        diagnostics=d,
        fidelity=fid,
        watertight=bool(proc.is_watertight),
        seconds=time.time() - t_start,
        glb_path=glb, stl_path=stl, stages=stages, notes=notes,
    )

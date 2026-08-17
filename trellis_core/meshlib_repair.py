"""
MeshLib SDK repair primitives, used as an alternative backend inside
printable_v2.py.

Two operations matter here, and the first is why this module exists.

**Orientation repair.** TRELLIS's mesh extraction emits scrambled face winding:
on a real job mesh 22.6% of the interior edges (66,095 of 291,787) are traversed
in the SAME direction by both adjacent faces. That is fatal downstream --
Manifold3D requires an oriented 2-manifold, MeshLib's own importer splits
vertices at every contradiction (97K verts -> 182K, 24,244 holes), and every
winding-number/SDF method silently produces garbage. Connectivity-based fixers
(PyMeshLab's re_orient_faces_coherently, trimesh's fix_winding) cannot repair
it because they propagate orientation across the same broken adjacency.
MeshLib's findDisorientedFaces decides per face by RAY PARITY instead --
geometry, not connectivity -- and clears 99.3% of it in ~0.1s.

**SDF rebuild.** With orientation fixed, rebuildMesh converts to a signed
distance field and back via dual marching cubes. Unlike trimesh's binary
voxelize + marching cubes (v1's fallback), the field is sub-voxel accurate, so
the result is both closer to the original surface and far cheaper in triangles.

Import is lazy and `available()` is the guard: MeshLib is an optional dep, and
nothing here may break a run that does not have it installed.
"""

import numpy as np


def available():
    """True if the MeshLib SDK can be imported."""
    try:
        import meshlib.mrmeshpy  # noqa: F401
        import meshlib.mrmeshnumpy  # noqa: F401
        return True
    except ImportError:
        return False


def _to_meshlib(vertices, faces):
    from meshlib import mrmeshnumpy as mn

    return mn.meshFromFacesVerts(
        np.ascontiguousarray(faces, dtype=np.int32),
        np.ascontiguousarray(vertices, dtype=np.float64),
    )


def _from_meshlib(mesh):
    from meshlib import mrmeshnumpy as mn

    # pack() drops the invalid verts/faces MeshLib leaves behind after edits;
    # getNumpyVerts would otherwise hand back those holes in the arrays.
    mesh.pack()
    verts = np.asarray(mn.getNumpyVerts(mesh), dtype=np.float64)
    faces = np.asarray(mn.getNumpyFaces(mesh.topology), dtype=np.int64)
    return verts, faces


def _disoriented_mask(vertices, faces):
    """Per-face ray-parity verdict from MeshLib, or None if it cannot be mapped back."""
    from meshlib import mrmeshpy as mm, mrmeshnumpy as mn

    mask = mn.getNumpyBitSet(mm.findDisorientedFaces(_to_meshlib(vertices, faces)))
    # MeshLib drops the odd unusable face on import; if that shifts the face
    # indexing we cannot map the verdict back onto our array safely.
    return mask if len(mask) == len(faces) else None


def fix_orientation(vertices, faces, notes, max_passes=2):
    """Give the mesh a coherent outward winding. Returns (faces, n_flipped).

    Ray parity alone is decided independently per face, so it comes back
    speckled: on a real job mesh it still left 3.6% of edges contradictory, in
    scattered patches. Those patches are not cosmetic -- the winding number
    downstream reads +-1 off across each one, so the rebuilt isosurface jumps to
    a different sheet (hull -> cabin wall, tens of voxels) and lands as a
    crater/scab on an otherwise smooth surface.

    So combine the two sources of truth instead of trusting either alone:

      1. propagate winding across face adjacency, which is noise-free but can
         only make each connected patch *self*-consistent, not correctly-facing;
      2. use ray parity for the one thing adjacency cannot decide -- whether a
         whole patch faces in or out -- by majority vote over its faces.

    Measured on the Benchy: 3.610% -> 0.043% contradictory edges, and the
    rebuild's out-of-surface error drops from 7.2% to 0.0%.
    """
    import trimesh

    faces = np.asarray(faces).copy()

    tm = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    trimesh.repair.fix_winding(tm)
    faces = np.asarray(tm.faces).copy()

    mask = _disoriented_mask(vertices, faces)
    if mask is None:
        msg = "meshlib face indexing shifted; falling back to per-face orientation repair"
        print(f"  {msg}")
        notes.append(msg)
        return _fix_orientation_per_face(vertices, faces, notes, max_passes)

    total = 0
    patches = trimesh.graph.connected_components(tm.face_adjacency, nodes=np.arange(len(faces)))
    for patch in patches:
        if mask[patch].mean() > 0.5:      # patch is inside-out as a whole
            faces[patch] = faces[patch][:, ::-1]
            total += len(patch)

    if total:
        msg = f"meshlib re-oriented {total:,} faces across {len(patches)} patch(es)"
        print(f"  {msg}")
        notes.append(msg)
    return faces, total


def _fix_orientation_per_face(vertices, faces, notes, max_passes=2):
    """Original per-face ray-parity flip; fallback when the patch vote cannot run."""
    faces = np.asarray(faces).copy()
    total = 0
    previous = None
    for _pass in range(max_passes):
        mask = _disoriented_mask(vertices, faces)
        if mask is None:
            break
        n = int(mask.sum())
        # Stops as soon as a pass stops helping: on self-touching sheets the last
        # few hundred faces oscillate between two contradictory states forever.
        if n == 0 or n == previous:
            break
        faces[mask] = faces[mask][:, ::-1]
        total += n
        previous = n

    if total:
        msg = f"meshlib re-oriented {total:,} faces (per-face ray parity)"
        print(f"  {msg}")
        notes.append(msg)
    return faces, total


def rebuild(vertices, faces, voxel_size, notes, close_holes=False):
    """Rebuild the mesh through a signed distance field (MeshLib rebuildMesh).

    Returns a manifold surface, but NOT necessarily a CLOSED one. This docstring
    used to promise "always returns a closed surface" and the caller relied on
    it. Measured on a real failed job whose mesh had 16.7% non-manifold and 4.8%
    open edges: HoleWindingNumber could not resolve inside from outside over
    large regions, and the extracted surface came back 0.000% non-manifold but
    22.5% OPEN. Manifold3D rejected it and the job died with no output at all.
    Callers must keep a guaranteed-closed rung (voxel remesh) below this one.

    HoleWindingNumber sign detection tolerates
    the holes and self-intersections that survive earlier stages -- but it reads
    the face winding, so fix_orientation() must have run first or the result is
    nonsense (measured: volume off by 2.4x and 5,921 disconnected bodies).

    close_holes defaults to FALSE, unlike MeshLib's own default. Virtually
    capping every hole before evaluating the winding number puts a phantom
    surface across each one, and the field on the far side of that cap comes out
    inside-out -- which shows up as crater/scab patches on the finished model.
    The generalized winding number already degrades gracefully across genuine
    holes, so it does not need the caps. Measured on the Benchy: turning this
    off took the surface >2 voxels off-target from 3.93% to 0.00%, and the
    rebuild from 30.3s to 6.2s.
    """
    from meshlib import mrmeshpy as mm

    settings = mm.RebuildMeshSettings()
    settings.voxelSize = float(voxel_size)
    settings.signMode = mm.SignDetectionModeShort.HoleWindingNumber
    settings.closeHolesInHoleWindingNumber = close_holes

    out = mm.rebuildMesh(mm.MeshPart(_to_meshlib(vertices, faces)), settings)
    v, f = _from_meshlib(out)
    msg = f"meshlib SDF rebuild at voxel {voxel_size:.6f}: {len(f):,} faces"
    print(f"  {msg}")
    notes.append(msg)
    return v, f

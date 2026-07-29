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


def fix_orientation(vertices, faces, notes, max_passes=2):
    """Flip the faces whose winding contradicts the ray-parity inside/outside test.

    Returns (faces, n_flipped). Vertices are never touched, so this is safe to
    run before anything else -- it only rewrites triangle winding.

    Iterates because the first pass changes what "inside" looks like, but stops
    as soon as a pass stops helping: on self-touching sheets the last few hundred
    faces oscillate between two equally-contradictory states forever.
    """
    from meshlib import mrmeshpy as mm, mrmeshnumpy as mn

    faces = np.asarray(faces).copy()
    total = 0
    previous = None
    for _pass in range(max_passes):
        mask = mn.getNumpyBitSet(mm.findDisorientedFaces(_to_meshlib(vertices, faces)))
        # MeshLib drops the odd unusable face on import; if that shifts the face
        # indexing we cannot map the mask back safely, so decline the pass.
        if len(mask) != len(faces):
            msg = (f"meshlib returned {len(mask)} face flags for {len(faces)} faces; "
                   "skipping orientation repair")
            print(f"  {msg}")
            notes.append(msg)
            break
        n = int(mask.sum())
        if n == 0 or n == previous:
            break
        faces[mask] = faces[mask][:, ::-1]
        total += n
        previous = n

    if total:
        msg = f"meshlib re-oriented {total:,} faces (ray parity)"
        print(f"  {msg}")
        notes.append(msg)
    return faces, total


def rebuild(vertices, faces, voxel_size, notes, close_holes=True):
    """Rebuild the mesh through a signed distance field (MeshLib rebuildMesh).

    Always returns a closed surface. HoleWindingNumber sign detection tolerates
    the holes and self-intersections that survive earlier stages -- but it reads
    the face winding, so fix_orientation() must have run first or the result is
    nonsense (measured: volume off by 2.4x and 5,921 disconnected bodies).
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

# Print-prep pipelines v2 / v3 (PyMeshLab + Manifold3D + MeshLib)

Experimental alternatives to `trellis_core/printable.py`, living side by side with it
so all three can be run on the same mesh and compared.

| | v1 (`printable.py`) | v2 (`printable_v2.py`) | v3 (`printable_v2.py`, `repair_backend="meshlib"`) |
|---|---|---|---|
| Clean | trimesh `merge_vertices` + degenerate-face drop | PyMeshLab: null faces, duplicate faces/vertices, unreferenced points | same as v2 |
| Orientation | none | none (PyMeshLab/trimesh reorienters fail, see below) | **MeshLib ray-parity repair** |
| Denoise | none | PyMeshLab Taubin (volume-preserving), sharp edges locked | same as v2 |
| Solidify | voxelize + orthographic flood fill (always) | Manifold3D exact booleans; trimesh voxel remesh as last resort | Manifold3D + **MeshLib SDF rebuild** as last resort |
| Guarantee | watertight *before* decimation | valid 2-manifold *after* every step, including export | same as v2 |

**v3 is the one to use.** On real TRELLIS meshes it is ~2× faster than both, 4–7×
smaller, and 20–25× closer to the original surface. Numbers below.

## Install

```bash
pip install -e ".[postproc-v2]"     # pymeshlab + manifold3d + meshlib
```

## Run the comparison

```bash
python compare_printable.py jobs/<id>/output/model.glb
python compare_printable.py in.glb --quiet --json metrics.json
python compare_printable.py in.glb --only v3 --taubin-steps 20 --crease-angle 45
```

All three write `<outdir>/v{1,2,3}.{glb,stl}`, then are measured with the *same*
ruler: fidelity is recomputed by the harness against one shared reference (the
input's largest component, pre-repair), and printability comes from the same
`printable.diagnostics()` for each.

## The pipeline

1. **Clean (PyMeshLab)** — `meshing_remove_null_faces`, `..._duplicate_faces`,
   `..._duplicate_vertices`, `..._unreferenced_vertices`.
2. **Denoise (PyMeshLab)** — `apply_coord_taubin_smoothing` (λ=0.5, μ=−0.53, 10
   steps). The λ/μ pair is what makes Taubin shrink-free, so volume survives.
   Sharp edges are locked separately: vertices on an edge with a dihedral angle
   above `--crease-angle` get scalar 0, the rest 1, and smoothing runs on the
   `q > 0.5` selection only. A cube stays a cube (verified: 0.8³ box → volume
   exactly 0.512, zero deviation).
3. **Solidify (Manifold3D)** — build a validated 2-manifold, boolean-union parts,
   fill enclosed cavities, drop floating fragments, optionally hollow.
4. **Export** — `.stl` for the slicer plus `.glb` with appearance re-baked.

### Repair ladder

Manifold3D rejects anything that is not an oriented 2-manifold, so stage 3 escalates:

1. direct conversion (with `Mesh.merge()` to stitch coincident-but-unshared verts);
2. PyMeshLab `meshing_repair_non_manifold_edges/vertices` + `meshing_close_holes`;
3. fan-patch of the small open boundary loops PyMeshLab declines (it refuses holes
   touching non-manifold vertices, and a handful of 3–5 vertex holes is enough to
   sink the whole mesh);
4. v1's voxel remesh — always watertight, but resamples the surface, so it is the
   last resort here rather than v1's default.

When rung 4 fires, v2 does two things v1 does not: it **re-runs the Taubin denoise on
the remesh** (the grid resampling introduces its own staircase artifacts, which the
first denoise pass never saw), and it **re-validates after decimation** — decimating
to the face budget can break watertightness, and a valid manifold wins over hitting
the target face count.

### Export hardening

`.stl` stores bare float32 triangles, so any two distinct vertices whose coordinates
collide at float32 get re-welded on import — by every slicer. v2 detects those and
displaces the extras along their normal by ~1e-5 of the model size (four orders of
magnitude below a 0.4 mm nozzle). Without this the exported STL picks up
non-manifold edges even though the mesh in memory is perfect.

## Measured results

Two real job meshes, identical settings, same machine. Mesh A =
`jobs/2c20e636…/output/model.glb` (195K faces after component filtering), mesh B =
`jobs/1467d882…/output/model.glb` (986K faces).

| Metric | A: v1 | A: v2 | A: v3 | B: v1 | B: v2 | B: v3 |
|---|---|---|---|---|---|---|
| Wall time | 27.4 s | 28.1 s | **11.8 s** | 73.2 s | 76.3 s | **32.3 s** |
| Faces | 1,000,000 | 999,984 | **140,480** | 1,000,000 | 999,546 | **251,842** |
| **Watertight (GLB)** | **no** | yes | yes | **no** | yes | yes |
| **Watertight (STL, as a slicer loads it)** | **no** | yes | yes | **no** | yes | yes |
| Disconnected bodies | 11 | 1 | 3 | 346 | 1 | 1 |
| Chamfer vs original | 0.106 % | 0.072 % | **0.006 %** | 0.111 % | 0.060 % | **0.005 %** |
| Hausdorff vs original | 3.92 % | 1.76 % | **0.47 %** | 10.01 % | 0.78 % | **0.45 %** |
| STL size | 48.8 MB | 48.8 MB | **6.9 MB** | 48.8 MB | 48.8 MB | **12.3 MB** |
| Overhang area | 13.3 % | 19.6 % | 20.3 % | 20.0 % | 21.4 % | 21.4 % |

v2 buys watertightness at v1's runtime and file size. v3 additionally halves the
runtime, cuts the file 4–7×, and lands 18–25× closer to the original surface —
because a correctly-oriented SDF rebuild needs far fewer triangles to describe the
same shape than a binary voxel grid does.

v1's Hausdorff of 10 % on mesh B is not a rounding artifact: its post-voxel
decimation shattered that model into 346 bodies and it never re-checked the result.

**On the overhang number**: v2 is not more overhung, it is more honest. With
`--taubin-steps 0` v2 reports 13.7 %, right next to v1's 13.3 %. The voxel staircase
has axis-aligned normals that dodge the overhang test; smoothing restores the true
surface slope, and the metric follows.

## Root cause: TRELLIS emits scrambled face winding

Every one of the six real job meshes surveyed failed to become a manifold through
rungs 1–3, and the reason turned out to be measurable and specific: **22.6 % of the
interior edges are traversed in the same direction by both adjacent faces**
(66,095 of 291,787 on mesh A; 550,656 faces needed flipping on mesh B). The mesh is
essentially closed — 411 open edges, 89 non-manifold edges — but its orientation is
noise.

That single defect explains every symptom:

- Manifold3D rejects it outright (it requires an *oriented* 2-manifold).
- MeshLib's importer splits a vertex at every contradiction: 97K verts → 182K, and
  24,244 holes appear out of nowhere.
- Winding-number/SDF methods read the winding, so they return confident garbage —
  MeshLib's `rebuildMesh` gave a volume 2.4× off across 5,921 disconnected bodies.
- `meshing_re_orient_faces_coherently` and `trimesh.repair.fix_winding` both fail,
  because they propagate orientation across the very adjacency that is broken.

MeshLib's `findDisorientedFaces` decides per face by **ray parity** — geometry, not
connectivity — and clears 99.3 % of it in ~0.1 s. That is the whole unlock: run it
first and every downstream tool starts behaving. A residual ~460 edges (0.16 %) sit
in genuinely ambiguous self-touching sheets and oscillate between two equally
contradictory states, so the ladder still ends at a remesh — but now it is a
*correctly signed* SDF rebuild instead of a blind voxel fill, which is where v3's
20× fidelity gain comes from.

The topology-preserving path (no remesh at all) is verified working on clean input
and will pay off if upstream mesh extraction ever emits coherent winding.

### Crater/scab artifacts, and the two bugs behind them

First real-world v3 output (a 3DBenchy) came back with crater-like scabs across
otherwise smooth hull panels. Measured against the raw generated mesh, **7.2 % of
the surface bulged more than 2 voxels outside it and 6.7 % dented more than 2
voxels into it, worst case 28 voxels** — far too big to be resampling error. Two
independent causes, both since fixed:

1. **Per-face orientation voting left speckle.** `findDisorientedFaces` decides
   each face independently by ray parity, which still left 3.6 % of edges
   contradictory, scattered in patches. The winding number then reads ±1 off
   across each patch and the isosurface jumps to a different sheet — on a Benchy,
   from the hull to the cabin wall, tens of voxels away. Fix: propagate winding
   over face adjacency first (noise-free, but only makes each patch
   self-consistent), then use ray parity solely to decide each connected patch's
   global sign by majority vote. 3.610 % → 0.043 % contradictory edges.
2. **Virtual hole closing inverted the field.** `closeHolesInHoleWindingNumber`
   (MeshLib's default, and separately what our own PyMeshLab repair rung did by
   stitching triangles over holes) lays a phantom cap across every hole; the
   field on the far side of that cap comes out inside-out. The generalized
   winding number degrades gracefully across genuine holes and does not need the
   caps. Fix: turn the flag off, and feed the SDF the *original* cleaned arrays
   rather than the repair ladder's hole-closed ones.

| | bulge >2 vox | dent >2 vox | p99 error | rebuild |
|---|---|---|---|---|
| Before (shipped briefly) | 7.18 % | 6.65 % | 10.0 vox | 30.4 s |
| Patch vote only | 3.93 % | 2.96 % | 15.1 vox | 30.3 s |
| **Both fixes (current)** | **0.23 %** | **0.28 %** | **1.2 vox** | **6.2 s** |

The flower mesh was unaffected either way (0.00 % / 0.01 %, p99 0.4 vox) — organic
surfaces hide the artifact, which is why the first round of testing missed it.
**Test print-prep changes on a mechanical shape with large flat/smooth panels.**

### Which MeshLib operations did *not* work

Recorded so nobody re-runs these:

| Attempt | Result |
|---|---|
| `fillHoles` on the imported mesh | Valid manifold, but the vertex-split had already shattered it into 8,900 shells; largest held 1/6 of the volume |
| Boolean-union those shells | Watertight, still only 18 % of the true volume |
| `offsetMesh` unsigned ±1–2 voxels | Watertight and manifold, but unsigned distance has no inside: yields a thin skin (volume 0.000049 vs 0.0088) |
| `rebuildMesh` before orientation repair | 5,921 bodies, volume 2.4× off |
| `rebuildMesh` **after** orientation repair | ✅ what v3 ships |

## Options beyond v1's

| Flag | Default | Effect |
|---|---|---|
| `--taubin-steps` | 10 | Denoise passes; 0 disables |
| `--taubin-lambda` / `--taubin-mu` | 0.5 / −0.53 | Tune both together or volume drifts |
| `--crease-angle` | 60° | Lock sharper edges out of smoothing; 0 smooths everything |
| `--min-fragment-ratio` | 0.005 | Drop shells below this fraction of the main body's **bounding-box diagonal** |
| `--hollow` | 0 (off) | Wall thickness via Minkowski erosion; cost scales with face count |
| `--multi-object` | off | v2/v3 boolean-**union** the objects into one solid (v1 concatenates shells) |

### Why fragments are judged by extent, not volume

An SDF rebuild can pinch a thin neck and turn a real limb into its own body. Those
whiskers have almost no volume but real spatial reach, so a volume threshold deletes
model geometry while looking harmless. Measured on mesh A, dropping shells under 1 %
of *volume* pushed Hausdorff from 0.40 % to 3.97 %; the same 1 % threshold on
*bounding-box diagonal* kept them and cost nothing. This matches how v1's
`significant_components()` already filters input components.

## Status

Not wired into `server/worker.py` or `make_printable.py` — `run_make_printable_v2()`
takes the same arguments and returns the same `PrintableResult` shape as
`run_make_printable()`, so switching is a one-line import change plus
`repair_backend="meshlib"`.

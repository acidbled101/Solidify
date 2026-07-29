# Print-prep pipeline v2 (PyMeshLab + Manifold3D)

Experimental alternative to `trellis_core/printable.py`, living side by side with it
so the two can be run on the same mesh and compared.

| | v1 (`printable.py`) | v2 (`printable_v2.py`) |
|---|---|---|
| Clean | trimesh `merge_vertices` + degenerate-face drop | PyMeshLab: null faces, duplicate faces/vertices, unreferenced points |
| Denoise | none | PyMeshLab Taubin (volume-preserving), sharp edges locked |
| Solidify | voxelize + orthographic flood fill (always) | Manifold3D exact booleans; voxel path only as a last resort |
| Guarantee | watertight *before* decimation | valid 2-manifold *after* every step, including export |

## Install

```bash
pip install -e ".[postproc-v2]"     # or: pip install pymeshlab manifold3d
```

## Run the comparison

```bash
python compare_printable.py jobs/<id>/output/model.glb
python compare_printable.py in.glb --quiet --json metrics.json
python compare_printable.py in.glb --only v2 --taubin-steps 20 --crease-angle 45
```

Both pipelines write `<outdir>/v1.{glb,stl}` and `<outdir>/v2.{glb,stl}`, then are
measured with the *same* ruler: fidelity is recomputed by the harness against one
shared reference (the input's largest component, pre-repair), and printability comes
from the same `printable.diagnostics()` for both.

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

| Metric | A: v1 | A: v2 | B: v1 | B: v2 |
|---|---|---|---|---|
| Wall time | 26.6 s | 27.9 s | 72.6 s | 79.0 s |
| Faces | 1,000,000 | 999,984 | 1,000,000 | 999,546 |
| **Watertight (GLB)** | **no** | **yes** | **no** | **yes** |
| **Watertight (STL, as a slicer loads it)** | **no** | **yes** | **no** | **yes** |
| Disconnected bodies | 11 | 1 | 346 | 1 |
| Chamfer vs original | 0.106 % | 0.072 % (−32 %) | 0.107 % | 0.060 % (−44 %) |
| Hausdorff vs original | 3.98 % | 1.77 % (−56 %) | 9.88 % | 0.86 % (−91 %) |
| Overhang area | 13.3 % | 19.6 % | 20.0 % | 21.4 % |

Same file size and 1–9 % more time, but v2 delivers a single watertight body where v1
delivers 11 and 346 non-watertight ones, at a fraction of the worst-case deviation.
The gap widens with mesh size: v1's post-voxel decimation is what shatters the model,
and it never re-checks the result.

**On the overhang number**: v2 is not more overhung, it is more honest. With
`--taubin-steps 0` v2 reports 13.7 %, right next to v1's 13.3 %. The voxel staircase
has axis-aligned normals that dodge the overhang test; smoothing restores the true
surface slope, and the metric follows.

## Known finding: TRELLIS meshes need the voxel rung

Every one of the six real job meshes surveyed failed to become a manifold through
rungs 1–3. They come out of the repair chain closed and edge-manifold (0 open edges,
0 non-manifold edges) but with an odd Euler characteristic and winding that neither
`meshing_re_orient_faces_coherently` nor `trimesh.repair.fix_winding` can make
coherent — self-touching sheets from the generator's mesh extraction.

So on today's TRELLIS output, v2's advantage does **not** come from preserving
original topology (rung 4 fires, same as v1). It comes from validating the result at
every step: post-remesh denoise, post-decimation re-repair, cavity/fragment removal
via exact booleans, and an STL that survives its own round-trip. The topology-
preserving path is verified working on clean input (synthetic and hand-punctured
meshes) and will pay off if upstream mesh extraction improves.

## Options beyond v1's

| Flag | Default | Effect |
|---|---|---|
| `--taubin-steps` | 10 | Denoise passes; 0 disables |
| `--taubin-lambda` / `--taubin-mu` | 0.5 / −0.53 | Tune both together or volume drifts |
| `--crease-angle` | 60° | Lock sharper edges out of smoothing; 0 smooths everything |
| `--hollow` | 0 (off) | Wall thickness via Minkowski erosion; cost scales with face count |
| `--multi-object` | off | v2 boolean-**unions** the objects into one solid (v1 concatenates shells) |

## Status

Not wired into `server/worker.py` or `make_printable.py` — `run_make_printable_v2()`
takes the same arguments and returns the same `PrintableResult` shape as
`run_make_printable()`, so switching is a one-line import change when the comparison
justifies it.

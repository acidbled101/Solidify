# Implementation Plan: Physics-Aware 3D Generative Pipeline

**Project:** Automated Physical Prototyping Pipeline
**Core Stack:** TRELLIS.2, DreamDPO, geometry post-processing
**Status:** Concept / planning — not yet implemented

---

## Two hardware tracks

Solidify runs TRELLIS.2 on Apple Silicon via PyTorch MPS — there
is no CUDA path here (see [README.md](README.md)). The original version of this
plan assumed a CUDA workstation (≥24GB dedicated VRAM, `flash_attn`, PyMeshLab,
Manifold3D). Since the hardware on hand right now is a Mac, this document is
split into two tracks:

- **Track A — Mac / MPS (priority, this repo).** Everything below is written
  against this repo's actual stack: `trellis_core/pipeline.py`,
  `backends/conv_none.py`, `backends/mesh_extract.py`, trimesh-only geometry
  processing. No PyMeshLab/Manifold3D dependency is assumed yet — Mac ARM64
  wheel availability for PyMeshLab is unreliable, so Track A leans on trimesh
  (already a dependency) and only reaches for Manifold3D if trimesh's own
  repair/boolean ops prove insufficient in practice.
- **Track B — CUDA (future).** Once the concept is validated on Mac, a second
  version of this plan will target a CUDA workstation (A100/3090/4090 class,
  ≥24GB VRAM), where `flash_attn`, PyMeshLab, and Manifold3D become reasonable
  choices again. Not started yet.

Everything from here down is Track A unless marked otherwise.

---

## Relationship to the existing print-prep pipeline

This repo already has a **non-differentiable, heuristic** print-prep pipeline in
[trellis_core/printable.py](trellis_core/printable.py) (used by
`make_printable.py` and the web server):

- `diagnostics()` — overhang % (face-normal vs. a critical angle) and thin-wall
  warnings (ray-cast wall thickness sampling).
- `fidelity()` — Chamfer/Hausdorff deviation and volume change vs. the
  pre-repair mesh.
- `repair_watertight()` — best-effort trimesh repair, falling back to a
  guaranteed-watertight voxel remesh + flood-fill solidify.

That pipeline runs **after** generation, as a fixed post-process — it reports
problems but can't steer generation away from them. The pipeline described
below is a different thing: it evaluates *candidate meshes mid-generation* to
steer the diffusion trajectory itself via preference optimization (DPO), using
a differentiable-in-spirit scoring function ($S$ below) that mirrors the same
physical concerns (overhang, thin walls, topology) but as a scalar reward for
ranking trajectory branches, not a report. Phase 4 (post-processing, below)
should call into the existing `repair_watertight` / `diagnostics` / `fidelity`
functions rather than reimplementing them — the DPO loop's job is to make the
*input* to that existing pipeline better, not to replace it.

---

## I. System Architecture & Tooling (Track A — Mac)

- **Hardware:** Apple Silicon (M1 or later), 24GB+ unified memory recommended.
  Unlike a CUDA card, VRAM is unified memory shared with the whole OS — this
  matters for the trajectory-branching step below (see Risks).
- **Generation Engine:** TRELLIS.2 via `trellis_core/pipeline.py`
  (`Trellis2ImageTo3DPipeline`, MPS device).
- **Optimization Framework:** DreamDPO-style inference-time preference
  optimization, adapted to branch during MPS sampling.
- **Geometry Processing:** trimesh (already a dependency) for the Geometric
  Judge and for Phase 4 post-processing, reusing `trellis_core/printable.py`.
  Manifold3D is a stretch goal *if* trimesh's manifold repair proves
  insufficient — it does ship macOS arm64 wheels, so it's not ruled out, just
  not assumed up front.

---

## II. The 3D Geometric Judge (Objective Function)

The core of the optimization loop evaluates intermediate meshes extracted from
candidate trajectory branches against physical constraints. The composite
score $S$ balances detail retention with manufacturing limits:

$$S = \alpha R_{Detail} - \left( \beta L_{OH} + \gamma L_{Th} + \delta L_{Topo} \right)$$

### 1. Detail Reward ($R_{Detail}$)
Preserves high-frequency features and sharp edges using the discrete
Laplace-Beltrami operator to calculate surface energy:
$$R_{Detail} = \sum_{v \in V} ||\Delta \mathbf{x}_v||^2$$

### 2. Overhang Penalty ($L_{OH}$)
Penalizes faces requiring support structures based on a critical hardware angle
$\theta_{crit}$ (e.g. 45°, matching `printable.py`'s existing
`--overhang-angle` default):
$$L_{OH} = \sum_{i=1}^{F} A_i \cdot \max\left(0, \, \mathbf{n}_i \cdot \mathbf{g} - \cos(\theta_{crit})\right)$$

### 3. Minimum Thickness Penalty ($L_{Th}$)
Aggressively penalizes cross-sections that fall below the printer's minimum
nozzle resolution $d_{min}$ (matches the spirit of `printable.py`'s
`min_wall_mm` ray-cast check):
$$L_{Th} = \sum_{p} \max\left(0, \, \left(\frac{d_{min} - d(p)}{d_{min}}\right)^2\right)$$

### 4. Topological Defect Penalty ($L_{Topo}$)
A strict scalar penalty applied to open boundaries and self-intersections to
prevent slicer failures:
$$L_{Topo} = w_{open} N_{open} + w_{intersect} N_{intersect}$$

The judge only needs to **rank** two candidate meshes (winner/loser) for the
DPO loss below — it does not need to be differentiable itself, so it can be
plain trimesh/numpy code (as `printable.py` already is), not a
GPU-differentiable objective.

---

## III. Step-by-Step Implementation Phases (Track A — Mac)

### Phase 1: Environment & Heuristic Calibration
1. Reuse the existing `.venv` from `setup.sh` — no new heavy deps required to
   start (trimesh/numpy/scipy are already installed).
2. Implement $L_{OH}$, $L_{Th}$, $L_{Topo}$ as thin wrappers around the
   existing `diagnostics()` logic in `trellis_core/printable.py`, rather than
   parallel reimplementations.
3. Calibrate $d_{min}$ and $\theta_{crit}$ against real prints from whatever
   printer this pipeline targets.

### Phase 2: Flow-Matching Interception
1. Hook into `run_generation()` in `trellis_core/pipeline.py` — this is where
   `pipeline.run(...)` currently calls the sampler in one shot; branching needs
   access to the sampler's per-step loop instead.
2. At a defined ODE timestep $t \in [0.3, 0.7]$, apply small noise
   perturbations to the latent state to create two parallel candidate
   trajectories ($\epsilon_A, \epsilon_B$).
3. Decode both branched latents through the SLat decoder into meshes via
   `backends/mesh_extract.py`, same as the existing single-trajectory path.

### Phase 3: The DreamDPO Optimization Loop
1. Score both candidate meshes with the Geometric Judge (Section II).
2. Assign the higher-scoring mesh as preferred ($y_w$), the lower as
   rejected ($y_l$).
3. Compute the DPO loss on the sampler's own velocity-prediction
   log-probabilities (not on the judge, which is non-differentiable by
   design):
   $$\mathcal{L}_{DPO}(\pi) = -\log \sigma \left( \beta_{dpo} \log \frac{P_\pi(y_w)}{P_{ref}(y_w)} - \beta_{dpo} \log \frac{P_\pi(y_l)}{P_{ref}(y_l)} \right)$$
4. Backpropagate through the sampler to steer the latent trajectory. This is
   the step most at risk on MPS — see Risks below.

### Phase 4: Post-Processing & Output
1. Extract the final mesh from the winning trajectory.
2. Hand it to the **existing** `trellis_core/printable.py` pipeline unchanged:
   `repair_watertight()` → `diagnostics()` → `fidelity()` →
   `export_glb_and_stl()`. Do not reimplement watertight repair or
   solidification here.
3. Route the finalized files wherever this pipeline's output is consumed
   (CLI, or the web server's job flow).

---

## IV. Known Risks & Mitigation Strategies (Track A — Mac)

| Risk Factor | Root Cause | Engineering Mitigation |
| :--- | :--- | :--- |
| **Unified-memory pressure during branching** | Two parallel decoded meshes + gradient state through the sampler share the same 24GB pool as the OS and everything else — there's no dedicated VRAM headroom like a CUDA card has. | Keep branch count at 2 (not more); free non-winning branch tensors immediately after scoring; consider running headless (see README's watchdog notes) to reclaim memory from WindowServer. |
| **Backprop through custom Metal kernels** | `mtlgemm`'s sparse-conv path (default per README) is a compiled forward Metal kernel; it's unclear it has a registered autograd backward. `backends/conv_none.py` is plain PyTorch gather/scatter and differentiates normally. | Force `SPARSE_CONV_BACKEND=none` for the branching/DPO step specifically (accepting the known slowdown) rather than assuming `mtlgemm` supports backprop; keep `mtlgemm` for the non-DPO fast path. |
| **GPU watchdog kills during long branched runs** | Two parallel trajectories roughly double per-step Metal kernel duration, and this repo already documents the macOS GPU-watchdog silently killing long kernels (`trellis_core/pipeline.py`'s `WatchdogError`). | Reuse the existing watchdog detection/handling; test headless first, per the README's documented workaround order. |
| **Topological Stalls** | Voxel-decoded meshes easily form micro-holes, triggering $L_{Topo}$ constantly during early sampling. | Use a tolerant $\delta$ early in the ODE flow; rely on the existing `repair_watertight()` voxel-fallback for final cleanup, not on the judge to force perfection mid-generation. |
| **Mode Collapse** | Overhang/thickness penalties overwhelm the generative prompt, yielding featureless blobs. | Schedule $\alpha$ (Detail Reward) high early in the ODE flow, ramping up structural penalties only in the final steps. |

---

## V. Track B — CUDA (future, not started)

Once Track A validates the concept end-to-end on Mac hardware, produce a
second version of this plan for a CUDA workstation, reintroducing:
- `flash_attn` for fused attention (vs. this repo's SDPA-padded fallback).
- PyMeshLab / Manifold3D for geometry processing, if trimesh's capabilities
  prove limiting on Track A.
- Larger branch counts and/or batched candidates, since dedicated VRAM removes
  the unified-memory contention problem above.

No further detail here until Track A is working.

"""Differentiable printability objectives, computed on the DUAL-GRID MESH the
shape decoder already produces -- not on the SLat latent.

WHY THIS EXISTS
---------------
`dpo_branch.slat_detail_proxy` is the only scalar the DPO steering loop can
differentiate today. It is the umbrella-Laplacian energy of the SLat *latent
features*: a stand-in for the judge's R_Detail term alone. It knows nothing
about overhang, wall thickness or topology, and measured across every real
branch on disk its change correlates with the judge's score change at only
r = +0.20 (n=17). The optimiser works; its objective is nearly uncorrelated
with the goal.

The received explanation for why nothing better was possible is that "mesh
extraction is not differentiable". That is true of the *implementation* and
false of the *mathematics*, and the distinction is the whole point of this
module.

    backends/mesh_extract.py::flexible_dual_grid_to_mesh, line 117:

        mesh_vertices = (coords.float() + dual_vertices) * voxel_size + aabb[0]

`dual_vertices` is `(1 + 2*margin) * sigmoid(h.feats[..., 0:3]) - margin`
(models/sc_vaes/fdg_vae.py:115) -- a smooth function of the decoder output,
which is a smooth function of the SLat, which is a smooth function of `delta`.
So **mesh vertex POSITIONS are differentiable w.r.t. delta**. This is not
marching cubes on a scalar field, where vertex positions come from a
non-differentiable table lookup plus an interpolation along an edge; it is a
dual-grid (FlexiCubes-family) decoder where the network *directly regresses*
each vertex's position.

What is genuinely non-differentiable is only the CONNECTIVITY:
  * `intersected = h.feats[..., 3:6] > 0`             (fdg_vae.py:116)
  * `coord_to_idx` python-dict neighbour lookup       (mesh_extract.py:88-108)
  * the quad-diagonal choice `torch.where(sw_02 > sw_13, ...)` (line 136-144)
  * `subdiv_binarized = subdiv.feats > 0` in every decoder up-block
    (sparse_unet_vae.py:159, 246)

All four are piecewise-constant integer/boolean decisions. Holding them fixed
and differentiating only through vertex positions is exactly the standard
DMTet / FlexiCubes / nvdiffrec training recipe, and it is correct everywhere
except on the measure-zero set where a flag flips.

CONSEQUENCE: three of the judge's four terms become *exactly* differentiable
w.r.t. delta, using the judge's own formulas rather than a proxy for them:

  R_Detail  sum_v ||x_v - mean_{u~v} x_u||^2         -> detail_energy()
  L_OH      sum_i A_i * relu(n_i.g - cos theta_crit) -> overhang_energy()
  L_Th      mean_p relu((d_min - d(p))/d_min)^2      -> thickness_energy()
                                                        (correspondence detached,
                                                         value differentiable)
  L_Topo    open/non-manifold edge RATE              -> NOT differentiable at all;
                                                        it is a pure function of
                                                        connectivity. topology_
                                                        soft_open_rate() is a
                                                        RELAXATION over the
                                                        `intersected` logits, not
                                                        the same number.

The price is not mathematical, it is memory and wall-clock: the decoder must
run WITHOUT torch.no_grad() (dpo_branch._decode_candidate currently wraps it),
which means one graph-building shape-decoder forward per gradient step. See
COST NOTES at the bottom of this docstring.

WHAT IS VALIDATED AND WHAT IS NOT
---------------------------------
  * The mesh-side maths in this file is validated offline against
    trellis_core.geometric_judge on 66 real decoded candidate meshes (see
    printability_proxy_test.py and the correlation table in the report). It
    reproduces the judge's R_Detail and L_OH to ~1e-6 relative.
  * The autograd chain from `dual_vertices` through
    backends.mesh_extract.flexible_dual_grid_to_mesh into these energies is
    verified by an executable CPU test (test_gradient_flows_through_extraction).
  * The autograd chain through the REAL shape decoder (sparse convs, MPS,
    checkpointing) is NOT verified -- it needs GPU time this module has not
    been given. Two known risks, both flagged in the plan's own risk table:
    (a) SparseConv3d's `flex_gemm` Metal kernel has no registered backward, so
    the decoder would have to run under SPARSE_CONV_BACKEND=none, which is
    slower; (b) memory. Treat "gradients reach delta through the real decoder"
    as an untested claim until someone runs it.

MEASURED (66 real decoded candidate meshes from devlab/traces, 23 distinct
forks, 54 within-fork candidate pairs; scratchpad scripts, all CPU, no GPU)
----------------------------------------------------------------------------
Reproduction check: recomputing the judge from the exported GLBs reproduces
its R_Detail, L_OH and L_Topo to <1e-6 relative on all 66 meshes, so the
offline dataset is faithful. ray_thickness() reproduces trimesh's own first-hit
distances to 1.7e-6 and thickness_energy() reproduces the judge's L_Th formula
on the same rays to 7e-7.

Within-fork paired deltas -- the design that matters, because steering has to
get the effect of a SMALL perturbation right (n = 54, bootstrap 95% CIs):

  proxy                     vs judge S     vs printability   vs L_OH   vs L_Th
  detail_energy (= what      +0.958***        -0.094 n.s.    +0.06 n.s. +0.09 n.s.
    the current proxy
    is a stand-in for)
  overhang_energy            -0.006 n.s.      -0.210 n.s.    +1.000***  +0.07 n.s.
  thickness_energy(8k rays)  -0.125 n.s.      -0.724***      +0.16 n.s. +0.715***
  printability_objective     +0.122 n.s.      +0.730***      -0.31*     -0.700***
  judge_objective (full S)   +0.979***        +0.104 n.s.    -0.03 n.s. -0.10 n.s.

Read the first and last rows together: an objective that tracks the judge's S
almost perfectly carries NO printability signal at all, because S is dominated
by R_Detail (within a fork, sd(dR_Detail)=0.035 vs sd(dL_OH)=0.0015). Steering
S and steering printability are close to orthogonal here. That is why
printability_objective() deliberately drops the R_Detail term.

Two caveats stated up front:
  * Those printability columns are against a DE-NOISED judge (L_Th averaged
    over 24 independent ray draws). Against the judge as it is actually
    deployed -- one 500-ray draw -- every printability proxy scores r between
    -0.15 and +0.12, none significant, at every proxy budget tried (500 to
    8000 rays). The judge's own within-fork Delta L_Th has a reliability of
    0.21 (CRN-paired, 24 seeds/pair), so |r| against it is capped near zero.
    The target is noise, not the proxy. FIX THE JUDGE'S RAY BUDGET (or make
    L_Th deterministic) before reading anything into a within-fork L_Th
    comparison -- including the preference LABELS the DPO loop is conditioned
    on, which are affected the same way.
  * L_Th appears on both sides of the printability comparison (same formula,
    different random draws), so agreement there is partly definitional. The
    non-circular results are the ones where the two sides differ: the
    detail/S rows above, and the overhang finding in
    unsupported_overhang_energy's docstring.

COST NOTES
----------
Peak activation memory is the binding constraint (the pipeline already OOM'd
once at 36 GiB). Mitigations, in the order they should be tried:
  1. dpo_branch.checkpointed_blocks() already walks `model.modules()` and
     flips any `use_checkpoint` flag -- SparseConvNeXtBlock3d,
     SparseResBlockC2S3d and friends all expose one, so the same helper works
     on the shape decoder unchanged.
  2. linearized_surrogate(): take ONE differentiable decode per fork, keep
     g = d(objective)/d(slat), then run the inner gradient loop against the
     first-order model  s(delta) ~ <g, slat(delta)>  which needs only
     flow-model forwards. Exact to first order inside the trust region
     project_delta_() already enforces.
  3. Subsample faces: L_OH and R_Detail are sums/means over faces and
     vertices; a uniform random subset is an unbiased estimator and shrinks
     the retained mesh-side graph proportionally. (It does NOT shrink the
     decoder's own activations, which dominate.)

Measured wall-clock on this hardware, from the trace timestamps: a no-grad
shape decode + judge + export is 62-75 s, and one existing gradient step
(flow model only, 2 continuation steps, checkpointed) is 33-83 s. A
graph-building decode plus its backward is roughly 2-3x a forward decode, so
naive per-step differentiable decoding would add ~150-200 s PER GRADIENT STEP
-- at the shipped 3 steps x 2 forks that is +15-20 min on a ~12 min run.
Mitigation 2 (linearized_surrogate) reduces that to one differentiable decode
per fork, i.e. +5-7 min total, and is the only version worth trying first.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def face_cross(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Un-normalized face normals, [F, 3]. ||cross_i|| == 2 * area_i.

    `faces` is an integer index tensor and is treated as a CONSTANT -- it
    carries no gradient and must not (it is the connectivity, which is
    piecewise-constant in delta; see this module's docstring).
    """
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    return torch.linalg.cross(v1 - v0, v2 - v0)


def face_areas(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    return 0.5 * face_cross(verts, faces).norm(dim=1)


# ---------------------------------------------------------------------------
# L_OH -- overhang
# ---------------------------------------------------------------------------


def overhang_energy(
    verts: torch.Tensor,
    faces: torch.Tensor,
    theta_crit_deg: float = 45.0,
    *,
    support_aware: bool = False,
    bed_clearance: float = 0.01,
    bed_softness: float = 0.004,
    z_min: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """L_OH, exactly as geometric_judge.overhang_penalty computes it:

        L_OH = sum_i A_i * relu(n_i . g - cos(theta_crit)),   g = (0, 0, -1)

    Written in the division-free form

        A_i * relu(n_i.g - c) = 0.5 * relu(-cross_iz - c * ||cross_i||)

    because A_i = ||cross_i||/2 and n_i.g = -cross_iz/||cross_i||. Avoiding the
    normalization is not cosmetic: ||cross_i|| -> 0 for a degenerate sliver
    triangle, and dividing by it produces an inf/NaN gradient that poisons the
    whole backward pass. Decoded mid-ODE meshes contain slivers routinely.

    Differentiable w.r.t. `verts` everywhere except (a) the relu kink and
    (b) exactly-degenerate triangles, both measure-zero.

    support_aware=False reproduces the judge BIT FOR BIT, including its known
    defect: the penalty is purely normal-based with no support test, so a face
    lying flat on the build plate (n.g = 1) is charged the full penalty even
    though it needs no support at all. A cube resting on the bed scores
    L_OH = 0.0732 rather than 0.

    support_aware=True multiplies each face by

        w_i = sigmoid((z_i - z_min - bed_clearance) / bed_softness)

    (z_i = face centroid height), which smoothly forgives faces sitting on or
    just above the build plate. This is the physically correct thing and the
    WRONG thing to correlate with the judge -- see the module report. Use it
    only if you have also fixed geometric_judge.overhang_penalty, otherwise the
    proxy and the judge disagree about a first-order effect and the gradient
    fights the preference label.

    z_min defaults to verts[:, 2].min(). Pass a detached constant if you do
    not want the bed height itself to receive gradient (recommended: a
    gradient that lowers the whole object to move the bed is not steering, it
    is cheating).
    """
    cross = face_cross(verts, faces)
    if cross.numel() == 0:
        return verts.sum() * 0.0
    cos_crit = math.cos(math.radians(theta_crit_deg))
    # 0.5 * relu(-cross_z - cos_crit * ||cross||)  ==  A * relu(n.g - cos_crit)
    excess = 0.5 * torch.relu(-cross[:, 2] - cos_crit * cross.norm(dim=1))
    if not support_aware:
        return excess.sum()

    if z_min is None:
        z_min = verts[:, 2].min().detach()
    zc = verts[faces].mean(dim=1)[:, 2]
    w = torch.sigmoid((zc - z_min - bed_clearance) / max(bed_softness, 1e-9))
    return (w * excess).sum()


def unsupported_overhang_energy(
    verts: torch.Tensor,
    faces: torch.Tensor,
    support_mask: torch.Tensor,
    theta_crit_deg: float = 45.0,
) -> torch.Tensor:
    """The overhang penalty an FDM slicer would actually charge:

        L_OH^sup = sum_{i : unsupported} A_i * relu(n_i . g - cos theta_crit)

    where `support_mask[i]` is True for faces that ARE supported and are
    therefore excluded -- i.e. faces resting on the build plate, and faces with
    material directly beneath them (an internal ceiling, or a bridge over
    existing structure). Build the mask with support_mask_from_raycast() below;
    it is a detached boolean, exactly like RayCorrespondence, and the energy
    stays differentiable in `verts` through A_i and n_i.

    MEASURED, and this is the central finding of the report: on the 66 real
    decoded candidate meshes on disk, only a median 5.3% (mean 8.7%) of the
    judge's L_OH comes from faces that genuinely need support. A median 66.9%
    comes from faces resting flat on the build plate, and ~47% from downward
    faces with material directly beneath them. As a result

        corr(L_OH_judge, L_OH_truly_unsupported) = +0.049 (Pearson, n=66, n.s.)
                                                 = -0.432 (Spearman, p=3e-4)

    The judge's L_OH and real support requirement are not merely
    imperfectly aligned; they are uncorrelated, and rank-wise mildly OPPOSED.
    A proxy therefore cannot serve both masters, and you must pick:

      * overhang_energy(support_aware=False) -> agrees with the judge, does not
        measure printability.
      * unsupported_overhang_energy()        -> measures printability, will
        DISAGREE with the judge and therefore fight the preference label the
        DPO loop is conditioned on.

    Using this one is only coherent if geometric_judge.overhang_penalty is
    fixed the same way at the same time, so proxy and judge keep pointing in
    the same direction.
    """
    cross = face_cross(verts, faces)
    if cross.numel() == 0:
        return verts.sum() * 0.0
    cos_crit = math.cos(math.radians(theta_crit_deg))
    excess = 0.5 * torch.relu(-cross[:, 2] - cos_crit * cross.norm(dim=1))
    keep = (~support_mask).to(excess.dtype)
    return (keep * excess).sum()


def support_mask_from_raycast(mesh, theta_crit_deg: float = 45.0,
                              bed_clearance: float = 0.01,
                              max_rays: int = 4000, seed: int = 0,
                              z_bed: Optional[float] = None):
    """Detached per-face support mask for unsupported_overhang_energy().

    A face counts as SUPPORTED if it rests on the build plate (centroid within
    `bed_clearance` of z_min) or if a ray fired straight down from its centroid
    hits the mesh again. Faces that are not downward-facing at all are marked
    supported too, since they contribute zero to the energy anyway and casting
    rays for them is wasted work.

    Non-differentiable by construction -- that is the point: it is the discrete
    correspondence, computed once per decode by trimesh's BVH, and held fixed
    while the gradient flows through the face areas and normals. Sampling is
    capped at `max_rays` contributing faces; unsampled contributing faces are
    treated as UNSUPPORTED (the conservative direction: it keeps a penalty
    rather than silently forgiving geometry that was never checked).

    z_bed defaults to the mesh's own z_min, which is right for a single decoded
    object standing on the plate (which is what this pipeline produces). Pass
    it explicitly when the mesh is a fragment whose lowest point is NOT on the
    plate, or the fragment's own underside will be forgiven as bed contact.

    Takes a trimesh.Trimesh, returns a torch bool tensor of length n_faces.
    """
    import numpy as np

    cos_crit = math.cos(math.radians(theta_crit_deg))
    ng = -mesh.face_normals[:, 2]
    down = (ng - cos_crit) > 0
    zc = mesh.triangles_center[:, 2]
    z0 = float(mesh.vertices[:, 2].min()) if z_bed is None else float(z_bed)
    on_bed = (zc - z0) < bed_clearance

    supported = ~down | on_bed
    idx = np.nonzero(down & ~on_bed)[0]
    if len(idx):
        sel = (np.random.default_rng(seed).choice(idx, max_rays, replace=False)
               if len(idx) > max_rays else idx)
        origins = mesh.triangles_center[sel] + np.array([0.0, 0.0, -1e-5])
        dirs = np.tile(np.array([0.0, 0.0, -1.0]), (len(sel), 1))
        hit = mesh.ray.intersects_any(origins, dirs)
        supported[sel[hit]] = True
    return torch.from_numpy(supported)


# ---------------------------------------------------------------------------
# R_Detail
# ---------------------------------------------------------------------------


def unique_edges(faces: torch.Tensor) -> torch.Tensor:
    """[E, 2] sorted unique undirected edges. Connectivity -> constant."""
    if faces.numel() == 0:
        return faces.new_zeros((0, 2))
    e = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], dim=0)
    e, _ = torch.sort(e, dim=1)
    return torch.unique(e, dim=0)


def detail_energy(
    verts: torch.Tensor,
    edges: torch.Tensor,
    reduction: str = "sum",
) -> torch.Tensor:
    """R_Detail = sum_v ||x_v - mean_{u in N(v)} x_u||^2 -- the uniform
    (umbrella) graph Laplacian energy of geometric_judge.detail_reward(),
    same operator, differentiable, in torch.

    reduction='sum' matches the judge. 'mean' is scale-free in vertex count,
    which matters if you compare meshes with different tessellation (the judge
    does not, and its docstring flags that as a known limitation).
    """
    n = verts.shape[0]
    if n == 0 or edges.numel() == 0:
        return verts.sum() * 0.0
    rows = torch.cat([edges[:, 0], edges[:, 1]])
    cols = torch.cat([edges[:, 1], edges[:, 0]])
    degree = torch.zeros(n, device=verts.device, dtype=verts.dtype).index_add_(
        0, rows, torch.ones_like(rows, dtype=verts.dtype)
    )
    has_nb = degree > 0
    degree_safe = torch.where(has_nb, degree, torch.ones_like(degree))
    nb_sum = torch.zeros_like(verts).index_add(0, rows, verts[cols])
    lap = verts - nb_sum / degree_safe.unsqueeze(1)
    lap = lap * has_nb.unsqueeze(1).to(verts.dtype)
    per_vertex = (lap ** 2).sum(dim=1)
    return per_vertex.sum() if reduction == "sum" else per_vertex.mean()


# ---------------------------------------------------------------------------
# L_Th -- wall thickness, with a detached ray correspondence
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RayCorrespondence:
    """Which face each sampled ray started from and which face it hit.

    Produced ONCE per differentiable decode by a non-differentiable ray cast
    (trimesh, or any BVH). Both fields are integer indices and carry no
    gradient -- that is the whole trick: the *correspondence* is a discrete
    argmin and cannot be differentiated, but the *distance* to the found
    triangle is a smooth function of vertex positions, so
    d(thickness)/d(verts) is well defined once the correspondence is pinned.
    Standard practice in differentiable ray tracing; correct except when the
    correspondence itself changes, which is again measure-zero in the interior
    of each cell and is exactly what the trust region bounds.
    """

    src_face: torch.Tensor  # [P] long -- face the ray was launched from
    hit_face: torch.Tensor  # [P] long -- face the ray first hit


def ray_thickness(
    verts: torch.Tensor,
    faces: torch.Tensor,
    corr: RayCorrespondence,
    eps_scale: float = 0.0,
) -> torch.Tensor:
    """Differentiable first-hit distances d(p), [P], for a fixed correspondence.

    Ray p starts just inside face s = src_face[p], travelling along its inward
    normal, and hits face h = hit_face[p]. With
        o = centroid(s) - eps * n_hat(s)      (origin)
        u = -n_hat(s)                         (direction)
        m = cross(h)                          (un-normalized normal of hit face)
        a = one vertex of face h
    the ray/plane intersection distance is

        d = ((a - o) . m) / (u . m)

    every factor of which is a smooth function of `verts`. Matches
    geometric_judge._thickness_arrays' geometry exactly (same origins, same
    directions, same "first hit" semantics) -- only the *search* for h is
    delegated to a detached BVH query.
    """
    cross = face_cross(verts, faces)
    nrm = cross.norm(dim=1, keepdim=True).clamp_min(1e-20)
    n_hat = cross / nrm

    s, h = corr.src_face, corr.hit_face
    centroid_s = verts[faces[s]].mean(dim=1)
    n_s = n_hat[s]
    o = centroid_s - eps_scale * n_s
    u = -n_s

    m = cross[h]
    a = verts[faces[h][:, 0]]
    denom = (u * m).sum(dim=1)
    # A ray parallel to its hit plane cannot happen for a real first hit; clamp
    # only to keep a pathological correspondence from producing an inf.
    denom = torch.where(denom.abs() < 1e-12, torch.full_like(denom, 1e-12), denom)
    return ((a - o) * m).sum(dim=1) / denom


def thickness_energy(
    verts: torch.Tensor,
    faces: torch.Tensor,
    corr: RayCorrespondence,
    d_min: float,
    eps_scale: float = 0.0,
) -> torch.Tensor:
    """L_Th = mean_p relu((d_min - d(p)) / d_min)^2 -- geometric_judge's own
    formula (thickness_penalty_detailed), on differentiable distances.

    Only rays that actually hit are represented in `corr`, matching the judge,
    which averages over successful rays only. If corr is empty this returns 0
    -- and, exactly as in the judge, 0 then means "unmeasured", NOT "thick".
    Do not feed an empty-correspondence 0 into a preference comparison without
    the judge's own fairness guard (rank_candidates drops gamma*L_Th from BOTH
    sides when either measurement is unreliable).
    """
    if corr.src_face.numel() == 0 or d_min <= 0:
        return verts.sum() * 0.0
    d = ray_thickness(verts, faces, corr, eps_scale)
    return (torch.relu((d_min - d) / d_min) ** 2).mean()


# ---------------------------------------------------------------------------
# L_Topo -- relaxation only (the judge's version is not differentiable at all)
# ---------------------------------------------------------------------------


def topology_soft_open_rate(
    intersected_logits: torch.Tensor,
    neighbourhood_complete: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """A RELAXATION of the judge's L_Topo open-boundary rate, over the decoder's
    raw `intersected` logits (h.feats[..., 3:6], before the `> 0` threshold).

    In flexible_dual_grid_to_mesh an intersected edge produces a quad only if
    all four voxels around it are present in the sparse set
    (`connected_voxel_valid`); an intersected edge whose neighbourhood is
    incomplete is DROPPED, and the resulting gap is what shows up downstream as
    an open boundary. So

        soft_open_rate = sum sigmoid(logits/T) * (1 - complete)
                         / (sum sigmoid(logits/T) + eps)

    is a smooth surrogate for "fraction of surface that fails to close".

    THIS IS NOT THE JUDGE'S NUMBER. The judge counts edges of the extracted
    triangle mesh with incidence 1 or >2; this counts would-be quads with an
    incomplete voxel neighbourhood. They are correlated by construction but
    not equal, and this one has never been validated against the other on real
    data (the traces on disk save meshes, not decoder logits). Ranked LAST for
    that reason -- see the report. Included because L_Topo is the one judge
    term with no exact differentiable counterpart, and a stated relaxation is
    better than silently dropping the term.
    """
    p = torch.sigmoid(intersected_logits / max(temperature, 1e-9)).reshape(-1)
    incomplete = (1.0 - neighbourhood_complete.reshape(-1).to(p.dtype))
    return (p * incomplete).sum() / (p.sum() + 1e-9)


# ---------------------------------------------------------------------------
# Composite objectives
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProxyWeights:
    """Mirrors geometric_judge.JudgeWeights for the terms that have a
    differentiable counterpart. Defaults match
    dpo_branch._default_judge_weights() (d_min=0.02, delta=0.05) so the proxy
    and the judge agree by default rather than by coincidence."""

    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0
    delta: float = 0.05
    theta_crit_deg: float = 45.0
    d_min: float = 0.02
    support_aware_overhang: bool = False
    bed_clearance: float = 0.01
    bed_softness: float = 0.004


def printability_objective(
    verts: torch.Tensor,
    faces: torch.Tensor,
    weights: ProxyWeights = ProxyWeights(),
    corr: Optional[RayCorrespondence] = None,
    soft_open_rate: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """PRINTABILITY ONLY -- the quantity the user actually asked to steer:

        P = -(beta*L_OH + gamma*L_Th + delta*L_Topo)

    Higher is better (it is a reward, sign-matched to the judge's S so it can
    be dropped straight into dpo_branch.preference_loss in place of
    slat_detail_proxy). Deliberately EXCLUDES R_Detail: the judge's S is
    numerically dominated by R_Detail (std 5.77 vs 0.062 for L_OH over one
    sweep), so any objective containing it ends up steering detail again --
    which is what the current proxy already does.

    corr=None drops the thickness term rather than silently scoring it 0 (the
    judge's own "0 means unmeasured, not good" hazard).
    """
    total = weights.beta * overhang_energy(
        verts, faces, weights.theta_crit_deg,
        support_aware=weights.support_aware_overhang,
        bed_clearance=weights.bed_clearance,
        bed_softness=weights.bed_softness,
    )
    if corr is not None:
        total = total + weights.gamma * thickness_energy(verts, faces, corr, weights.d_min)
    if soft_open_rate is not None:
        total = total + weights.delta * soft_open_rate
    return -total


def judge_objective(
    verts: torch.Tensor,
    faces: torch.Tensor,
    edges: Optional[torch.Tensor] = None,
    weights: ProxyWeights = ProxyWeights(),
    corr: Optional[RayCorrespondence] = None,
    soft_open_rate: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """The judge's full S = alpha*R_Detail - (beta*L_OH + gamma*L_Th +
    delta*L_Topo), differentiably, for callers who want to track S rather than
    printability. Use printability_objective() if the goal is FDM printability
    -- see TRAP 2 in the report: this one correlates with S almost perfectly
    and with printability barely at all, because R_Detail swamps it.
    """
    if edges is None:
        edges = unique_edges(faces)
    return (
        weights.alpha * detail_energy(verts, edges)
        + printability_objective(verts, faces, weights, corr, soft_open_rate)
    )


# ---------------------------------------------------------------------------
# First-order surrogate (memory mitigation #2, see module docstring)
# ---------------------------------------------------------------------------


def linearized_surrogate(
    slat_feats: torch.Tensor,
    grad_at_anchor: torch.Tensor,
    anchor_feats: torch.Tensor,
) -> torch.Tensor:
    """<g, x - x0>: the first-order model of an objective around an anchor.

    Pay ONE differentiable decode per fork to obtain
    g = d(printability_objective)/d(slat_feats) at the branch point, then run
    the whole inner steering loop against this, which needs only flow-model
    forwards -- no decoder, no mesh extraction, no ray casting. Exact to first
    order, and the pipeline already confines delta to a hard RMS trust region
    (dpo_branch.project_delta_), which is precisely the regime where a
    first-order model is the right approximation.

    Turns the cost of printability steering from
    `num_delta_grad_steps` differentiable decodes into ONE.
    """
    return (grad_at_anchor.detach() * (slat_feats - anchor_feats.detach())).sum()

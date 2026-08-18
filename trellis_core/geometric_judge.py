"""
3D Geometric Judge -- physics-aware scoring function for ranking candidate
meshes during inference-time preference optimization.

See 3D_PRINTING_AWARE_PIPELINE.md, Section II, for the score definition:

    S = alpha * R_Detail - (beta * L_OH + gamma * L_Th + delta * L_Topo)

This module only needs to RANK two candidate meshes (winner/loser) for the
DPO loss in the trajectory-branching hook -- it does not need to be
differentiable itself, so it's plain trimesh/numpy, same as the existing
non-ML print-prep pipeline in trellis_core/printable.py. Where a metric
overlaps with printable.py's diagnostics() (overhang angle convention,
thin-wall ray casting), the same conventions are used here so scores from
this module and printable.py's diagnostics agree on what "overhang" and
"thin wall" mean -- but the numbers aren't pulled from printable.py directly,
since diagnostics() reports percentages/counts for a human report while this
module needs continuous, differentiable-in-spirit penalty magnitudes for
ranking.

Known limitations (see review notes, kept here rather than lost in a PR
description):
  - R_Detail uses a uniform (umbrella) Laplacian, which responds to
    tessellation density/regularity as well as genuine shape detail -- a
    coarser, more irregular mesh of the same shape can score *higher*. This
    only matters when comparing candidates with meaningfully different face
    counts; branches decoded from the same decoder at the same step should be
    comparably tessellated.
  - L_Topo's N_intersect term is a non-manifold-edge count, not real
    triangle-triangle self-intersection testing (that needs an extra
    dependency this repo doesn't have, e.g. python-fcl). It catches
    fan/T-junction defects but reads as zero for two clean surface sheets
    passing through each other.
  - theta_crit is measured as a half-angle from straight-down, so it
    coincides with the conventional "angle from vertical" FDM overhang rule
    only at exactly 45 degrees; raising theta_crit flags MORE faces, not
    fewer. Matches printable.py's existing convention exactly (deliberate),
    just non-obvious when calibrating (plan Phase 1.3).
"""

import math
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
import trimesh


@dataclass(frozen=True)
class _DetailRewardArrays:
    laplacian_mag: np.ndarray  # per-vertex ||Δx_v||^2, length == len(mesh.vertices)
    vertex_degree: np.ndarray  # per-vertex neighbor count, same length


def _detail_reward_arrays(mesh: trimesh.Trimesh) -> _DetailRewardArrays:
    """The actual R_Detail computation. detail_reward() below is a one-line
    wrapper that sums laplacian_mag -- kept as a separate function so the
    per-vertex arrays are available to callers that want them (e.g. for a
    detail-energy heatmap) without a second pass over the mesh."""
    verts = mesh.vertices
    n = len(verts)
    if n == 0:
        return _DetailRewardArrays(np.zeros(0), np.zeros(0))

    edges = mesh.edges_unique
    if len(edges) == 0:
        return _DetailRewardArrays(np.zeros(n), np.zeros(n))
    i, j = edges[:, 0], edges[:, 1]
    rows = np.concatenate([i, j])
    cols = np.concatenate([j, i])

    degree = np.bincount(rows, minlength=n).astype(np.float64)
    has_neighbors = degree > 0
    degree_safe = np.where(has_neighbors, degree, 1.0)

    neighbor_sum = np.zeros_like(verts)
    np.add.at(neighbor_sum, rows, verts[cols])
    neighbor_mean = neighbor_sum / degree_safe[:, None]

    laplacian = verts - neighbor_mean
    laplacian[~has_neighbors] = 0.0

    return _DetailRewardArrays(
        laplacian_mag=np.einsum("ij,ij->i", laplacian, laplacian),
        vertex_degree=degree,
    )


def detail_reward(mesh: trimesh.Trimesh) -> float:
    """R_Detail: discrete Laplace-Beltrami surface energy, sum_v ||Δx_v||^2.

    Uses the uniform (umbrella) graph Laplacian -- Δx_v = x_v - mean(neighbors)
    -- rather than a cotangent Laplacian, since cotangent weights need
    consistent per-triangle angles that a mid-generation candidate mesh
    (possibly non-manifold, self-intersecting) isn't guaranteed to have.
    Vectorized via plain numpy (bincount + add.at); do not replace with a
    per-vertex Python loop, it's too slow for real mesh sizes, and avoid
    reintroducing a scipy sparse matrix here -- it was measured to allocate
    ~100MB transiently per call at 400K vertices, which matters given the
    plan's #1 flagged risk is unified-memory pressure during branching.
    """
    return float(_detail_reward_arrays(mesh).laplacian_mag.sum())


@dataclass(frozen=True)
class _OverhangArrays:
    overhang_cos: np.ndarray  # per-face n_i . g  (g = (0,0,-1)); 1.0 = straight down
    face_areas: np.ndarray    # per-face area, same indexing as overhang_cos


def _overhang_arrays(mesh: trimesh.Trimesh) -> _OverhangArrays:
    if len(mesh.faces) == 0:
        return _OverhangArrays(np.zeros(0), np.zeros(0))
    normal_z = mesh.face_normals[:, 2]
    n_dot_g = -normal_z  # g = (0, 0, -1)
    return _OverhangArrays(overhang_cos=n_dot_g, face_areas=mesh.area_faces)


def overhang_penalty(mesh: trimesh.Trimesh, theta_crit_deg: float = 45.0) -> float:
    """L_OH: area-weighted overhang penalty.

    L_OH = sum_i A_i * max(0, n_i . g - cos(theta_crit))

    Gravity vector g = (0, 0, -1) (print bed at z_min, "up" is +z), matching
    trellis_core/printable.py's diagnostics() overhang convention
    (normal_z < -cos(overhang_angle) there is the binary-mask version of the
    same n . g > cos(theta_crit) condition used here).
    """
    arrays = _overhang_arrays(mesh)
    if len(arrays.overhang_cos) == 0:
        return 0.0
    cos_crit = math.cos(math.radians(theta_crit_deg))
    excess = np.maximum(0.0, arrays.overhang_cos - cos_crit)
    return float(np.sum(arrays.face_areas * excess))


@dataclass(frozen=True)
class ThicknessResult:
    penalty: float
    valid: bool  # False if ray sampling failed / every ray missed -- penalty is 0.0 by construction, not because the mesh is fine


@dataclass(frozen=True)
class _ThicknessArrays:
    wall_depths: np.ndarray  # first-hit ray distance, successful rays only (world units, not d_min-normalized)
    rays_requested: int      # len(sample_idx) -- how many faces were sampled
    rays_hit: int             # how many of those rays returned a first hit


def _thickness_arrays(
    mesh: trimesh.Trimesh,
    n_ray_samples: int = 500,
    rng: Optional[np.random.Generator] = None,
) -> _ThicknessArrays:
    """Ray-cast sampling only, independent of d_min -- wall_depths is the raw
    measured thickness at each sampled point; thickness_penalty_detailed()
    below applies the d_min-relative penalty formula to this. NOTE: does not
    replicate the n_faces==0 / d_min<=0 short-circuit (that stays in the
    caller, which must avoid touching rng at all in the d_min<=0 case -- see
    thickness_penalty_detailed's docstring)."""
    n_faces = len(mesh.faces)
    if n_faces == 0:
        return _ThicknessArrays(np.zeros(0), 0, 0)

    rng = rng or np.random.default_rng()
    sample_idx = rng.choice(n_faces, size=min(n_ray_samples, n_faces), replace=False)
    eps = max(mesh.scale * 1e-4, 1e-9)
    origins = mesh.triangles_center[sample_idx] - mesh.face_normals[sample_idx] * eps
    directions = -mesh.face_normals[sample_idx]

    try:
        locations, index_ray, _index_tri = mesh.ray.intersects_location(origins, directions)
    except Exception as e:
        print(f"  (thickness_penalty: ray sampling skipped: {e})")
        return _ThicknessArrays(np.zeros(0), len(sample_idx), 0)
    if len(locations) == 0:
        return _ThicknessArrays(np.zeros(0), len(sample_idx), 0)

    dists = np.linalg.norm(locations - origins[index_ray], axis=1)
    first_hit = {}
    for d, ray_i in zip(dists, index_ray):
        if ray_i not in first_hit or d < first_hit[ray_i]:
            first_hit[ray_i] = d

    if not first_hit:
        return _ThicknessArrays(np.zeros(0), len(sample_idx), 0)
    d = np.fromiter(first_hit.values(), dtype=np.float64)
    return _ThicknessArrays(wall_depths=d, rays_requested=len(sample_idx), rays_hit=len(d))


def thickness_penalty_detailed(
    mesh: trimesh.Trimesh,
    d_min: float,
    n_ray_samples: int = 500,
    rng: Optional[np.random.Generator] = None,
) -> ThicknessResult:
    """L_Th: minimum-wall-thickness penalty, mean over sampled points.

    L_Th = mean_p max(0, (d_min - d(p)) / d_min)^2

    The plan (Section II.3) writes this as a sum over sampled points p, but a
    literal sum is not comparable across candidate meshes with different face
    counts: n_ray_samples = min(500, n_faces), so a coarser candidate samples
    (and therefore sums over) fewer points, making it look artificially
    thinner-penalty-free relative to a finer one of identical true geometry.
    Averaging makes the score an unbiased per-point estimate that's
    comparable regardless of tessellation, at the cost of being a literal
    reading of the plan's formula -- deliberate deviation, not an oversight.

    d_min must be in the SAME coordinate units as the mesh. Meshes decoded
    straight out of trellis_core/pipeline.py live in the normalized
    [-0.5, 0.5]^3 unit cube (see pipeline.py's aabb=[[-0.5,...],[0.5,...]]),
    not millimeters -- use normalized_thickness_threshold() to convert a
    real-world d_min_mm into this space once you know the intended print
    size. Passing a millimeter-scale d_min (e.g. 1.0) directly against a
    unit-cube mesh makes this penalty saturate at/near its maximum for
    almost any real geometry and dominate the composite score.

    Returns a ThicknessResult so a 0.0 penalty from ray-cast failure can be
    told apart from a genuinely thick mesh (see .valid).
    """
    if len(mesh.faces) == 0 or d_min <= 0:
        return ThicknessResult(0.0, False)
    arrays = _thickness_arrays(mesh, n_ray_samples, rng)
    if arrays.rays_hit == 0:
        return ThicknessResult(0.0, False)
    excess = np.maximum(0.0, (d_min - arrays.wall_depths) / d_min)
    return ThicknessResult(float(np.mean(excess ** 2)), True)


def normalized_thickness_threshold(d_min_mm: float, print_size_mm: float) -> float:
    """Convert a physical minimum-wall-thickness target (mm) into the
    normalized [-0.5, 0.5]^3 unit-cube space TRELLIS.2 decodes into.

    print_size_mm is the longest edge of the physical print you intend to
    slice this model at (the scale factor a slicer would apply to map the
    unit cube to a real object) -- generation time doesn't otherwise know the
    intended physical scale.
    """
    if print_size_mm <= 0:
        raise ValueError("print_size_mm must be positive")
    return d_min_mm / print_size_mm


@dataclass(frozen=True)
class _TopologyArrays:
    edge_incidence: np.ndarray  # per-edge face count (1=open boundary, 2=manifold, >2=non-manifold)
    n_open: int
    n_nonmanifold: int
    n_edges: int


def _topology_arrays(mesh: trimesh.Trimesh) -> _TopologyArrays:
    if len(mesh.faces) == 0:
        return _TopologyArrays(np.zeros(0, dtype=np.int64), 0, 0, 0)
    counts = np.bincount(mesh.edges_unique_inverse)
    n_edges = len(counts)
    if n_edges == 0:
        return _TopologyArrays(counts, 0, 0, 0)
    n_open = int(np.sum(counts == 1))
    n_nonmanifold = int(np.sum(counts > 2))
    return _TopologyArrays(edge_incidence=counts, n_open=n_open, n_nonmanifold=n_nonmanifold, n_edges=n_edges)


def topology_penalty(
    mesh: trimesh.Trimesh,
    w_open: float = 1.0,
    w_intersect: float = 1.0,
) -> float:
    """L_Topo: penalty for open boundaries and self-intersections, as a RATE.

    L_Topo = w_open * (N_open / N_edges) + w_intersect * (N_intersect / N_edges)

    The plan (Section II.4) writes this as raw counts (w_open*N_open +
    w_intersect*N_intersect), but a raw count is not comparable in scale to
    R_Detail/L_OH/L_Th (all O(1) on a unit-cube mesh): a mid-ODE voxel mesh
    can easily carry 10^2-10^4 boundary edges (plan's own "Topological
    Stalls" risk row), which would make L_Topo dominate the composite score S
    regardless of alpha/beta/gamma/delta -- effectively reducing S to "whichever
    candidate has fewer open edges" and making the detail/overhang/thickness
    terms irrelevant to ranking. Dividing by the mesh's own edge count turns
    this into a rate in roughly [0, ~a few], the same order of magnitude as
    the other three terms, while still growing with genuine topological
    damage. Deliberate deviation from the plan's literal formula, not an
    oversight -- same reasoning as thickness_penalty_detailed's sum->mean
    change.

    N_open counts boundary edges (edges belonging to exactly one face) --
    a mesh only ever watertight has zero of these. True triangle-triangle
    self-intersection testing needs an extra dependency (e.g. python-fcl)
    this repo doesn't have; N_intersect is approximated by counting
    non-manifold edges (shared by more than two faces) via
    mesh.edges_unique_inverse, a cheap, dependency-free proxy for the same
    class of problem (geometry that will confuse a slicer), not a literal
    self-intersection count -- it reads as zero for two clean surface sheets
    passing through each other with no shared edge.
    """
    arrays = _topology_arrays(mesh)
    if arrays.n_edges == 0:
        return 0.0
    return float(
        w_open * (arrays.n_open / arrays.n_edges)
        + w_intersect * (arrays.n_nonmanifold / arrays.n_edges)
    )


@dataclass(frozen=True)
class JudgeWeights:
    """Frozen so a caller scheduling weights over the ODE trajectory (plan's
    Mode Collapse mitigation: high alpha early, ramp up beta/gamma/delta late)
    must build a new instance (dataclasses.replace(weights, alpha=...))
    rather than mutating a shared default in place -- a mutable default here
    previously poisoned every subsequent call in the process, including
    score_mesh's and rank_candidates' own separate default instances.
    """
    alpha: float = 1.0    # detail reward weight
    beta: float = 1.0     # overhang penalty weight
    gamma: float = 1.0    # thickness penalty weight
    delta: float = 1.0    # topology penalty weight
    theta_crit_deg: float = 45.0
    # Placeholder assuming a ~100mm print with a 1mm min wall target (0.01 in
    # the normalized unit cube). Recompute with normalized_thickness_threshold()
    # for the actual intended print size -- see thickness_penalty_detailed().
    d_min: float = 0.01
    w_open: float = 1.0
    w_intersect: float = 1.0
    n_ray_samples: int = 500
    # Candidate meshes get decimated to at most this many faces before
    # scoring (via fast_simplification, already a project dependency) --
    # thickness_penalty's ray casting was measured at ~40s on an 800K-face
    # mesh (trimesh's pure-Python ray/rtree fallback rebuilds a bounds tree
    # per call; installing `embreex` would help but isn't assumed present).
    # Ranking only needs relative signal, not full-resolution precision.
    max_faces_for_scoring: Optional[int] = 20000


def with_weights(base: JudgeWeights, **overrides) -> JudgeWeights:
    """Convenience wrapper around dataclasses.replace for weight scheduling."""
    return replace(base, **overrides)


@dataclass
class JudgeScore:
    total: float
    detail_reward: float
    overhang_penalty: float
    thickness_penalty: float
    thickness_valid: bool
    topology_penalty: float


@dataclass
class JudgeDetails:
    """The per-face/per-vertex/per-edge arrays each penalty function computes
    internally and normally throws away after reducing to a scalar. Not used
    by the DPO ranking logic itself -- exists so a caller building a
    visualization (see devlab/) can show histograms and per-element
    breakdowns instead of only the four aggregate penalty scalars in
    JudgeScore. All arrays are relative to the (possibly decimated) mesh that
    was actually scored -- see faces_scored."""

    # detail reward (R_Detail)
    laplacian_mag: np.ndarray   # per-vertex ||Δx_v||^2
    vertex_degree: np.ndarray   # per-vertex neighbor count
    # overhang (L_OH)
    overhang_cos: np.ndarray    # per-face n_i . g; 1.0 = straight down, -1.0 = straight up
    face_areas: np.ndarray      # per-face area, same indexing as overhang_cos
    # thickness (L_Th)
    wall_depths: np.ndarray     # first-hit ray distance, successful rays only (world units)
    rays_requested: int
    rays_hit: int
    # topology (L_Topo)
    edge_incidence: np.ndarray  # per-edge face count (1=open, 2=manifold, >2=non-manifold)
    n_open: int
    n_nonmanifold: int
    n_edges: int
    # bookkeeping
    faces_scored: int           # face count of the mesh actually scored (post-decimation)


def _prepare_for_scoring(mesh: trimesh.Trimesh, max_faces: Optional[int]) -> trimesh.Trimesh:
    """Decimate a copy of mesh to at most max_faces before scoring. Does not
    touch the caller's mesh -- the DPO loop still hands the undecimated
    winner on to decoding/post-processing, only the judge sees the smaller
    copy."""
    if max_faces is None or len(mesh.faces) <= max_faces:
        return mesh
    import fast_simplification
    ratio = 1.0 - (max_faces / len(mesh.faces))
    verts, faces = fast_simplification.simplify(mesh.vertices, mesh.faces, ratio)
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def score_mesh_detailed(
    mesh: trimesh.Trimesh,
    weights: Optional[JudgeWeights] = None,
    rng: Optional[np.random.Generator] = None,
):
    """Like score_mesh(), but also returns a JudgeDetails with the raw
    per-face/per-vertex/per-edge arrays each penalty computes internally and
    normally discards -- for visualization/debugging (see devlab/).

    This is the ONE implementation of the S = alpha*R_Detail - (beta*L_OH +
    gamma*L_Th + delta*L_Topo) computation; score_mesh() below is a
    zero-cost wrapper that drops the JudgeDetails half of the return value.
    There is deliberately no separate/duplicated scoring math anywhere else.

    Returns (JudgeScore, JudgeDetails).
    """
    weights = weights or JudgeWeights()
    mesh = _prepare_for_scoring(mesh, weights.max_faces_for_scoring)

    detail_arrays = _detail_reward_arrays(mesh)
    r_detail = float(detail_arrays.laplacian_mag.sum())

    oh_arrays = _overhang_arrays(mesh)
    if len(oh_arrays.overhang_cos) == 0:
        l_oh = 0.0
    else:
        cos_crit = math.cos(math.radians(weights.theta_crit_deg))
        excess = np.maximum(0.0, oh_arrays.overhang_cos - cos_crit)
        l_oh = float(np.sum(oh_arrays.face_areas * excess))

    # Mirrors thickness_penalty_detailed's short-circuit exactly: when
    # d_min <= 0, rng must stay untouched (no ray sampling at all), not just
    # short-circuited after consuming random draws.
    if weights.d_min <= 0:
        th_arrays = _ThicknessArrays(np.zeros(0), 0, 0)
        l_th = ThicknessResult(0.0, False)
    else:
        th_arrays = _thickness_arrays(mesh, weights.n_ray_samples, rng)
        if th_arrays.rays_hit == 0:
            l_th = ThicknessResult(0.0, False)
        else:
            excess = np.maximum(0.0, (weights.d_min - th_arrays.wall_depths) / weights.d_min)
            l_th = ThicknessResult(float(np.mean(excess ** 2)), True)

    topo_arrays = _topology_arrays(mesh)
    if topo_arrays.n_edges == 0:
        l_topo = 0.0
    else:
        l_topo = float(
            weights.w_open * (topo_arrays.n_open / topo_arrays.n_edges)
            + weights.w_intersect * (topo_arrays.n_nonmanifold / topo_arrays.n_edges)
        )

    total = (
        weights.alpha * r_detail
        - weights.beta * l_oh
        - weights.gamma * l_th.penalty
        - weights.delta * l_topo
    )
    score = JudgeScore(
        total=total,
        detail_reward=r_detail,
        overhang_penalty=l_oh,
        thickness_penalty=l_th.penalty,
        thickness_valid=l_th.valid,
        topology_penalty=l_topo,
    )
    details = JudgeDetails(
        laplacian_mag=detail_arrays.laplacian_mag,
        vertex_degree=detail_arrays.vertex_degree,
        overhang_cos=oh_arrays.overhang_cos,
        face_areas=oh_arrays.face_areas,
        wall_depths=th_arrays.wall_depths,
        rays_requested=th_arrays.rays_requested,
        rays_hit=th_arrays.rays_hit,
        edge_incidence=topo_arrays.edge_incidence,
        n_open=topo_arrays.n_open,
        n_nonmanifold=topo_arrays.n_nonmanifold,
        n_edges=topo_arrays.n_edges,
        faces_scored=len(mesh.faces),
    )
    return score, details


def score_mesh(
    mesh: trimesh.Trimesh,
    weights: Optional[JudgeWeights] = None,
    rng: Optional[np.random.Generator] = None,
) -> JudgeScore:
    """Compute S = alpha*R_Detail - (beta*L_OH + gamma*L_Th + delta*L_Topo)."""
    score, _details = score_mesh_detailed(mesh, weights, rng)
    return score


def rank_candidates(
    mesh_a: trimesh.Trimesh,
    mesh_b: trimesh.Trimesh,
    weights: Optional[JudgeWeights] = None,
    rng: Optional[np.random.Generator] = None,
) -> tuple:
    """Rank two candidate meshes for a DPO step.

    Returns (winner_index, score_a, score_b) where winner_index is 0 for
    mesh_a or 1 for mesh_b -- the higher-scoring mesh is preferred (y_w),
    the other rejected (y_l), per the plan's Phase 3.

    Uses a shared RNG (common random numbers) for both candidates' thickness
    ray sampling, so the two scores are a paired comparison over the same
    sampled directions/faces-by-index rather than independently noisy
    estimates -- meaningfully reduces variance when the two branches are
    near-identical, which is the expected case right after a small DPO
    perturbation.

    Raises ValueError if BOTH candidates score non-finite (e.g. both
    diverged) -- there is no sane winner to return in that case. If only one
    is non-finite, the finite candidate wins automatically (a NaN/inf score
    must never be preferred just because it compares False on both sides of
    `>=`).

    A failed thickness ray-cast (JudgeScore.thickness_valid=False) defaults
    L_Th to 0.0 -- the BEST possible value, not "no problem". This matters in
    exactly the case that's common mid-generation: a mesh riddled with holes
    (the plan's own "Topological Stalls" risk row) is precisely the kind of
    mesh whose inward rays escape through a hole and hit nothing, so without
    this guard the MORE broken candidate would score a free pass on L_Th and
    could win on that alone. If either candidate's thickness measurement is
    unreliable, gamma*L_Th is dropped from BOTH sides' comparison value (not
    just the invalid one) before picking a winner -- an unmeasured candidate
    must not win by default just because its unmeasured penalty reads as
    zero. This only adjusts the WINNER decision; the .total each JudgeScore
    reports is unchanged.
    """
    winner_index, score_a, score_b, _cmp_a, _cmp_b, _details_a, _details_b = rank_candidates_detailed(
        mesh_a, mesh_b, weights, rng,
    )
    return winner_index, score_a, score_b


def rank_candidates_detailed(
    mesh_a: trimesh.Trimesh,
    mesh_b: trimesh.Trimesh,
    weights: Optional[JudgeWeights] = None,
    rng: Optional[np.random.Generator] = None,
):
    """Like rank_candidates(), but also returns (cmp_a, cmp_b, details_a,
    details_b): the actual comparison values used to pick the winner (which
    can differ from score_a.total/score_b.total under the thickness-fairness
    adjustment below -- see rank_candidates' docstring) and the JudgeDetails
    for each candidate, for visualization. rank_candidates() is a thin
    wrapper that drops these four extra values; this is the one
    implementation of the ranking logic.

    Returns (winner_index, score_a, score_b, cmp_a, cmp_b, details_a, details_b).
    """
    weights = weights or JudgeWeights()
    shared_rng = rng or np.random.default_rng()
    seed_state = shared_rng.bit_generator.state

    score_a_rng = np.random.Generator(type(shared_rng.bit_generator)())
    score_a_rng.bit_generator.state = seed_state
    score_a, details_a = score_mesh_detailed(mesh_a, weights, score_a_rng)

    score_b_rng = np.random.Generator(type(shared_rng.bit_generator)())
    score_b_rng.bit_generator.state = seed_state
    score_b, details_b = score_mesh_detailed(mesh_b, weights, score_b_rng)

    thickness_reliable = score_a.thickness_valid and score_b.thickness_valid
    cmp_a = score_a.total if thickness_reliable else score_a.total + weights.gamma * score_a.thickness_penalty
    cmp_b = score_b.total if thickness_reliable else score_b.total + weights.gamma * score_b.thickness_penalty

    a_finite = math.isfinite(cmp_a)
    b_finite = math.isfinite(cmp_b)
    if not a_finite and not b_finite:
        raise ValueError(
            "rank_candidates: both candidates scored non-finite -- "
            "no valid winner (likely both branches diverged)."
        )
    if a_finite != b_finite:
        winner_index = 0 if a_finite else 1
    else:
        winner_index = 0 if cmp_a >= cmp_b else 1
    return winner_index, score_a, score_b, cmp_a, cmp_b, details_a, details_b

"""
CPU-only unit tests for evaluation/geometric_judge.py.

This repo has no pytest suite (test/ holds sample input images), so this is a
self-contained script: plain `assert`s plus a __main__ runner, matching
experiments/dpo_inference_steering/dpo_branch_test.py's style.

    python trellis_core/geometric_judge_test.py

Needs trimesh, numpy, scipy (already project dependencies) and rtree +
fast_simplification for the ray-casting / decimation paths specifically --
install them the same way setup.sh does if missing.
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dataclasses
import numpy as np
import trimesh

from . import geometric_judge as gj


def test_overhang_sign_convention():
    """Straight-down face gets max penalty; straight-up face gets zero."""
    down = trimesh.Trimesh(
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 2, 1]], process=False,
    )
    assert math.isclose(down.face_normals[0][2], -1.0, abs_tol=1e-6)
    up = trimesh.Trimesh(
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]], process=False,
    )
    assert math.isclose(up.face_normals[0][2], 1.0, abs_tol=1e-6)
    assert gj.overhang_penalty(down, 45.0) > 0
    assert gj.overhang_penalty(up, 45.0) == 0.0
    print("  overhang_penalty: straight-down face penalized, straight-up face is not")


def test_topology_penalty_is_a_rate():
    box = trimesh.creation.box(extents=[1, 1, 1])
    assert gj.topology_penalty(box) == 0.0

    tri = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]], process=False)
    assert gj.topology_penalty(tri) == 1.0  # all 3 edges are boundary -> rate 1.0, not count 3

    fan = trimesh.Trimesh(
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        faces=[[0, 1, 2], [0, 1, 3], [1, 0, 2]],  # 3rd face reuses edge (0,1) -> non-manifold
        process=False,
    )
    assert 0 < gj.topology_penalty(fan) < 3.0, "a raw-count regression would read as 3.0 here, not a rate"
    print("  topology_penalty: returns a per-edge RATE (bounded, small), not a raw edge count")


def test_thickness_unit_cube_semantics():
    box = trimesh.creation.box(extents=[1, 1, 1])
    w = gj.JudgeWeights(d_min=gj.normalized_thickness_threshold(1.0, 100.0))
    assert w.d_min == 0.01
    score = gj.score_mesh(box, w)
    assert score.thickness_penalty == 0.0 and score.thickness_valid

    thin = trimesh.creation.box(extents=[1, 1, 0.005])
    score_thin = gj.score_mesh(thin, w)
    assert score_thin.thickness_penalty > 0 and score_thin.thickness_valid
    print("  thickness_penalty_detailed: unit-cube d_min correctly flags a genuinely thin wall")


def test_thickness_tessellation_invariance():
    coarse = trimesh.creation.box(extents=[1, 1, 0.04])
    fine = coarse.subdivide().subdivide()
    lt_coarse = gj.thickness_penalty_detailed(coarse, d_min=0.1, rng=np.random.default_rng(0))
    lt_fine = gj.thickness_penalty_detailed(fine, d_min=0.1, rng=np.random.default_rng(0))
    assert math.isclose(lt_coarse.penalty, lt_fine.penalty, rel_tol=1e-9)
    print("  thickness_penalty_detailed: mean-normalized, invariant to tessellation density")


def test_judge_weights_frozen():
    w = gj.JudgeWeights()
    try:
        w.alpha = 999.0
        raise SystemExit("FAIL: JudgeWeights should be a frozen dataclass")
    except dataclasses.FrozenInstanceError:
        pass
    w2 = gj.with_weights(w, alpha=999.0)
    assert w.alpha == 1.0 and w2.alpha == 999.0
    print("  JudgeWeights: frozen (mutation raises), with_weights() builds a new instance")


def test_rank_candidates_nan_guard():
    box = trimesh.creation.box(extents=[1, 1, 1])
    bad = box.copy()
    bad.vertices[0] = [np.nan, np.nan, np.nan]

    winner, _, _ = gj.rank_candidates(box, bad)
    assert winner == 0, "finite mesh must win regardless of argument order"
    winner2, _, _ = gj.rank_candidates(bad, box)
    assert winner2 == 1, "finite mesh must win regardless of argument order"

    try:
        gj.rank_candidates(bad, bad)
        raise SystemExit("FAIL: both-NaN should raise ValueError")
    except ValueError:
        pass
    print("  rank_candidates: a NaN-scored mesh never wins; both-NaN raises ValueError")


def test_rank_candidates_thickness_fairness():
    """A mesh whose thickness ray-cast FAILS must not win purely because its
    unmeasured L_Th defaults to 0.0 (the best possible value) -- see the
    fairness adjustment in rank_candidates' docstring. Regression test for a
    bug a holistic review caught after the per-file reviews had all passed:
    thickness_valid was computed but never consulted by rank_candidates."""
    # A: genuinely thin (valid measurement, real penalty)
    mesh_a = trimesh.creation.box(extents=[1, 1, 0.005])
    # B: an open half-shell -- inward rays near the opening escape and never
    # hit anything, so the ray cast comes back invalid.
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.5)
    keep = sphere.triangles_center[:, 2] > 0
    mesh_b = trimesh.Trimesh(vertices=sphere.vertices, faces=sphere.faces[keep], process=False)

    w = gj.JudgeWeights(d_min=gj.normalized_thickness_threshold(1.0, 100.0), gamma=50.0)
    score_a = gj.score_mesh(mesh_a, w, rng=np.random.default_rng(0))
    score_b = gj.score_mesh(mesh_b, w, rng=np.random.default_rng(0))
    assert score_a.thickness_valid and not score_b.thickness_valid, (
        "test setup check failed -- construct a different degenerate mesh if "
        "trimesh's ray backend changes behavior"
    )
    assert score_b.total > score_a.total, (
        "test setup check: B's raw .total must exceed A's for this to test "
        "anything (otherwise B never had a chance to win unfairly)"
    )

    winner, _, _ = gj.rank_candidates(mesh_a, mesh_b, w, rng=np.random.default_rng(0))
    assert winner == 0, "the mesh with a FAILED thickness measurement must not win on that alone"
    print("  rank_candidates: a failed thickness measurement can no longer win the comparison")


def test_decimation_before_scoring():
    big = trimesh.creation.icosphere(subdivisions=5)
    assert len(big.faces) > 2000
    score = gj.score_mesh(big, gj.JudgeWeights(max_faces_for_scoring=2000))
    assert math.isfinite(score.total)
    print("  score_mesh: max_faces_for_scoring decimates before scoring without erroring")


TESTS = [
    test_overhang_sign_convention,
    test_topology_penalty_is_a_rate,
    test_thickness_unit_cube_semantics,
    test_thickness_tessellation_invariance,
    test_judge_weights_frozen,
    test_rank_candidates_nan_guard,
    test_rank_candidates_thickness_fairness,
    test_decimation_before_scoring,
]


def main():
    failures = []
    for fn in TESTS:
        print(f"{fn.__name__}:")
        try:
            fn()
        except AssertionError as e:
            failures.append((fn.__name__, e))
            print(f"  FAILED: {e}")
    print()
    if failures:
        print(f"{len(failures)}/{len(TESTS)} FAILED")
        return 1
    print(f"All {len(TESTS)} tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

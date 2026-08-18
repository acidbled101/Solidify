"""Tests for trellis_core.printability_proxy.

Two things are being asserted, and they are different:

  1. NUMERIC AGREEMENT with trellis_core.geometric_judge. The proxy is only
     interesting if it computes the judge's own quantity, not a lookalike.
  2. GRADIENT FLOW through backends.mesh_extract.flexible_dual_grid_to_mesh --
     the actual extraction routine the real decoder calls, not a stand-in.
     This is the claim the whole design rests on ("mesh extraction is not
     differentiable" is true of the connectivity and false of the vertex
     positions), so it is tested by running it, not by reading it.

Run: .venv/bin/python -m pytest trellis_core/printability_proxy_test.py -q
"""

import math

import numpy as np
import torch
import trimesh

from trellis_core import geometric_judge as gj
from trellis_core import printability_proxy as pp


def _random_mesh(seed=0, n=400):
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(n, 3)) * 0.2
    hull = trimesh.convex.convex_hull(trimesh.PointCloud(pts))
    # jitter so faces are not all clean hull planes
    v = np.asarray(hull.vertices) + rng.normal(size=hull.vertices.shape) * 0.01
    return trimesh.Trimesh(vertices=v, faces=np.asarray(hull.faces), process=False)


def _torch_mesh(mesh):
    return (torch.tensor(np.asarray(mesh.vertices), dtype=torch.float64),
            torch.tensor(np.asarray(mesh.faces), dtype=torch.long))


# ---------------------------------------------------------------------------
# 1. numeric agreement with the judge
# ---------------------------------------------------------------------------


def test_overhang_matches_judge():
    for seed in range(5):
        mesh = _random_mesh(seed)
        v, f = _torch_mesh(mesh)
        ours = float(pp.overhang_energy(v, f, 45.0))
        theirs = gj.overhang_penalty(mesh, 45.0)
        assert abs(ours - theirs) <= 1e-9 + 1e-7 * abs(theirs), (ours, theirs)


def test_overhang_matches_judge_at_other_thresholds():
    mesh = _random_mesh(3)
    v, f = _torch_mesh(mesh)
    for theta in (10.0, 30.0, 45.0, 60.0, 80.0):
        ours = float(pp.overhang_energy(v, f, theta))
        theirs = gj.overhang_penalty(mesh, theta)
        assert abs(ours - theirs) <= 1e-9 + 1e-7 * abs(theirs), (theta, ours, theirs)


def test_detail_matches_judge():
    for seed in range(5):
        mesh = _random_mesh(seed)
        v, f = _torch_mesh(mesh)
        edges = pp.unique_edges(f)
        # trimesh's edges_unique is the same set; assert that too, since the
        # judge's R_Detail is defined over it.
        assert edges.shape[0] == len(mesh.edges_unique)
        ours = float(pp.detail_energy(v, edges))
        theirs = gj.detail_reward(mesh)
        assert abs(ours - theirs) <= 1e-9 + 1e-7 * abs(theirs), (ours, theirs)


def test_overhang_division_free_form_survives_degenerate_triangles():
    """A sliver triangle has ||cross|| -> 0. The naive form n = cross/||cross||
    produces NaN gradients there; the division-free form must not."""
    v = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0],
                      [0.0, 1.0, 0.3]], dtype=torch.float64, requires_grad=True)
    f = torch.tensor([[0, 1, 2], [0, 1, 3]], dtype=torch.long)  # first is degenerate
    loss = pp.overhang_energy(v, f, 45.0)
    loss.backward()
    assert torch.isfinite(v.grad).all(), v.grad


def test_support_aware_forgives_the_build_plate():
    """The judge's known defect: a cube resting on the bed is charged for its
    bottom face. support_aware=True must remove that and leave the rest."""
    cube = trimesh.creation.box(extents=(0.4, 0.4, 0.4))
    cube.apply_translation([0, 0, 0.2])  # sits on z = 0
    v, f = _torch_mesh(cube)
    naive = float(pp.overhang_energy(v, f, 45.0))
    fixed = float(pp.overhang_energy(v, f, 45.0, support_aware=True,
                                     bed_clearance=0.01, bed_softness=0.002))
    assert naive > 0.0, "the judge really does charge a flat-on-bed face"
    assert abs(naive - gj.overhang_penalty(cube, 45.0)) < 1e-9
    # The bed face is forgiven by sigmoid(-clearance/softness) = sigmoid(-5),
    # i.e. 99.3% of the penalty removed. It is a SOFT gate on purpose: a hard
    # z-threshold has zero gradient everywhere, which is useless for steering.
    assert fixed < 0.01 * naive, (naive, fixed)

    # ...and it does not forgive a genuine overhang. Same cube lifted clear of
    # the plate: the bottom face is now unsupported and must still be charged.
    lifted = cube.copy()
    lifted.apply_translation([0, 0, 0.5])
    v2, f2 = _torch_mesh(lifted)
    still = float(pp.overhang_energy(v2, f2, 45.0, support_aware=True,
                                     bed_clearance=0.01, bed_softness=0.002,
                                     z_min=torch.tensor(0.0, dtype=torch.float64)))
    assert abs(still - naive) < 1e-9, (still, naive)


# ---------------------------------------------------------------------------
# 2. gradient flow through the REAL extraction routine
# ---------------------------------------------------------------------------


def _toy_dual_grid(n=6):
    """A dense n^3 block of voxels with per-voxel dual vertices and all three
    axis edges intersected -- the same tensor shapes fdg_vae.py hands to
    flexible_dual_grid_to_mesh."""
    coords = torch.stack(torch.meshgrid(
        torch.arange(n), torch.arange(n), torch.arange(n), indexing="ij"
    ), dim=-1).reshape(-1, 3).int()
    torch.manual_seed(0)
    logits = torch.randn(coords.shape[0], 3, dtype=torch.float32) * 0.5
    intersected = logits > 0
    return coords, logits, intersected


def test_gradient_flows_through_extraction():
    """THE LOAD-BEARING TEST.

    Runs backends/mesh_extract.py's flexible_dual_grid_to_mesh -- the exact
    function trellis2's FlexiDualGridVaeDecoder.forward calls -- and shows that
    d(overhang_energy)/d(raw decoder output) is finite and non-zero, despite
    the routine containing .item() calls, python dict lookups, boolean masking
    and a torch.where on a comparison.

    Those operations only ever touch INDICES. `mesh_vertices` is built by
    `(coords.float() + dual_vertices) * voxel_size + aabb[0]`, which is pure
    tensor arithmetic, so autograd walks straight back through it.
    """
    from backends.mesh_extract import flexible_dual_grid_to_mesh

    coords, _logits, intersected = _toy_dual_grid()
    # This is fdg_vae.py:115 verbatim, with h_raw standing in for h.feats.
    h_raw = torch.randn(coords.shape[0], 3, dtype=torch.float32) * 0.3
    h_raw.requires_grad_(True)
    voxel_margin = 0.5
    dual_vertices = (1 + 2 * voxel_margin) * torch.sigmoid(h_raw) - voxel_margin

    verts, faces = flexible_dual_grid_to_mesh(
        coords, dual_vertices, intersected, None,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], grid_size=6, train=False,
    )
    assert faces.shape[0] > 0
    assert verts.requires_grad, "mesh vertices must stay attached to the graph"
    assert verts.grad_fn is not None

    loss = pp.overhang_energy(verts, faces, 45.0)
    assert loss.item() > 0
    loss.backward()

    assert h_raw.grad is not None
    assert torch.isfinite(h_raw.grad).all()
    assert h_raw.grad.abs().max() > 0, "gradient reached the raw decoder output"


def test_support_mask_and_unsupported_energy():
    """A flat plate floating in the air is 100% unsupported; the same plate
    with a solid box underneath it is not."""
    plate = trimesh.creation.box(extents=(0.4, 0.4, 0.02))
    plate.apply_translation([0, 0, 0.5])
    v, f = _torch_mesh(plate)
    naive = float(pp.overhang_energy(v, f, 45.0))
    mask = pp.support_mask_from_raycast(plate, 45.0, bed_clearance=0.01, z_bed=0.0)
    lifted = float(pp.unsupported_overhang_energy(v, f, mask, 45.0))
    assert abs(lifted - naive) < 1e-9, (lifted, naive)

    pillar = trimesh.creation.box(extents=(0.4, 0.4, 0.5))
    pillar.apply_translation([0, 0, 0.245])
    both = trimesh.util.concatenate([plate, pillar])
    v2, f2 = _torch_mesh(both)
    mask2 = pp.support_mask_from_raycast(both, 45.0, bed_clearance=0.01)
    propped = float(pp.unsupported_overhang_energy(v2, f2, mask2, 45.0))
    naive2 = float(pp.overhang_energy(v2, f2, 45.0))
    assert naive2 > 0.05, naive2               # the judge charges plate + pillar bottom
    assert propped < 0.05 * naive2, (propped, naive2)  # nothing actually needs support


def test_gradient_flows_for_every_energy():
    from backends.mesh_extract import flexible_dual_grid_to_mesh

    coords, _logits, intersected = _toy_dual_grid()
    h_raw = (torch.randn(coords.shape[0], 3) * 0.3).requires_grad_(True)
    dual_vertices = 2.0 * torch.sigmoid(h_raw) - 0.5
    verts, faces = flexible_dual_grid_to_mesh(
        coords, dual_vertices, intersected, None,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], grid_size=6, train=False,
    )
    edges = pp.unique_edges(faces)
    corr = pp.RayCorrespondence(
        src_face=torch.arange(0, faces.shape[0], 7),
        hit_face=torch.arange(1, faces.shape[0], 7)[: len(range(0, faces.shape[0], 7))],
    )
    smask = torch.zeros(faces.shape[0], dtype=torch.bool)
    smask[::3] = True
    for name, loss in [
        ("overhang", pp.overhang_energy(verts, faces)),
        ("overhang_support_aware", pp.overhang_energy(verts, faces, support_aware=True)),
        ("unsupported_overhang", pp.unsupported_overhang_energy(verts, faces, smask)),
        ("detail", pp.detail_energy(verts, edges)),
        ("thickness", pp.thickness_energy(verts, faces, corr, d_min=0.02)),
        ("printability", pp.printability_objective(verts, faces, corr=corr)),
    ]:
        g = torch.autograd.grad(loss, h_raw, retain_graph=True, allow_unused=True)[0]
        assert g is not None, name
        assert torch.isfinite(g).all(), name
        assert g.abs().max() > 0, name


def test_ray_thickness_reproduces_a_known_slab():
    """Two parallel plates 0.05 apart: a ray fired inward from the top plate
    must measure 0.05."""
    v = torch.tensor([
        [0.0, 0.0, 0.05], [1.0, 0.0, 0.05], [0.0, 1.0, 0.05],   # top, normal +z
        [0.0, 0.0, 0.00], [1.0, 0.0, 0.00], [0.0, 1.0, 0.00],   # bottom
    ], dtype=torch.float64)
    f = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.long)
    corr = pp.RayCorrespondence(src_face=torch.tensor([0]), hit_face=torch.tensor([1]))
    d = pp.ray_thickness(v, f, corr)
    assert abs(float(d[0]) - 0.05) < 1e-12, d

    # and the penalty is the judge's formula on that distance
    got = float(pp.thickness_energy(v, f, corr, d_min=0.02))
    assert got == 0.0  # 0.05 > d_min -> no penalty
    got = float(pp.thickness_energy(v, f, corr, d_min=0.10))
    assert abs(got - ((0.10 - 0.05) / 0.10) ** 2) < 1e-12, got


def test_linearized_surrogate_is_first_order_exact():
    torch.manual_seed(0)
    x0 = torch.randn(50, 8, dtype=torch.float64)
    g = torch.randn(50, 8, dtype=torch.float64)
    x = x0 + torch.randn(50, 8, dtype=torch.float64) * 1e-3
    s = pp.linearized_surrogate(x, g, x0)
    assert abs(float(s) - float((g * (x - x0)).sum())) < 1e-12


if __name__ == "__main__":
    import sys
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:
                fails += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    sys.exit(1 if fails else 0)

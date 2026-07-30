"""
CPU-only, plain-torch unit tests for trellis_core/dpo_branch.py.

This repo has no pytest suite (test/ holds sample input images), so this is a
self-contained script: plain `assert`s plus a __main__ runner.

    python trellis_core/dpo_branch_test.py

What it can and cannot cover:

  * COVERED -- everything hardware-independent: the timestep schedule, the
    sampler-params split, the voxel adjacency construction, the differentiable
    detail proxy, the preference loss, the trust-region projection, and the
    end-to-end "gradients reach delta and nothing else" property of the
    differentiable Euler continuation.

  * NOT COVERED -- anything needing MPS or the real TRELLIS.2 weights:
    sample_shape_slat_with_dpo_branch() itself, SparseTensor interop,
    pipeline.decode_shape_slat(), and whether conv_none actually has a working
    autograd backward on Metal. Those must be validated on the Mac.

The stand-in sampler below re-implements FlowEulerSampler's undecorated helper
methods verbatim (from TRELLIS.2/trellis2/pipelines/samplers/flow_euler.py) so
the tests exercise the same arithmetic the real code will, and plain dense
[N, C] tensors stand in for SparseTensor.feats -- a faithful substitute,
because every SparseTensor operation dpo_branch relies on (__sub__, __rmul__,
.replace) is elementwise on feats with coords held fixed.
"""

import math
import os
import sys

# Runnable as `python trellis_core/dpo_branch_test.py` from the repo root or
# from anywhere: put the repo root on sys.path so `trellis_core` imports as a
# package (which is what triggers bootstrap.configure_env_and_paths()).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn

from trellis_core import dpo_branch


# ---------------------------------------------------------------------------
# Dummy stand-ins
# ---------------------------------------------------------------------------


class DummyFlowModel(nn.Module):
    """Single Linear over concatenated (x, t) returning a same-shaped velocity."""

    def __init__(self, channels: int):
        super().__init__()
        self.lin = nn.Linear(channels + 1, channels)

    def forward(self, x, t, cond=None, **kwargs):
        t_col = (t.reshape(-1, 1) / 1000.0).to(x.dtype)
        return self.lin(torch.cat([x, t_col], dim=1))


class DummySampler:
    """FlowEulerSampler's undecorated helpers only, copied verbatim."""

    def __init__(self, sigma_min: float = 1e-5):
        self.sigma_min = sigma_min

    def _v_to_xstart_eps(self, x_t, t, v):
        assert x_t.shape == v.shape
        eps = (1 - t) * v + x_t
        x_0 = (1 - self.sigma_min) * x_t - (self.sigma_min + (1 - self.sigma_min) * t) * v
        return x_0, eps

    def _inference_model(self, model, x_t, t, cond=None, **kwargs):
        t = torch.tensor([1000 * t] * x_t.shape[0], device=x_t.device, dtype=torch.float32)
        return model(x_t, t, cond, **kwargs)

    def _get_model_prediction(self, model, x_t, t, cond=None, **kwargs):
        pred_v = self._inference_model(model, x_t, t, cond, **kwargs)
        pred_x_0, pred_eps = self._v_to_xstart_eps(x_t=x_t, t=t, v=pred_v)
        return pred_x_0, pred_eps, pred_v


def cube_coords(n: int) -> torch.Tensor:
    """[n^3, 4] coords for a solid n x n x n block in batch 0."""
    r = torch.arange(n)
    grid = torch.stack(torch.meshgrid(r, r, r, indexing="ij"), dim=-1).reshape(-1, 3)
    return torch.cat([torch.zeros(grid.shape[0], 1, dtype=torch.long), grid], dim=1).int()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_build_t_pairs():
    pairs = dpo_branch.build_t_pairs(10)
    assert len(pairs) == 10
    assert math.isclose(pairs[0][0], 1.0)
    assert math.isclose(pairs[-1][1], 0.0)
    # contiguous and strictly descending
    for (t, t_prev), (nt, _) in zip(pairs, pairs[1:]):
        assert t > t_prev
        assert math.isclose(t_prev, nt)

    # rescale_t must reproduce flow_euler.sample()'s exact formula
    steps, rescale = 8, 3.0
    expected = np.linspace(1, 0, steps + 1)
    expected = rescale * expected / (1 + (rescale - 1) * expected)
    got = dpo_branch.build_t_pairs(steps, rescale)
    for i, (t, t_prev) in enumerate(got):
        assert math.isclose(t, expected[i], rel_tol=1e-12, abs_tol=1e-12)
        assert math.isclose(t_prev, expected[i + 1], rel_tol=1e-12, abs_tol=1e-12)
    print("  build_t_pairs: schedule matches flow_euler.sample() exactly")


def test_split_sampler_params():
    steps, rescale_t, extra = dpo_branch.split_sampler_params({
        "steps": 25,
        "rescale_t": 2.0,
        "verbose": True,
        "tqdm_desc": "x",
        "guidance_strength": 3.5,
        "guidance_interval": (0.0, 0.5),
        "some_future_mixin_kwarg": 7,
    })
    assert steps == 25 and rescale_t == 2.0
    # everything not consumed by sample() itself must pass straight through,
    # including kwargs from mixins we cannot know about ahead of time.
    assert extra == {
        "guidance_strength": 3.5,
        "guidance_interval": (0.0, 0.5),
        "some_future_mixin_kwarg": 7,
    }
    steps, rescale_t, extra = dpo_branch.split_sampler_params({})
    assert (steps, rescale_t, extra) == (50, 1.0, {})
    print("  split_sampler_params: unknown mixin kwargs pass through untouched")


def test_voxel_neighbor_pairs():
    # 2x2x2 block: each voxel has exactly 3 axis-aligned neighbours.
    coords = cube_coords(2)
    src, tgt = dpo_branch.voxel_neighbor_pairs(coords)
    assert src.shape == tgt.shape
    degree = torch.bincount(tgt, minlength=coords.shape[0])
    assert torch.all(degree == 3), degree

    # an isolated voxel far away must get degree 0, and must NOT be spuriously
    # matched by a borrow in the packed key (the -1 offset case).
    far = torch.tensor([[0, 500, 500, 500]], dtype=torch.int32)
    coords2 = torch.cat([coords, far], dim=0)
    src2, tgt2 = dpo_branch.voxel_neighbor_pairs(coords2)
    degree2 = torch.bincount(tgt2, minlength=coords2.shape[0])
    assert degree2[-1].item() == 0
    assert torch.all(degree2[:-1] == 3)

    # voxel at the origin (0,0,0): its -1 neighbours are out of range and must
    # not wrap onto anything real.
    origin_only = torch.tensor([[0, 0, 0, 0]], dtype=torch.int32)
    s3, t3 = dpo_branch.voxel_neighbor_pairs(origin_only)
    assert s3.numel() == 0 and t3.numel() == 0

    # multi-batch: voxels in different batches are never neighbours
    a = cube_coords(2)
    b = cube_coords(2).clone()
    b[:, 0] = 1
    src4, tgt4 = dpo_branch.voxel_neighbor_pairs(torch.cat([a, b], dim=0))
    degree4 = torch.bincount(tgt4, minlength=16)
    assert torch.all(degree4 == 3), degree4
    print("  voxel_neighbor_pairs: degrees correct, no key collisions, batch-isolated")


def test_slat_detail_proxy():
    coords = cube_coords(4)
    src, tgt = dpo_branch.voxel_neighbor_pairs(coords)
    n, c = coords.shape[0], 3

    # constant features -> zero Laplacian energy
    flat = torch.ones(n, c)
    assert float(dpo_branch.slat_detail_proxy(flat, src, tgt)) == 0.0

    # high-frequency checkerboard -> strictly positive, and larger than a
    # smooth linear ramp (this is what makes it a "detail" reward)
    parity = (coords[:, 1] + coords[:, 2] + coords[:, 3]).long() % 2
    checker = (parity.float() * 2 - 1).unsqueeze(1).repeat(1, c)
    ramp = coords[:, 1].float().unsqueeze(1).repeat(1, c) / 4.0
    s_checker = float(dpo_branch.slat_detail_proxy(checker, src, tgt))
    s_ramp = float(dpo_branch.slat_detail_proxy(ramp, src, tgt))
    assert s_checker > s_ramp > -1e-12, (s_checker, s_ramp)
    assert s_checker > 0.0

    # differentiable
    feats = torch.randn(n, c, requires_grad=True)
    dpo_branch.slat_detail_proxy(feats, src, tgt).backward()
    assert feats.grad is not None and float(feats.grad.abs().sum()) > 0

    # empty input must not explode
    empty_src, empty_tgt = dpo_branch.voxel_neighbor_pairs(torch.zeros(0, 4, dtype=torch.int32))
    assert float(dpo_branch.slat_detail_proxy(torch.zeros(0, c), empty_src, empty_tgt)) == 0.0
    print(f"  slat_detail_proxy: checker={s_checker:.4f} > ramp={s_ramp:.4f} > flat=0, grads flow")


def test_preference_loss_signs():
    s_ref = torch.tensor(1.0)
    higher = torch.tensor(2.0)
    lower = torch.tensor(0.0)

    # delta won: a bigger margin (higher proxy) must give a SMALLER loss
    assert float(dpo_branch.preference_loss(higher, s_ref, True, 1.0)) < \
           float(dpo_branch.preference_loss(lower, s_ref, True, 1.0))
    # delta lost: the ordering flips
    assert float(dpo_branch.preference_loss(lower, s_ref, False, 1.0)) < \
           float(dpo_branch.preference_loss(higher, s_ref, False, 1.0))
    # a zero margin is exactly -log(0.5) either way
    tie = float(dpo_branch.preference_loss(s_ref, s_ref, True, 1.0))
    assert math.isclose(tie, -math.log(0.5), rel_tol=1e-6)
    print("  preference_loss: sign flips with the judge's verdict; tie == -log 0.5")


def test_project_delta():
    d = torch.full((100, 4), 1.0, requires_grad=True)
    dpo_branch.project_delta_(d, 0.25)
    assert math.isclose(float(d.detach().pow(2).mean().sqrt()), 0.25, rel_tol=1e-5)
    # already inside the trust region -> untouched
    small = torch.full((10, 2), 0.01, requires_grad=True)
    before = small.detach().clone()
    dpo_branch.project_delta_(small, 0.5)
    assert torch.allclose(small.detach(), before)
    print("  project_delta_: RMS clamped to the trust region, no-op when inside")


def _chain(seed=0, n_cube=4, channels=4, steps=3):
    """Build a dummy (sampler, model, base_feats, adjacency, t_pairs) chain."""
    torch.manual_seed(seed)
    coords = cube_coords(n_cube)
    src, tgt = dpo_branch.voxel_neighbor_pairs(coords)
    model = DummyFlowModel(channels)
    sampler = DummySampler()
    base = torch.randn(coords.shape[0], channels)
    pairs = dpo_branch.build_t_pairs(10)[4 : 4 + steps]  # a t~0.6 branch window
    return sampler, model, base, src, tgt, pairs


def test_gradients_reach_delta_only():
    """(a) gradients flow into delta and NOT into the dummy model's params."""
    sampler, model, base, src, tgt, pairs = _chain()

    with dpo_branch.frozen_parameters(model):
        assert not any(p.requires_grad for p in model.parameters())
        delta = (torch.randn_like(base) * 0.02).requires_grad_(True)
        _x, pred_x0 = dpo_branch.differentiable_continuation(
            sampler, model, base + delta, pairs, None
        )
        assert pred_x0.requires_grad, "no autograd graph was built through the flow model"
        proxy = dpo_branch.slat_detail_proxy(pred_x0, src, tgt)
        proxy.backward()

    assert delta.grad is not None
    assert float(delta.grad.abs().sum()) > 0, "delta received a zero gradient"
    for name, p in model.named_parameters():
        assert p.grad is None, f"model parameter {name} received a gradient"
    # requires_grad flags restored on exit
    assert all(p.requires_grad for p in model.parameters())

    # and with grad disabled there is no graph at all
    _x2, pred2 = dpo_branch.differentiable_continuation(
        sampler, model, base, pairs, None, enable_grad=False
    )
    assert not pred2.requires_grad
    print("  differentiable_continuation: grad -> delta only; model params untouched")


def _run_steer(delta_won, seed=0, num_steps=8):
    sampler, model, base, src, tgt, pairs = _chain(seed=seed)
    torch.manual_seed(seed + 1)
    eps_b = torch.randn_like(base) * 0.02
    delta = eps_b.clone().requires_grad_(True)

    with dpo_branch.frozen_parameters(model):
        _x_ref, pred_ref = dpo_branch.differentiable_continuation(
            sampler, model, base, pairs, None, enable_grad=False
        )
        proxy_ref = dpo_branch.slat_detail_proxy(pred_ref, src, tgt).detach()

        def continue_fn(d):
            _x, pred = dpo_branch.differentiable_continuation(sampler, model, base + d, pairs, None)
            return dpo_branch.slat_detail_proxy(pred, src, tgt)

        losses, proxies, rms = dpo_branch.steer_delta(
            delta, continue_fn, proxy_ref, delta_won,
            beta=50.0, num_steps=num_steps, lr=0.005, max_rms=0.06,
        )
    return delta.detach(), eps_b, float(proxy_ref), losses, proxies, rms


def test_steer_delta_when_delta_wins():
    """(b) loss decreases and the proxy is pushed UP when delta is the winner."""
    delta, eps_b, proxy_ref, losses, proxies, rms = _run_steer(True)
    assert losses[-1] < losses[0], losses
    assert proxies[-1] > proxies[0], proxies
    assert all(r <= 0.06 + 1e-6 for r in rms), rms
    print(f"  steer_delta(winner=delta): loss {losses[0]:.5f} -> {losses[-1]:.5f}, "
          f"proxy {proxies[0]:.5f} -> {proxies[-1]:.5f} (ref {proxy_ref:.5f})")
    return delta, eps_b, losses, proxies


def test_steer_delta_when_delta_loses():
    """(c) delta gets pushed BACK / the proxy is driven DOWN when delta loses.

    Note on the literal wording of the requirement: with a signed-margin
    preference loss the loss value decreases in BOTH cases -- that is what
    minimizing -log sigma(beta * sign * margin) does. The observable that
    distinguishes winner from loser is the DIRECTION delta moves and the
    direction the proxy is driven, which is what is asserted here (and the
    winner/loser updates are checked to be anti-correlated below).
    """
    delta, eps_b, proxy_ref, losses, proxies, rms = _run_steer(False)
    assert proxies[-1] < proxies[0], proxies
    assert all(r <= 0.06 + 1e-6 for r in rms), rms
    print(f"  steer_delta(loser=delta):  loss {losses[0]:.5f} -> {losses[-1]:.5f}, "
          f"proxy {proxies[0]:.5f} -> {proxies[-1]:.5f} (ref {proxy_ref:.5f})")
    return delta, eps_b, losses, proxies


def test_winner_and_loser_updates_are_opposed():
    """The same starting delta must move in opposite directions for the two verdicts."""
    d_win, eps_b, _, _ = test_steer_delta_when_delta_wins()
    d_lose, eps_b2, _, _ = test_steer_delta_when_delta_loses()
    assert torch.allclose(eps_b, eps_b2), "the two runs must start from the same delta"

    upd_win = (d_win - eps_b).flatten()
    upd_lose = (d_lose - eps_b).flatten()
    cos = float(torch.dot(upd_win, upd_lose) / (upd_win.norm() * upd_lose.norm()))
    assert cos < -0.9, f"winner/loser updates should be near-antiparallel, got cos={cos:.4f}"
    print(f"  winner vs loser update direction: cos = {cos:.4f} (anti-parallel)")


def test_config_clamps():
    assert dpo_branch.DPOBranchConfig(t_branch=0.9).effective_t_branch() == 0.7
    assert dpo_branch.DPOBranchConfig(t_branch=0.1).effective_t_branch() == 0.3
    assert dpo_branch.DPOBranchConfig(t_branch=0.42).effective_t_branch() == 0.42
    cfg = dpo_branch.DPOBranchConfig(branch_noise_scale=0.04)
    assert cfg.effective_delta_lr() == 0.01
    assert dpo_branch.DPOBranchConfig(delta_lr=0.5).effective_delta_lr() == 0.5
    # judge weights must be per-config instances, not a shared mutable default
    a, b = dpo_branch.DPOBranchConfig(), dpo_branch.DPOBranchConfig()
    assert a.judge_weights is not b.judge_weights
    assert a.judge_weights.d_min == 0.02  # unit-cube units, not millimetres
    print("  DPOBranchConfig: t_branch clamped to [0.3, 0.7], lr derived, weights not shared")


def test_select_branch_window():
    """Table-test steps in {1,2,3,4,12} -- steps=1 used to crash downstream
    with an opaque IndexError before select_branch_window existed."""
    for steps in (1, 2, 3, 4, 12):
        pairs = dpo_branch.build_t_pairs(steps)
        i_branch, k = dpo_branch.select_branch_window(pairs, 0.5, 2)
        assert 0 <= i_branch <= len(pairs) - k, (steps, i_branch, k)
        assert 1 <= k <= len(pairs)
        assert len(pairs[i_branch : i_branch + k]) == k

    # nominal case (this repo's actual benchmarked step count): t_branch=0.5
    # must land EXACTLY on t=0.5.
    pairs12 = dpo_branch.build_t_pairs(12)
    i_branch, k = dpo_branch.select_branch_window(pairs12, 0.5, 2)
    assert (i_branch, k) == (6, 2)
    assert math.isclose(pairs12[i_branch][0], 0.5)

    # nearest-t search must not undershoot: at steps=12, t_branch=0.3, the
    # nearest schedule point is 0.333 (index 8), not 0.25 (index 9) -- a prior
    # "first start <= t_branch" version would have picked 0.25, silently
    # undershooting the requested branch point.
    i_branch, _ = dpo_branch.select_branch_window(pairs12, 0.3, 2)
    assert math.isclose(pairs12[i_branch][0], 1 / 3, rel_tol=1e-9), pairs12[i_branch][0]

    # steps=1: must not crash, and must return a usable (in-range) window
    pairs1 = dpo_branch.build_t_pairs(1)
    assert dpo_branch.select_branch_window(pairs1, 0.5, 2) == (0, 1)

    # empty schedule -> a clear error, not an opaque one several lines later
    try:
        dpo_branch.select_branch_window([], 0.5, 2)
        raise SystemExit("FAIL: empty t_pairs should raise ValueError")
    except ValueError:
        pass

    print("  select_branch_window: steps in {1,2,3,4,12} all produce valid, in-range windows")


def test_backfill_sampler_defaults():
    class FakeCfgSampler:
        def sample(self, model, noise, cond, neg_cond, steps=50, rescale_t=1.0,
                   guidance_strength=3.0, verbose=True, **kwargs):
            pass

    class FakeGuidanceIntervalSampler:
        def sample(self, model, noise, cond, neg_cond, steps=50, rescale_t=1.0,
                   guidance_strength=3.0, guidance_interval=(0.0, 1.0), verbose=True, **kwargs):
            pass

    filled = dpo_branch.backfill_sampler_defaults(FakeCfgSampler(), {})
    assert filled.get("guidance_strength") == 3.0
    assert "model" not in filled and "cond" not in filled and "steps" not in filled

    # must never override an explicitly supplied value
    filled2 = dpo_branch.backfill_sampler_defaults(FakeCfgSampler(), {"guidance_strength": 7.0})
    assert filled2["guidance_strength"] == 7.0

    filled3 = dpo_branch.backfill_sampler_defaults(FakeGuidanceIntervalSampler(), {})
    assert filled3.get("guidance_interval") == (0.0, 1.0)
    assert filled3.get("guidance_strength") == 3.0
    print("  backfill_sampler_defaults: fills missing guidance defaults, never overrides supplied values")


def test_sparse_conv_backend_restores_env():
    """The env override must be scoped and restored (TRELLIS.2 may be absent)."""
    import os
    os.environ["SPARSE_CONV_BACKEND"] = "flex_gemm"
    with dpo_branch.sparse_conv_backend("none"):
        assert os.environ["SPARSE_CONV_BACKEND"] == "none"
    assert os.environ["SPARSE_CONV_BACKEND"] == "flex_gemm"

    del os.environ["SPARSE_CONV_BACKEND"]
    with dpo_branch.sparse_conv_backend("none"):
        assert os.environ["SPARSE_CONV_BACKEND"] == "none"
    assert "SPARSE_CONV_BACKEND" not in os.environ
    print("  sparse_conv_backend: env var set inside, restored (incl. unset) outside")


TESTS = [
    test_build_t_pairs,
    test_split_sampler_params,
    test_voxel_neighbor_pairs,
    test_slat_detail_proxy,
    test_preference_loss_signs,
    test_project_delta,
    test_select_branch_window,
    test_backfill_sampler_defaults,
    test_config_clamps,
    test_sparse_conv_backend_restores_env,
    test_gradients_reach_delta_only,
    test_winner_and_loser_updates_are_opposed,
]


def main():
    torch.set_num_threads(1)
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

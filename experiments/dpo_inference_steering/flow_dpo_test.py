"""
CPU-only, plain-torch unit tests for experiments/dpo_inference_steering/flow_dpo.py.

Same self-contained-script convention as dpo_branch_test.py and
geometric_judge_test.py (this repo has no pytest suite):

    python trellis_core/flow_dpo_test.py

What these tests are FOR
------------------------
flow_dpo.py makes three claims that are easy to state and easy to get wrong.
Each has a test that would actually fail if the claim were false:

  1. "The score is an exact affine function of the velocity."
     -> test_score_identity_is_exact checks it against a closed-form Gaussian
        marginal, analytically, to ~1e-6. Not a plausibility check.

  2. "The SDE family preserves the ODE's marginals."
     -> test_sde_preserves_marginals integrates BOTH on an analytic field and
        asserts the endpoint means/stds agree while the per-sample endpoints
        do NOT. Asserting only the first would pass for a broken SDE that
        silently fell back to the ODE, so both directions are checked.

  3. "The cheap velocity-space loss equals the real trajectory loss."
     -> test_velocity_form_matches_trajectory_form derives nothing and
        assumes nothing: it computes both and compares. This is the test that
        catches a sign inversion, which is otherwise silent and trains the
        model backwards.

Plus the two the brief asks for directly: a DPO loss over a dummy batch of
preferred/rejected trajectories, and a gradient test showing the vector field
moves toward the preferred path and away from the rejected one.

test_existing_preference_loss_is_a_special_case ties this module to the code
already shipping in dpo_branch.py, so the two cannot silently diverge.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from . import flow_dpo
from .dpo_branch import preference_loss


# ---------------------------------------------------------------------------
# An analytic flow: p_0 = N(0, I) -> p_1 = N(mu, s^2 I) under the linear path.
# Everything about it is available in closed form, so tests can check exact
# values rather than "looks about right".
# ---------------------------------------------------------------------------

MU, SIGMA_DATA = 2.0, 0.5


def _a2(t):
    """Var of x_t = (1-t) x_0 + t x_1 with x_0 ~ N(0,I), x_1 ~ N(mu, s^2 I)."""
    return (1.0 - t) ** 2 + (t ** 2) * (SIGMA_DATA ** 2)


def analytic_velocity(x, t):
    """v(x,t) = E[x_1 - x_0 | x_t = x], exact for the Gaussian-to-Gaussian case.

    Jointly Gaussian, so the conditional expectation is the linear regression
        E[u|x] = E[u] + Cov(x,u)/Var(x) * (x - E[x])
    with Cov(x_t, x_1 - x_0) = t*s^2 - (1-t) and E[x_t] = t*mu.
    """
    t = t.view(-1, *([1] * (x.dim() - 1))) if torch.is_tensor(t) and t.dim() > 0 else t
    c = (t * SIGMA_DATA ** 2 - (1.0 - t)) / _a2(t)
    return MU + c * (x - t * MU)


def analytic_score(x, t):
    """grad log p_t(x) for p_t = N(t*mu, a_t^2 I)."""
    t = t.view(-1, *([1] * (x.dim() - 1))) if torch.is_tensor(t) and t.dim() > 0 else t
    return -(x - t * MU) / _a2(t)


# ---------------------------------------------------------------------------
# 1. The ODE <-> SDE mapping
# ---------------------------------------------------------------------------


def test_score_identity_is_exact():
    """score_from_velocity must reproduce the true score of the marginal.

    This is the identity that makes the SDE free (no second network). If it
    were only approximately right, the SDE drift would be biased and every
    downstream claim would inherit the bias.
    """
    path = flow_dpo.LinearPath()
    x = torch.randn(64, 3) * 1.5 + 0.4
    for t_val in [0.0, 0.1, 0.35, 0.5, 0.77, 0.9]:
        t = torch.full((64,), t_val)
        v = analytic_velocity(x, t)
        got = path.score_from_velocity(x, v, t)
        want = analytic_score(x, t)
        err = (got - want).abs().max()
        assert err < 1e-5, f"t={t_val}: score identity off by {err:.3e}"
    print("  score = -(x_t - t*v)/(1-t) exact to <1e-5 across t in [0, 0.9]")


def test_noise_recovery_is_exact():
    """x_0 = x_t - t*v, the other half of the affine relationship."""
    path = flow_dpo.LinearPath()
    x0, x1 = torch.randn(128, 2), torch.randn(128, 2) + MU
    for t_val in [0.0, 0.25, 0.6, 0.95]:
        t = torch.full((128,), t_val)
        xt = path.interpolate(x0, x1, t)
        v = path.target_velocity(x0, x1)
        err = (path.noise_from_velocity(xt, v, t) - x0).abs().max()
        assert err < 1e-5, f"t={t_val}: noise recovery off by {err:.3e}"
    print("  x_0 = x_t - t*v exact to <1e-5")


def test_sde_preserves_marginals():
    """The load-bearing test for the whole conceptual argument.

    Integrate the SAME analytic field as an ODE (sigma=0) and as an SDE
    (sigma>0). The theory predicts identical distributions but different
    sample paths. Both halves are asserted: a broken SDE that ignored its
    noise term would pass the first assertion and fail the second.
    """
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(7)
    x0 = torch.randn(20000, 1)

    gap = flow_dpo.sde_ode_marginal_gap(
        analytic_velocity, x0, steps=200, sigma=0.6, generator=g
    )

    # Same law: the endpoint mean and spread must agree.
    assert gap["mean_shift"] < 0.05, f"marginal mean drifted by {gap['mean_shift']:.4f}"
    assert 0.90 < gap["std_ratio"] < 1.10, f"marginal std ratio {gap['std_ratio']:.4f} != 1"

    # Different paths: per-sample endpoints must genuinely disagree, or the
    # "SDE" is a no-op and the test above proves nothing.
    assert gap["per_sample_rmse"] > 0.1, (
        f"per-sample RMSE {gap['per_sample_rmse']:.4f} too small -- "
        "the SDE collapsed to the ODE and this test is vacuous"
    )

    # And the ODE endpoint should actually be the data distribution.
    assert abs(gap["ode_mean_norm"] - MU) < 0.1, f"ODE endpoint mean {gap['ode_mean_norm']:.3f} != {MU}"
    print(
        f"  same law (mean shift {gap['mean_shift']:.4f}, std ratio {gap['std_ratio']:.3f}) "
        f"but different paths (per-sample RMSE {gap['per_sample_rmse']:.3f})"
    )


def test_sigma_zero_is_the_ode():
    """sigma=0 must return the plain velocity -- the ODE is a family member."""
    x, t = torch.randn(16, 3), torch.rand(16)
    v = torch.randn(16, 3)
    assert torch.equal(flow_dpo.sde_drift(v, x, t, sigma=0.0), v)
    print("  sigma=0 drift is exactly v (ODE recovered, not approximated)")


# ---------------------------------------------------------------------------
# 2. The DPO objectives
# ---------------------------------------------------------------------------


def test_dpo_loss_on_dummy_batch():
    """The requested mock suite: DPO loss over a dummy preferred/rejected batch.

    Checks the three properties any correct pairwise preference loss has:
      - it is finite and positive
      - a model that matches the winner better than the reference scores
        LOWER loss than one that matches the loser better
      - a model identical to the reference sits at exactly -log(0.5)
    """
    torch.manual_seed(0)
    B, D = 8, 5
    v_target_w, v_target_l = torch.randn(B, D), torch.randn(B, D)
    v_ref_w, v_ref_l = torch.randn(B, D), torch.randn(B, D)

    # A model that IS the reference: margin is exactly 0 -> loss = -log sigmoid(0)
    loss_tie = flow_dpo.dpo_loss_velocity(
        v_ref_w, v_ref_w, v_target_w, v_ref_l, v_ref_l, v_target_l, beta=0.5
    )
    assert abs(float(loss_tie) - -torch.log(torch.tensor(0.5))) < 1e-6, float(loss_tie)

    # A model that leans toward the winner vs one that leans toward the loser.
    lean_w = v_ref_w + 0.9 * (v_target_w - v_ref_w)
    lean_l = v_ref_l + 0.9 * (v_target_l - v_ref_l)
    loss_good = flow_dpo.dpo_loss_velocity(
        lean_w, v_ref_w, v_target_w, v_ref_l, v_ref_l, v_target_l, beta=0.5
    )
    loss_bad = flow_dpo.dpo_loss_velocity(
        v_ref_w, v_ref_w, v_target_w, lean_l, v_ref_l, v_target_l, beta=0.5
    )
    assert torch.isfinite(loss_good) and loss_good > 0
    assert loss_good < loss_tie < loss_bad, (
        f"ordering violated: good={float(loss_good):.4f} tie={float(loss_tie):.4f} bad={float(loss_bad):.4f}"
    )
    print(
        f"  loss(lean->winner)={float(loss_good):.4f} < tie={float(loss_tie):.4f} "
        f"< loss(lean->loser)={float(loss_bad):.4f}"
    )


def test_velocity_form_matches_trajectory_form():
    """The cheap velocity loss must equal the explicit Gaussian-density loss.

    dpo_loss_velocity is *derived* from dpo_loss_trajectory by substituting
    x_next = x_t + v_target*dt and cancelling the 1/(2 sigma^2 dt) prefactor
    into beta. That derivation is only worth anything if it is true, so this
    computes both and compares -- including that the sign convention agrees,
    which is the failure mode that silently inverts an alignment run.

    beta_traj = beta_vel * 2 * sigma^2 * dt is the prefactor being absorbed.
    """
    torch.manual_seed(3)
    B, D = 16, 4
    dt, sigma, beta_vel = 0.02, 0.7, 0.3
    path = flow_dpo.LinearPath()

    t = torch.full((B,), 0.4)
    x_t_w, x_t_l = torch.randn(B, D), torch.randn(B, D)
    v_target_w, v_target_l = torch.randn(B, D), torch.randn(B, D)
    v_theta_w, v_theta_l = torch.randn(B, D), torch.randn(B, D)
    v_ref_w, v_ref_l = torch.randn(B, D), torch.randn(B, D)

    # The observed next state IS the target-velocity step. That substitution
    # is what turns transition densities into velocity errors.
    x_next_w = x_t_w + flow_dpo.sde_drift(v_target_w, x_t_w, t, sigma, path) * dt
    x_next_l = x_t_l + flow_dpo.sde_drift(v_target_l, x_t_l, t, sigma, path) * dt

    # The exact prefactor. drift is affine in v with slope k(t), so
    #   logratio = -(dt * k^2 / (2 sigma^2)) * (err_theta - err_ref)
    # and matching the two forms needs beta_traj * dt * k^2 / (2 sigma^2) = beta_vel.
    k = float(flow_dpo.drift_velocity_slope(0.4, sigma))
    beta_traj = beta_vel * 2.0 * (sigma ** 2) / (dt * k ** 2)

    loss_traj = flow_dpo.dpo_loss_trajectory(
        v_theta_w, v_ref_w, x_t_w, x_next_w,
        v_theta_l, v_ref_l, x_t_l, x_next_l,
        t, beta=beta_traj, dt=dt, sigma=sigma, path=path,
    )
    loss_vel = flow_dpo.dpo_loss_velocity(
        v_theta_w, v_ref_w, v_target_w,
        v_theta_l, v_ref_l, v_target_l,
        beta=beta_vel,
    )
    err = abs(float(loss_traj) - float(loss_vel))
    assert err < 1e-4, f"forms disagree by {err:.3e}: traj={float(loss_traj):.6f} vel={float(loss_vel):.6f}"
    print(f"  trajectory form {float(loss_traj):.6f} == velocity form {float(loss_vel):.6f} (|d|={err:.2e}), k(0.4)={k:.4f}")


def test_beta_conversion_is_timestep_dependent():
    """The prefactor k(t)^2 is NOT constant in t -- a global beta is a compromise.

    This is the honest counterweight to "the SDE mapping is free". At sigma=0
    (the ODE) k == 1 everywhere and the velocity-space reduction is exact. The
    moment you inject the noise that gives the model a density, the conversion
    between the true trajectory loss and its cheap form becomes t-dependent,
    and it diverges as t -> 1.
    """
    sigma = 0.7
    ks = {t: float(flow_dpo.drift_velocity_slope(t, sigma)) for t in [0.1, 0.5, 0.9, 0.99]}
    assert all(abs(float(flow_dpo.drift_velocity_slope(t, 0.0)) - 1.0) < 1e-9
               for t in [0.1, 0.5, 0.9, 0.99]), "sigma=0 must give k==1 at every t"
    assert ks[0.99] > 10 * ks[0.1], f"k should blow up near t=1, got {ks}"
    print("  sigma=0: k==1 for all t (reduction exact).  sigma=0.7: k = "
          + ", ".join(f"{t}->{v:.2f}" for t, v in ks.items()) + "  (a global beta is a compromise)")


def test_existing_preference_loss_is_a_special_case():
    """Ties this module to the preference_loss already shipping in dpo_branch.py.

    dpo_branch's inference-time steering compares ONE perturbed candidate
    against ONE detached reference under a scalar proxy score, i.e. the
    single-pair, scalar-reward reduction of the general pairwise form:

        -log sigmoid(beta * (s_delta - s_ref))

    Constructing velocity tensors whose squared-error differences equal those
    scalars must reproduce dpo_branch's number exactly. If someone later
    changes the sign or the temperature convention in either file, this fails.
    """
    beta = 2.0
    s_delta, s_ref = torch.tensor(0.8, requires_grad=True), torch.tensor(0.3)

    existing = preference_loss(s_delta, s_ref, delta_branch_won=True, beta=beta)

    # Build a pair whose (err_theta - err_ref) difference is exactly -(s_delta - s_ref):
    # dpo_loss_velocity's inner bracket enters with a minus sign, so a
    # *lower* error on the winner maps to a *higher* proxy score.
    zero = torch.zeros(1, 1)
    v_target = torch.zeros(1, 1)
    v_theta_w = torch.zeros(1, 1)
    v_ref_w = (s_delta - s_ref).reshape(1, 1).sqrt()  # err_ref - err_theta = s_delta - s_ref
    general = flow_dpo.dpo_loss_velocity(
        v_theta_w, v_ref_w, v_target,
        zero, zero, v_target,          # loser pair contributes exactly 0
        beta=beta,
    )
    err = abs(float(existing.detach()) - float(general.detach()))
    assert err < 1e-6, f"dpo_branch.preference_loss={float(existing.detach()):.6f} != general form {float(general.detach()):.6f}"
    print(f"  dpo_branch.preference_loss reproduced exactly by the general form ({float(existing.detach()):.6f})")


def test_reference_free_gradient_is_per_sample_parallel():
    """Dropping the reference rescales the update per sample; it never rotates it.

    Working the derivative out by hand: both objectives differentiate the same
    bracket, and only err(theta, .) depends on theta, so

        dL/dtheta = c_i * 2 * (v*_l - v*_w),      c_i = positive sigmoid weight

    for BOTH forms. The reference only enters through the scalar c_i (it shifts
    where the sigmoid sits), so per-sample gradient directions are *exactly*
    parallel and only the per-sample weighting differs.

    That is the precise version of "reference-free is a cheaper approximation":
    it reweights which pairs get attention, it does not aim somewhere else.
    The batch-level cosine is <1 purely because of that reweighting, which is
    why this asserts per-sample parallelism and reports the batch cosine as an
    observation rather than pretending it should be 1.
    """
    torch.manual_seed(11)
    B, D = 32, 6
    v_target_w, v_target_l = torch.randn(B, D), torch.randn(B, D)
    v_ref_w, v_ref_l = torch.randn(B, D), torch.randn(B, D)

    theta = torch.randn(B, D, requires_grad=True)
    loss_ref = flow_dpo.dpo_loss_velocity(theta, v_ref_w, v_target_w, theta, v_ref_l, v_target_l, beta=0.4)
    (g_ref,) = torch.autograd.grad(loss_ref, theta)

    theta2 = theta.detach().clone().requires_grad_(True)
    loss_free = flow_dpo.dpo_loss_reference_free(theta2, v_target_w, theta2, v_target_l, beta=0.4)
    (g_free,) = torch.autograd.grad(loss_free, theta2)

    per_sample_cos = F.cosine_similarity(g_ref, g_free, dim=1)
    assert float(per_sample_cos.min()) > 1.0 - 1e-5, (
        f"per-sample gradients not parallel (min cos={float(per_sample_cos.min()):.6f})"
    )
    batch_cos = float(F.cosine_similarity(g_ref.flatten(), g_free.flatten(), dim=0))
    ratio = (g_free.norm(dim=1) / g_ref.norm(dim=1).clamp_min(1e-12))
    print(
        f"  per-sample cos = {float(per_sample_cos.min()):.6f} (exactly parallel); "
        f"batch cos {batch_cos:.3f}; per-sample magnitude ratio "
        f"{float(ratio.min()):.2f}-{float(ratio.max()):.2f} (the reweighting)"
    )


def test_implicit_reward_ranks_winner_above_loser():
    """DPO's implicit reward must rank the pair it was aligned on.

    There is NO unconditional per-sample guarantee, and it is worth being
    precise about why. Writing theta = v_ref + a(v_w - v_ref) with opposed
    targets v_l = -v_w and expanding r_w > r_l, everything cancels down to

        <v_ref, v_w>  <  ||v_w||^2

    -- a condition on the REFERENCE, not on how opposed the targets are. A
    reference that already over-shoots the winner's direction can be scored
    above the aligned model on the winner. So:

      - neutral reference (v_ref = 0): the condition is 0 < ||v_w||^2, true
        always -> a genuine per-sample guarantee.
      - arbitrary reference: holds only in distribution. ~88% here. Asserting
        100% would be asserting something false, and the first draft of this
        test did exactly that.

    This is the sharp edge under "DPO needs no reward model": the implicit
    reward is defined *relative to a reference*, so it inherits the
    reference's pathologies rather than escaping them.
    """
    torch.manual_seed(5)
    B, D = 256, 3

    # Neutral reference + opposed targets: per-sample guarantee.
    v_w = torch.randn(B, D)
    zero_ref = torch.zeros(B, D)
    theta = zero_ref + 0.8 * (v_w - zero_ref)
    r_w = flow_dpo.implicit_reward(theta, zero_ref, v_w, beta=0.5)
    r_l = flow_dpo.implicit_reward(theta, zero_ref, -v_w, beta=0.5)
    frac_neutral = float((r_w > r_l).float().mean())
    assert frac_neutral == 1.0, f"neutral reference should rank 100%, got {frac_neutral:.1%}"

    # Arbitrary reference: distributional only, and the derived condition
    # must predict exactly which samples fail.
    v_ref = torch.randn(B, D)
    theta2 = v_ref + 0.8 * (v_w - v_ref)
    r_w2 = flow_dpo.implicit_reward(theta2, v_ref, v_w, beta=0.5)
    r_l2 = flow_dpo.implicit_reward(theta2, v_ref, -v_w, beta=0.5)
    ranked = r_w2 > r_l2
    predicted = (v_ref * v_w).sum(dim=1) < (v_w ** 2).sum(dim=1)
    assert torch.equal(ranked, predicted), "derived condition does not predict the failures"
    assert float(r_w2.mean()) > float(r_l2.mean()), "mean reward ordering must still hold"
    print(
        f"  neutral reference: {frac_neutral:.0%} (per-sample guarantee); "
        f"arbitrary reference: {float(ranked.float().mean()):.1%}, and the derived "
        "condition <v_ref,v_w> < ||v_w||^2 predicts every failure exactly"
    )


# ---------------------------------------------------------------------------
# 3. The gradient actually moves the field the right way
# ---------------------------------------------------------------------------


def test_gradient_moves_field_toward_preferred_away_from_rejected():
    """The requested core test, stated as a measurable before/after.

    Take a real (tiny) velocity network, do ONE DPO step, and assert BOTH:
      - its distance to the preferred velocity DECREASED
      - its distance to the rejected velocity INCREASED

    The second assertion is the one that matters. A loss that merely fit the
    winner (plain supervised regression on the preferred sample) would pass
    the first and fail the second; only a genuinely *contrastive* objective
    pushes away from the rejected path.
    """
    torch.manual_seed(0)
    B, D = 64, 3
    net = torch.nn.Sequential(torch.nn.Linear(D + 1, 32), torch.nn.Tanh(), torch.nn.Linear(32, D))
    ref = torch.nn.Sequential(torch.nn.Linear(D + 1, 32), torch.nn.Tanh(), torch.nn.Linear(32, D))
    ref.load_state_dict(net.state_dict())
    for p in ref.parameters():
        p.requires_grad_(False)

    x = torch.randn(B, D)
    t = torch.rand(B, 1)
    inp = torch.cat([x, t], dim=1)
    v_pref = torch.randn(B, D)
    v_rej = torch.randn(B, D)

    def dist(net_out, target):
        return float(((net_out - target) ** 2).sum(dim=1).mean())

    with torch.no_grad():
        before_pref, before_rej = dist(net(inp), v_pref), dist(net(inp), v_rej)

    opt = torch.optim.SGD(net.parameters(), lr=0.05)
    for _ in range(20):
        opt.zero_grad()
        out = net(inp)
        with torch.no_grad():
            out_ref = ref(inp)
        loss = flow_dpo.dpo_loss_velocity(out, out_ref, v_pref, out, out_ref, v_rej, beta=0.05)
        loss.backward()
        opt.step()

    with torch.no_grad():
        after_pref, after_rej = dist(net(inp), v_pref), dist(net(inp), v_rej)

    assert after_pref < before_pref, f"distance to PREFERRED grew: {before_pref:.4f} -> {after_pref:.4f}"
    assert after_rej > before_rej, f"distance to REJECTED shrank: {before_rej:.4f} -> {after_rej:.4f}"
    print(
        f"  ||v-v_pref||^2 {before_pref:.3f} -> {after_pref:.3f} (down)   "
        f"||v-v_rej||^2 {before_rej:.3f} -> {after_rej:.3f} (up)"
    )


def test_gradient_vanishes_when_preference_is_tied():
    """Identical winner and loser => no preference information => no update.

    Guards against a loss that drifts on ties, which would mean the objective
    contains a term unrelated to the preference signal.
    """
    torch.manual_seed(2)
    B, D = 16, 4
    v_target = torch.randn(B, D)
    v_ref = torch.randn(B, D)
    theta = torch.randn(B, D, requires_grad=True)
    loss = flow_dpo.dpo_loss_velocity(theta, v_ref, v_target, theta, v_ref, v_target, beta=0.7)
    (g,) = torch.autograd.grad(loss, theta)
    assert g.abs().max() < 1e-6, f"tied preference produced a gradient of {float(g.abs().max()):.3e}"
    print("  tied preference produces exactly zero gradient")


# ---------------------------------------------------------------------------
# 4. Diagnostics used to make claims in the dashboard
# ---------------------------------------------------------------------------


def test_path_curvature_matches_analytic_second_derivative():
    """The straightness metric must return the true ||x''||^2, not a scaled one.

    The dashboard uses this number to decide whether DPO straightened
    anything, so it is checked against a closed form rather than a vibe. For
    x(t) = (t, 3t^2) the exact answer is ||x''||^2 = |(0, 6)|^2 = 36, and it
    must be 36 at ANY step count -- a discretisation-dependent answer would
    mean the dt normalisation is wrong (it was: dt^2 instead of dt^4, which
    understated curvature ~1000x at 33 steps while still looking plausible).
    """
    for steps in [17, 33, 129]:
        t = torch.linspace(0, 1, steps).view(-1, 1, 1)
        straight = t * torch.tensor([[[1.0, 2.0]]])
        assert float(flow_dpo.path_curvature(straight).max()) < 1e-6, "straight line must read 0"

        bent = torch.cat([t, (t ** 2) * 3.0], dim=2)
        kappa = float(flow_dpo.path_curvature(bent).mean())
        assert abs(kappa - 36.0) < 0.5, f"steps={steps}: curvature {kappa:.3f} != analytic 36.0"
    print("  straight line reads 0; x=(t,3t^2) reads 36.0 == analytic ||x''||^2, at 17/33/129 steps")


def test_tilted_density_ratio_is_normalised():
    """exp(r/beta) weights must average to 1 so they are usable as importance weights."""
    r = torch.randn(1000) * 2.0
    w = flow_dpo.tilted_density_ratio(r, beta=0.5)
    assert abs(float(w.mean()) - 1.0) < 1e-4, float(w.mean())
    assert float(w.min()) >= 0.0
    print(f"  tilt weights normalised (mean={float(w.mean()):.6f}, max={float(w.max()):.2f})")


def test_tortuosity_is_scale_free_and_curvature_is_not():
    """The metric bug that inverted a published conclusion, pinned as a test.

    path_curvature scales as (displacement)^2. Doubling the size of a path
    without changing its SHAPE must therefore quadruple raw curvature while
    leaving tortuosity untouched. If that invariance is ever lost, the
    straightness claims in experiments/dpo_inference_steering/THEORY.md become unsupported again.
    """
    t = torch.linspace(0, 1, 65).view(-1, 1, 1)
    bent = torch.cat([t, torch.sin(t * 3.14159) * 0.4], dim=2)

    k1 = float(flow_dpo.path_curvature(bent).mean())
    tort1 = float(flow_dpo.path_tortuosity(bent).mean())
    kn1 = float(flow_dpo.path_curvature_normalized(bent).mean())

    scaled = bent * 2.0  # identical shape, twice the size
    k2 = float(flow_dpo.path_curvature(scaled).mean())
    tort2 = float(flow_dpo.path_tortuosity(scaled).mean())
    kn2 = float(flow_dpo.path_curvature_normalized(scaled).mean())

    assert abs(k2 / k1 - 4.0) < 0.01, f"raw curvature should scale 4x, got {k2/k1:.3f}"
    assert abs(tort2 - tort1) < 1e-5, f"tortuosity must be scale-free, moved {tort1:.6f}->{tort2:.6f}"
    assert abs(kn2 - kn1) < 1e-4, f"normalised curvature must be scale-free, {kn1:.6f}->{kn2:.6f}"

    straight = t * torch.tensor([[[1.0, 2.0]]])
    assert abs(float(flow_dpo.path_tortuosity(straight).mean()) - 1.0) < 1e-5, "straight line must be 1.0"
    print(f"  doubling a path: raw kappa x{k2/k1:.2f} (confounded), "
          f"tortuosity {tort1:.4f}->{tort2:.4f} (invariant); straight line = 1.0")


TESTS = [
    test_score_identity_is_exact,
    test_noise_recovery_is_exact,
    test_sde_preserves_marginals,
    test_sigma_zero_is_the_ode,
    test_dpo_loss_on_dummy_batch,
    test_velocity_form_matches_trajectory_form,
    test_beta_conversion_is_timestep_dependent,
    test_existing_preference_loss_is_a_special_case,
    test_reference_free_gradient_is_per_sample_parallel,
    test_implicit_reward_ranks_winner_above_loser,
    test_gradient_moves_field_toward_preferred_away_from_rejected,
    test_gradient_vanishes_when_preference_is_tied,
    test_path_curvature_matches_analytic_second_derivative,
    test_tortuosity_is_scale_free_and_curvature_is_not,
    test_tilted_density_ratio_is_normalised,
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

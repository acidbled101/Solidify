"""Preference alignment for continuous flow models: the ODE->SDE mapping.

This module is the honest core of the "why does DPO work on a flow model"
question. It is deliberately free of any TRELLIS.2 type so it can be unit
tested on CPU with toy tensors (same rationale as
trellis_core/dpo_branch.py's steer_delta).

THE PROBLEM
-----------
DPO's derivation needs a *policy* -- a conditional distribution whose
likelihood ratio pi_theta/pi_ref is finite and differentiable. A flow model
trained by flow matching gives you a deterministic ODE

    dx/dt = v_theta(x, t),      x_0 ~ N(0, I),   x_1 ~ data

whose induced distribution over trajectories is a Dirac delta given x_0.
log pi_theta(tau) is then either +inf or -inf, so the DPO log-ratio is
undefined pointwise. You cannot just "apply DPO" to an ODE.

THE MAPPING
-----------
The escape is that the ODE is only one member of a whole family of processes
that share its time marginals p_t. For any sigma_t >= 0,

    dx = [ v(x,t) + (sigma_t^2 / 2) * s(x,t) ] dt + sigma_t dW        (1)

has the same p_t as the ODE, where s(x,t) = grad_x log p_t(x). Proof is one
line of Fokker-Planck: for dx = f dt + sigma dW,

    d_t p = -div(f p) + (sigma^2/2) laplacian(p)

and substituting f = v + (sigma^2/2) grad log p gives

    -div(v p) - (sigma^2/2) div(p grad log p) + (sigma^2/2) laplacian(p)
  = -div(v p) - (sigma^2/2) laplacian(p) + (sigma^2/2) laplacian(p)
  = -div(v p)

which is exactly the continuity equation the ODE satisfies. sigma_t = 0
recovers the ODE. So (1) is a family of samplers, all with identical
marginals, indexed by how much noise you inject.

WHY THAT IS USEFUL, AND WHAT IT COSTS
-------------------------------------
Take sigma_t > 0 and discretise (1) with step dt. The transition kernel is
Gaussian:

    pi_theta(x_{t+dt} | x_t) = N( x_t + drift_theta(x_t,t) dt , sigma_t^2 dt I )

Now log pi_theta is a finite quadratic, the DPO log-ratio is a *difference of
squared errors*, and everything is differentiable. That is the entire trick.

The honest caveat, stated here rather than buried: the equivalence in (1) is
at the level of **marginals p_t, not path measures**. The ODE and the SDE
induce genuinely different distributions over trajectories -- they agree on
where you are at time t, not on how you got there. DPO on the SDE therefore
optimises a preference over *SDE* paths and you then sample with... whichever
you choose. Any preference signal that depends on the path rather than the
endpoint is being optimised under a different process than the one you deploy.
For endpoint rewards (which is what a mesh judge is -- it scores the final
geometry, not the trajectory) the marginal-level equivalence at t=1 is exactly
what is needed and the gap closes. For path-dependent rewards it does not.
See sde_ode_marginal_gap() for a numerical measurement of this gap.

THE FREE LUNCH FOR LINEAR PATHS
-------------------------------
Equation (1) needs the score s(x,t), which normally means a second network.
For the linear/rectified path x_t = (1-t) x_0 + t x_1 it does not: the score
is an exact affine function of the velocity the model already predicts (see
score_from_velocity). So the SDE costs zero extra parameters and one extra
elementwise op. This is what makes the mapping practical rather than merely
true.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F

__all__ = [
    "LinearPath",
    "sde_drift",
    "drift_velocity_slope",
    "sde_transition",
    "gaussian_logprob",
    "trajectory_logratio",
    "dpo_loss_trajectory",
    "dpo_loss_velocity",
    "dpo_loss_reference_free",
    "implicit_reward",
    "tilted_density_ratio",
    "path_curvature",
    "path_tortuosity",
    "path_curvature_normalized",
    "sde_ode_marginal_gap",
]


# ---------------------------------------------------------------------------
# The probability path
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LinearPath:
    """The rectified-flow / linear interpolant path, t: 0 (noise) -> 1 (data).

        x_t = (1 - t) x_0 + t x_1,     x_0 ~ N(0, I),  x_1 ~ data
        v   = x_1 - x_0                (constant along a conditional path)

    Every identity below is exact for this path -- none is an approximation.
    TRELLIS.2's shape-SLat sampler uses this same linear interpolant (see
    build_t_pairs in trellis_core/dpo_branch.py), which is why this is the
    path worth being careful about.

    `eps` guards the t -> 1 endpoint where the conditional variance (1-t)^2
    collapses and the score diverges like 1/(1-t). That divergence is real,
    not a numerical artifact: at t=1 the marginal is the data distribution
    and its score is genuinely unbounded for a discrete/degenerate dataset.
    """

    eps: float = 1e-4

    def interpolate(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = self._broadcast(t, x1)
        return (1.0 - t) * x0 + t * x1

    def target_velocity(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """The flow-matching regression target. Independent of t for this path."""
        return x1 - x0

    def noise_from_velocity(
        self, x_t: torch.Tensor, v: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Recover x_0 from (x_t, v). Exact:

            x_t = (1-t) x_0 + t x_1  and  x_1 = v + x_0   =>   x_t = x_0 + t v
        """
        return x_t - self._broadcast(t, x_t) * v

    def score_from_velocity(
        self, x_t: torch.Tensor, v: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """grad_x log p_t(x_t), as an exact affine function of the velocity.

        Conditional on x_1, x_t ~ N(t x_1, (1-t)^2 I), so

            grad log p_t(x_t | x_1) = -(x_t - t x_1) / (1-t)^2
                                    = -(1-t) x_0 / (1-t)^2
                                    = -x_0 / (1-t)
                                    = -(x_t - t v) / (1-t)

        Taking the conditional expectation of both sides over p(x_1 | x_t)
        turns the left side into the marginal score and the right side into
        the same expression in the *marginal* velocity -- which is precisely
        what a flow-matching-trained v_theta regresses to. So the identity
        survives the lift from conditional to marginal, and no separate score
        network is ever needed.
        """
        t_b = self._broadcast(t, x_t)
        one_minus_t = (1.0 - t_b).clamp_min(self.eps)
        return -(x_t - t_b * v) / one_minus_t

    @staticmethod
    def _broadcast(t: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        """Right-pad a per-sample t with singleton dims so it broadcasts."""
        if not torch.is_tensor(t):
            t = torch.as_tensor(t, dtype=like.dtype, device=like.device)
        if t.dim() == 0:
            return t
        return t.view(t.shape[0], *([1] * (like.dim() - 1)))


# ---------------------------------------------------------------------------
# The marginal-preserving SDE family
# ---------------------------------------------------------------------------


def sde_drift(
    v: torch.Tensor,
    x_t: torch.Tensor,
    t: torch.Tensor,
    sigma: float,
    path: Optional[LinearPath] = None,
) -> torch.Tensor:
    """Drift of equation (1): v + (sigma^2 / 2) * score.

    sigma = 0 returns the plain ODE velocity, so the ODE really is the
    sigma -> 0 member of this family rather than a separate special case.
    """
    if sigma == 0.0:
        return v
    path = path or LinearPath()
    score = path.score_from_velocity(x_t, v, t)
    return v + 0.5 * (sigma ** 2) * score


def drift_velocity_slope(t, sigma: float, path: Optional[LinearPath] = None):
    """d(drift)/d(v) = 1 + sigma^2 t / (2 (1-t)).  Affine, so this is a scalar.

    Substituting score = -(x - t v)/(1-t) into drift = v + (sigma^2/2) score
    gives drift = k(t) v - (sigma^2/2) x/(1-t), so the drift is affine in v
    with slope k(t) above.

    This matters more than it looks. The practical velocity-space DPO loss is
    obtained from the true trajectory loss by absorbing a prefactor into
    beta -- and that prefactor contains k(t)^2. Since k(t) -> infinity as
    t -> 1, **a single global beta is exactly correct at only one timestep**
    whenever sigma > 0. At sigma = 0 (the ODE) k == 1 for all t and the
    reduction is exact everywhere.

    So the noise that made DPO applicable at all is the same noise that makes
    its cheap form t-inhomogeneous. That is a real cost of the ODE->SDE
    mapping, not a rounding error, and it is why flow_dpo_test pins the
    conversion at a stated t rather than pretending beta is free.
    """
    path = path or LinearPath()
    if not torch.is_tensor(t):
        t = torch.as_tensor(t, dtype=torch.float32)
    one_minus_t = (1.0 - t).clamp_min(path.eps)
    return 1.0 + (sigma ** 2) * t / (2.0 * one_minus_t)


def sde_transition(
    v: torch.Tensor,
    x_t: torch.Tensor,
    t: torch.Tensor,
    dt: float,
    sigma: float,
    path: Optional[LinearPath] = None,
) -> Tuple[torch.Tensor, float]:
    """One Euler-Maruyama step of (1) as a Gaussian: returns (mean, std).

    This is the object that makes DPO applicable at all -- the point where a
    deterministic model becomes a distribution with a tractable density.
    """
    mean = x_t + sde_drift(v, x_t, t, sigma, path) * dt
    std = sigma * math.sqrt(abs(dt))
    return mean, std


def gaussian_logprob(x: torch.Tensor, mean: torch.Tensor, std: float) -> torch.Tensor:
    """log N(x; mean, std^2 I), summed over all non-batch dims.

    The normalising constant is kept (rather than dropped as "it cancels")
    because dpo_loss_reference_free does NOT have a matching term to cancel
    against; only the reference-based ratio does.
    """
    if std <= 0:
        raise ValueError("gaussian_logprob needs std > 0; sigma=0 is the ODE, which has no density")
    d = x[0].numel() if x.dim() > 1 else x.numel()
    sq = ((x - mean) ** 2).flatten(start_dim=1).sum(dim=1) if x.dim() > 1 else ((x - mean) ** 2).sum()
    return -0.5 * sq / (std ** 2) - d * math.log(std * math.sqrt(2.0 * math.pi))


# ---------------------------------------------------------------------------
# DPO objectives
# ---------------------------------------------------------------------------


def trajectory_logratio(
    v_theta: torch.Tensor,
    v_ref: torch.Tensor,
    x_t: torch.Tensor,
    x_next: torch.Tensor,
    t: torch.Tensor,
    dt: float,
    sigma: float,
    path: Optional[LinearPath] = None,
) -> torch.Tensor:
    """log pi_theta(x_next | x_t) - log pi_ref(x_next | x_t), per sample.

    Both kernels share sigma, so the log-normalisers cancel exactly and this
    reduces to a difference of squared errors scaled by 1/(2 sigma^2 dt).
    """
    path = path or LinearPath()
    mean_theta, std = sde_transition(v_theta, x_t, t, dt, sigma, path)
    mean_ref, _ = sde_transition(v_ref, x_t, t, dt, sigma, path)
    return gaussian_logprob(x_next, mean_theta, std) - gaussian_logprob(x_next, mean_ref, std)


def dpo_loss_trajectory(
    v_theta_w: torch.Tensor, v_ref_w: torch.Tensor, x_t_w: torch.Tensor, x_next_w: torch.Tensor,
    v_theta_l: torch.Tensor, v_ref_l: torch.Tensor, x_t_l: torch.Tensor, x_next_l: torch.Tensor,
    t: torch.Tensor,
    *,
    beta: float,
    dt: float,
    sigma: float,
    path: Optional[LinearPath] = None,
) -> torch.Tensor:
    """DPO computed the *explicit* way: real Gaussian transition log-densities.

        L = -log sigmoid( beta * [ logratio(winner) - logratio(loser) ] )

    Slower and noisier than dpo_loss_velocity, and only usable because the
    SDE mapping gave the model a density at all. Its job here is to be the
    ground truth that the cheap form is checked against
    (see flow_dpo_test.test_velocity_form_matches_trajectory_form).
    """
    path = path or LinearPath()
    lr_w = trajectory_logratio(v_theta_w, v_ref_w, x_t_w, x_next_w, t, dt, sigma, path)
    lr_l = trajectory_logratio(v_theta_l, v_ref_l, x_t_l, x_next_l, t, dt, sigma, path)
    return -F.logsigmoid(beta * (lr_w - lr_l)).mean()


def dpo_loss_velocity(
    v_theta_w: torch.Tensor, v_ref_w: torch.Tensor, v_target_w: torch.Tensor,
    v_theta_l: torch.Tensor, v_ref_l: torch.Tensor, v_target_l: torch.Tensor,
    *,
    beta: float,
) -> torch.Tensor:
    """The practical Diffusion-DPO-style reduction, in velocity space.

        L = -log sigmoid( -beta * [ (||v_th - v*_w||^2 - ||v_rf - v*_w||^2)
                                   -(||v_th - v*_l||^2 - ||v_rf - v*_l||^2) ] )

    Reading it: the bracket is "how much worse than the reference is theta on
    the winner, minus how much worse it is on the loser". Driving the loss
    down pushes theta to fit the winner's velocity better than the reference
    does, and the loser's velocity worse -- which is the behaviour the unit
    tests assert directly.

    Derivation, not assertion: substitute x_next = x_t + v_target*dt into
    trajectory_logratio. Every term is a squared error against v_target and
    the 1/(2 sigma^2 dt) prefactor is a positive constant, absorbed into
    beta. sigma has cancelled out entirely -- which is why the practical form
    never needs you to pick one. The test named above verifies this
    numerically instead of taking my word for it.

    NOTE the sign: this bracket is a difference of *errors*, so it enters
    with a minus. Getting this backwards trains the model to prefer the
    rejected sample and is the single easiest way to silently invert an
    alignment run; test_velocity_form_matches_trajectory_form catches it.
    """
    def err(a, b):
        return ((a - b) ** 2).flatten(start_dim=1).sum(dim=1)

    inner_w = err(v_theta_w, v_target_w) - err(v_ref_w, v_target_w)
    inner_l = err(v_theta_l, v_target_l) - err(v_ref_l, v_target_l)
    return -F.logsigmoid(-beta * (inner_w - inner_l)).mean()


def dpo_loss_reference_free(
    v_theta_w: torch.Tensor, v_target_w: torch.Tensor,
    v_theta_l: torch.Tensor, v_target_l: torch.Tensor,
    *,
    beta: float,
    gamma: float = 0.0,
) -> torch.Tensor:
    """SimPO-style reference-free alignment: drop pi_ref, add a margin.

        L = -log sigmoid( -beta * [ ||v_th - v*_w||^2 - ||v_th - v*_l||^2 ] - gamma )

    Why this is worth having in *this* codebase specifically: the
    reference-based forms need a second frozen copy of the flow model and a
    second forward pass per preference pair. We measured the activation cost
    of one backward-enabled forward through TRELLIS.2's SLat flow model at
    ~9.5 GB (see checkpointed_blocks in trellis_core/dpo_branch.py, and the
    OOM it was written to fix). Dropping the reference halves the forwards
    that must be retained and removes a whole model's weights from memory.
    On a 36.27 GiB unified-memory ceiling that is the difference between
    running and not.

    What it costs, stated plainly: pi_ref is what bounds the KL to the base
    model. Without it nothing stops the aligned field drifting arbitrarily
    far to chase reward, so `gamma` (a fixed margin) and early stopping are
    doing the regularisation that a principled KL term used to do. That is a
    weaker guarantee, not an equivalent one. The toy experiment measures the
    consequence -- mode entropy collapse -- rather than assuming it away.
    """
    def err(a, b):
        return ((a - b) ** 2).flatten(start_dim=1).sum(dim=1)

    margin = err(v_theta_w, v_target_w) - err(v_theta_l, v_target_l)
    return -F.logsigmoid(-beta * margin - gamma).mean()


def implicit_reward(
    v_theta: torch.Tensor, v_ref: torch.Tensor, v_target: torch.Tensor, *, beta: float
) -> torch.Tensor:
    """DPO's implicit reward r(x) = beta * log(pi_theta/pi_ref), in velocity form.

    The reason DPO needs no reward model: r is *defined* by the policy pair,
    so the optimal policy of the KL-regularised objective,

        pi*(x) ~ pi_ref(x) exp(r(x)/beta),

    is reachable by optimising pi directly. There is no separate network to
    overfit, and no reward-model-vs-policy distribution mismatch, because the
    reward is a reparameterisation of the policy rather than an estimate of
    something external.

    This is exactly the quantity the dashboard's reward-shift histogram
    plots, so it is computed here rather than re-derived in JS.
    """
    def err(a, b):
        return ((a - b) ** 2).flatten(start_dim=1).sum(dim=1)

    return -beta * (err(v_theta, v_target) - err(v_ref, v_target))


def tilted_density_ratio(reward: torch.Tensor, beta: float) -> torch.Tensor:
    """exp(r/beta), normalised to mean 1 -- the importance weight from p to p*.

    Useful as a diagnostic: if this is near-uniform the alignment did nothing,
    and if it is near-one-hot the aligned model has collapsed onto whatever
    the reward likes most. Both failure modes are invisible in the loss curve.
    """
    w = torch.exp((reward - reward.max()) / max(beta, 1e-8))
    return w / w.mean().clamp_min(1e-12)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def path_curvature(traj: torch.Tensor) -> torch.Tensor:
    """Mean squared discrete acceleration along each trajectory.

        kappa = mean_i || x_{i+1} - 2 x_i + x_{i-1} ||^2 / dt^2

    Zero iff the path is a straight line traversed at constant speed. This is
    the standard straightness measure from the rectified-flow literature, and
    it exists here so the claim "DPO straightens probability paths" can be
    *measured* rather than asserted -- the toy experiment reports it before
    and after alignment, in both directions.

    `traj` is (steps, batch, dim). Returns per-sample curvature (batch,).

    Note the dt^4: the second difference approximates x'' * dt^2, so squaring
    it and recovering ||x''||^2 divides by dt^4, not dt^2. Getting this wrong
    understates curvature by a factor of 1/dt^2 (~1000x at 33 steps) while
    still looking like a plausible number, which is exactly why
    flow_dpo_test.test_path_curvature_matches_analytic_second_derivative
    checks it against a closed-form value instead of just "straight is 0".

    WARNING -- this quantity is NOT a measure of straightness on its own. It
    scales as (displacement)^2, so a model that moves its endpoints further
    away registers higher curvature even if its paths are straighter in shape.
    Preference alignment does exactly that when it concentrates mass, so
    comparing raw kappa between a base and an aligned model is confounded by
    the very mode collapse you are usually also measuring. Use
    path_tortuosity() (scale-free) for "did the paths get straighter", and
    keep this one only for a fixed endpoint distribution.

    Measured on this repo's own alignment run: raw kappa 9.45 -> 168.90
    (17.9x "worse") while tortuosity went 1.942 -> 1.615 (1.20x BETTER). The
    two disagree in direction. The first draft of report/flow_dpo_theory.md
    reported the raw ratio as evidence that DPO fails to straighten paths;
    that conclusion was an artifact of this scaling.
    """
    if traj.shape[0] < 3:
        return torch.zeros(traj.shape[1], device=traj.device, dtype=traj.dtype)
    dt = 1.0 / (traj.shape[0] - 1)
    accel = traj[2:] - 2.0 * traj[1:-1] + traj[:-2]
    return (accel ** 2).flatten(start_dim=2).sum(dim=2).mean(dim=0) / (dt ** 4)


def path_tortuosity(traj: torch.Tensor) -> torch.Tensor:
    """Arc length divided by straight-line distance. Scale-free; 1.0 == straight.

    The measure to use when asking "did alignment straighten the paths".
    Invariant to any global rescaling of the space, so unlike path_curvature it
    cannot be gamed by moving the endpoints further apart.

    `traj` is (steps, batch, dim). Returns per-sample tortuosity (batch,).
    """
    if traj.shape[0] < 2:
        return torch.ones(traj.shape[1], device=traj.device, dtype=traj.dtype)
    seg = (traj[1:] - traj[:-1]).flatten(start_dim=2).norm(dim=2).sum(dim=0)
    chord = (traj[-1] - traj[0]).flatten(start_dim=1).norm(dim=1)
    return seg / chord.clamp_min(1e-9)


def path_curvature_normalized(traj: torch.Tensor) -> torch.Tensor:
    """path_curvature divided by squared displacement -- the scale-free version.

    Reported alongside path_tortuosity because the two answer slightly
    different questions (pointwise bending vs total detour) and agreeing
    across both is what makes a straightness claim credible.
    """
    disp = (traj[-1] - traj[0]).flatten(start_dim=1).pow(2).sum(dim=1)
    return path_curvature(traj) / disp.clamp_min(1e-9)


@torch.no_grad()
def sde_ode_marginal_gap(
    velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    steps: int,
    sigma: float,
    path: Optional[LinearPath] = None,
    generator: Optional[torch.Generator] = None,
) -> dict:
    """Integrate the same field as an ODE and as an SDE; compare endpoints.

    The theory says the two agree in *distribution* at every t, not
    per-sample. This measures both, so the distinction is visible rather than
    glossed:

      - `per_sample_rmse`   should be LARGE. Different paths, same law.
      - `mean_shift` / `std_ratio` should be ~0 / ~1. That is the claim.

    Reporting only the first would make the mapping look broken; reporting
    only the second would hide that trajectory-level rewards are not covered
    by it. The dashboard shows both side by side.
    """
    path = path or LinearPath()
    dt = 1.0 / steps
    x_ode = x0.clone()
    x_sde = x0.clone()

    for i in range(steps):
        t = torch.full((x0.shape[0],), i * dt, device=x0.device, dtype=x0.dtype)
        v_ode = velocity_fn(x_ode, t)
        x_ode = x_ode + v_ode * dt

        v_sde = velocity_fn(x_sde, t)
        mean, std = sde_transition(v_sde, x_sde, t, dt, sigma, path)
        noise = torch.randn(x_sde.shape, device=x_sde.device, dtype=x_sde.dtype, generator=generator)
        x_sde = mean + std * noise

    return {
        "per_sample_rmse": float(((x_ode - x_sde) ** 2).mean().sqrt()),
        "mean_shift": float((x_ode.mean(dim=0) - x_sde.mean(dim=0)).norm()),
        "std_ratio": float(x_sde.std(dim=0).mean() / x_ode.std(dim=0).mean().clamp_min(1e-12)),
        "ode_mean_norm": float(x_ode.mean(dim=0).norm()),
        "sigma": sigma,
    }

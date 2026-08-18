"""Weight-space DPO for TRELLIS.2's shape-SLat flow model.

This is the DreamDPO/Diffusion-DPO objective in its *training* form: a frozen
reference copy of the flow model, a dataset of (condition, winner latent, loser
latent) triples, and gradient descent on the flow model's own weights. It
replaces the inference-time latent steering in `dpo_branch.py` entirely --
nothing here perturbs a mid-ODE latent, and no differentiable proxy for the
judge is needed, because the judge only ever produced the dataset labels
offline.


WHY THIS FORM AND NOT DREAMDPO'S EQ. 6
--------------------------------------
DreamDPO optimizes a *3D representation* theta against a frozen 2D diffusion
prior, so its gradient is the prior's own denoising error and the preference
only picks a sign. TRELLIS.2 has no such prior available at training time --
but it does not need one, because here the thing being optimized IS the
generative model. The reference model plays the role DreamDPO's prior plays:
it is the known-good direction, and the preference tilts away from it.

Concretely, DPO's implicit reward

    r(x) = beta * log( pi_theta(x) / pi_ref(x) )

is well defined the moment pi_theta has a density. A deterministic Euler ODE
does not (see flow_dpo.py section 1.1), so we use the standard diffusion-DPO
route: bound the trajectory log-ratio by its per-timestep regression form.
For a flow-matching model that bound is a difference of squared VELOCITY
errors, which is exactly what `flow_dpo.dpo_loss_velocity` derives, and it is
the same shape as DreamDPO's eq. 4 with `v` in place of `eps`.


PATH CONVENTION -- MUST MATCH THE SAMPLER
-----------------------------------------
Taken from TRELLIS.2/trellis2/pipelines/samplers/flow_euler.py, NOT from a
generic rectified-flow reference. t runs 1 (noise) -> 0 (data):

    x_t = (1 - t) * x_0 + (sigma_min + (1 - sigma_min) * t) * eps
    v   = (1 - sigma_min) * eps - x_0                       <- regression target

Check against the sampler's own `_v_to_xstart_eps`:
    eps = (1 - t) * v + x_t
        = (1-t)[(1-sm)eps - x_0] + (1-t)x_0 + (sm + (1-sm)t) eps
        = eps [ (1-sm)(1-t) + sm + (1-sm)t ] = eps                     OK

The model is called with t scaled by 1000 (`_inference_model`), so
`predict_velocity` below does the same. Getting either of these wrong silently
trains against a shifted schedule and looks like "DPO just doesn't help".


LATENT SPACE
------------
The flow model operates in NORMALIZED SLat space. `sample_shape_slat` applies
`slat = slat * std + mean` AFTER sampling, so a latent captured from the
pipeline output is de-normalized. Datasets built for this objective must store
`(slat - mean) / std`, i.e. what the flow model actually emits. Use
`normalize_slat` / `denormalize_slat` below rather than doing it by hand.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

# These primitives were extracted to training/flow_matching.py: the shipped
# trainer needed them and should not import from a retired experiment. The
# dependency now runs the correct way round -- this module depends on them.
from training.flow_matching import (  # noqa: F401  (re-exported for callers)
    noise_latent, velocity_target, sample_timesteps,
)
def normalize_slat(feats: torch.Tensor, normalization: Dict[str, Any]) -> torch.Tensor:
    """De-normalized SLat features -> the flow model's own space."""
    mean = torch.as_tensor(normalization["mean"], device=feats.device, dtype=feats.dtype)
    std = torch.as_tensor(normalization["std"], device=feats.device, dtype=feats.dtype)
    return (feats - mean[None]) / std[None]


def denormalize_slat(feats: torch.Tensor, normalization: Dict[str, Any]) -> torch.Tensor:
    mean = torch.as_tensor(normalization["mean"], device=feats.device, dtype=feats.dtype)
    std = torch.as_tensor(normalization["std"], device=feats.device, dtype=feats.dtype)
    return feats * std[None] + mean[None]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class FlowDPOConfig:
    """Knobs for the objective.

    beta:            preference temperature. Errors below are MEAN-reduced, so
                     beta must absorb the factor Diffusion-DPO writes as
                     `beta * T` -- with T=1000 timesteps their effective value
                     is O(2e3 - 5e3). A beta that is too small makes the
                     sigmoid linear (every pair contributes equally, including
                     the mislabelled ones); too large saturates it (only the
                     already-correct pairs contribute, and the gradient dies).
                     This is the single hyperparameter worth sweeping first.

    sigma_min:       MUST equal the pipeline's `shape_slat_sampler_params`
                     sigma_min. Read it off the loaded pipeline; do not guess.

    sft_weight:      coefficient on an auxiliary plain flow-matching loss on
                     the WINNER only (the RPO / "DPO+SFT" hybrid). Zero
                     reproduces vanilla DPO. Non-zero is strongly recommended
                     here: the DPO objective is invariant to degrading BOTH
                     branches as long as the loser degrades faster, and this
                     repo has already measured that exact failure -- reward
                     rising while L_Th degraded 7.4x, and all 3000 aligned
                     samples collapsing onto one geometry UNDER the
                     reference-based objective (report/flow_dpo_theory.md
                     2.3(b),(c)). The KL anchor did not prevent it because the
                     velocity-space surrogate contains no term penalising
                     distance from the reference. This term does.

    shared_noise:    draw ONE eps and ONE t and apply them to both the winner
                     and the loser. This is the training-time analogue of
                     DreamDPO's "same timestep, different noise" ablation and
                     of `geometric_judge.rank_candidates`' common random
                     numbers: it makes the comparison paired, cancelling the
                     (large) variance from the t/eps draw. Requires the pair to
                     share a sparse support -- see `assert_pairable`.

    t_sampling:      'logit_normal' (rectified-flow standard; concentrates
                     samples in the mid-schedule where the velocity actually
                     carries shape information) or 'uniform'.

    t_max:           clamp t away from 1. The exact trajectory log-ratio picks
                     up a k(t) = 1 + sigma^2 t / (2(1-t)) prefactor that the
                     per-timestep form absorbs into beta; measured k(0.99) =
                     25.26 (flow_dpo_test.test_beta_conversion_is_timestep_
                     dependent), so the near-t=1 samples are exactly where a
                     single global beta is most wrong.
    """

    beta: float = 5000.0
    sigma_min: float = 1e-5
    sft_weight: float = 0.0
    shared_noise: bool = True
    t_sampling: str = "logit_normal"
    t_min: float = 0.0
    t_max: float = 0.99
    logit_normal_mean: float = 0.0
    logit_normal_std: float = 1.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _per_sample_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    sample_index: Optional[torch.Tensor] = None,
    num_samples: Optional[int] = None,
) -> torch.Tensor:
    """Mean squared error reduced to one scalar PER SAMPLE.

    Mean- not sum-reduced, so `beta` means the same thing for a 2k-voxel object
    and a 40k-voxel one. With sum reduction the effective temperature would
    scale with token count and large objects would dominate every batch.

    `sample_index` maps rows of a sparse [N, C] feature tensor to their batch
    element (this is `SparseTensor.coords[:, 0]`). Without it, dim 0 is taken
    to be the batch dimension (the dense case).
    """
    se = (pred.float() - target.float()).pow(2)
    if sample_index is None:
        return se.reshape(se.shape[0], -1).mean(dim=1)

    per_token = se.reshape(se.shape[0], -1).mean(dim=1)
    n = int(num_samples if num_samples is not None else int(sample_index.max()) + 1)
    total = torch.zeros(n, device=per_token.device, dtype=per_token.dtype)
    total = total.index_add(0, sample_index, per_token)
    count = torch.zeros(n, device=per_token.device, dtype=per_token.dtype)
    count = count.index_add(0, sample_index, torch.ones_like(per_token))
    return total / count.clamp_min(1.0)


def predict_velocity(model, x_t, t: torch.Tensor, cond, **kwargs):
    """Call the flow model the way FlowEulerSampler._inference_model does.

    The x1000 scaling is not cosmetic -- `t_embedder` was trained against it.
    `t` is a per-sample tensor in [0, 1]; a SparseTensor's batch size is read
    off its coords rather than its feature rows.
    """
    batch = x_t.shape[0] if not hasattr(x_t, "coords") else int(x_t.coords[:, 0].max()) + 1
    t_model = (1000.0 * t).to(device=_device_of(x_t), dtype=torch.float32)
    if t_model.numel() == 1 and batch > 1:
        t_model = t_model.expand(batch)
    return model(x_t, t_model, cond, **kwargs)


def _device_of(x) -> torch.device:
    return x.feats.device if hasattr(x, "feats") else x.device


def _feats(x) -> torch.Tensor:
    return x.feats if hasattr(x, "feats") else x


def assert_pairable(x0_w, x0_l) -> None:
    """The winner and loser must live on the SAME sparse support.

    This is the structural precondition the whole objective rests on, and it is
    the one most easily violated when a dataset is built by running the full
    pipeline twice: `sample_sparse_structure` is a SEPARATE flow model, so two
    independent runs of the same prompt produce different voxel coordinate
    sets. Then N_w != N_l, `shared_noise` is impossible, and the two
    "densities" being compared are over different index sets -- the log-ratio
    is not a log-ratio of anything.

    Build pairs by sampling the structure stage ONCE per condition and drawing
    both candidates from the shape-SLat flow model on those fixed coords.
    """
    fw, fl = _feats(x0_w), _feats(x0_l)
    if fw.shape != fl.shape:
        raise ValueError(
            f"winner/loser latents have different shapes {tuple(fw.shape)} vs "
            f"{tuple(fl.shape)} -- they were almost certainly generated from "
            "different sparse-structure samples. Fix the dataset builder: "
            "sample coords once per condition, then sample the SLat flow "
            "model N times on those same coords."
        )
    if hasattr(x0_w, "coords") and hasattr(x0_l, "coords"):
        if not torch.equal(x0_w.coords, x0_l.coords):
            raise ValueError(
                "winner/loser latents share a shape but not a coordinate set; "
                "the pair is not a paired comparison."
            )


# ---------------------------------------------------------------------------
# The objective
# ---------------------------------------------------------------------------


@dataclass
class FlowDPOOutput:
    loss: torch.Tensor
    dpo_loss: torch.Tensor
    sft_loss: torch.Tensor
    margin: torch.Tensor              # the pre-sigmoid logit, per sample
    implicit_reward_w: torch.Tensor   # beta * (err_ref_w - err_theta_w)
    implicit_reward_l: torch.Tensor
    accuracy: torch.Tensor            # fraction with r_w > r_l
    err_theta_w: torch.Tensor
    err_theta_l: torch.Tensor
    err_ref_w: torch.Tensor
    err_ref_l: torch.Tensor

    def metrics(self) -> Dict[str, float]:
        return {
            "loss": float(self.loss.detach()),
            "dpo_loss": float(self.dpo_loss.detach()),
            "sft_loss": float(self.sft_loss.detach()),
            "margin": float(self.margin.detach().mean()),
            "reward_w": float(self.implicit_reward_w.detach().mean()),
            "reward_l": float(self.implicit_reward_l.detach().mean()),
            "reward_acc": float(self.accuracy.detach()),
            "err_theta_w": float(self.err_theta_w.detach().mean()),
            "err_theta_l": float(self.err_theta_l.detach().mean()),
            "err_ref_w": float(self.err_ref_w.detach().mean()),
            "err_ref_l": float(self.err_ref_l.detach().mean()),
        }


def flow_dpo_loss(
    model,
    x0_w,
    x0_l,
    cond,
    cfg: FlowDPOConfig,
    *,
    ref_model=None,
    ref_velocity_w: Optional[torch.Tensor] = None,
    ref_velocity_l: Optional[torch.Tensor] = None,
    t: Optional[torch.Tensor] = None,
    eps_w: Optional[torch.Tensor] = None,
    eps_l: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
    weight_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    pair_weight: Optional[torch.Tensor] = None,
    **model_kwargs,
) -> FlowDPOOutput:
    """DPO on the flow model's weights, from a preference pair of clean latents.

        L = -log sigma( -beta * [ (e_theta^w - e_ref^w) - (e_theta^l - e_ref^l) ] )

    where e_*^y = || v_*(x_t^y, t, c) - v_target^y ||^2, mean-reduced.

    Read the sign: minimising L drives the bracket NEGATIVE, i.e. it pushes
    e_theta^w below e_ref^w (the model gets better than the reference at
    reconstructing the winner) and e_theta^l above e_ref^l (worse than the
    reference at the loser). That is the whole mechanism, and it is DreamDPO's
    eq. 4 with velocity in place of noise and a real reference model in place
    of SimPO's margin.

    Args:
        model:        the flow model being trained (shape-SLat flow model).
        x0_w, x0_l:   clean NORMALIZED latents, winner and loser. SparseTensor
                      or dense tensor. Must be pairable (same support).
        cond:         conditioning, exactly as the sampler passes it -- for
                      TRELLIS.2 image-to-3D that is `pipeline.get_cond(img)['cond']`.
                      The SAME cond for both branches; a pair across different
                      conditions is not a preference pair.
        ref_model:    frozen reference. Optional if both `ref_velocity_*` are
                      supplied (see below).
        ref_velocity_w/l:
                      precomputed reference velocities for THIS (t, eps) draw.
                      Supplying these lets the reference model stay off the
                      device entirely -- the single biggest memory win
                      available here, since it removes one full model copy and
                      two forward passes from the step. It requires the (t,
                      eps) draws to be fixed per pair and cached alongside the
                      dataset; pass them back in via `t`/`eps_w`/`eps_l`.
        t, eps_w, eps_l:
                      override the random draws (for caching, or for tests).
        weight_fn:    optional w(t) timestep weighting, applied to the bracket.
        pair_weight:  per-pair scalar weight. Use it to weight by the judge's
                      score gap, or pass 0 to drop a pair -- this is where
                      DreamDPO's tau threshold lives in a dataset setting.

    Returns FlowDPOOutput; `.loss` is what you call .backward() on.
    """
    assert_pairable(x0_w, x0_l)

    fw, fl = _feats(x0_w), _feats(x0_l)
    device = fw.device
    sample_index = x0_w.coords[:, 0].long() if hasattr(x0_w, "coords") else None
    num_samples = int(sample_index.max()) + 1 if sample_index is not None else fw.shape[0]

    # ---- draws -----------------------------------------------------------
    if t is None:
        t = sample_timesteps(num_samples, cfg, device=device, dtype=torch.float32, generator=generator)
    t = t.to(device=device, dtype=torch.float32).reshape(-1)
    if t.numel() == 1:
        t = t.expand(num_samples)

    if eps_w is None:
        eps_w = torch.randn(fw.shape, device=device, dtype=fw.dtype, generator=generator)
    if eps_l is None:
        # Shared noise makes this a PAIRED comparison: the t/eps draw is by far
        # the largest variance component in the bracket, and sharing cancels it
        # exactly. DreamDPO's ablation reaches the same conclusion from the
        # other direction -- pairs that differ too much stop being informative.
        eps_l = eps_w if cfg.shared_noise else torch.randn(
            fl.shape, device=device, dtype=fl.dtype, generator=generator
        )

    # per-token t, so a batch can mix timesteps
    t_tok = t[sample_index].unsqueeze(1) if sample_index is not None else t.reshape(-1, *([1] * (fw.dim() - 1)))

    xt_w = noise_latent(fw, eps_w, t_tok, cfg.sigma_min)
    xt_l = noise_latent(fl, eps_l, t_tok, cfg.sigma_min)
    vt_w = velocity_target(fw, eps_w, cfg.sigma_min)
    vt_l = velocity_target(fl, eps_l, cfg.sigma_min)

    def _wrap(like, feats):
        return like.replace(feats) if hasattr(like, "replace") else feats

    # ---- policy ----------------------------------------------------------
    v_theta_w = _feats(predict_velocity(model, _wrap(x0_w, xt_w), t, cond, **model_kwargs))
    v_theta_l = _feats(predict_velocity(model, _wrap(x0_l, xt_l), t, cond, **model_kwargs))

    # ---- reference -------------------------------------------------------
    if ref_velocity_w is None or ref_velocity_l is None:
        if ref_model is None:
            raise ValueError("supply either ref_model or both ref_velocity_w and ref_velocity_l")
        with torch.no_grad():
            ref_velocity_w = _feats(predict_velocity(ref_model, _wrap(x0_w, xt_w), t, cond, **model_kwargs))
            ref_velocity_l = _feats(predict_velocity(ref_model, _wrap(x0_l, xt_l), t, cond, **model_kwargs))
    ref_velocity_w = ref_velocity_w.detach()
    ref_velocity_l = ref_velocity_l.detach()

    # ---- errors ----------------------------------------------------------
    e_theta_w = _per_sample_mse(v_theta_w, vt_w, sample_index, num_samples)
    e_theta_l = _per_sample_mse(v_theta_l, vt_l, sample_index, num_samples)
    e_ref_w = _per_sample_mse(ref_velocity_w, vt_w, sample_index, num_samples)
    e_ref_l = _per_sample_mse(ref_velocity_l, vt_l, sample_index, num_samples)

    bracket = (e_theta_w - e_ref_w) - (e_theta_l - e_ref_l)
    if weight_fn is not None:
        bracket = bracket * weight_fn(t).to(bracket.dtype)

    logits = -cfg.beta * bracket
    per_pair = -F.logsigmoid(logits)
    if pair_weight is not None:
        per_pair = per_pair * pair_weight.to(per_pair.dtype).reshape(-1)
    dpo_loss = per_pair.mean()

    # ---- optional anchor -------------------------------------------------
    # Plain flow matching on the winner. Without it the objective is invariant
    # to a uniform degradation of both branches (only the DIFFERENCE appears),
    # which is the documented Diffusion-DPO failure and the one already
    # measured on this repo's 2-D experiment.
    sft_loss = e_theta_w.mean() if cfg.sft_weight else torch.zeros((), device=device)
    loss = dpo_loss + cfg.sft_weight * sft_loss

    r_w = cfg.beta * (e_ref_w - e_theta_w)
    r_l = cfg.beta * (e_ref_l - e_theta_l)

    return FlowDPOOutput(
        loss=loss,
        dpo_loss=dpo_loss.detach(),
        sft_loss=sft_loss.detach(),
        margin=logits.detach(),
        implicit_reward_w=r_w.detach(),
        implicit_reward_l=r_l.detach(),
        accuracy=(r_w > r_l).float().mean().detach(),
        err_theta_w=e_theta_w.detach(),
        err_theta_l=e_theta_l.detach(),
        err_ref_w=e_ref_w.detach(),
        err_ref_l=e_ref_l.detach(),
    )


# ---------------------------------------------------------------------------
# Smoke test -- run this file directly
# ---------------------------------------------------------------------------


def _smoke() -> None:
    torch.manual_seed(0)
    cfg = FlowDPOConfig(beta=100.0, sigma_min=1e-5, sft_weight=0.0, shared_noise=True)

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(8, 8)

        def forward(self, x, t, cond, **kw):
            return self.lin(x) + t.reshape(-1, 1)[:, :1] * 0.0

    model, ref = Toy(), Toy()
    ref.load_state_dict(model.state_dict())
    for p in ref.parameters():
        p.requires_grad_(False)

    x_w = torch.randn(4, 8)
    x_l = torch.randn(4, 8)
    cond = torch.zeros(4, 1)

    # 1. theta == ref  =>  bracket == 0  =>  loss == log 2
    out = flow_dpo_loss(model, x_w, x_l, cond, cfg, ref_model=ref)
    assert abs(float(out.loss) - torch.log(torch.tensor(2.0))) < 1e-4, float(out.loss)
    assert abs(float(out.margin.mean())) < 1e-4
    print(f"identical models -> loss {float(out.loss):.6f} (log 2 = 0.693147)  OK")

    # 2. descending the objective must widen the margin on a FIXED draw
    fixed_t, fixed_eps = torch.full((4,), 0.5), torch.randn(4, 8)
    kw = dict(ref_model=ref, t=fixed_t, eps_w=fixed_eps)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    first = flow_dpo_loss(model, x_w, x_l, cond, cfg, **kw)
    for _ in range(50):
        opt.zero_grad(set_to_none=True)
        flow_dpo_loss(model, x_w, x_l, cond, cfg, **kw).loss.backward()
        opt.step()
    last = flow_dpo_loss(model, x_w, x_l, cond, cfg, **kw)
    print(f"margin {float(first.margin.mean()):+.4f} -> {float(last.margin.mean()):+.4f}, "
          f"loss {float(first.loss):.4f} -> {float(last.loss):.4f}")
    assert float(last.margin.mean()) > float(first.margin.mean()), "margin did not widen"
    assert float(last.loss) < float(first.loss), "loss did not fall"
    assert float(last.accuracy) == 1.0, "implicit reward does not rank winner above loser"
    print("gradient direction OK (implicit reward now ranks winner above loser)")

    # 3. the documented pathology: vanilla DPO is invariant to degrading BOTH
    #    branches, so the winner's ABSOLUTE error is free to rise. This is not
    #    a bug in the loss -- it is why sft_weight exists.
    d_w = float(last.err_theta_w.mean() - last.err_ref_w.mean())
    d_l = float(last.err_theta_l.mean() - last.err_ref_l.mean())
    print(f"  err_w vs ref {d_w:+.4f}, err_l vs ref {d_l:+.4f}  (only d_w < d_l is guaranteed)")
    assert d_w < d_l

    # 4. ... and the anchor term controls it
    model_sft, opt_sft = Toy(), None
    model_sft.load_state_dict(ref.state_dict())
    opt_sft = torch.optim.SGD(model_sft.parameters(), lr=1e-3)
    cfg_sft = FlowDPOConfig(beta=100.0, sigma_min=1e-5, sft_weight=1.0, shared_noise=True)
    for _ in range(50):
        opt_sft.zero_grad(set_to_none=True)
        flow_dpo_loss(model_sft, x_w, x_l, cond, cfg_sft, **kw).loss.backward()
        opt_sft.step()
    anchored = flow_dpo_loss(model_sft, x_w, x_l, cond, cfg_sft, **kw)
    drift_plain = float(last.err_theta_w.mean() - last.err_ref_w.mean())
    drift_anchor = float(anchored.err_theta_w.mean() - anchored.err_ref_w.mean())
    print(f"  winner-error drift: sft_weight=0 {drift_plain:+.4f}  vs  sft_weight=1 {drift_anchor:+.4f}")
    assert drift_anchor < drift_plain, "anchor did not reduce winner drift"

    # 5. pairability is enforced
    try:
        flow_dpo_loss(model, x_w, torch.randn(4, 6), cond, cfg, ref_model=ref)
    except ValueError as e:
        print(f"mismatched support rejected OK: {str(e)[:60]}...")
    else:
        raise AssertionError("mismatched support was not rejected")

    print("\nall smoke checks passed")


if __name__ == "__main__":
    _smoke()

"""
Phase 2/3 of 3D_PRINTING_AWARE_PIPELINE.md: flow-matching interception plus a
DreamDPO-style, inference-time preference step on the shape-SLat trajectory.

This module is a drop-in alternative to
`Trellis2ImageTo3DPipeline.sample_shape_slat` specifically -- the method used
by the '512' and '1024' (non-cascade) pipeline_type paths
(trellis2_image_to_3d.py's run(), ~line 546-565). generate.py's own default is
'1024_cascade', which calls the differently-shaped `sample_shape_slat_cascade`
(two flow models, an upsample step, a different return signature) instead --
adapting this module for the cascade path is unstarted follow-up work, not
covered here. It runs the ordinary Euler flow sampler down to a
branch timestep, forks two candidate trajectories, ranks their decoded meshes
with trellis_core.geometric_judge, uses that discrete preference to steer a
small learnable latent perturbation by gradient descent, and then resumes the
ordinary full-speed sampler for the remaining steps.

Nothing here is wired into trellis_core/pipeline.py's run_generation() -- that
integration is deliberately a later phase. The signature below mirrors
sample_shape_slat's so it can be substituted without redesign.


STATED INTERPRETATION OF THE PLAN (please sanity-check this choice)
------------------------------------------------------------------
The plan's Phase 3 writes the loss in literal language-model DPO notation:

    L_DPO(pi) = -log sigma( beta * log P_pi(y_w)/P_ref(y_w)
                          - beta * log P_pi(y_l)/P_ref(y_l) )

Two things make that formula un-implementable as written in this setting:

1. A deterministic Euler ODE has no trajectory likelihood. `P_pi(y)` is not
   defined for a flow-matching sampler run at inference time -- there is no
   stochastic policy whose log-density we could evaluate on a decoded mesh.
2. The plan explicitly frames the whole thing as *inference-time* preference
   optimization: no training loop, no optimizer over model weights, no
   persisted checkpoint. Diffusion-DPO (Wallace et al.) -- the actual
   weight-space version of this loss -- requires a frozen reference *copy* of
   the flow model, a real training loop, and a dataset of preference pairs.
   Implementing that here would contradict the plan's own framing (and would
   not fit in unified memory on the target Mac alongside the live model).

So this module implements **inference-time latent steering**, not weight
fine-tuning. The only tensor that ever carries `requires_grad=True` is a
per-voxel latent perturbation `delta` at the branch point; every flow-model
parameter is explicitly frozen for the duration and restored afterwards.

The DPO formula is mapped onto that setting term by term:

    P_pi(y)   ->  s(x_B)   the differentiable proxy score of the *steerable*
                           branch (the one carrying `delta`)
    P_ref(y)  ->  s(x_A)   the same proxy on the *reference* branch (delta-free,
                           detached -- it is exactly what the vanilla sampler
                           would have produced)
    log ratio ->  s(x_B) - s(x_A.detach())
    y_w / y_l ->  a sign, supplied by geometric_judge.rank_candidates()

Because there is only ONE steerable branch (not a policy and a reference model
evaluated on both y_w and y_l), the two log-ratio terms of the DPO formula
collapse into a single signed margin:

    sign  = +1 if the judge preferred the delta branch, else -1
    L     = -log sigma( beta_dpo * sign * (s(x_B) - s(x_A).detach()) )

Minimizing L therefore performs gradient *ascent* on the proxy when the judge
picked the delta branch, and gradient *descent* when it picked the reference
branch. Read that descent case literally: it drives s(x_B) DOWN, in whatever
direction -grad(s) happens to point for the CURRENT delta -- it is not a
restoring force back toward delta=0 / eps_A. If the losing eps_B already
happened to lower the detail proxy, "descent" amplifies it rather than
reversing it. The hard RMS trust region (project_delta_) is what keeps this
bounded, not the loss.

The proxy `s` is `slat_detail_proxy()` below: the umbrella-Laplacian energy of
the SLat features over the voxel grid's own 6-neighbour adjacency. It is the
latent-space analogue of geometric_judge.detail_reward()'s R_Detail (which is
the same operator on mesh vertex positions over mesh edges) -- deliberately so,
because R_Detail is the one term of the judge's composite score S that has a
meaningful, differentiable counterpart in SLat space. Overhang, wall thickness
and topology only exist after mesh extraction, which is non-differentiable
(backends/mesh_extract.py does `.item()` and Python dict lookups on CPU). Those
three terms therefore enter only through the judge's discrete verdict, i.e.
through the *sign* of the margin, never through the gradient.

HONEST LIMITATIONS (expanded after independent review -- read this section,
it's the most important one)
  * This is an approximation of the plan's DPO formula, not an implementation
    of Diffusion-DPO's weight-space loss.
  * THE GRADIENT SIGNAL AND THE JUDGE'S REASONING ARE LIKELY UNCORRELATED, NOT
    JUST "PARTIAL". The gradient only ever comes from slat_detail_proxy
    (R_Detail's latent analogue). But S = alpha*R_Detail - (beta*L_OH +
    gamma*L_Th + delta*L_Topo) is dominated by L_Topo in practice: a mid-ODE
    voxel mesh can carry 10^2-10^4 boundary edges (plan's own "Topological
    Stalls" risk row), and even with topology_penalty() now reporting a
    per-EDGE RATE (not a raw count -- see geometric_judge.py) and delta
    softened to 0.05 here, L_Topo can still dominate S for a noisy
    mid-generation mesh. In the modal case the judge's verdict is mostly "which
    branch has fewer open boundaries" -- a fact about mesh-extraction topology
    that has NO reason to correlate with, and could easily anti-correlate with
    (more high-frequency latent detail plausibly means MORE micro-holes at a
    given decode resolution), the direction slat_detail_proxy's gradient
    pushes delta in. Nothing here measures or guarantees that correlation.
    What actually keeps this from being harmful is the re-ranking gate at the
    end (the optimized branch is re-decoded and re-judged against the
    reference; if it lost, we resume from the reference instead) -- so the
    worst case is wasted compute, not a worse output. Read the steering step
    as "a detail-energy-biased proposal, kept only if the judge independently
    prefers it after the fact" -- NOT as gradient descent toward what the
    judge rewarded. If you build on this, instrument
    DPOBranchReport.delta_branch_won_initial vs .delta_branch_won_final across
    many runs; if the optimized branch doesn't win noticeably more than the
    unoptimized initial candidate did, the proxy gradient is contributing
    nothing and the branch step reduces to expensive re-sampling.
  * The judge is run ONCE per branch point to obtain the preference label; the
    label is then held fixed across the `num_delta_grad_steps` gradient steps
    (re-decoding a mesh per gradient step would cost a full decoder forward
    each time, which the plan's own risk table rules out on unified memory).
  * NOT VERIFIED ON REAL HARDWARE. Every TRELLIS.2 call site here was written
    against the source in TRELLIS.2/ but never executed -- there is no MPS
    device or model checkpoint on the machine this was written on. Points
    flagged `UNVERIFIED` in comments below need confirming on the Mac.
"""

import contextlib
import importlib
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import bootstrap  # noqa: F401  (ensures env/path setup already ran)

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from tqdm import tqdm

from . import geometric_judge


# ---------------------------------------------------------------------------
# Config / result
# ---------------------------------------------------------------------------


def _default_judge_weights() -> geometric_judge.JudgeWeights:
    """Judge weights calibrated for MID-GENERATION latent-space meshes.

    A decoded SLat mesh lives in the [-0.5, 0.5] unit cube, not in millimetres.
    geometric_judge.JudgeWeights' own default is already unit-cube-normalized
    (d_min=0.01, ~a 1mm wall on a 100mm print -- see that module's docstring),
    but 0.02 is used here instead: mid-ODE candidate meshes are coarser and
    noisier than a final decode, so a slightly more forgiving threshold avoids
    flagging every wall as too thin before the shape has converged.

    delta (the topology weight) is also softened from JudgeWeights' own
    default: per the plan's "Topological Stalls" risk row, mid-ODE voxel
    meshes are riddled with micro-holes. geometric_judge.topology_penalty()
    now reports L_Topo as a per-edge RATE rather than a raw count (see that
    function's docstring), which already brings it to the same rough order of
    magnitude as R_Detail/L_OH/L_Th -- delta=0.05 further de-emphasizes it on
    top of that, specifically for this noisier mid-generation setting.
    """
    return geometric_judge.JudgeWeights(d_min=0.02, delta=0.05)


@dataclass
class DPOBranchConfig:
    """Knobs for the branch + preference step.

    t_branch:            ODE timestep to fork at. The plan specifies t in
                         [0.3, 0.7]; values outside are clamped to that range.
    branch_noise_scale:  std of the gaussian perturbation seeding the delta
                         branch, in normalized SLat units.
    continuation_steps:  how many Euler steps to run differentiably from each
                         branched state before decoding + judging. Keep small
                         (2-4): every step is one flow-model forward held in
                         the autograd graph.
    dpo_beta:            beta_dpo in the preference loss -- scales the raw
                         gradient under the plain-SGD optimizer steer_delta()
                         uses (deliberately not Adam, whose m/sqrt(v)
                         normalization would make beta nearly inert once the
                         preference-loss sigmoid saturates).
    num_delta_grad_steps: gradient steps taken on `delta` for one judge verdict.
    delta_lr:            SGD lr for `delta`. None -> branch_noise_scale / 4,
                         a starting point for a step relative to the initial
                         perturbation's scale, not a hard guarantee -- the
                         actual step also scales with beta and the proxy's
                         gradient magnitude now that the optimizer is SGD.
    delta_max_norm_ratio: hard trust region. After every gradient step delta's
                         RMS is projected back to at most
                         branch_noise_scale * this, so the steered latent can
                         never wander far enough out of distribution to break
                         the decoder. Replaces a soft L2 penalty (whose weight
                         would need tuning against an unknown proxy scale).
    decode_resolution:   resolution handed to pipeline.decode_shape_slat() for
                         the two candidate decodes. Should match the stage the
                         flow model belongs to (512 for the LR/cascade stage).
    judge_weights:       weights for geometric_judge.score_mesh().
    force_conv_none:     force SPARSE_CONV_BACKEND=none for the differentiable
                         segment (plan's Risk table).
    verbose:             progress bars / prints.
    seed:                optional seed for the branch perturbations, for
                         reproducible branching.
    """

    t_branch: float = 0.5
    branch_noise_scale: float = 0.02
    continuation_steps: int = 2
    dpo_beta: float = 1.0
    num_delta_grad_steps: int = 3
    delta_lr: Optional[float] = None
    delta_max_norm_ratio: float = 3.0
    decode_resolution: int = 512
    judge_weights: geometric_judge.JudgeWeights = field(default_factory=_default_judge_weights)
    force_conv_none: bool = True
    verbose: bool = True
    seed: Optional[int] = None

    # The plan pins the branch point to this window; see Section III, Phase 2.
    T_BRANCH_MIN: float = 0.3
    T_BRANCH_MAX: float = 0.7

    def effective_t_branch(self) -> float:
        return float(min(max(self.t_branch, self.T_BRANCH_MIN), self.T_BRANCH_MAX))

    def effective_delta_lr(self) -> float:
        return float(self.delta_lr if self.delta_lr is not None else self.branch_noise_scale / 4.0)


@dataclass
class DPOBranchReport:
    """What the branch step actually did -- returned alongside the SLat so a
    caller (and a human reviewing a run) can see whether steering helped."""

    branch_t: float
    resume_t: float
    delta_branch_won_initial: bool
    delta_branch_won_final: bool
    resumed_from_delta_branch: bool
    score_reference: Optional[float]
    score_delta_initial: Optional[float]
    score_delta_final: Optional[float]
    loss_history: List[float]
    proxy_reference: float
    proxy_delta_history: List[float]
    delta_rms_history: List[float]
    notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Sparse-conv backend override (plan's Risk table, row 2)
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def sparse_conv_backend(backend: str = "none"):
    """Temporarily force TRELLIS.2's sparse-conv backend, restoring on exit.

    Why: the plan's Risk table says `mtlgemm`/`flex_gemm`'s sparse-conv path is
    a compiled forward Metal kernel with no known registered autograd backward,
    while backends/conv_none.py is plain gather/scatter PyTorch and
    differentiates normally. We do not verify that claim here (no MPS on the
    machine this was written on) -- we just take the documented mitigation and
    scope it as narrowly as possible.

    Setting the env var alone is NOT enough: trellis2.modules.sparse.config
    reads SPARSE_CONV_BACKEND once at import time into a module global `CONV`,
    so the env var is only honoured by anything that has not imported yet. We
    therefore also flip `config.CONV` via its own set_conv_backend() and set
    the env var (for lazily-imported consumers), restoring both afterwards.

    We additionally pre-register the backend module in conv.conv._backends:
    SparseConv3d.__init__ lazily imports the backend it was CONSTRUCTED with,
    but SparseConv3d.forward() does a bare `_backends[config.CONV]` lookup with
    no import guard -- flipping CONV under an already-constructed layer would
    otherwise KeyError on the first forward.

    Safe with respect to SparseTensor's storage: in basic.py's __init__, only
    'torchsparse' and 'spconv' change the underlying data representation; both
    'flex_gemm' and 'none' fall into the same plain-dict branch, so tensors
    created under one are readable under the other.

    Note: reading models/structured_latent_flow.py, SLatFlowModel is built from
    sp.SparseLinear + ModulatedSparseTransformerCrossBlock and does not appear
    to instantiate SparseConv3d at all -- the sparse convs live in the shape
    DECODER (sc_vaes/fdg_vae.py), which we never backprop through. So for the
    shape-SLat stage this override is most likely a no-op and is applied
    defensively, exactly as the plan's risk table prescribes.

    UNVERIFIED -- confirm on real Mac hardware that (a) a shape-SLat flow-model
    forward under CONV='none' produces the same values as under 'flex_gemm'
    (weight layouts are documented to match: conv_none.py's init comment says
    it permutes to flex_gemm's (Co, Kd, Kh, Kw, Ci) layout), and (b) flipping
    CONV under already-constructed modules mid-run is in fact benign.
    """
    prev_env = os.environ.get("SPARSE_CONV_BACKEND")
    prev_conv = None
    sparse_config = None

    os.environ["SPARSE_CONV_BACKEND"] = backend
    try:
        try:
            sparse_config = importlib.import_module("trellis2.modules.sparse.config")
            conv_dispatch = importlib.import_module("trellis2.modules.sparse.conv.conv")
            prev_conv = sparse_config.CONV
            if backend not in conv_dispatch._backends:
                conv_dispatch._backends[backend] = importlib.import_module(
                    f"trellis2.modules.sparse.conv.conv_{backend}"
                )
            sparse_config.set_conv_backend(backend)
        except Exception as e:  # pragma: no cover - depends on TRELLIS.2 presence
            # Not fatal: worst case we run the differentiable segment on the
            # default backend and (per the risk table) it may fail to build a
            # backward graph. Say so loudly rather than silently continuing.
            # A "No module named ...conv_none" here means patches/mps_compat.py
            # has not been run against this TRELLIS.2 checkout -- that script is
            # what copies backends/conv_none.py into the trellis2 tree.
            print(f"[dpo_branch] could not switch sparse-conv backend to '{backend}': {e}")
        yield
    finally:
        if sparse_config is not None and prev_conv is not None:
            sparse_config.set_conv_backend(prev_conv)
        if prev_env is None:
            os.environ.pop("SPARSE_CONV_BACKEND", None)
        else:
            os.environ["SPARSE_CONV_BACKEND"] = prev_env


@contextlib.contextmanager
def frozen_parameters(model):
    """Set requires_grad=False on every parameter of `model`, then restore.

    The whole point of inference-time steering is that NO model weight is ever
    updated. This makes that structural rather than a matter of discipline:
    even if a stray .backward() reached the flow model, there would be nowhere
    for the gradient to accumulate.
    """
    saved = []
    for p in model.parameters():
        saved.append((p, p.requires_grad))
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p, flag in saved:
            p.requires_grad_(flag)


# ---------------------------------------------------------------------------
# Differentiable pieces -- pure torch, no TRELLIS.2 types (unit-testable)
# ---------------------------------------------------------------------------


def voxel_neighbor_pairs(coords: torch.Tensor, grid_stride: int = 2048) -> Tuple[torch.Tensor, torch.Tensor]:
    """6-neighbour adjacency over a sparse voxel grid, as (src, tgt) index pairs.

    `coords` is SparseTensor.coords: [N, 4] integer (batch, x, y, z). Returns
    index tensors such that voxel `tgt[k]` has voxel `src[k]` as an axis-aligned
    neighbour. Fully vectorized (sort + searchsorted on a packed integer key) --
    deliberately NOT the Python dict loop that backends/conv_none.py uses to
    build its 27-neighbourhood, because this runs inside the sampling loop.

    Key packing is ((b * S + x) * S + y) * S + z with S = grid_stride. Coords
    are documented as living in [0, 1023] (SparseTensor's class docstring), and
    a -1 offset borrows into coordinate value S-1 = 2047, which can never be an
    actual voxel -- so out-of-range neighbours are guaranteed not to collide
    with real ones. UNVERIFIED on real data: if any coordinate ever reaches
    2047 the borrow guarantee breaks; raise grid_stride if that happens.
    """
    if coords.numel() == 0:
        empty = torch.zeros(0, dtype=torch.long, device=coords.device)
        return empty, empty

    c = coords.long()
    b, x, y, z = c[:, 0], c[:, 1], c[:, 2], c[:, 3]
    S = grid_stride
    key = ((b * S + x) * S + y) * S + z
    order = torch.argsort(key)
    sorted_key = key[order]
    n = sorted_key.numel()

    src_chunks: List[torch.Tensor] = []
    tgt_chunks: List[torch.Tensor] = []
    for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
        nkey = ((b * S + (x + dx)) * S + (y + dy)) * S + (z + dz)
        pos = torch.searchsorted(sorted_key, nkey).clamp_(max=n - 1)
        found = sorted_key[pos] == nkey
        tgt_chunks.append(torch.nonzero(found, as_tuple=False).squeeze(1))
        src_chunks.append(order[pos[found]])

    return torch.cat(src_chunks), torch.cat(tgt_chunks)


def slat_detail_proxy(
    feats: torch.Tensor,
    neighbor_src: torch.Tensor,
    neighbor_tgt: torch.Tensor,
) -> torch.Tensor:
    """Differentiable R_Detail analogue in SLat feature space.

    The umbrella (uniform graph) Laplacian energy of the per-voxel features
    over the voxel adjacency graph, mean-reduced over voxels:

        s(f) = mean_v || f_v - mean_{u in N(v)} f_u ||^2

    This is exactly geometric_judge.detail_reward()'s operator -- the same
    uniform Laplacian, chosen there over a cotangent Laplacian for the same
    robustness reason -- but applied to latent features over the fixed voxel
    grid instead of vertex positions over extracted mesh edges. That means it
    is available at every ODE timestep, needs no mesh extraction, and carries
    a gradient back to whatever produced `feats`.

    Mean- rather than sum-reduced so the value does not scale with token count
    (the branch's two candidates share `coords`, so it would cancel anyway, but
    it keeps `dpo_beta` interpretable across resolutions).

    Isolated voxels (no active neighbour) contribute zero.
    """
    n = feats.shape[0]
    if n == 0 or neighbor_tgt.numel() == 0:
        return feats.sum() * 0.0

    zeros = torch.zeros_like(feats)
    summed = zeros.index_add(0, neighbor_tgt, feats[neighbor_src])
    degree = torch.zeros(n, device=feats.device, dtype=feats.dtype).index_add(
        0, neighbor_tgt, torch.ones_like(neighbor_tgt, dtype=feats.dtype)
    )
    has_neighbors = degree > 0
    degree_safe = torch.where(has_neighbors, degree, torch.ones_like(degree))

    laplacian = feats - summed / degree_safe.unsqueeze(1)
    laplacian = laplacian * has_neighbors.unsqueeze(1).to(feats.dtype)
    return (laplacian ** 2).sum(dim=1).mean()


def preference_loss(
    proxy_delta: torch.Tensor,
    proxy_reference: torch.Tensor,
    delta_branch_won: bool,
    beta: float,
) -> torch.Tensor:
    """The single-branch reduction of the plan's DPO loss (see module docstring).

        L = -log sigma( beta * sign * (s_delta - s_ref) ),  sign = +/-1

    `proxy_reference` should already be detached: the reference branch plays
    P_ref's role and must not receive gradient.
    """
    sign = 1.0 if delta_branch_won else -1.0
    margin = beta * sign * (proxy_delta - proxy_reference)
    return -F.logsigmoid(margin)


def project_delta_(delta: torch.Tensor, max_rms: float) -> None:
    """Hard trust region: rescale `delta` in place so its RMS is <= max_rms.

    A projection rather than a soft L2 penalty because the proxy's numeric
    scale is unknown a priori, so any penalty weight would be an untunable
    guess; a projection is scale-free and gives a hard guarantee that the
    steered latent stays within a known radius of the unperturbed trajectory.
    """
    if max_rms <= 0 or delta.numel() == 0:
        return
    with torch.no_grad():
        rms = delta.pow(2).mean().sqrt()
        if rms > max_rms:
            delta.mul_(max_rms / (rms + 1e-12))


def steer_delta(
    delta: torch.Tensor,
    continue_fn: Callable[[torch.Tensor], torch.Tensor],
    proxy_reference: torch.Tensor,
    delta_branch_won: bool,
    *,
    beta: float,
    num_steps: int,
    lr: float,
    max_rms: float,
) -> Tuple[List[float], List[float], List[float]]:
    """Run `num_steps` Adam steps on `delta` alone under the preference loss.

    `continue_fn(delta) -> proxy_delta` must rebuild the differentiable
    continuation from the branch point with the current delta and return the
    scalar proxy score of the resulting candidate. It is called once per
    gradient step (the flow model is re-run; the mesh decoder and the judge are
    NOT -- the preference label is fixed for the whole inner loop).

    Deliberately free of any TRELLIS.2 type so it can be unit-tested with a
    dummy model on CPU. Returns (loss_history, proxy_history, delta_rms_history).
    """
    # Plain SGD, not Adam: Adam's m/sqrt(v) normalization makes the per-step
    # update magnitude ~lr regardless of the raw gradient's scale, which makes
    # `beta` (the preference loss's temperature) nearly inert once the sigmoid
    # saturates -- it only changes how FEW steps it takes to saturate, not the
    # step size once there. Under SGD the update is -lr*grad, so beta (which
    # scales the loss directly) scales the gradient and therefore the actual
    # step size, restoring it as a meaningful knob, and the step naturally
    # shrinks as the margin becomes favorable (sigma saturates towards 0)
    # instead of walking to the trust-region edge every time.
    optimizer = torch.optim.SGD([delta], lr=lr)
    losses: List[float] = []
    proxies: List[float] = []
    rms: List[float] = []

    for _ in range(num_steps):
        optimizer.zero_grad(set_to_none=True)
        proxy_delta = continue_fn(delta)
        loss = preference_loss(proxy_delta, proxy_reference, delta_branch_won, beta)
        loss.backward()
        optimizer.step()
        project_delta_(delta, max_rms)

        losses.append(float(loss.detach()))
        proxies.append(float(proxy_delta.detach()))
        rms.append(float(delta.detach().pow(2).mean().sqrt()))

    return losses, proxies, rms


# ---------------------------------------------------------------------------
# Sampler plumbing
# ---------------------------------------------------------------------------


def build_t_pairs(steps: int, rescale_t: float = 1.0) -> List[Tuple[float, float]]:
    """Reproduce FlowEulerSampler.sample()'s timestep schedule exactly.

    Copied from flow_euler.py so that a branched run visits the same t values
    as an unbranched one -- resuming must continue the ORIGINAL schedule, not
    restart a shorter one from t_branch.
    """
    t_seq = np.linspace(1, 0, steps + 1)
    t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
    t_seq = t_seq.tolist()
    return [(t_seq[i], t_seq[i + 1]) for i in range(steps)]


def select_branch_window(
    t_pairs: Sequence[Tuple[float, float]],
    t_branch: float,
    k: int,
) -> Tuple[int, int]:
    """Pick (i_branch, k): branch at t_pairs[i_branch], run k pairs
    differentiably (t_pairs[i_branch : i_branch + k]).

    Picks the i whose start-t is NEAREST t_branch -- a prior version picked
    the first start <= t_branch, which silently undershoots the plan's
    t in [0.3, 0.7] window (e.g. at steps=12 it would branch at t=0.25 when
    asked for t_branch=0.3). k is only ever clamped DOWN to fit the schedule
    (never up), then i_branch is clamped to [0, len(t_pairs) - k] so the
    continuation always has exactly k pairs available.

    Raises ValueError on an empty schedule (steps=0) -- the caller previously
    crashed on this several lines later with an opaque IndexError instead of
    a clear message (steps is a free, unvalidated CLI int -- see generate.py's
    --steps and trellis_core/pipeline.py's sampler_overrides passthrough).

    Does NOT guarantee the returned branch_t lands inside [0.3, 0.7] -- for a
    very coarse schedule (few total steps) there may be no t in that window at
    all. Callers should check the returned window against their own bounds and
    warn/log if it's out of range; this function only guarantees a
    non-crashing, nearest-available choice.
    """
    n = len(t_pairs)
    if n < 1:
        raise ValueError("select_branch_window: t_pairs is empty (steps=0?)")
    k = max(1, min(int(k), n))
    starts = [p[0] for p in t_pairs]
    i_branch = min(range(n), key=lambda i: abs(starts[i] - t_branch))
    i_branch = max(0, min(i_branch, n - k))
    return i_branch, k


def split_sampler_params(sampler_params: Dict[str, Any]) -> Tuple[int, float, Dict[str, Any]]:
    """Split a merged sampler_params dict into (steps, rescale_t, inference_kwargs).

    `steps`, `rescale_t`, `verbose` and `tqdm_desc` are consumed by
    FlowEulerSampler.sample() itself; EVERYTHING else (neg_cond,
    guidance_strength, guidance_interval, guidance_rescale, ...) is forwarded
    verbatim down to _inference_model. We never assume which mixins are active
    -- FlowEulerCfgSampler vs FlowEulerGuidanceIntervalSampler is only decided
    when the pretrained pipeline.json loads -- so the pass-through is total,
    mirroring how trellis_core/pipeline.py already splats sampler_overrides in.
    """
    params = dict(sampler_params)
    steps = int(params.pop("steps", 50))
    rescale_t = float(params.pop("rescale_t", 1.0))
    params.pop("verbose", None)
    params.pop("tqdm_desc", None)
    return steps, rescale_t, params


def describe_sampler(sampler) -> str:
    """Human-readable note of which guidance mixins the loaded sampler has."""
    return " <- ".join(c.__name__ for c in type(sampler).__mro__[:-1])


def backfill_sampler_defaults(sampler, inference_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Fill in any guidance kwargs the active sampler's mixins require but
    that inference_kwargs doesn't already supply, using the DEFAULTS declared
    on the sampler's own .sample() method.

    This module calls sampler._get_model_prediction()/_inference_model()
    directly (see differentiable_continuation()'s docstring for why), which
    bypasses FlowEulerSampler.sample()'s own default-argument layer.
    Concretely: ClassifierFreeGuidanceSamplerMixin._inference_model requires
    guidance_strength (no default at that layer -- flow_euler.py:141 only
    defaults it to 3.0 on FlowEulerCfgSampler.sample()'s SIGNATURE), and
    GuidanceIntervalSamplerMixin._inference_model requires guidance_interval
    the same way (defaulted to (0.0, 1.0) only on
    FlowEulerGuidanceIntervalSampler.sample(), flow_euler.py:183). If
    shape_slat_sampler_params (loaded from the pretrained pipeline.json)
    happens to omit either key -- plausible, since upstream code that always
    goes through .sample() would never notice the gap -- calling
    _get_model_prediction directly would raise a bare
    "missing required positional argument" TypeError with no indication of
    why. We never hardcode which mixins are active (that's only known once
    the real pipeline.json loads), so this introspects type(sampler).sample's
    signature and only backfills params that (a) that signature declares with
    a default and (b) the caller didn't already supply.
    """
    import inspect

    try:
        sig = inspect.signature(type(sampler).sample)
    except (TypeError, ValueError):
        return inference_kwargs
    filled = dict(inference_kwargs)
    for name, param in sig.parameters.items():
        if name in filled or param.default is inspect.Parameter.empty:
            continue
        if name in ("self", "model", "noise", "cond", "neg_cond", "steps", "verbose"):
            continue
        filled[name] = param.default
    return filled


def differentiable_continuation(
    sampler,
    model,
    x_t,
    t_pairs: Sequence[Tuple[float, float]],
    cond=None,
    enable_grad: bool = True,
    **inference_kwargs,
):
    """Euler-step forward under ambient grad, returning (x_final, pred_x_0_final).

    This is the key trick from the plan's Phase 2: FlowEulerSampler.sample() and
    .sample_once() are both decorated @torch.no_grad(), so nothing called
    through them can ever produce a gradient. Their helpers
    (_get_model_prediction / _inference_model / _v_to_xstart_eps) are plain
    undecorated methods, so re-implementing sample_once()'s three lines here --

        pred_x_0, pred_eps, pred_v = sampler._get_model_prediction(...)
        x_prev = x_t - (t - t_prev) * pred_v

    -- under torch.enable_grad() gives a real autograd path back through the
    flow model's velocity prediction to whatever x_t depends on.

    Works unchanged for a SparseTensor (its __sub__/__rmul__ operate on .feats
    and preserve .coords) and for a plain dense tensor, which is what the unit
    test exercises.

    `enable_grad=False` runs the identical arithmetic under no_grad -- used for
    the reference branch and for the final re-run, where the graph would be
    dead weight in unified memory.
    """
    pred_x_0 = None
    ctx = torch.enable_grad() if enable_grad else torch.no_grad()
    with ctx:
        for t, t_prev in t_pairs:
            pred_x_0, _pred_eps, pred_v = sampler._get_model_prediction(
                model, x_t, t, cond, **inference_kwargs
            )
            x_t = x_t - (t - t_prev) * pred_v
    return x_t, pred_x_0


# ---------------------------------------------------------------------------
# Mesh / judge glue
# ---------------------------------------------------------------------------


def _to_trimesh(mesh) -> Optional[trimesh.Trimesh]:
    """trellis2 Mesh -> trimesh.Trimesh, or None if the decode came back empty.

    An empty decode is the signature of the macOS GPU-watchdog bug that
    trellis_core/pipeline.py already documents; here it just means "this branch
    is unrankable", which the caller degrades gracefully on rather than
    aborting a whole generation.
    """
    verts = mesh.vertices.detach().cpu().numpy()
    faces = mesh.faces.detach().cpu().numpy()
    if verts.shape[0] == 0 or faces.shape[0] == 0:
        return None
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def _decode_candidate(pipeline, slat_normalized, normalization, resolution) -> Optional[trimesh.Trimesh]:
    """Un-normalize a (detached) SLat estimate and decode it to a trimesh.

    We decode the sampler's pred_x_0 -- its current estimate of the CLEAN
    latent -- not the noisy x_t, because at t ~= 0.5 the raw state is still
    half noise and the decoder would return garbage for both branches equally,
    making the ranking meaningless.

    decode_shape_slat() is not itself @torch.no_grad() (unlike decode_latent),
    so we wrap it explicitly and hand it a detached tensor: mesh extraction is
    non-differentiable anyway (backends/mesh_extract.py uses .item() and Python
    dict lookups), and building a graph through the decoder would waste a large
    amount of the unified memory the plan's risk table is worried about.
    """
    with torch.no_grad():
        slat = _denormalize(slat_normalized.detach(), normalization)
        meshes, _subs = pipeline.decode_shape_slat(slat, resolution)
    if not meshes:
        return None
    return _to_trimesh(meshes[0])


def _denormalize(slat, normalization: Optional[dict]):
    """slat * std + mean, matching sample_shape_slat's final two lines."""
    if not normalization:
        return slat
    std = torch.tensor(normalization["std"])[None].to(slat.device)
    mean = torch.tensor(normalization["mean"])[None].to(slat.device)
    return slat * std + mean


def _rank(mesh_a, mesh_b, weights, rng):
    """rank_candidates with graceful degradation when a branch failed to decode.

    Returns (delta_branch_won: bool, score_a, score_b) where branch A is the
    reference and branch B is the delta branch.
    """
    if mesh_a is None and mesh_b is None:
        return False, None, None
    if mesh_a is None:
        return True, None, geometric_judge.score_mesh(mesh_b, weights, rng)
    if mesh_b is None:
        return False, geometric_judge.score_mesh(mesh_a, weights, rng), None
    winner, score_a, score_b = geometric_judge.rank_candidates(mesh_a, mesh_b, weights, rng)
    return winner == 1, score_a, score_b


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def sample_shape_slat_with_dpo_branch(
    pipeline,
    cond: dict,
    flow_model,
    coords: torch.Tensor,
    sampler_params: dict = {},
    dpo_config: Optional[DPOBranchConfig] = None,
    return_report: bool = False,
):
    """Drop-in alternative to Trellis2ImageTo3DPipeline.sample_shape_slat().

    Takes the same arguments and returns the same type (a de-normalized shape
    SLat SparseTensor), plus a DPOBranchConfig and an optional report, so a
    later phase can slot it into pipeline.py behind a flag without redesign.
    `dpo_config=None` means "use DPOBranchConfig()" -- it still branches; to
    get stock behaviour, call pipeline.sample_shape_slat() instead.

    Cost relative to sample_shape_slat(), on top of the usual `steps` model
    forwards: `continuation_steps * (2 + num_delta_grad_steps)` extra flow
    forwards (of which num_delta_grad_steps also carry a backward), and 3
    shape-decoder forwards (reference, initial candidate, optimized candidate).

    Args:
        pipeline: the loaded Trellis2ImageTo3DPipeline.
        cond: conditioning dict as produced by pipeline.get_cond() -- {'cond',
            'neg_cond'}. 'cond' is passed positionally to the sampler helpers
            exactly as sample_once() does; everything else rides along as
            inference kwargs.
        flow_model: the shape-SLat flow model for this stage.
        coords: [N, 4] sparse-structure coordinates. Fixed for the whole ODE --
            branching only ever perturbs feats, never coords.
        sampler_params: per-call overrides, merged over
            pipeline.shape_slat_sampler_params exactly as sample_shape_slat does.
        dpo_config: branch/preference settings.
        return_report: if True, return (slat, DPOBranchReport).

    Returns:
        SparseTensor (or (SparseTensor, DPOBranchReport)).
    """
    from trellis2.modules.sparse import SparseTensor  # noqa: WPS433 (must follow bootstrap)

    cfg = dpo_config or DPOBranchConfig()
    sampler = pipeline.shape_slat_sampler
    device = pipeline.device
    notes: List[str] = [f"sampler mro: {describe_sampler(sampler)}"]

    merged = {**(pipeline.shape_slat_sampler_params or {}), **(sampler_params or {})}
    steps, rescale_t, inference_kwargs = split_sampler_params(merged)
    t_pairs = build_t_pairs(steps, rescale_t)

    # 'cond' goes positionally (like sample_once); 'neg_cond' etc. as kwargs.
    cond_kwargs = dict(cond or {})
    cond_tensor = cond_kwargs.pop("cond", None)
    inference_kwargs = {**cond_kwargs, **inference_kwargs}
    inference_kwargs = backfill_sampler_defaults(sampler, inference_kwargs)

    generator = None
    if cfg.seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    # This module judges/steers using ONLY the first sample's decoded mesh
    # (_decode_candidate/_to_trimesh take meshes[0]), while `delta` and the
    # detail proxy span every sample's voxels in `coords` -- with
    # num_samples>1 a verdict from sample 0's geometry would silently steer
    # (and select the resume state for) every other sample in the batch too.
    # Not currently reachable anywhere in this repo (generate.py/server never
    # pass num_samples>1, and Trellis2ImageTo3DPipeline.run() defaults to 1),
    # so fail loudly rather than silently mis-steering if that ever changes.
    assert int(coords[:, 0].max().item()) == 0, (
        "sample_shape_slat_with_dpo_branch only supports num_samples=1 -- "
        "the judge/steering only look at sample 0's decoded mesh."
    )

    noise_feats = torch.randn(coords.shape[0], flow_model.in_channels, generator=generator).to(device)
    x = SparseTensor(feats=noise_feats, coords=coords)

    if pipeline.low_vram:
        flow_model.to(device)

    # ---- choose the branch index on the ORIGINAL schedule -------------------
    t_branch_requested = cfg.effective_t_branch()
    i_branch, k = select_branch_window(t_pairs, t_branch_requested, cfg.continuation_steps)

    # ---- Phase 2a: normal no_grad sampling down to the branch point ---------
    for t, t_prev in tqdm(
        t_pairs[:i_branch], desc="Sampling shape SLat (pre-branch)", disable=not cfg.verbose
    ):
        x = sampler.sample_once(flow_model, x, t, t_prev, cond_tensor, **inference_kwargs).pred_x_prev

    branch_t = t_pairs[i_branch][0]
    cont_pairs = t_pairs[i_branch : i_branch + k]
    resume_pairs = t_pairs[i_branch + k :]
    resume_t = cont_pairs[-1][1]

    if not (cfg.T_BRANCH_MIN <= branch_t <= cfg.T_BRANCH_MAX):
        notes.append(
            f"branch_t={branch_t:.3f} falls outside the plan's "
            f"[{cfg.T_BRANCH_MIN}, {cfg.T_BRANCH_MAX}] window -- the sampler "
            f"schedule (steps={len(t_pairs)}) was too coarse to hit it exactly."
        )
        if cfg.verbose:
            print(f"[dpo_branch] warning: {notes[-1]}")

    x_branch = x.detach()
    base_feats = x_branch.feats.detach()
    neighbor_src, neighbor_tgt = voxel_neighbor_pairs(x_branch.coords)
    normalization = getattr(pipeline, "shape_slat_normalization", None)

    def _proxy(slat) -> torch.Tensor:
        return slat_detail_proxy(_denormalize(slat, normalization).feats, neighbor_src, neighbor_tgt)

    def _continue(delta_tensor, enable_grad: bool = True):
        """One (optionally differentiable) continuation from the branch point."""
        branched = x_branch.replace(base_feats + delta_tensor)
        return differentiable_continuation(
            sampler, flow_model, branched, cont_pairs, cond_tensor,
            enable_grad=enable_grad, **inference_kwargs,
        )

    # ---- Phase 2b: fork two candidates -------------------------------------
    # Branch A (reference) is the UNPERTURBED trajectory, i.e. eps_A = 0: it is
    # exactly what the vanilla sampler would have produced, which is what makes
    # it a meaningful stand-in for P_ref. Branch B gets eps_B ~ N(0, scale^2),
    # and that perturbation is what becomes the learnable `delta`.
    eps_b = torch.randn(base_feats.shape, generator=generator).to(base_feats.device) * cfg.branch_noise_scale
    delta = eps_b.clone().to(base_feats.dtype).requires_grad_(True)

    # sparse_conv_backend is scoped ONLY around steer_delta below -- that is
    # the single call in this function that runs .backward(). Everything else
    # here (both initial continuations, both decodes) is enable_grad=False /
    # no_grad, so forcing the slower conv backend for them would cost real
    # decode time (a full 512-res decode on backends/conv_none.py's Python
    # dict-loop neighbor map) for zero benefit -- see sparse_conv_backend's
    # docstring point about this being a no-op for the flow model but NOT for
    # anything that runs the decoder while the override is active.
    backend_ctx = sparse_conv_backend("none") if cfg.force_conv_none else contextlib.nullcontext()

    with frozen_parameters(flow_model):
        assert not any(p.requires_grad for p in flow_model.parameters()), \
            "flow model parameters must stay frozen: this is inference-time steering, not fine-tuning"

        # Reference branch: eps_A = 0, no grad, computed ONCE and kept -- the
        # sampler is deterministic, so re-running it later would give the same
        # tensors back for the price of another k flow-model forwards.
        x_ref, pred_x0_ref = differentiable_continuation(
            sampler, flow_model, x_branch, cont_pairs, cond_tensor,
            enable_grad=False, **inference_kwargs,
        )
        proxy_ref = _proxy(pred_x0_ref).detach()

        # No grad here: this first candidate exists only to be decoded and
        # judged, and a k-step autograd graph we never backward through would
        # sit in unified memory for the whole (expensive) decode.
        _x_b, pred_x0_b = _continue(delta.detach(), enable_grad=False)

        # ---- Phase 3: decode + judge (both non-differentiable) -------------
        if pipeline.low_vram:
            # Free the flow model from the device before the decoder lands on
            # it -- the plan's unified-memory risk row.
            flow_model.cpu()
        mesh_ref = _decode_candidate(pipeline, pred_x0_ref, normalization, cfg.decode_resolution)
        mesh_delta = _decode_candidate(pipeline, pred_x0_b, normalization, cfg.decode_resolution)
        if pipeline.low_vram:
            flow_model.to(device)

        delta_won, score_ref, score_delta = _rank(mesh_ref, mesh_delta, cfg.judge_weights, rng)
        decode_ok = mesh_ref is not None and mesh_delta is not None
        if not decode_ok:
            notes.append(
                "a candidate failed to decode (empty mesh) -- skipping gradient "
                "steering entirely (there is no information to steer with) and "
                "falling back to whichever branch decoded, reference if neither did."
            )
        del mesh_delta

        if cfg.verbose:
            print(
                f"[dpo_branch] t={branch_t:.3f}  judge preferred "
                f"{'delta' if delta_won else 'reference'} branch  "
                f"(S_ref={None if score_ref is None else round(score_ref.total, 4)}, "
                f"S_delta={None if score_delta is None else round(score_delta.total, 4)})"
            )

        if not decode_ok:
            loss_history, proxy_history, rms_history = [], [], []
            delta_won_final, score_delta_final = delta_won, score_delta
            fallback_is_delta = mesh_ref is None  # reference failed but delta decoded
            x = (_x_b if fallback_is_delta else x_ref).detach()
            del _x_b, pred_x0_b, pred_x0_ref, x_ref, mesh_ref
        else:
            del _x_b, pred_x0_b, pred_x0_ref

            # ---- Phase 3: gradient steps on `delta` only, conv backend
            # forced to 'none' only for this call (see comment above) --------
            with backend_ctx:
                loss_history, proxy_history, rms_history = steer_delta(
                    delta,
                    lambda d: _proxy(_continue(d)[1]),
                    proxy_ref,
                    delta_won,
                    beta=cfg.dpo_beta,
                    num_steps=int(cfg.num_delta_grad_steps),
                    lr=cfg.effective_delta_lr(),
                    max_rms=cfg.branch_noise_scale * cfg.delta_max_norm_ratio,
                )

            # ---- pick the state to resume from ----------------------------
            # The gradient steps moved delta, so the optimized branch is a NEW
            # candidate the judge has not seen. Re-run the continuation once
            # (no grad), decode it, and re-rank against the reference mesh we
            # already have, so we always resume from a state the judge
            # actually prefers -- "extract the final mesh from the winning
            # trajectory", Phase 4.1.
            x_final_delta, pred_x0_final = _continue(delta.detach(), enable_grad=False)

            if pipeline.low_vram:
                flow_model.cpu()
            mesh_delta_final = _decode_candidate(pipeline, pred_x0_final, normalization, cfg.decode_resolution)
            if pipeline.low_vram:
                flow_model.to(device)

            delta_won_final, _score_ref_final, score_delta_final = _rank(
                mesh_ref, mesh_delta_final, cfg.judge_weights, rng
            )
            del mesh_ref, mesh_delta_final

            x = (x_final_delta if delta_won_final else x_ref).detach()
            del x_final_delta, x_ref, pred_x0_final

    if cfg.verbose:
        print(
            f"[dpo_branch] resuming from {'delta' if delta_won_final else 'reference'} "
            f"branch at t={resume_t:.3f} ({len(resume_pairs)} steps left)"
        )

    # ---- Phase 2c: resume the normal, full-speed sampler --------------------
    # Back on the default (fast) conv backend and back under @torch.no_grad()
    # via sample_once, continuing the ORIGINAL t schedule from where the
    # differentiable segment stopped.
    for t, t_prev in tqdm(
        resume_pairs, desc="Sampling shape SLat (post-branch)", disable=not cfg.verbose
    ):
        x = sampler.sample_once(flow_model, x, t, t_prev, cond_tensor, **inference_kwargs).pred_x_prev

    if pipeline.low_vram:
        flow_model.cpu()

    slat = _denormalize(x, normalization)

    if not return_report:
        return slat

    report = DPOBranchReport(
        branch_t=branch_t,
        resume_t=resume_t,
        delta_branch_won_initial=delta_won,
        delta_branch_won_final=delta_won_final,
        resumed_from_delta_branch=bool(delta_won_final),
        score_reference=None if score_ref is None else score_ref.total,
        score_delta_initial=None if score_delta is None else score_delta.total,
        score_delta_final=None if score_delta_final is None else score_delta_final.total,
        loss_history=loss_history,
        proxy_reference=float(proxy_ref),
        proxy_delta_history=proxy_history,
        delta_rms_history=rms_history,
        notes=notes,
    )
    return slat, report

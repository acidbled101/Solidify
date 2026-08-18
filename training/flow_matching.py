"""Flow-matching primitives shared by the supervised fine-tune and the retired
preference-steering experiment.

These five names used to live in the DPO experiment, and sft_train.py imported
them from there -- so the shipped trainer depended on a line of work that had
been measured and abandoned. Nothing about them is DPO-specific: they are the
path parameterisation of the rectified flow itself, plus a memory knob.

The parameterisation mirrors TRELLIS.2's own FlowEulerSampler exactly, which is
the whole point -- train on a different path than the sampler integrates and the
adapter is fitted to a model that is never run.

    x_t = (1-t) x_0 + (sigma_min + (1-sigma_min) t) eps
    v   = (1-sigma_min) eps - x_0

sigma_min MUST equal the pipeline's shape_slat_sampler.sigma_min; sft_train.py
reads it off the loaded pipeline rather than hardcoding it.

training/cuda/sft_cuda.py (now training/cuda/) inlines exactly these definitions so it
can run standalone on a rented GPU box with this repo absent. If you change the
maths here, change it there too -- adapters must load back into the Mac
inference path, so the two must agree.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# Path parameterisation
# ---------------------------------------------------------------------------


def noise_latent(
    x_0: torch.Tensor,
    eps: torch.Tensor,
    t: torch.Tensor,
    sigma_min: float,
) -> torch.Tensor:
    """x_t = (1-t) x_0 + (sigma_min + (1-sigma_min) t) eps.

    `t` broadcasts against x_0: pass a scalar, or a per-sample/per-token
    column so a batch can carry different timesteps.
    """
    return (1.0 - t) * x_0 + (sigma_min + (1.0 - sigma_min) * t) * eps


def velocity_target(
    x_0: torch.Tensor,
    eps: torch.Tensor,
    sigma_min: float,
) -> torch.Tensor:
    """v = (1-sigma_min) eps - x_0. Independent of t on the linear path."""
    return (1.0 - sigma_min) * eps - x_0


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class FlowMatchConfig:
    """Timestep sampling and path knobs.

    sigma_min:        must equal the pipeline's shape_slat_sampler.sigma_min.
    t_sampling:       'logit_normal' is the rectified-flow standard and
                      concentrates samples near the middle of the trajectory,
                      where the velocity field is hardest to fit; 'uniform'
                      spreads them evenly.
    t_max:            clamps t away from 1, where the path degenerates.

    The DPO-only fields (beta, shared_noise, sft_weight) stayed behind in the
    experiment's own config -- see experiments/dpo_inference_steering/.
    """

    sigma_min: float = 1e-5
    t_sampling: str = "logit_normal"
    t_min: float = 0.0
    t_max: float = 0.99
    logit_normal_mean: float = 0.0
    logit_normal_std: float = 1.0


def sample_timesteps(
    num: int,
    cfg: "FlowMatchConfig",
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample t in [t_min, t_max], shape [num]."""
    if cfg.t_sampling == "uniform":
        t = torch.rand(num, device=device, dtype=dtype, generator=generator)
    elif cfg.t_sampling == "logit_normal":
        u = torch.randn(num, device=device, dtype=dtype, generator=generator)
        t = torch.sigmoid(cfg.logit_normal_mean + cfg.logit_normal_std * u)
    else:
        raise ValueError(f"unknown t_sampling: {cfg.t_sampling!r}")
    return t.clamp(cfg.t_min, cfg.t_max)


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


@contextmanager
def checkpointed_blocks(model, enabled: bool = True):
    """Temporarily turn on gradient checkpointing for every submodule of
    `model` that supports it, restoring the original flags on exit.

    This is what makes a backward pass over the flow model fit in memory.
    Measured on the real shape-SLat flow model (SLatFlowModel: 30 blocks,
    model_channels 1536, mlp_ratio 5.33 -> 8192-wide MLP, bfloat16): one
    graph-building forward retains ~9.5 GB of activations, against PyTorch's
    ~36.3 GiB MPS ceiling. Flipping the flag drops the retained set to roughly
    one block-input tensor per block (~1 GB total), at the cost of recomputing
    each block's forward during backward.

    Numerically this changes nothing -- checkpointing recomputes exactly the
    same forward -- so it is safe to scope tightly around the one call that
    runs .backward().

    Applied by walking submodules and flipping any `use_checkpoint` attribute
    rather than hardcoding block types: SLatFlowModel builds self.blocks from a
    config, and the attribute is read per-forward (not baked in at
    construction), so a runtime flip is honoured immediately.
    """
    if not enabled:
        yield
        return
    flipped = []
    try:
        for module in model.modules():
            if getattr(module, "use_checkpoint", None) is False:
                module.use_checkpoint = True
                flipped.append(module)
        if not flipped:
            # Not fatal (the segment may still fit for a small enough voxel
            # count), but it is the difference between ~1 GB and ~38 GB of
            # retained activations -- never let that pass silently.
            print(
                "[flow_matching] warning: no submodule exposed a use_checkpoint "
                "flag; gradient checkpointing is NOT active and the backward "
                "pass may exhaust unified memory."
            )
        yield
    finally:
        for module in flipped:
            module.use_checkpoint = False

"""Minimal LoRA for TRELLIS.2's flow models -- portable (CUDA / MPS / CPU).

This is the same adapter as trellis_core/lora.py, with the Mac-only commentary
trimmed and nothing about the MATH changed. That matters: adapters trained on
the A100 must load back into the Mac inference path, and the checkpoints
already on the Mac must load here, so the parameter names, shapes, scaling and
target-module selection are kept byte-compatible.

WHY LoRA RATHER THAN A FULL FINE-TUNE
-------------------------------------
The base flow model is 1.29B parameters. A full fine-tune needs fp32 gradients
and two AdamW moments for all of them -- ~15 GB of optimizer state alone,
before activations. LoRA at rank 16 over the attention and MLP projections
trains ~17M parameters instead (~69 MB fp32), which fits anywhere and keeps
the frozen base weights exactly available as a reference model by disabling
the adapters.
"""

import math
from typing import Dict, Iterable, List, Sequence

import torch
import torch.nn as nn


DEFAULT_TARGETS = ("to_qkv", "to_out", "to_q", "to_kv", "mlp.0", "mlp.2")


class LoRALinear(nn.Module):
    """Wraps a frozen Linear (or SparseLinear) with a trainable low-rank update.

        y = W x + (B A x) * (alpha / rank)

    `A` is Kaiming-initialised and `B` is zero, so the adapter contributes
    exactly nothing at step 0 and training starts from the base model's own
    behaviour rather than from a perturbed one.

    TRELLIS.2's `SparseLinear` subclasses `nn.Linear` but its forward takes and
    returns a `SparseTensor`, so the wrapper unwraps `.feats` on the way in and
    re-wraps on the way out. That one difference is why peft is not used here.
    """

    def __init__(self, base: nn.Linear, rank: int = 16, alpha: float = 32.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = rank
        self.scaling = alpha / rank
        self.enabled = True
        dev = base.weight.device
        self.lora_A = nn.Parameter(torch.zeros(rank, base.in_features, device=dev, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        out = self.base(x)
        if not self.enabled:
            return out
        feats = x.feats if hasattr(x, "feats") else x
        # The low-rank branch is computed in the BASE MODEL'S dtype (bf16 for
        # the 30 transformer blocks) while the parameters themselves stay fp32
        # so AdamW keeps full-precision moments. Casting per call is cheap next
        # to the matmuls. Autograd flows back through the cast normally.
        dt = feats.dtype
        delta = (feats @ self.lora_A.to(dt).t()) @ self.lora_B.to(dt).t()
        delta = delta * self.scaling
        if hasattr(out, "feats"):
            return out.replace(out.feats + delta.to(out.feats.dtype))
        return out + delta.to(out.dtype)


def apply_lora(
    model: nn.Module,
    rank: int = 16,
    alpha: float = 32.0,
    targets: Sequence[str] = DEFAULT_TARGETS,
    only_blocks: bool = True,
) -> List[str]:
    """Replace matching Linear submodules with LoRA-wrapped versions, in place.

    `only_blocks` restricts adaptation to the transformer stack, leaving the
    timestep embedder, the input/output projections and the adaLN modulation
    untouched -- those are where a small perturbation does the most damage to
    the sampler's calibration, and they are a rounding error in parameter count.

    Freezes the ENTIRE base model first. Wrapping alone would only freeze the
    Linears that get adapted, leaving the embedder and projections trainable,
    so an optimizer built over `model.parameters()` would quietly full
    fine-tune several hundred million weights while the run reported itself as
    LoRA. Freezing up front makes the adapters the only thing carrying
    gradient by construction.
    """
    for p in model.parameters():
        p.requires_grad_(False)

    adapted: List[str] = []
    for name, module in list(model.named_modules()):
        if only_blocks and not name.startswith("blocks."):
            continue
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if not isinstance(child, nn.Linear) or isinstance(child, LoRALinear):
                continue
            if not any(full.endswith(t) for t in targets):
                continue
            setattr(module, child_name, LoRALinear(child, rank=rank, alpha=alpha))
            adapted.append(full)
    return adapted


def lora_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    for m in model.modules():
        if isinstance(m, LoRALinear):
            yield m.lora_A
            yield m.lora_B


def set_lora_enabled(model: nn.Module, enabled: bool) -> None:
    """Toggle every adapter. Disabled, the model is bit-for-bit the frozen base
    again -- which is how eval compares tuned against base without loading a
    second copy of a 1.3B model."""
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.enabled = enabled


class disabled_lora:
    """Context manager: run a block against the untouched base model."""

    def __init__(self, model: nn.Module):
        self.model = model

    def __enter__(self):
        set_lora_enabled(self.model, False)
        return self.model

    def __exit__(self, *exc):
        set_lora_enabled(self.model, True)
        return False


def lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    out = {}
    for name, m in model.named_modules():
        if isinstance(m, LoRALinear):
            out[f"{name}.lora_A"] = m.lora_A.detach().cpu()
            out[f"{name}.lora_B"] = m.lora_B.detach().cpu()
    return out


def load_lora_state_dict(model: nn.Module, state: Dict[str, torch.Tensor]) -> int:
    n = 0
    by_name = {name: m for name, m in model.named_modules() if isinstance(m, LoRALinear)}
    for key, tensor in state.items():
        mod_name, _, which = key.rpartition(".")
        m = by_name.get(mod_name)
        if m is None:
            continue
        target = m.lora_A if which == "lora_A" else m.lora_B
        with torch.no_grad():
            target.copy_(tensor.to(target.device, target.dtype))
        n += 1
    return n


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in lora_parameters(model))

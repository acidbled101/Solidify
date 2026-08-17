"""Minimal LoRA for TRELLIS.2's flow models.

WHY NOT peft
------------
peft is not installed here, and it expects HuggingFace module conventions.
TRELLIS.2's transformer blocks mix plain `nn.Linear` with `SparseLinear`, whose
forward takes and returns a `SparseTensor` rather than a plain tensor. peft's
`Linear` adapter assumes tensor in / tensor out and would break on those.

`SparseLinear` does subclass `nn.Linear`, so one wrapper covers both -- it just
has to unwrap `.feats` on the way in and re-wrap on the way out. That is the
entire difference, and it is ~15 lines, which is less work than fighting a
dependency that does not want to be here.

WHY LoRA AT ALL
---------------
The base model is 1.29B parameters (2.41 GB bf16). A full fine-tune needs
gradients and Adam moments for all of them -- roughly 15 GB of optimizer state
alone on top of activations, against a 36 GB unified-memory ceiling shared with
the OS. LoRA at rank 16 over attention and MLP projections trains a few million
parameters instead, and has the useful side effect that the frozen base weights
remain exactly available as a reference model (disable the adapters), which is
what a later DPO stage would need.
"""

import math
import os
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn


DEFAULT_TARGETS = ("to_qkv", "to_out", "to_q", "to_kv", "mlp.0", "mlp.2")


class LoRALinear(nn.Module):
    """Wraps a frozen Linear (or SparseLinear) with a trainable low-rank update.

        y = W x + (B A x) * (alpha / rank)

    `A` is Kaiming-initialised and `B` is zero, so the adapter contributes
    exactly nothing at step 0 and training starts from the base model's own
    behaviour rather than from a perturbed one.
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
        # Compute in the base model's dtype, not fp32. The parameters stay fp32
        # so AdamW keeps full-precision moments, but MPS refuses a matmul whose
        # accumulator and destination dtypes differ -- an fp32 LoRA branch
        # against a bf16 model aborts the process with
        #   "Destination NDArray and Accumulator NDArray cannot have different
        #    datatype in MPSNDArrayMatrixMultiplication"
        # rather than raising a Python error. Casting per call is cheap next to
        # the matmuls themselves and gradients flow back through it normally.
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
    untouched. Those last few are where a small perturbation does the most
    damage to the sampler's calibration, and they are a rounding error in
    parameter count anyway.

    Returns the list of adapted module paths.

    Freezes the ENTIRE base model first. Wrapping only freezes the Linears that
    get adapted, which leaves the timestep embedder, adaLN modulation and the
    input/output projections trainable -- so an optimizer built over
    `model.parameters()` would quietly full-fine-tune several hundred million
    weights while the run reported itself as LoRA. Freezing up front makes the
    adapters the only thing carrying gradient by construction.
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
    """Toggle every adapter.

    Disabled, the model is bit-for-bit the frozen base again -- which is how a
    reference model is obtained for free, and how eval can compare tuned
    against base without loading a second copy.
    """
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

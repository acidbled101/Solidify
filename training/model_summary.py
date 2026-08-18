"""Dump the flow model's layer structure as JSON, marking what trains.

Answers "what exactly is being fine-tuned" without having to trust a summary
line. Every parameter tensor is listed with its shape, dtype, count and
`trainable` flag, so a LoRA run can be audited rather than assumed --
`apply_lora` freezes the whole base model and then adds adapters, and this is
what verifies that actually happened.

Loads the flow model alone on CPU (2.41 GB) rather than the whole pipeline
(~10 GB), so it is safe to run beside a dataset build or a training job on a
32 GB machine.

    PYTHONPATH=. python -m trellis_core.model_summary --lora --out model.json
"""

import argparse
import collections
import json
import os
import sys
from typing import Any, Dict, List

from trellis_core import bootstrap  # noqa: F401

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CKPT = "microsoft/TRELLIS.2-4B/ckpts/slat_flow_img2shape_dit_1_3B_512_bf16"


def summarize(model, lora_applied: bool) -> Dict[str, Any]:
    from trellis_core import lora as lora_mod

    lora_modules = {n for n, m in model.named_modules() if isinstance(m, lora_mod.LoRALinear)}

    params: List[Dict[str, Any]] = []
    for name, p in model.named_parameters():
        owner = name.rsplit(".", 1)[0]
        # A LoRA-wrapped Linear becomes "<path>.base.weight"; strip that so the
        # entry reports the module the user actually recognises.
        base_owner = owner[:-5] if owner.endswith(".base") else owner
        params.append({
            "name": name,
            "module": base_owner,
            "shape": list(p.shape),
            "dtype": str(p.dtype).replace("torch.", ""),
            "parameters": int(p.numel()),
            "trainable": bool(p.requires_grad),
            "lora_adapter": ".lora_" in name,
            "lora_adapted_module": base_owner in lora_modules,
        })

    total = sum(p["parameters"] for p in params)
    trainable = sum(p["parameters"] for p in params if p["trainable"])

    # Group by block so 30 identical transformer blocks do not need reading
    # thirty times over.
    groups: Dict[str, Dict[str, int]] = collections.OrderedDict()
    for p in params:
        n = p["name"]
        if n.startswith("blocks."):
            idx = n.split(".")[1]
            key = f"blocks.{idx}"
        else:
            key = n.split(".")[0]
        g = groups.setdefault(key, {"parameters": 0, "trainable": 0, "tensors": 0})
        g["parameters"] += p["parameters"]
        g["trainable"] += p["parameters"] if p["trainable"] else 0
        g["tensors"] += 1

    trainable_modules = sorted({p["module"] for p in params if p["trainable"]})

    return {
        "checkpoint": DEFAULT_CKPT,
        "architecture": type(model).__name__,
        "config": {
            "in_channels": getattr(model, "in_channels", None),
            "out_channels": getattr(model, "out_channels", None),
            "model_channels": getattr(model, "model_channels", None),
            "num_blocks": len(getattr(model, "blocks", []) or []),
            "dtype": str(getattr(model, "dtype", "")).replace("torch.", ""),
        },
        "totals": {
            "parameters": total,
            "trainable_parameters": trainable,
            "frozen_parameters": total - trainable,
            "trainable_fraction": round(trainable / total, 6) if total else 0.0,
            "parameter_tensors": len(params),
            "trainable_tensors": sum(1 for p in params if p["trainable"]),
        },
        "lora": {
            "applied": lora_applied,
            "adapted_modules": len(lora_modules),
            "adapted_module_names": sorted(lora_modules),
        },
        "trainable_modules": trainable_modules,
        "groups": groups,
        "parameters": params,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lora", action="store_true", help="apply LoRA before dumping")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--out", default=os.path.join(REPO, "report", "model_layers.json"))
    ap.add_argument("--full", action="store_true",
                    help="include every parameter tensor (default: also included)")
    args = ap.parse_args(argv)

    from trellis2 import models
    from trellis_core import lora as lora_mod

    print(f"loading {args.ckpt} (CPU) ...", flush=True)
    model = models.from_pretrained(args.ckpt)   # stays on CPU
    if args.lora:
        lora_mod.apply_lora(model, rank=args.rank, alpha=args.alpha)

    data = summarize(model, args.lora)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(data, f, indent=2)

    t = data["totals"]
    print(f"\n{data['architecture']}  {t['parameters']/1e9:.3f}B parameters, "
          f"{data['config']['num_blocks']} blocks")
    print(f"trainable: {t['trainable_parameters']/1e6:.2f}M "
          f"({100*t['trainable_fraction']:.3f}%) in {t['trainable_tensors']} tensors")
    print(f"frozen   : {t['frozen_parameters']/1e9:.3f}B")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Shared utility functions for compression operations.
"""

from __future__ import annotations

import torch.nn as nn


def get_mlp_modules(layer: nn.Module) -> dict[str, nn.Linear]:
    mlp = getattr(layer, "mlp", None)
    if mlp is None:
        raise ValueError(f"Cannot find MLP in layer: {type(layer)}")
    modules = {}
    for name in ["gate_proj", "up_proj", "down_proj"]:
        mod = getattr(mlp, name, None)
        if mod is not None and isinstance(mod, nn.Linear):
            modules[name] = mod
    if not modules:
        for name, mod in mlp.named_modules():
            if isinstance(mod, nn.Linear):
                modules[name] = mod
    return modules


def get_attention_module(layer: nn.Module) -> nn.Module:
    for name in ["self_attn", "attention"]:
        attn = getattr(layer, name, None)
        if attn is not None:
            return attn
    raise ValueError(f"Cannot find attention in layer: {type(layer)}")


def get_intermediate_size(layer: nn.Module) -> int:
    mods = get_mlp_modules(layer)
    if "gate_proj" in mods:
        return mods["gate_proj"].out_features
    if "up_proj" in mods:
        return mods["up_proj"].out_features
    raise ValueError("Cannot determine intermediate size")

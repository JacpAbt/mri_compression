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
    for name in ["self_attn", "attn", "attention"]:
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


def get_mlp_submodule(layer: nn.Module) -> nn.Module:
    """Return the MLP sub-module from a transformer layer (mlp or feed_forward)."""
    for name in ["mlp", "feed_forward"]:
        m = getattr(layer, name, None)
        if m is not None:
            return m
    raise ValueError(f"Cannot find MLP sub-module in layer: {type(layer)}")


def resolve_layer(model: nn.Module, layer_idx: int, inspector=None) -> nn.Module:
    """Resolve a transformer layer object, architecture-agnostically.

    Tries ``inspector.layer_path`` first; falls back to the three most common
    attribute paths used by Llama/Qwen, GPT-2, and GPT-NeoX family models.

    Args:
        model: The full causal-LM model.
        layer_idx: Zero-based layer index.
        inspector: Optional ``ModelInspector`` instance (provides ``layer_path``).

    Returns:
        The ``nn.Module`` for that transformer block.

    Raises:
        RuntimeError: When the architecture cannot be detected automatically.
    """
    if inspector is not None and hasattr(inspector, "layer_path"):
        obj = model
        for part in inspector.layer_path.split("."):
            obj = getattr(obj, part)
        return obj[layer_idx]

    for path in ["model.layers", "transformer.h", "gpt_neox.layers"]:
        try:
            obj = model
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj[layer_idx]
        except AttributeError:
            continue

    raise RuntimeError(
        f"Cannot resolve transformer layer {layer_idx}: unknown architecture. "
        "Pass inspector= for architecture-aware resolution."
    )

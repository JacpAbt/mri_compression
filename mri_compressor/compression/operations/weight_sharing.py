"""
Weight Sharing Operation
=========================
Makes two consecutive layers share MLP weights.

For high-CKA layer pairs identified by Study 17, the layers produce nearly
identical representations, so layer B's MLP projections can be replaced with
references to layer A's parameters.  This saves memory at inference time
without any additional approximation error.

Architecture-agnostic: works for both gated MLPs (gate_proj/up_proj/down_proj)
and standard MLPs (c_fc/c_proj, fc1/fc2, dense_h_to_4h/dense_4h_to_h).
"""

from __future__ import annotations
import logging

import torch
import torch.nn as nn

from .._utils import resolve_layer

logger = logging.getLogger(__name__)

# Gated MLP (Llama/Qwen SwiGLU style)
_GATED_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
# Standard MLP candidate names (GPT-2, GPT-NeoX, BERT-style, etc.)
_STANDARD_CANDIDATES = (
    ("c_fc", "c_proj"),        # GPT-2
    ("fc1", "fc2"),            # many standard MLPs
    ("dense_h_to_4h", "dense_4h_to_h"),  # GPT-NeoX / Falcon
)


def _get_mlp_projection_names(mlp: nn.Module) -> tuple[str, ...]:
    """Return the projection attribute names present in *mlp*."""
    # Gated architecture check (prefer)
    if all(hasattr(mlp, p) for p in _GATED_PROJECTIONS):
        return _GATED_PROJECTIONS
    # Standard architectures
    for up_name, down_name in _STANDARD_CANDIDATES:
        if hasattr(mlp, up_name) and hasattr(mlp, down_name):
            return (up_name, down_name)
    # Generic fallback: share all Linear children
    return tuple(
        name for name, mod in mlp.named_children() if isinstance(mod, nn.Linear)
    )


def _get_mlp(layer: nn.Module) -> nn.Module:
    """Return the MLP sub-module from a transformer block."""
    for attr in ("mlp", "feed_forward"):
        m = getattr(layer, attr, None)
        if m is not None:
            return m
    raise ValueError(f"Cannot find MLP sub-module in layer: {type(layer)}")


class WeightSharer:
    @staticmethod
    @torch.no_grad()
    def share_mlp_weights(model, layer_a_idx, layer_b_idx, device, inspector=None):
        """
        Make layer B's MLP share weights with layer A.

        For high-CKA pairs, layer B's MLP projections become references
        to layer A's parameters.  This saves memory at inference time.

        Args:
            model: The full model.
            layer_a_idx: Source layer index (keeps its weights).
            layer_b_idx: Target layer index (gets shared weights).
            device: Torch device.
            inspector: Optional ``ModelInspector`` for architecture-agnostic
                layer resolution.  Falls back to common attribute paths.

        Returns:
            dict with sharing stats::

                {
                    "params_saved": int,
                    "projections_shared": list,
                }
        """
        layer_a = resolve_layer(model, layer_a_idx, inspector)
        layer_b = resolve_layer(model, layer_b_idx, inspector)

        mlp_a = _get_mlp(layer_a)
        mlp_b = _get_mlp(layer_b)

        proj_names = _get_mlp_projection_names(mlp_a)

        params_saved = 0
        projections_shared: list[str] = []

        for proj_name in proj_names:
            mod_a = getattr(mlp_a, proj_name, None)
            mod_b = getattr(mlp_b, proj_name, None)

            if mod_a is None or mod_b is None:
                logger.warning(
                    "Projection %s not found in one of the layers (%d, %d); skipping",
                    proj_name, layer_a_idx, layer_b_idx,
                )
                continue

            n_weight_params = mod_b.weight.numel()
            n_bias_params = mod_b.bias.numel() if mod_b.bias is not None else 0

            # Point layer B's weight to layer A's Parameter object
            mod_b.weight = mod_a.weight

            if mod_a.bias is not None:
                mod_b.bias = mod_a.bias
            elif mod_b.bias is not None:
                mod_b.bias = None

            params_saved += n_weight_params + n_bias_params
            projections_shared.append(proj_name)

        logger.info(
            "Shared MLP weights: layer %d -> layer %d  |  projections=%s  |  params_saved=%d",
            layer_a_idx, layer_b_idx, projections_shared, params_saved,
        )

        return {
            "params_saved": params_saved,
            "projections_shared": projections_shared,
        }

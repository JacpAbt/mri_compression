"""
Depth Pruning Operation
=========================
Zero out an entire transformer layer, making it a no-op.
The residual connection means input passes through unchanged.
"""

from __future__ import annotations
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class DepthPruner:

    @staticmethod
    @torch.no_grad()
    def prune_layer(layer: nn.Module):
        """Zero all linear weights in the layer."""
        for mod in layer.modules():
            if isinstance(mod, nn.Linear):
                mod.weight.data.zero_()
                if mod.bias is not None:
                    mod.bias.data.zero_()
        # Also zero LayerNorm if present (so it becomes identity-like)
        # Actually, keep LayerNorm -- zeroing MLP+Attn is sufficient
        # since the residual connection preserves the input.

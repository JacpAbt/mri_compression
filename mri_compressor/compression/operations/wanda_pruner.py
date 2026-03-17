"""
Wanda-guided Structured Pruning Operation
============================================
Prunes neurons based on Wanda importance scores (activation norm * weight norm).
"""

from __future__ import annotations
import logging
from typing import Any, Optional

import torch

from .._utils import get_mlp_modules
from .dead_removal import DeadNeuronRemover

logger = logging.getLogger(__name__)


class WandaPruner:
    @staticmethod
    @torch.no_grad()
    def prune(
        layer,
        activations,
        target_sparsity,
        device,
        precomputed_importance: Optional[Any] = None,
    ):
        """
        Prune neurons using Wanda importance scores.

        Args:
            layer: Transformer layer module.
            activations: Activation tensor [n_tokens, n_neurons].
            target_sparsity: Fraction of neurons to prune (0.0 to 1.0).
            device: Torch device.
            precomputed_importance: Optional precomputed importance scores
                (e.g., from Study 3). If provided, use these instead of
                recomputing from activations/weights. When None, falls back
                to the standard Wanda computation.

        Returns:
            (n_pruned, remaining_activations)
        """
        mlp = get_mlp_modules(layer)

        if precomputed_importance is not None:
            # Use precomputed importance scores directly
            if isinstance(precomputed_importance, torch.Tensor):
                importance = precomputed_importance.to(device)
            else:
                importance = torch.tensor(precomputed_importance, device=device, dtype=torch.float32)
        else:
            # Standard Wanda: activation_norm * weight_norm
            act_norm = activations.to(device).norm(dim=0)
            w_norm = mlp["down_proj"].weight.data.norm(dim=0) if "down_proj" in mlp else torch.ones_like(act_norm)
            importance = act_norm * w_norm

        n = importance.shape[0]
        n_keep = max(1, int(n * (1 - target_sparsity)))
        _, top = importance.topk(n_keep)
        keep = torch.zeros(n, dtype=torch.bool, device=device)
        keep[top] = True
        keep_cpu = keep.cpu()  # single transfer for both uses
        n_pruned = DeadNeuronRemover._shrink_mlp(layer, keep_cpu, device)
        return n_pruned, activations[:, keep_cpu]

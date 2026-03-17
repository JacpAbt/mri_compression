"""
Static Neuron Folding Operation
================================
Folds input-invariant neurons into bias terms.

Neurons identified by Study 20 as "foldable" have nearly constant activation
regardless of input.  Their contribution can be absorbed into the down_proj
bias, after which the corresponding gate_proj / up_proj rows are zeroed out
and the columns removed from the activations tensor.
"""

from __future__ import annotations
import logging
from typing import List

import torch
import torch.nn as nn

from .._utils import get_mlp_modules

logger = logging.getLogger(__name__)


class StaticNeuronFolder:
    @staticmethod
    @torch.no_grad()
    def fold(layer, activations, foldable_indices, device):
        """
        Fold static neurons into bias terms.

        For every neuron index listed in *foldable_indices*:
        1.  Compute the mean activation across calibration data.
        2.  Absorb that constant into the down_proj bias:
                ``down_proj.bias += mean_act * down_proj.weight[:, neuron_idx]``
        3.  Zero out the gate_proj and up_proj rows for that neuron (its
            contribution is now captured in the bias).
        4.  Update the activations tensor to remove the foldable columns.

        Args:
            layer: Transformer layer module.
            activations: Calibration activations tensor ``[N, D]``.
            foldable_indices: List of neuron indices to fold.
            device: Torch device.

        Returns:
            ``(n_folded, updated_activations)`` -- number of neurons folded and
            the remaining activations tensor with foldable columns removed.
        """
        if not foldable_indices:
            return 0, activations

        n_neurons = activations.shape[1]
        foldable_indices = [i for i in foldable_indices if 0 <= i < n_neurons]
        if not foldable_indices:
            return 0, activations

        mlp_modules = get_mlp_modules(layer)
        down_proj: nn.Linear = mlp_modules["down_proj"]

        # ------------------------------------------------------------------
        # Ensure down_proj has a bias; create one (zeros) if missing.
        # ------------------------------------------------------------------
        if down_proj.bias is None:
            down_proj.bias = nn.Parameter(
                torch.zeros(down_proj.out_features, device=device, dtype=down_proj.weight.dtype)
            )

        # ------------------------------------------------------------------
        # For each foldable neuron, absorb its constant into the bias.
        # ------------------------------------------------------------------
        for neuron_idx in foldable_indices:
            mean_act = activations[:, neuron_idx].mean().item()

            # bias += mean_act * down_proj.weight[:, neuron_idx]
            down_proj.bias.data.add_(
                mean_act * down_proj.weight.data[:, neuron_idx]
            )

            # Zero out gate_proj and up_proj rows for this neuron.
            for name in ("gate_proj", "up_proj"):
                if name in mlp_modules:
                    mod = mlp_modules[name]
                    mod.weight.data[neuron_idx].zero_()
                    if mod.bias is not None:
                        mod.bias.data[neuron_idx] = 0.0

        n_folded = len(foldable_indices)
        logger.debug("Folded %d static neurons into down_proj bias", n_folded)

        # ------------------------------------------------------------------
        # Remove foldable columns from activations tensor.
        # ------------------------------------------------------------------
        keep_mask = torch.ones(n_neurons, dtype=torch.bool)
        for idx in foldable_indices:
            keep_mask[idx] = False
        updated_activations = activations[:, keep_mask]

        return n_folded, updated_activations

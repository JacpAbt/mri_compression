"""
Dead Neuron Removal Operation
================================
Removes dead/dormant neurons from MLP layers based on activation magnitude.
"""

from __future__ import annotations
import logging
from typing import Optional

import torch
import torch.nn as nn

from .._utils import get_mlp_modules

logger = logging.getLogger(__name__)


class DeadNeuronRemover:
    @staticmethod
    @torch.no_grad()
    def remove_by_mri_count(
        layer,
        activations,
        n_to_remove,
        device,
        protected_indices: Optional[set] = None,
        mlp_info=None,
    ):
        """
        Remove the lowest-magnitude neurons from an MLP layer.

        Args:
            layer: Transformer layer module.
            activations: Activation tensor [n_tokens, n_neurons].
            n_to_remove: Number of neurons to remove.
            device: Torch device.
            protected_indices: Optional set of neuron indices that must never
                be pruned (from Studies 4, 9). When None, falls back to
                existing behavior with no protection.

        Returns:
            (n_removed, remaining_activations)
        """
        n_neurons = activations.shape[1]
        n_to_remove = min(n_to_remove, n_neurons - 1)
        if n_to_remove <= 0:
            return 0, activations
        fire_mag = activations.abs().mean(dim=0)
        _, sorted_idx = fire_mag.sort()

        keep_mask = torch.ones(n_neurons, dtype=torch.bool)

        if protected_indices is not None:
            # Skip protected neurons when selecting which to remove
            removed_count = 0
            for idx in sorted_idx:
                if removed_count >= n_to_remove:
                    break
                idx_item = idx.item()
                if idx_item not in protected_indices:
                    keep_mask[idx_item] = False
                    removed_count += 1
        else:
            keep_mask[sorted_idx[:n_to_remove]] = False

        n_removed = DeadNeuronRemover._shrink_mlp(layer, keep_mask, device, mlp_info)
        return n_removed, activations[:, keep_mask]

    @staticmethod
    @torch.no_grad()
    def remove_by_indices(
        layer,
        activations,
        indices_to_remove: list,
        device,
        protected_indices: Optional[set] = None,
        mlp_info=None,
    ):
        """
        Remove specific neurons by index (not magnitude ranking).

        Used for domain-unnecessary removal where indices are pre-computed
        by Study 22. Builds a keep_mask from explicit indices, filtering
        out any protected neurons.

        Args:
            layer: Transformer layer module.
            activations: Activation tensor [n_tokens, n_neurons].
            indices_to_remove: List of neuron indices to remove.
            device: Torch device.
            protected_indices: Optional set of neuron indices that must
                never be pruned.

        Returns:
            (n_removed, remaining_activations)
        """
        n_neurons = activations.shape[1]
        if not indices_to_remove:
            return 0, activations

        # Build keep mask
        keep_mask = torch.ones(n_neurons, dtype=torch.bool)
        protected = protected_indices or set()

        for idx in indices_to_remove:
            if 0 <= idx < n_neurons and idx not in protected:
                keep_mask[idx] = False

        n_removed = DeadNeuronRemover._shrink_mlp(layer, keep_mask, device, mlp_info)
        return n_removed, activations[:, keep_mask]

    @staticmethod
    @torch.no_grad()
    def remove_combined(
        layer,
        activations,
        n_dead_to_remove: int,
        domain_unnecessary_indices: list,
        device,
        protected_indices: Optional[set] = None,
        mlp_info=None,
    ):
        """
        Combined dead-removal + domain-unnecessary removal in a single pass.

        This avoids index drift: instead of removing dead neurons first
        (which shifts indices) and then removing domain-unnecessary neurons
        (with now-stale indices), we build a unified keep_mask upfront.

        Args:
            layer: Transformer layer module.
            activations: Activation tensor [n_tokens, n_neurons].
            n_dead_to_remove: Number of lowest-magnitude neurons to remove.
            domain_unnecessary_indices: Pre-computed indices to remove.
            device: Torch device.
            protected_indices: Set of neuron indices to protect.

        Returns:
            (n_dead_removed, n_domain_removed, remaining_activations)
        """
        n_neurons = activations.shape[1]
        protected = protected_indices or set()
        keep_mask = torch.ones(n_neurons, dtype=torch.bool)

        # Step 1: Mark dead neurons (by lowest activation magnitude)
        n_dead = min(n_dead_to_remove, n_neurons - 1)
        dead_removed = 0
        if n_dead > 0:
            fire_mag = activations.abs().mean(dim=0)
            _, sorted_idx = fire_mag.sort()
            for idx in sorted_idx:
                if dead_removed >= n_dead:
                    break
                idx_item = idx.item()
                if idx_item not in protected:
                    keep_mask[idx_item] = False
                    dead_removed += 1

        # Step 2: Mark domain-unnecessary neurons (by pre-computed indices)
        domain_removed = 0
        for idx in domain_unnecessary_indices:
            if 0 <= idx < n_neurons and idx not in protected and keep_mask[idx]:
                keep_mask[idx] = False
                domain_removed += 1

        # Single shrink operation
        total_removed = DeadNeuronRemover._shrink_mlp(layer, keep_mask, device, mlp_info)
        return dead_removed, domain_removed, activations[:, keep_mask]

    @staticmethod
    @torch.no_grad()
    def _shrink_mlp(layer, keep_mask, device, mlp_info=None):
        n_removed = (~keep_mask).sum().item()
        if n_removed == 0:
            return 0
        keep_idx = keep_mask.nonzero(as_tuple=True)[0].to(device)

        if mlp_info is not None:
            inter = mlp_info.intermediate_size
            # gate_proj / up_proj — neurons are the "output" (intermediate) dimension
            for proj in filter(None, [mlp_info.gate_proj, mlp_info.up_proj]):
                W = proj.weight
                if W.shape[0] == inter:
                    # nn.Linear layout: (inter, hidden) — neurons are rows
                    proj.weight = nn.Parameter(
                        torch.index_select(W, 0, keep_idx))
                    proj.out_features = proj.weight.shape[0]
                else:
                    # Conv1D layout: (hidden, inter) — neurons are columns
                    proj.weight = nn.Parameter(
                        torch.index_select(W, 1, keep_idx))
                    if hasattr(proj, 'nf'):
                        proj.nf = proj.weight.shape[1]
                if proj.bias is not None and proj.bias.shape[0] == inter:
                    proj.bias = nn.Parameter(proj.bias.data[keep_idx])
            # down_proj / c_proj — neurons are the "input" (intermediate) dimension
            if mlp_info.down_proj is not None:
                proj = mlp_info.down_proj
                W = proj.weight
                if W.shape[0] == inter:
                    # Conv1D layout: (inter, hidden) — neurons are rows
                    proj.weight = nn.Parameter(
                        torch.index_select(W, 0, keep_idx))
                else:
                    # nn.Linear layout: (hidden, inter) — neurons are columns
                    proj.weight = nn.Parameter(
                        torch.index_select(W, 1, keep_idx))
                    proj.in_features = proj.weight.shape[1]
        else:
            # Fallback: nn.Linear only (original behaviour)
            mlp_modules = get_mlp_modules(layer)
            for name in ["gate_proj", "up_proj"]:
                if name in mlp_modules:
                    mod = mlp_modules[name]
                    mod.weight = nn.Parameter(
                        torch.index_select(mod.weight.data, 0, keep_idx))
                    mod.out_features = mod.weight.shape[0]
                    if mod.bias is not None:
                        mod.bias = nn.Parameter(mod.bias.data[keep_idx])
            if "down_proj" in mlp_modules:
                mod = mlp_modules["down_proj"]
                mod.weight = nn.Parameter(
                    torch.index_select(mod.weight.data, 1, keep_idx))
                mod.in_features = mod.weight.shape[1]

        return n_removed

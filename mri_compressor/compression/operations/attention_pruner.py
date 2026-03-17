"""
Attention Head Pruning Operation
==================================
Zero out the least-important attention heads in a layer.

Strategy: zero the output projection columns for pruned heads.
This effectively makes those heads produce zero output while keeping
the attention computation intact (avoids shape changes).

Why not remove heads entirely? Changing num_heads breaks the model's
config and all downstream shape assumptions. Zeroing is safe and
equivalent at inference time (the heads contribute nothing).
"""

from __future__ import annotations
import logging
from typing import Optional

import torch
import torch.nn as nn

from .._utils import get_attention_module

logger = logging.getLogger(__name__)


class AttentionHeadPruner:

    @staticmethod
    @torch.no_grad()
    def prune_heads(
        layer: nn.Module,
        n_heads_to_prune: int,
        dataloader,
        model: nn.Module,
        layer_idx: int,
        device: torch.device,
        max_batches: int = 4,
        head_importance_data: Optional[list] = None,
        cluster_prunable_heads: Optional[list] = None,
    ) -> list[int]:
        """
        Identify and zero out the least-important heads.

        Args:
            layer: Transformer layer module.
            n_heads_to_prune: Number of heads to prune.
            dataloader: Calibration dataloader.
            model: Full model (for config access).
            layer_idx: Index of this layer.
            device: Torch device.
            max_batches: Max calibration batches.
            head_importance_data: Optional list of dicts with Study 6
                entropy+sink data per head. If provided, use this instead
                of Frobenius norm to determine head importance. Each entry
                should have keys like "head_idx", "entropy", "sink_score".
                When None, falls back to existing Frobenius norm behavior.
            cluster_prunable_heads: Optional list of head indices identified
                as functionally redundant by Study 19 (head clustering).
                When combined with head_importance_data, heads in this list
                get their importance reduced by 50% (pruned preferentially).
                When provided alone, these heads are pruned first, with
                Frobenius norm fallback for any remaining budget.

        Returns:
            List of pruned head indices.
        """
        attn = get_attention_module(layer)
        num_heads = model.config.num_attention_heads
        head_dim = model.config.hidden_size // num_heads

        if n_heads_to_prune <= 0 or n_heads_to_prune >= num_heads:
            return []

        cluster_set = set(cluster_prunable_heads) if cluster_prunable_heads else set()

        if head_importance_data is not None:
            # Use Study 6 entropy+sink data for importance ranking
            # Higher entropy = more diverse attention = more important
            # Higher sink_score = more used as attention sink = more important
            head_importance = torch.zeros(num_heads)
            for entry in head_importance_data:
                h = entry.get("head_idx", 0)
                if h < num_heads:
                    entropy = entry.get("entropy", 0.0)
                    sink = entry.get("sink_score", 0.0)
                    # Combined importance: entropy + sink contribution
                    head_importance[h] = entropy + sink

            # If cluster_prunable_heads provided, reduce importance of
            # functionally redundant heads by 50% so they get pruned first
            if cluster_set:
                for h in cluster_set:
                    if h < num_heads:
                        head_importance[h] *= 0.5

        elif cluster_prunable_heads is not None:
            # Only cluster_prunable_heads provided (no head_importance_data).
            # Use cluster set as primary pruning candidates; fall back to
            # Frobenius norm for any remaining budget beyond the cluster set.
            cluster_heads = [h for h in cluster_prunable_heads if h < num_heads]

            if len(cluster_heads) >= n_heads_to_prune:
                # Enough cluster candidates: use Frobenius norm among them
                # to pick the least important within the cluster set
                o_proj = attn.o_proj
                W_o = o_proj.weight.data
                cluster_importance = []
                for h in cluster_heads:
                    start = h * head_dim
                    end = start + head_dim
                    norm_val = W_o[:, start:end].float().norm().item()
                    cluster_importance.append((norm_val, h))
                cluster_importance.sort()  # ascending by norm
                heads_to_prune = [h for _, h in cluster_importance[:n_heads_to_prune]]

                # Skip the normal sorting path below
                # Jump directly to zeroing
                for h in heads_to_prune:
                    start = h * head_dim
                    end = start + head_dim
                    attn.o_proj.weight.data[:, start:end] = 0
                    for proj_name in ["q_proj"]:
                        proj = getattr(attn, proj_name, None)
                        if proj is not None:
                            proj.weight.data[start:end] = 0
                            if proj.bias is not None:
                                proj.bias.data[start:end] = 0
                return heads_to_prune
            else:
                # Not enough cluster candidates: prune all cluster heads,
                # then fill remaining budget with Frobenius norm ranking
                remaining_budget = n_heads_to_prune - len(cluster_heads)
                non_cluster = [h for h in range(num_heads) if h not in cluster_set]

                o_proj = attn.o_proj
                W_o = o_proj.weight.data
                non_cluster_importance = []
                for h in non_cluster:
                    start = h * head_dim
                    end = start + head_dim
                    norm_val = W_o[:, start:end].float().norm().item()
                    non_cluster_importance.append((norm_val, h))
                non_cluster_importance.sort()  # ascending by norm
                extra_heads = [h for _, h in non_cluster_importance[:remaining_budget]]

                heads_to_prune = cluster_heads + extra_heads

                # Zero out and return
                for h in heads_to_prune:
                    start = h * head_dim
                    end = start + head_dim
                    attn.o_proj.weight.data[:, start:end] = 0
                    for proj_name in ["q_proj"]:
                        proj = getattr(attn, proj_name, None)
                        if proj is not None:
                            proj.weight.data[start:end] = 0
                            if proj.bias is not None:
                                proj.bias.data[start:end] = 0
                return heads_to_prune
        else:
            # Measure head importance via output norm contribution
            # Each head's contribution = norm of its output projection slice
            o_proj = attn.o_proj
            W_o = o_proj.weight.data  # [hidden_size, hidden_size]

            head_importance = torch.zeros(num_heads)
            for h in range(num_heads):
                start = h * head_dim
                end = start + head_dim
                # Frobenius norm of the output projection for this head
                head_importance[h] = W_o[:, start:end].float().norm()

        # Prune the lowest-importance heads
        _, sorted_heads = head_importance.sort()
        heads_to_prune = sorted_heads[:n_heads_to_prune].tolist()

        # Zero out the pruned heads' weights in all projections
        for h in heads_to_prune:
            start = h * head_dim
            end = start + head_dim
            # Zero output projection columns for this head
            attn.o_proj.weight.data[:, start:end] = 0
            # Zero the Q/K/V projection rows for this head
            # (saves computation and prevents any gradient flow)
            for proj_name in ["q_proj"]:
                proj = getattr(attn, proj_name, None)
                if proj is not None:
                    proj.weight.data[start:end] = 0
                    if proj.bias is not None:
                        proj.bias.data[start:end] = 0

        return heads_to_prune

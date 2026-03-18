"""
Attention Head Pruning Operation
==================================
Zero out the least-important attention heads in a layer.

Strategy: zero the output projection columns AND the query/key/value projection
rows for each pruned head.  Zeroing o_proj columns ensures the head contributes
nothing to the residual stream — this matches what Study 10 measured (full head
removal) so the PPL safety budget computed there remains valid.

For GQA models (Qwen/Llama with num_kv_heads < num_heads), k/v rows are shared
across multiple query heads and are therefore left intact.  Only q_proj rows
and o_proj columns are zeroed for GQA pruned heads.

Why not remove heads entirely? Changing num_heads breaks the model's config and
all downstream shape assumptions. Zeroing is safe and equivalent at inference
time (the heads contribute nothing).
"""

from __future__ import annotations
import logging
from typing import Optional

import torch
import torch.nn as nn

from .._utils import get_attention_module

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _zero_head(
    attn: nn.Module,
    head_idx: int,
    head_dim: int,
    num_heads: int,
    num_kv_heads: int,
) -> None:
    """Zero all projections that exclusively belong to ``head_idx``.

    The o_proj column zeroing is the *primary* operation: it makes the head
    contribute exactly zero to the residual stream, matching the measurement
    methodology used in Study 10.

    For MHA (``num_kv_heads == num_heads``) k/v rows are also zeroed since
    each head owns its own KV slot.  For GQA models, k/v rows are shared
    across query heads and are therefore left alone.
    """
    start = head_idx * head_dim
    end = start + head_dim

    # --- o_proj: zero the *columns* that this head writes to the residual ---
    o_proj = getattr(attn, "o_proj", None) or getattr(attn, "c_proj", None)
    if o_proj is not None:
        o_proj.weight.data[:, start:end] = 0.0

    # --- q_proj: zero the *rows* for this head ---
    q_proj = getattr(attn, "q_proj", None)
    if q_proj is not None:
        q_proj.weight.data[start:end] = 0.0
        if q_proj.bias is not None:
            q_proj.bias.data[start:end] = 0.0
    # GPT-2 uses a single c_attn that holds Q/K/V concatenated — skip
    # (it is shared and cannot be zeroed per-head safely here).

    # --- k_proj / v_proj: only zero when head exclusively owns its KV slot ---
    # GQA: group_size > 1 means multiple Q heads share one KV head. Zeroing
    # the shared k/v rows would damage *other* still-alive Q heads in the group.
    group_size = max(1, num_heads // num_kv_heads) if num_kv_heads > 0 else 1
    if group_size == 1:  # MHA: safe to zero k/v
        for proj_name in ("k_proj", "v_proj"):
            proj = getattr(attn, proj_name, None)
            if proj is not None:
                proj.weight.data[start:end] = 0.0
                if proj.bias is not None:
                    proj.bias.data[start:end] = 0.0


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
        attn_info=None,
    ) -> list[int]:
        """
        Identify and zero out the least-important heads.

        Args:
            layer: Transformer layer module.
            n_heads_to_prune: Number of heads to prune.
            dataloader: Calibration dataloader.
            model: Full model (for config access fallback).
            layer_idx: Index of this layer.
            device: Torch device.
            max_batches: Max calibration batches.
            head_importance_data: Optional list of dicts with Study 6
                entropy+sink data per head.
            cluster_prunable_heads: Optional list of head indices identified
                as functionally redundant by Study 19.
            attn_info: Optional ``AttentionInfo`` from ``ModelInspector``.
                Provides ``num_heads``, ``num_kv_heads``, and ``head_dim``
                for architecture-agnostic pruning and GQA handling.

        Returns:
            List of pruned head indices.
        """
        attn = get_attention_module(layer)

        # --- Resolve head count and head dimension ---
        if attn_info is not None:
            num_heads = attn_info.num_heads
            num_kv_heads = attn_info.num_kv_heads
            head_dim = attn_info.head_dim
        else:
            num_heads = getattr(model.config, "num_attention_heads", None)
            num_kv_heads = getattr(model.config, "num_key_value_heads", num_heads)
            if num_heads is None:
                raise RuntimeError(
                    f"Cannot determine num_attention_heads for layer {layer_idx}. "
                    "Pass attn_info= for architecture-aware pruning."
                )
            head_dim = getattr(model.config, "hidden_size", num_heads * 64) // num_heads

        if n_heads_to_prune <= 0 or n_heads_to_prune >= num_heads:
            return []

        cluster_set = set(cluster_prunable_heads) if cluster_prunable_heads else set()

        # ------------------------------------------------------------------ #
        #  Build head_importance scores                                        #
        # ------------------------------------------------------------------ #
        if head_importance_data is not None:
            # Use Study 6 entropy+sink data for importance ranking
            head_importance = torch.zeros(num_heads)
            for entry in head_importance_data:
                h = entry.get("head_idx", 0)
                if h < num_heads:
                    entropy = entry.get("entropy", 0.0)
                    sink = entry.get("sink_score", 0.0)
                    head_importance[h] = entropy + sink

            # Reduce importance of cluster-redundant heads
            if cluster_set:
                for h in cluster_set:
                    if h < num_heads:
                        head_importance[h] *= 0.5

        elif cluster_prunable_heads is not None:
            # Only cluster info provided — use Frobenius norm within the set
            o_proj = getattr(attn, "o_proj", None) or getattr(attn, "c_proj", None)
            cluster_heads = [h for h in cluster_prunable_heads if h < num_heads]

            if len(cluster_heads) >= n_heads_to_prune:
                # Enough cluster candidates: rank by Frobenius norm
                W_o = o_proj.weight.data
                cluster_importance = sorted(
                    (W_o[:, h * head_dim:(h + 1) * head_dim].float().norm().item(), h)
                    for h in cluster_heads
                )
                heads_to_prune = [h for _, h in cluster_importance[:n_heads_to_prune]]
                for h in heads_to_prune:
                    _zero_head(attn, h, head_dim, num_heads, num_kv_heads)
                return heads_to_prune

            else:
                # Not enough: prune all cluster heads, fill rest by Frobenius norm
                remaining_budget = n_heads_to_prune - len(cluster_heads)
                non_cluster = [h for h in range(num_heads) if h not in cluster_set]
                W_o = o_proj.weight.data
                non_cluster_importance = sorted(
                    (W_o[:, h * head_dim:(h + 1) * head_dim].float().norm().item(), h)
                    for h in non_cluster
                )
                extra_heads = [h for _, h in non_cluster_importance[:remaining_budget]]
                heads_to_prune = cluster_heads + extra_heads
                for h in heads_to_prune:
                    _zero_head(attn, h, head_dim, num_heads, num_kv_heads)
                return heads_to_prune

        else:
            # Fallback: Frobenius norm of o_proj columns as importance proxy
            o_proj = getattr(attn, "o_proj", None) or getattr(attn, "c_proj", None)
            if o_proj is None:
                logger.warning(f"Layer {layer_idx}: no o_proj/c_proj found; skipping head pruning")
                return []
            W_o = o_proj.weight.data
            head_importance = torch.zeros(num_heads)
            for h in range(num_heads):
                head_importance[h] = W_o[:, h * head_dim:(h + 1) * head_dim].float().norm()

        # ------------------------------------------------------------------ #
        #  Select and zero the lowest-importance heads                         #
        # ------------------------------------------------------------------ #
        _, sorted_heads = head_importance.sort()
        heads_to_prune = sorted_heads[:n_heads_to_prune].tolist()

        for h in heads_to_prune:
            _zero_head(attn, h, head_dim, num_heads, num_kv_heads)

        return heads_to_prune

"""
Studies Hybrid Attention: Architecture-Agnostic Attention Analysis
==================================================================

Core utilities and study functions for hybrid models with DeltaNet linear-attention
(e.g. Qwen3.5 3:1 DeltaNet:softmax layout). All three entry points work for both
standard softmax-attention layers and DeltaNet linear-attention layers.

Core Utilities:
  _find_mlp_module           - find MLP sub-module from transformer layer
  _find_attn_output_proj     - generic output projection finder (any attn type)
  _collect_attn_contribution - residual decomposition → attn contribution [tokens, d_model]

Study 6 Extension:
  LinearAttentionChannelReport dataclass
  run_linear_attention_analysis - channel-group magnitude for DeltaNet layers

Study 23 (new):
  LinearAttentionRankReport dataclass
  run_linear_attention_rank_analysis - SVD rank of attention contributions (all layers)
"""

from __future__ import annotations
import gc
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.amp import autocast

logger = logging.getLogger(__name__)


# =============================================================================
# Core Architecture-Agnostic Utilities
# =============================================================================

def _find_mlp_module(layer: nn.Module) -> Optional[nn.Module]:
    """Find the MLP sub-module from a transformer layer."""
    for attr in ["mlp", "feed_forward", "ff", "ffn"]:
        m = getattr(layer, attr, None)
        if m is not None:
            return m
    return None


def _find_attn_output_proj(layer: nn.Module, d_model: int) -> Optional[nn.Linear]:
    """
    Find the output projection of any attention mechanism in a transformer layer.

    Strategy:
    1. Try named attention attributes (self_attn, attn, attention, self_attention)
       → search within that sub-module for nn.Linear with out_features == d_model.
    2. Try generic fallback: any child with BOTH an input proj (q_proj/c_attn) AND
       an output proj (o_proj/c_proj).
    3. Try any non-MLP child whose sub-modules contain a Linear with out_features==d_model.

    Returns the last matching nn.Linear found within the candidate (most likely to be
    the output projection), or None if not found.
    """
    candidate = None

    # Step 1: named attention attributes
    for attr in ["self_attn", "attn", "attention", "self_attention"]:
        if hasattr(layer, attr):
            candidate = getattr(layer, attr)
            break

    # Step 2: generic AND fallback (q+o projections)
    if candidate is None:
        for _, mod in layer.named_children():
            has_in = hasattr(mod, "q_proj") or hasattr(mod, "c_attn")
            has_out = hasattr(mod, "o_proj") or hasattr(mod, "c_proj")
            if has_in and has_out:
                candidate = mod
                break

    # Step 3: any non-MLP child with a Linear of out_features == d_model
    if candidate is None:
        mlp_names = {"mlp", "feed_forward", "ff", "ffn"}
        for name, mod in layer.named_children():
            if name.lower() in mlp_names:
                continue
            for _, submod in mod.named_modules():
                if isinstance(submod, nn.Linear) and submod.out_features == d_model:
                    candidate = mod
                    break
            if candidate is not None:
                break

    if candidate is None:
        return None

    # Scan candidate's sub-modules for the last nn.Linear with out_features == d_model
    result = None
    for _, mod in candidate.named_modules():
        if isinstance(mod, nn.Linear) and mod.out_features == d_model:
            result = mod
    return result


@torch.no_grad()
def _collect_attn_contribution(
    inspector,
    layer_idx: int,
    dataset,
    batch_size: int,
    max_batches: int,
) -> torch.Tensor:
    """
    Collect the attention contribution for one layer via residual decomposition.

    Method (architecture-agnostic, works for softmax AND DeltaNet):
      Pre-norm residual block: y = x + attn(...) + mlp(...)
      Therefore: attn_contribution = (y − x) − mlp_output

    Three hooks are registered:
      - Pre-hook on the full transformer layer  → captures x (layer input)
      - Post-hook on the full transformer layer → captures y (layer output)
      - Post-hook on the MLP sub-module        → captures mlp_out

    Returns:
        Tensor of shape [total_tokens, d_model] on CPU in float32.
    """
    from ..data_utils import get_dataloader

    model = inspector.model
    device = next(model.parameters()).device

    # Resolve the layer via the inspector's layer path
    obj = model
    for part in inspector.layer_path.split("."):
        obj = getattr(obj, part)
    layer = obj[layer_idx]

    # Find MLP sub-module
    mlp = _find_mlp_module(layer)
    if mlp is None:
        raise RuntimeError(
            f"Cannot find MLP in layer {layer_idx} for attention contribution collection"
        )

    hook_data: dict = {"x": None, "y": None, "mlp_out": None}

    def layer_pre_hook(module, args, kwargs=None):
        # First positional arg is hidden_states [batch, seq, d_model]
        x = args[0] if isinstance(args, (tuple, list)) else args
        hook_data["x"] = x.detach()

    def layer_post_hook(module, args, output, kwargs=None):
        # Layer output may be a tuple; first element is hidden_states
        y = output[0] if isinstance(output, (tuple, list)) else output
        hook_data["y"] = y.detach()

    def mlp_post_hook(module, args, output, kwargs=None):
        mlp_out = output[0] if isinstance(output, (tuple, list)) else output
        hook_data["mlp_out"] = mlp_out.detach()

    h_pre = layer.register_forward_pre_hook(layer_pre_hook)
    h_post = layer.register_forward_hook(layer_post_hook)
    h_mlp = mlp.register_forward_hook(mlp_post_hook)

    all_contribs: list[torch.Tensor] = []
    loader = get_dataloader(dataset, batch_size=batch_size)
    model.eval()
    use_cuda = str(device).startswith("cuda")

    try:
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            ids = batch["input_ids"].to(device)
            mask = batch.get("attention_mask", torch.ones_like(ids)).to(device)
            if use_cuda:
                with autocast("cuda", dtype=torch.bfloat16):
                    model(input_ids=ids, attention_mask=mask)
            else:
                model(input_ids=ids, attention_mask=mask)

            if (hook_data["x"] is not None
                    and hook_data["y"] is not None
                    and hook_data["mlp_out"] is not None):
                x = hook_data["x"].float()
                y = hook_data["y"].float()
                mlp_out = hook_data["mlp_out"].float()

                # Only proceed if shapes match (all [batch, seq, d_model])
                if x.shape == y.shape == mlp_out.shape:
                    contrib = (y - x) - mlp_out          # [batch, seq, d_model]
                    contrib_flat = contrib.reshape(-1, contrib.shape[-1])  # [tokens, d_model]
                    all_contribs.append(contrib_flat.cpu())

                hook_data["x"] = hook_data["y"] = hook_data["mlp_out"] = None
    finally:
        h_pre.remove()
        h_post.remove()
        h_mlp.remove()

    if not all_contribs:
        raise RuntimeError(
            f"No attention contribution data collected for layer {layer_idx}. "
            "Check that the model follows a standard residual-block structure."
        )

    result = torch.cat(all_contribs, dim=0)
    del all_contribs
    gc.collect()
    return result


# =============================================================================
# Study 6 Extension: Linear Attention Channel Group Analysis
# =============================================================================

@dataclass
class LinearAttentionChannelReport:
    """Channel-group magnitude report for a DeltaNet (linear-attention) layer."""
    layer_idx: int
    group_idx: int
    channels: Tuple[int, int]       # (start, end) channel slice in d_model
    output_magnitude: float         # mean |attn_contribution| for this channel group
    output_fraction: float          # group_mag / total_layer_attn_mag
    layer_type: str = "linear"      # Distinguishes from HeadImportanceReport


def run_linear_attention_analysis(
    inspector,
    dataset,
    batch_size: int = 4,
    max_batches: int = 16,
) -> List[LinearAttentionChannelReport]:
    """
    Study 6 Extension: Channel-group magnitude analysis for DeltaNet layers.

    For each DeltaNet layer (inspector.attn_layers[i] is None):
      - Collects attention contribution via residual decomposition [tokens, d_model]
      - Splits d_model into channel groups analogous to "virtual attention heads"
      - num_groups taken from the first standard-attention layer's num_heads (default 8)
      - Reports the magnitude fraction each group contributes

    Groups with output_fraction < 0.3 / num_groups are flagged as low-magnitude
    candidates for output-projection row zeroing (see compression/diagnostician.py).
    """
    print("  Study 6 extension: Linear attention channel-group analysis...")

    d_model = inspector.model.config.hidden_size

    # Derive num_groups from the first standard-attention layer; fall back to 8
    num_groups = 8
    for ai in inspector.attn_layers:
        if ai is not None:
            num_groups = ai.num_heads
            break
    d_group = d_model // num_groups

    reports: List[LinearAttentionChannelReport] = []

    for layer_idx in range(inspector.num_layers):
        # Only DeltaNet (linear-attention) layers
        if inspector.attn_layers[layer_idx] is not None:
            continue

        try:
            contrib = _collect_attn_contribution(
                inspector, layer_idx, dataset, batch_size, max_batches
            )
        except Exception as e:
            logger.warning(f"  Layer {layer_idx}: contribution collection failed: {e}")
            continue

        # [total_tokens, d_model] → per-group magnitude
        total_mag = contrib.abs().mean().item()
        safe_total = max(total_mag, 1e-12)

        layer_group_mags: list[float] = []
        for g in range(num_groups):
            s = g * d_group
            e = min((g + 1) * d_group, d_model)
            group_mag = contrib[:, s:e].abs().mean().item()
            frac = group_mag / safe_total
            layer_group_mags.append(group_mag)
            reports.append(LinearAttentionChannelReport(
                layer_idx=layer_idx,
                group_idx=g,
                channels=(s, e),
                output_magnitude=group_mag,
                output_fraction=frac,
            ))

        g_min = min(layer_group_mags)
        g_max = max(layer_group_mags)
        g_mean = sum(layer_group_mags) / len(layer_group_mags)
        print(f"    Layer {layer_idx:2d} (linear attn): "
              f"group_mag min={g_min:.4f}, max={g_max:.4f}, mean={g_mean:.4f}")

        del contrib
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not reports:
        print("    No DeltaNet layers found — all layers have standard attention.")

    return reports


# =============================================================================
# Study 23: Linear Attention Rank Analysis (all layers)
# =============================================================================

@dataclass
class LinearAttentionRankReport:
    """Eigenspectrum rank analysis of attention contributions for one layer."""
    layer_idx: int
    layer_type: str              # "standard" or "linear"
    effective_rank: float        # participation ratio = (Σλ)² / Σλ²
    rank_90: int                 # principal components for 90% cumulative variance
    rank_95: int                 # principal components for 95% cumulative variance
    rank_99: int                 # principal components for 99% cumulative variance
    total_dim: int               # d_model (full dimensionality)
    compression_ratio: float     # rank_95 / total_dim; < 0.35 → SVD factorization candidate


def run_linear_attention_rank_analysis(
    inspector,
    dataset,
    batch_size: int = 4,
    max_batches: int = 16,
) -> List[LinearAttentionRankReport]:
    """
    Study 23: Rank analysis of attention contributions across ALL layers.

    For each layer (standard AND DeltaNet):
      1. Collect attention contribution via residual decomposition → [tokens, d_model]
      2. Compute eigenspectrum of the covariance matrix [d_model, d_model]
      3. Report effective rank (participation ratio) and rank thresholds at 90/95/99%
      4. compression_ratio = rank_95 / d_model

    Compression signal:
      compression_ratio < 0.35 → attention output uses < 35% of d_model dimensions
      to capture 95% of variance → candidate for SVD low-rank factorization of
      the output projection.
    """
    print("\n" + "=" * 80)
    print("STUDY 23: Linear Attention Rank Analysis")
    print("=" * 80)

    d_model = inspector.model.config.hidden_size
    reports: List[LinearAttentionRankReport] = []

    standard_r95: list[int] = []
    linear_r95: list[int] = []

    for layer_idx in range(inspector.num_layers):
        layer_type = "standard" if inspector.attn_layers[layer_idx] is not None else "linear"

        try:
            contrib = _collect_attn_contribution(
                inspector, layer_idx, dataset, batch_size, max_batches
            )
        except Exception as e:
            logger.warning(f"  Layer {layer_idx}: contribution collection failed: {e}")
            continue

        contrib = contrib.float()
        X = contrib - contrib.mean(dim=0)          # center: [tokens, d_model]
        tokens = X.shape[0]

        # Covariance matrix [d_model, d_model]
        # Use float64 for numerical stability of eigvalsh
        cov = (X.double().T @ X.double()) / max(tokens - 1, 1)

        # Eigenvalues in ascending order → flip to descending, clamp negatives
        S = torch.linalg.eigvalsh(cov).flip(0).clamp(min=0.0).float()

        total = S.sum().clamp(min=1e-12)
        cumfrac = S.cumsum(0) / total

        # Participation ratio (effective rank)
        s_sq_sum = (S ** 2).sum().clamp(min=1e-12)
        effective_rank = float(total ** 2 / s_sq_sum)

        # Rank thresholds
        rank_90 = max(1, min(int((cumfrac < 0.90).sum().item()) + 1, d_model))
        rank_95 = max(1, min(int((cumfrac < 0.95).sum().item()) + 1, d_model))
        rank_99 = max(1, min(int((cumfrac < 0.99).sum().item()) + 1, d_model))

        compression_ratio = round(rank_95 / d_model, 4)
        candidate_tag = " [low-rank candidate]" if compression_ratio < 0.35 else ""

        report = LinearAttentionRankReport(
            layer_idx=layer_idx,
            layer_type=layer_type,
            effective_rank=round(effective_rank, 2),
            rank_90=rank_90,
            rank_95=rank_95,
            rank_99=rank_99,
            total_dim=d_model,
            compression_ratio=compression_ratio,
        )
        reports.append(report)

        (standard_r95 if layer_type == "standard" else linear_r95).append(rank_95)

        print(f"  Layer {layer_idx:2d} ({layer_type:8s}): "
              f"eff_rank={effective_rank:6.1f}, "
              f"r@90%={rank_90:4d}, r@95%={rank_95:4d}, r@99%={rank_99:4d}, "
              f"d={d_model}, ratio={compression_ratio:.3f}"
              + candidate_tag)

        del contrib, X, cov, S, cumfrac
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- Summary ---
    if standard_r95:
        mean_s = sum(standard_r95) / len(standard_r95)
        print(f"\n  Standard attention layers ({len(standard_r95)}): "
              f"mean rank@95% = {mean_s:.1f} / {d_model} "
              f"(ratio = {mean_s/d_model:.3f})")
    if linear_r95:
        mean_l = sum(linear_r95) / len(linear_r95)
        print(f"  Linear attention layers  ({len(linear_r95)}): "
              f"mean rank@95% = {mean_l:.1f} / {d_model} "
              f"(ratio = {mean_l/d_model:.3f})")

    candidates = [r for r in reports if r.compression_ratio < 0.35]
    if candidates:
        cand_ids = [r.layer_idx for r in candidates]
        print(f"  Low-rank candidates (ratio < 0.35, {len(candidates)} layers): {cand_ids}")
    else:
        print("  No low-rank candidates (all layers use > 35% of d_model for 95% variance).")

    return reports

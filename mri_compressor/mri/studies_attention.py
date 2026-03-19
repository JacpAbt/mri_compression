"""
Study 6: Attention Head Importance.

Analyzes attention head specialization and importance. From Voita et al.
(2019): "Specialized heads do the heavy lifting, the rest can be pruned."

Measures:
- Entropy: Low entropy = head is focused, likely specialized
- First-token attention: High = attention sink behavior (Xiao et al., 2023)
- Concentration: How peaked is the attention distribution?
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

from ..model_utils import ModelInspector
from ..data_utils import TextDataset, get_dataloader


# =============================================================================
# Study 6: Attention Head Importance
# =============================================================================

@dataclass
class HeadImportanceReport:
    """Importance scores for attention heads."""
    layer_idx: int
    head_idx: int
    # Metrics
    mean_entropy: float              # average entropy of attention distribution
    first_token_attention: float     # fraction of attention going to first token (attention sink)
    max_attention_concentration: float  # max attention to any single position (averaged)
    ablation_ppl_delta: Optional[float]  # perplexity change when this head is zeroed out


def run_attention_head_importance(
    inspector: ModelInspector,
    dataset: TextDataset,
    batch_size: int = 4,
    max_batches: int = 16,
    do_ablation: bool = False,  # Set True for full ablation (slow but definitive)
) -> List[HeadImportanceReport]:
    """
    Study 6: Analyze attention head specialization and importance.

    From Voita et al. (2019): "Specialized heads do the heavy lifting,
    the rest can be pruned."

    We measure:
    - Entropy: Low entropy = head is focused, likely specialized
    - First-token attention: High = attention sink behavior (Xiao et al., 2023)
    - Concentration: How peaked is the attention distribution?

    Note: For GPT-2 we can get attention weights directly.
    For newer models we may need output_attentions=True.
    """
    print("\n" + "="*80)
    print("STUDY 6: Attention Head Importance")
    print("="*80)

    # Collect attention patterns
    dataloader = get_dataloader(dataset, batch_size=batch_size)

    # We'll accumulate statistics across batches
    head_stats = defaultdict(lambda: {
        "entropy_sum": 0.0, "first_token_sum": 0.0, "max_attn_sum": 0.0, "count": 0
    })

    with torch.no_grad():
        # Force eager attention implementation to get actual attention weights
        # SDPA (the default in newer transformers) doesn't return weights
        original_attn_impl = getattr(inspector.model.config, '_attn_implementation', None)
        try:
            inspector.model.config._attn_implementation = "eager"
        except Exception:
            pass

        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            input_ids = batch["input_ids"].to(inspector.device)

            try:
                outputs = inspector.model(input_ids=input_ids, output_attentions=True)
            except Exception as e:
                print(f"  WARNING: output_attentions failed: {e}")
                print("  Skipping Study 6.")
                if original_attn_impl is not None:
                    inspector.model.config._attn_implementation = original_attn_impl
                return []

            # outputs.attentions is a tuple of (batch, num_heads, seq, seq) per layer
            if outputs.attentions is None:
                print("  WARNING: Model did not return attention weights. Skipping.")
                if original_attn_impl is not None:
                    inspector.model.config._attn_implementation = original_attn_impl
                return []

            for layer_idx, attn_weights in enumerate(outputs.attentions):
                # Some layers may return None (e.g. SDPA fallback)
                if attn_weights is None:
                    continue

                # attn_weights: (batch, num_heads, seq, seq)
                B, H, S, _ = attn_weights.shape

                for h in range(H):
                    head_attn = attn_weights[:, h, :, :]  # (B, S, S)

                    # Entropy of attention distribution (per query position, averaged)
                    # Clamp to avoid log(0)
                    log_attn = torch.log(head_attn.clamp(min=1e-10))
                    entropy = -(head_attn * log_attn).sum(dim=-1).mean().item()

                    # Attention to first token (attention sink)
                    first_token_attn = head_attn[:, :, 0].mean().item()

                    # Max attention concentration
                    max_attn = head_attn.max(dim=-1).values.mean().item()

                    key = (layer_idx, h)
                    head_stats[key]["entropy_sum"] += entropy
                    head_stats[key]["first_token_sum"] += first_token_attn
                    head_stats[key]["max_attn_sum"] += max_attn
                    head_stats[key]["count"] += 1

    # Restore original attention implementation
    if original_attn_impl is not None:
        inspector.model.config._attn_implementation = original_attn_impl

    if not head_stats:
        print("  WARNING: No attention weights were collected. Skipping.")
        return []

    reports = []
    for (layer_idx, head_idx), stats in sorted(head_stats.items()):
        n = stats["count"]
        if n == 0:
            continue
        report = HeadImportanceReport(
            layer_idx=layer_idx,
            head_idx=head_idx,
            mean_entropy=stats["entropy_sum"] / n,
            first_token_attention=stats["first_token_sum"] / n,
            max_attention_concentration=stats["max_attn_sum"] / n,
            ablation_ppl_delta=None,
        )
        reports.append(report)

    # Print summary per layer (standard-attention layers only)
    for layer_idx in range(inspector.num_layers):
        layer_reports = [r for r in reports if r.layer_idx == layer_idx]
        if not layer_reports:
            # DeltaNet / linear-attention layer — no standard head data
            continue
        avg_entropy = np.mean([r.mean_entropy for r in layer_reports])
        avg_first = np.mean([r.first_token_attention for r in layer_reports])
        max_first = max([r.first_token_attention for r in layer_reports])
        min_entropy_head = min(layer_reports, key=lambda r: r.mean_entropy)
        print(f"  Layer {layer_idx:2d}: avg_entropy={avg_entropy:.2f}, "
              f"avg_first_token_attn={avg_first:.3f}, "
              f"max_first_token_attn={max_first:.3f}, "
              f"most_focused_head={min_entropy_head.head_idx}")

    # --- Study 6 Extension: analyze DeltaNet / linear-attention layers ---
    # run_linear_attention_analysis only processes layers where attn_layers[i] is None,
    # so it is a no-op on purely standard-attention models.
    try:
        from .studies_hybrid_attention import run_linear_attention_analysis
        linear_reports = run_linear_attention_analysis(
            inspector, dataset, batch_size, max_batches
        )
        reports.extend(linear_reports)
    except Exception as e:
        import traceback
        print(f"  WARNING: linear attention extension failed: {e}")
        traceback.print_exc()

    return reports

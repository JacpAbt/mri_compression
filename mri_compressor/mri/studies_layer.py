"""
Study 10: Layer Redundancy Analysis.

Measures each layer's contribution by removing it (zeroing its output in the
residual stream). This reveals:
- Which layers are "load-bearing" vs redundant
- Whether attention or MLP matters more at each depth
- Safe targets for layer pruning/skipping
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

from ..model_utils import ModelInspector
from ..data_utils import TextDataset, get_dataloader, evaluate_perplexity


# =============================================================================
# Study 10: Layer Redundancy Analysis
# =============================================================================

@dataclass
class LayerRedundancyReport:
    """How much each layer contributes to the model's output."""
    layer_idx: int
    component: str         # "mlp" or "attention" or "full"
    ppl_without: float     # perplexity with this component zeroed
    ppl_delta: float       # change from baseline
    residual_norm: float   # L2 norm of this layer's contribution to residual stream


def run_layer_redundancy(
    inspector: ModelInspector,
    dataset: TextDataset,
    batch_size: int = 4,
    max_eval_batches: int = 8,
) -> List[LayerRedundancyReport]:
    """
    Study 10: Measure each layer's contribution by removing it.

    Method: For each layer, zero out its MLP and/or attention output
    (effectively making it a no-op in the residual stream).

    This reveals:
    - Which layers are "load-bearing" vs redundant
    - Whether attention or MLP matters more at each depth
    - Safe targets for layer pruning/skipping
    """
    print("\n" + "="*80)
    print("STUDY 10: Layer Redundancy Analysis")
    print("="*80)

    eval_loader = get_dataloader(dataset, batch_size=batch_size)
    baseline_ppl = evaluate_perplexity(inspector.model, eval_loader,
                                        inspector.device, max_batches=max_eval_batches)
    print(f"  Baseline perplexity: {baseline_ppl:.2f}")

    reports = []
    layers = inspector.get_layers()

    for layer_idx in range(inspector.num_layers):
        layer = layers[layer_idx]

        # --- Zero out MLP ---
        mlp_info = inspector.mlp_layers[layer_idx]

        # Save and zero the down_proj bias and weight contribution
        # Simplest approach: hook that zeros MLP output
        mlp_zeroed = [False]

        def make_mlp_zero_hook(flag):
            def hook_fn(module, input, output):
                if flag[0]:
                    return torch.zeros_like(output)
                return output
            return hook_fn

        # Find the MLP module
        mlp_module = None
        for attr in ['mlp', 'feed_forward']:
            if hasattr(layer, attr):
                mlp_module = getattr(layer, attr)
                break

        if mlp_module is not None:
            hook = mlp_module.register_forward_hook(make_mlp_zero_hook(mlp_zeroed))
            mlp_zeroed[0] = True
            ppl_no_mlp = evaluate_perplexity(inspector.model, eval_loader,
                                              inspector.device, max_batches=max_eval_batches)
            mlp_zeroed[0] = False
            hook.remove()

            reports.append(LayerRedundancyReport(
                layer_idx=layer_idx, component="mlp",
                ppl_without=ppl_no_mlp, ppl_delta=ppl_no_mlp - baseline_ppl,
                residual_norm=0.0,
            ))

        # --- Zero out Attention ---
        attn_zeroed = [False]
        attn_module = None
        for attr in ['self_attn', 'attn', 'attention']:
            if hasattr(layer, attr):
                attn_module = getattr(layer, attr)
                break

        if attn_module is not None:
            hook = attn_module.register_forward_hook(
                lambda m, i, o, flag=attn_zeroed: (
                    (torch.zeros_like(o[0]),) + o[1:] if flag[0] and isinstance(o, tuple)
                    else (torch.zeros_like(o) if flag[0] else o)
                )
            )
            attn_zeroed[0] = True
            ppl_no_attn = evaluate_perplexity(inspector.model, eval_loader,
                                               inspector.device, max_batches=max_eval_batches)
            attn_zeroed[0] = False
            hook.remove()

            reports.append(LayerRedundancyReport(
                layer_idx=layer_idx, component="attention",
                ppl_without=ppl_no_attn, ppl_delta=ppl_no_attn - baseline_ppl,
                residual_norm=0.0,
            ))

    # Print summary
    for layer_idx in range(inspector.num_layers):
        layer_reports = [r for r in reports if r.layer_idx == layer_idx]
        mlp_r = next((r for r in layer_reports if r.component == "mlp"), None)
        attn_r = next((r for r in layer_reports if r.component == "attention"), None)

        mlp_delta = f"{mlp_r.ppl_delta:+.2f}" if mlp_r else "N/A"
        attn_delta = f"{attn_r.ppl_delta:+.2f}" if attn_r else "N/A"
        print(f"  Layer {layer_idx:2d}: MLP PPL delta={mlp_delta}, "
              f"Attn PPL delta={attn_delta}")

    return reports

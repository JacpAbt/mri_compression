"""
Studies 3 and 9: Wanda Importance Scores and Critical Neuron Search.

Study 3 - Wanda Importance Scores: Compute per-neuron importance using
    the Wanda metric (weight magnitude * activation norm), streaming
    one layer at a time.

Study 9 - Critical Neuron Search: Find "super weights" and critical neurons
    whose removal causes maximum model degradation. Inspired by Yu et al.
    (2024) and "The Achilles' Heel of LLMs" (2025).
"""

import torch
import torch.nn as nn
import numpy as np
import gc
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

from ..model_utils import ModelInspector, collect_single_layer
from ..data_utils import TextDataset, get_dataloader, evaluate_perplexity


# =============================================================================
# Study 3: Wanda Importance Scores (streaming version)
# =============================================================================

def compute_wanda_scores(
    inspector: ModelInspector,
    dataset: TextDataset,
    batch_size: int = 4,
    max_batches: int = 16,
) -> Dict[int, torch.Tensor]:
    """Study 3: STREAMING Wanda scores — one layer at a time."""
    print("\n" + "="*80)
    print("STUDY 3: Wanda Importance Scores")
    print("="*80)

    wanda_scores = {}
    for layer_idx in range(inspector.num_layers):
        act = collect_single_layer(inspector, dataset, layer_idx,
                                   batch_size=batch_size, max_batches=max_batches)

        # Activation norm per neuron
        act_norm = act.float().norm(p=2, dim=0)  # (intermediate_size,)

        # Weight magnitude of down_proj
        down_w = inspector.mlp_layers[layer_idx].down_proj.weight.data.cpu().float()
        intermediate_size = act_norm.shape[0]
        if down_w.shape[0] == intermediate_size:
            weight_norm = down_w.abs().mean(dim=1)
        else:
            weight_norm = down_w.abs().mean(dim=0)

        score = weight_norm * act_norm
        wanda_scores[layer_idx] = score

        top5 = score.topk(5).indices.tolist()
        bot5 = score.topk(5, largest=False).indices.tolist()
        print(f"  Layer {layer_idx:2d}: mean={score.mean():.4f}, "
              f"max={score.max():.4f}, top5={top5}, bot5={bot5}")

        del act, act_norm, down_w, weight_norm
        gc.collect()

    return wanda_scores


# =============================================================================
# Study 9: Critical Neuron Search
# =============================================================================

@dataclass
class CriticalNeuronReport:
    """Identifies the most critical neurons whose removal causes maximum damage."""
    layer_idx: int
    neuron_idx: int
    # Impact metrics
    ppl_increase_single: float      # PPL increase when only this neuron is zeroed
    ppl_increase_rank: int          # rank among all neurons in this layer
    weight_norm: float              # L2 norm of this neuron's outgoing weights
    activation_mean: float          # mean absolute activation


def run_critical_neuron_search(
    inspector: ModelInspector,
    dataset: TextDataset,
    batch_size: int = 4,
    max_eval_batches: int = 8,
    top_k_per_layer: int = 5,       # test top-k candidates per layer
    candidate_method: str = "weight_norm",  # how to select candidates
) -> List[CriticalNeuronReport]:
    """
    Study 9: Find the most critical neurons in the model.

    Inspired by "Super Weights" (Yu et al., 2024): "Pruning a single super weight
    completely destroys the model's ability to generate text."

    And "The Achilles' Heel of LLMs" (2025): "Disabling as few as three neurons
    can catastrophically impair a 72B-parameter model."

    Method:
    1. Rank neurons by a fast heuristic (weight norm, activation magnitude)
    2. Test top candidates by zeroing them out one at a time
    3. Measure perplexity impact

    This reveals the vulnerability structure of the model.
    """
    print("\n" + "="*80)
    print("STUDY 9: Critical Neuron Search")
    print("="*80)

    # First, get baseline perplexity
    eval_loader = get_dataloader(dataset, batch_size=batch_size)
    baseline_ppl = evaluate_perplexity(inspector.model, eval_loader,
                                        inspector.device, max_batches=max_eval_batches)
    print(f"  Baseline perplexity: {baseline_ppl:.2f}")

    all_reports = []

    for layer_idx in range(inspector.num_layers):
        mlp = inspector.mlp_layers[layer_idx]
        down_w = mlp.down_proj.weight.data

        # Detect layout: nn.Linear is (hidden, intermediate), Conv1D is (intermediate, hidden)
        is_conv1d = (down_w.shape[0] == mlp.intermediate_size)

        # Select candidates based on heuristic
        if candidate_method == "weight_norm":
            if is_conv1d:
                neuron_importance = down_w.norm(dim=1)  # norm per row
            else:
                neuron_importance = down_w.norm(dim=0)  # norm per column
        elif candidate_method == "max_weight":
            if is_conv1d:
                neuron_importance = down_w.abs().max(dim=1).values
            else:
                neuron_importance = down_w.abs().max(dim=0).values
        else:
            if is_conv1d:
                neuron_importance = down_w.norm(dim=1)
            else:
                neuron_importance = down_w.norm(dim=0)

        # Get top-k candidates
        top_indices = neuron_importance.topk(top_k_per_layer).indices.tolist()

        for neuron_idx in top_indices:
            # Zero out this neuron's weights in down_proj
            if is_conv1d:
                original_weights = down_w[neuron_idx, :].clone()
                down_w[neuron_idx, :] = 0
            else:
                original_weights = down_w[:, neuron_idx].clone()
                down_w[:, neuron_idx] = 0

            # Measure perplexity
            ppl = evaluate_perplexity(inspector.model, eval_loader,
                                       inspector.device, max_batches=max_eval_batches)

            # Restore
            if is_conv1d:
                down_w[neuron_idx, :] = original_weights
            else:
                down_w[:, neuron_idx] = original_weights

            ppl_increase = ppl - baseline_ppl

            report = CriticalNeuronReport(
                layer_idx=layer_idx,
                neuron_idx=neuron_idx,
                ppl_increase_single=ppl_increase,
                ppl_increase_rank=0,  # filled in later
                weight_norm=neuron_importance[neuron_idx].item(),
                activation_mean=0.0,  # could fill from study 4
            )
            all_reports.append(report)

        # Print top finding for this layer
        layer_reports = [r for r in all_reports if r.layer_idx == layer_idx]
        worst = max(layer_reports, key=lambda r: r.ppl_increase_single)
        print(f"  Layer {layer_idx:2d}: most critical neuron {worst.neuron_idx}, "
              f"PPL increase: {worst.ppl_increase_single:+.2f}")

    # Sort and rank
    all_reports.sort(key=lambda r: r.ppl_increase_single, reverse=True)
    for i, r in enumerate(all_reports):
        r.ppl_increase_rank = i + 1

    # Print overall top-10 most critical neurons
    print(f"\n  TOP 10 Most Critical Neurons (baseline PPL={baseline_ppl:.2f}):")
    for r in all_reports[:10]:
        print(f"    #{r.ppl_increase_rank}: Layer {r.layer_idx}, Neuron {r.neuron_idx}, "
              f"PPL increase: {r.ppl_increase_single:+.2f}, "
              f"weight_norm: {r.weight_norm:.4f}")

    return all_reports

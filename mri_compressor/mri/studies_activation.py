"""
Studies 1 and 4: Activation Profiling and Massive Activation Scan.

Study 1 - Activation Profiling: Comprehensive statistics on MLP activation
    distributions per layer (streaming, one layer at a time).

Study 4 - Massive Activation Scan: Find input-agnostic outlier activations
    (Sun et al., 2024) that function as indispensable bias terms.
"""

import torch
import numpy as np
import gc
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

from ..model_utils import ModelInspector, collect_single_layer
from ..data_utils import TextDataset, get_dataloader


# =============================================================================
# Study 1: Activation Profiling (streaming version)
# =============================================================================

@dataclass
class ActivationProfile:
    """Statistics about activation distributions in a single MLP layer."""
    layer_idx: int
    mean: float
    std: float
    median: float
    pct_near_zero: float
    pct_negative: float
    pct_exactly_zero: float
    natural_sparsity: float
    kurtosis: float
    skewness: float
    max_val: float
    min_val: float
    top1_ratio: float
    top10_ratio: float
    gini_coefficient: float


def compute_activation_profile(activations: torch.Tensor, layer_idx: int) -> ActivationProfile:
    """Compute comprehensive statistics on activation tensor. (unchanged logic)"""
    act = activations.float()
    abs_act = act.abs()

    mean_val = act.mean().item()
    std_val = act.std().item()
    median_val = act.median().item()
    max_val = act.max().item()
    min_val = act.min().item()

    threshold = 0.01 * abs_act.max().item()
    pct_near_zero = (abs_act < threshold).float().mean().item()
    pct_negative = (act < 0).float().mean().item()
    pct_exactly_zero = (act == 0).float().mean().item()

    abs_median = abs_act.median().item()
    natural_sparsity = (abs_act < abs_median).float().mean().item()

    centered = act - act.mean()
    var = centered.pow(2).mean()
    kurtosis = (centered.pow(4).mean() / var.pow(2)).item() - 3
    skewness = (centered.pow(3).mean() / var.pow(1.5)).item()

    flat = abs_act.flatten()
    mean_abs = flat.mean().item()
    top1_ratio = flat.max().item() / (mean_abs + 1e-10)
    top10_vals, _ = flat.topk(min(10, len(flat)))
    top10_ratio = top10_vals.mean().item() / (mean_abs + 1e-10)

    sorted_abs, _ = flat.sort()
    n = len(sorted_abs)
    gini = (2 * torch.arange(1, n+1, device=sorted_abs.device).float() @ sorted_abs - (n + 1) * sorted_abs.sum()) / (n * sorted_abs.sum() + 1e-10)

    return ActivationProfile(
        layer_idx=layer_idx,
        mean=mean_val, std=std_val, median=median_val,
        pct_near_zero=pct_near_zero, pct_negative=pct_negative,
        pct_exactly_zero=pct_exactly_zero, natural_sparsity=natural_sparsity,
        kurtosis=kurtosis, skewness=skewness,
        max_val=max_val, min_val=min_val,
        top1_ratio=top1_ratio, top10_ratio=top10_ratio,
        gini_coefficient=gini.item(),
    )


def run_activation_profiling(
    inspector: ModelInspector,
    dataset: TextDataset,
    batch_size: int = 4,
    max_batches: int = 16,
) -> List[ActivationProfile]:
    """Study 1: STREAMING version — one layer at a time."""
    print("\n" + "="*80)
    print("STUDY 1: Activation Profiling")
    print("="*80)

    profiles = []
    for layer_idx in range(inspector.num_layers):
        act = collect_single_layer(inspector, dataset, layer_idx,
                                   batch_size=batch_size, max_batches=max_batches)
        profile = compute_activation_profile(act, layer_idx)
        profiles.append(profile)
        print(f"  Layer {layer_idx:2d}: near_zero={profile.pct_near_zero:.1%}, "
              f"kurtosis={profile.kurtosis:.1f}, top1_ratio={profile.top1_ratio:.1f}, "
              f"gini={profile.gini_coefficient:.3f}")
        del act
        gc.collect()

    return profiles


# =============================================================================
# Study 4: Massive Activation Scan (streaming version)
# =============================================================================

@dataclass
class MassiveActivationReport:
    layer_idx: int
    neuron_mean_activation: torch.Tensor
    neuron_variance: torch.Tensor
    massive_neuron_indices: List[int]
    massive_neuron_ratios: List[float]
    input_agnostic_indices: List[int]
    input_agnostic_values: List[float]


def run_massive_activation_scan(
    inspector: ModelInspector,
    dataset: TextDataset,
    batch_size: int = 4,
    max_batches: int = 16,
    massive_threshold: float = 50.0,
) -> List[MassiveActivationReport]:
    """Study 4: STREAMING massive activation scan."""
    print("\n" + "="*80)
    print("STUDY 4: Massive Activation Scan")
    print("="*80)

    reports = []
    for layer_idx in range(inspector.num_layers):
        act = collect_single_layer(inspector, dataset, layer_idx,
                                   batch_size=batch_size, max_batches=max_batches)

        neuron_mean = act.abs().mean(dim=0)
        neuron_var = act.var(dim=0)

        layer_median = neuron_mean.median().item()

        massive_mask = neuron_mean > (massive_threshold * layer_median)
        massive_indices = massive_mask.nonzero(as_tuple=True)[0].tolist()
        massive_ratios = (neuron_mean[massive_mask] / (layer_median + 1e-10)).tolist()

        cv = neuron_var.sqrt() / (neuron_mean + 1e-10)
        agnostic_mask = (cv < 0.1) & (neuron_mean > 10 * layer_median)
        agnostic_indices = agnostic_mask.nonzero(as_tuple=True)[0].tolist()
        agnostic_values = neuron_mean[agnostic_mask].tolist()

        report = MassiveActivationReport(
            layer_idx=layer_idx,
            neuron_mean_activation=neuron_mean,
            neuron_variance=neuron_var,
            massive_neuron_indices=massive_indices,
            massive_neuron_ratios=massive_ratios,
            input_agnostic_indices=agnostic_indices,
            input_agnostic_values=agnostic_values,
        )
        reports.append(report)

        n_massive = len(massive_indices)
        n_agnostic = len(agnostic_indices)
        if n_massive > 0 or n_agnostic > 0:
            print(f"  Layer {layer_idx:2d}: {n_massive} massive neurons, "
                  f"{n_agnostic} input-agnostic. "
                  f"Top ratio: {max(massive_ratios) if massive_ratios else 0:.0f}x")
        else:
            print(f"  Layer {layer_idx:2d}: No massive activations found "
                  f"(max ratio: {neuron_mean.max().item() / (layer_median + 1e-10):.1f}x)")

        del act
        gc.collect()

    return reports

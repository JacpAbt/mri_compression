"""
Study 8: Sparsity Structure Analysis.

Analyzes the structure and patterns of which neurons fire and when.
Key questions:
- Is sparsity uniform across token positions, or do certain positions
  activate more neurons? (e.g., first token, punctuation)
- Do neurons fire in correlated groups (clusters)?
- Is the sparsity pattern consistent across different inputs,
  or is it truly input-adaptive?

This determines whether static masks (same for all inputs) suffice or
dynamic/input-dependent masks (CATS) are needed.

Streaming version: processes one layer at a time for memory efficiency.
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
# Study 8: Sparsity Structure Analysis (streaming version)
# =============================================================================

@dataclass
class SparsityStructureReport:
    layer_idx: int
    token_sparsity_variance: float
    position_dependent_sparsity: torch.Tensor
    neuron_specialization_score: float
    co_activation_clusters: int
    activation_consistency: float


def run_sparsity_structure_analysis(
    inspector: ModelInspector,
    dataset: TextDataset,
    batch_size: int = 4,
    max_batches: int = 16,
) -> List[SparsityStructureReport]:
    """Study 8: STREAMING sparsity structure — one layer at a time."""
    print("\n" + "="*80)
    print("STUDY 8: Sparsity Structure Analysis")
    print("="*80)

    reports = []
    for layer_idx in range(inspector.num_layers):
        # preserve_shape=True to keep (batch, seq_len, D) for position analysis
        all_act = collect_single_layer(inspector, dataset, layer_idx,
                                       batch_size=batch_size, max_batches=max_batches,
                                       preserve_shape=True)

        B, S, D = all_act.shape
        firing = (all_act.abs() > 0.01).float()

        # 1. Position-dependent sparsity
        position_sparsity = 1.0 - firing.mean(dim=(0, 2))
        token_sparsity_var = position_sparsity.var().item()

        # 2. Neuron activation consistency
        neuron_activation_rate = firing.reshape(-1, D).mean(dim=0)
        consistency_per_neuron = (neuron_activation_rate - 0.5).abs() * 2
        activation_consistency = consistency_per_neuron.mean().item()

        # 3. Specialization entropy
        rates = neuron_activation_rate.clamp(0.001, 0.999)
        entropy = -(rates * rates.log() + (1-rates) * (1-rates).log()).mean().item()

        # 4. Co-activation clustering (sampled)
        sample_size = min(256, D)
        sampled_idx = torch.randperm(D)[:sample_size]
        sampled_firing = firing.reshape(-1, D)[:, sampled_idx]

        if sampled_firing.shape[0] > 1000:
            token_sample = torch.randperm(sampled_firing.shape[0])[:1000]
            sampled_firing = sampled_firing[token_sample]

        corr = torch.corrcoef(sampled_firing.T)
        n_high_corr = (corr.abs() > 0.5).float().sum().item() / sample_size

        report = SparsityStructureReport(
            layer_idx=layer_idx,
            token_sparsity_variance=token_sparsity_var,
            position_dependent_sparsity=position_sparsity,
            neuron_specialization_score=entropy,
            co_activation_clusters=int(n_high_corr),
            activation_consistency=activation_consistency,
        )
        reports.append(report)

        print(f"  Layer {layer_idx:2d}: pos_sparsity_var={token_sparsity_var:.4f}, "
              f"consistency={activation_consistency:.3f}, "
              f"specialization_entropy={entropy:.3f}, "
              f"co_activation_degree={n_high_corr:.1f}")

        del all_act, firing, sampled_firing, corr
        gc.collect()

    return reports

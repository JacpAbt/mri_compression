"""
Study 5: Dead/Dormant Neuron Analysis.

Census of never/rarely-firing neurons (Voita et al., 2023). Categorizes
each neuron by how often it fires across all inputs: dead, dormant, rare,
or hyperactive. This reveals candidates for free pruning (dead/dormant)
vs structural neurons (hyperactive = possibly massive activations).

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
# Study 5: Dead/Dormant Neuron Analysis (streaming version)
# =============================================================================

@dataclass
class DeadNeuronReport:
    layer_idx: int
    total_neurons: int
    dead_count: int
    dormant_count: int
    rare_count: int
    hyperactive_count: int
    activation_rate: torch.Tensor


def run_dead_neuron_analysis(
    inspector: ModelInspector,
    dataset: TextDataset,
    batch_size: int = 4,
    max_batches: int = 16,
    fire_threshold: float = 0.01,
) -> List[DeadNeuronReport]:
    """Study 5: STREAMING dead/dormant neuron census."""
    print("\n" + "="*80)
    print("STUDY 5: Dead/Dormant Neuron Analysis")
    print("="*80)

    reports = []
    for layer_idx in range(inspector.num_layers):
        act = collect_single_layer(inspector, dataset, layer_idx,
                                   batch_size=batch_size, max_batches=max_batches)

        firing = (act.abs() > fire_threshold).float()
        activation_rate = firing.mean(dim=0)

        total = activation_rate.shape[0]
        dead = (activation_rate == 0).sum().item()
        dormant = ((activation_rate > 0) & (activation_rate < 0.01)).sum().item()
        rare = ((activation_rate >= 0.01) & (activation_rate < 0.10)).sum().item()
        hyperactive = (activation_rate > 0.99).sum().item()

        report = DeadNeuronReport(
            layer_idx=layer_idx,
            total_neurons=total,
            dead_count=int(dead),
            dormant_count=int(dormant),
            rare_count=int(rare),
            hyperactive_count=int(hyperactive),
            activation_rate=activation_rate,
        )
        reports.append(report)

        print(f"  Layer {layer_idx:2d}: dead={dead:4.0f} ({dead/total:.1%}), "
              f"dormant={dormant:4.0f} ({dormant/total:.1%}), "
              f"rare={rare:4.0f}, hyperactive={hyperactive:4.0f} ({hyperactive/total:.1%})")

        del act, firing
        gc.collect()

    return reports

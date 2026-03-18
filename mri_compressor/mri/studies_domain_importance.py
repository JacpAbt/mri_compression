"""
Study 22: Domain-Conditional Wanda Importance
===============================================
Computes Wanda scores (weight_norm * activation_norm) per neuron,
per domain, per layer.

Key insight from Study 11: ALL neurons fire universally across domains
(domain_specificity ~ 0). But firing != mattering. A neuron can fire
identically for English and Code but be critical only for Code.

This study measures per-domain *importance* via Wanda scores, revealing:
- Domain-critical neurons (top 10% by domain Wanda score)
- Domain-unnecessary neurons (bottom 20% AND below global median)

The latter can be safely removed for domain-specific compression.
"""

import gc
import logging
import torch
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

from ..model_utils import ModelInspector, collect_single_layer
from .studies_domain import load_domain_datasets
from ..data_utils import TextDataset

logger = logging.getLogger(__name__)


@dataclass
class DomainWandaReport:
    """Per-layer domain-conditional importance report."""
    layer_idx: int
    domains: list
    domain_wanda_scores: Dict[str, torch.Tensor]  # domain -> (intermediate_size,)
    global_mean_wanda: torch.Tensor                # mean across domains
    n_domain_critical: Dict[str, int]              # top 10% per domain
    n_domain_unnecessary: Dict[str, int]           # bottom 20% AND below global median


def compute_domain_wanda_scores_streaming(
    inspector: ModelInspector,
    domain_datasets: Dict[str, TextDataset],
    batch_size: int = 4,
    max_batches: int = 16,
    prior_results: Optional[Dict] = None,
) -> Dict[str, Dict[int, torch.Tensor]]:
    """
    Compute per-domain Wanda scores layer-by-layer (streaming).

    Outer loop: layers. Inner loop: domains.
    One layer at a time to minimize memory usage.

    Wanda score for neuron j in domain d at layer l:
        wanda[d][l][j] = ||activation_j||_2 * ||down_proj_weight[:, j]||_1

    Returns:
        dict[domain][layer_idx] -> tensor(intermediate_size,)
    """
    domains = sorted(domain_datasets.keys())
    domain_wanda = {d: {} for d in domains}

    for layer_idx in range(inspector.num_layers):
        if layer_idx % 6 == 0:
            print(f"    Layer {layer_idx}/{inspector.num_layers}...")

        # Get weight norms once per layer (shared across domains).
        # Use L2 norm to match canonical Wanda: w_norm = ||W[:, j]||_2.
        # The original used L1 (.abs().mean()), which is inconsistent with the
        # L2 activation norm used below; L2 × L2 matches Sun et al. (Wanda paper).
        down_w = inspector.mlp_layers[layer_idx].down_proj.weight.data.cpu().float()
        # down_proj: [hidden_size, intermediate_size]
        # Weight norm per neuron (column norm)
        intermediate_size = down_w.shape[1] if down_w.shape[0] < down_w.shape[1] else down_w.shape[0]
        if down_w.shape[0] == intermediate_size:
            # Conv1D layout: (intermediate, hidden) -> L2 row norms
            weight_norm = down_w.norm(p=2, dim=1)
        else:
            # Standard layout: (hidden, intermediate) -> L2 column norms
            weight_norm = down_w.norm(p=2, dim=0)

        for domain_name in domains:
            dataset = domain_datasets[domain_name]
            act = collect_single_layer(
                inspector, dataset, layer_idx,
                batch_size=batch_size, max_batches=max_batches,
            )
            # act: (N, D) in float32 on CPU

            # Activation L2 norm per neuron across all tokens
            act_norm = act.float().norm(p=2, dim=0)  # (intermediate_size,)

            # Wanda score = weight_norm * activation_norm
            score = weight_norm * act_norm
            domain_wanda[domain_name][layer_idx] = score

            del act, act_norm

        del down_w, weight_norm
        gc.collect()
        torch.cuda.empty_cache()

    return domain_wanda


def compute_domain_necessity(
    domain_wanda: Dict[str, Dict[int, torch.Tensor]],
    num_layers: int,
    critical_pct: float = 0.90,
    unnecessary_pct: float = 0.20,
) -> List[DomainWandaReport]:
    """
    From per-domain Wanda scores, compute per-layer necessity reports.

    Args:
        domain_wanda: dict[domain][layer_idx] -> tensor(intermediate_size,)
        num_layers: Number of transformer layers.
        critical_pct: Percentile threshold for domain-critical (top 10%).
        unnecessary_pct: Fraction threshold for domain-unnecessary (bottom 20%).

    Returns:
        List of DomainWandaReport, one per layer.
    """
    domains = sorted(domain_wanda.keys())
    reports = []

    for layer_idx in range(num_layers):
        scores = {}
        for d in domains:
            scores[d] = domain_wanda[d].get(layer_idx)
            if scores[d] is None:
                continue

        # Skip layers with missing data
        available = [d for d in domains if scores.get(d) is not None]
        if not available:
            continue

        # Stack into matrix: (num_domains, intermediate_size)
        score_matrix = torch.stack([scores[d] for d in available])
        global_mean = score_matrix.mean(dim=0)

        # Per-layer safety gate: median of the cross-domain mean Wanda score for
        # this specific layer. Computed fresh for each layer so early layers (with
        # higher activation magnitudes) don't contaminate the threshold for late layers.
        per_layer_median = global_mean.median().item()

        n_critical = {}
        n_unnecessary = {}

        for d_idx, d in enumerate(available):
            d_scores = scores[d]
            n_neurons = d_scores.shape[0]

            # Domain-critical: top (1 - critical_pct) fraction
            # critical_pct=0.90 means top 10%
            k_critical = max(1, int(n_neurons * (1.0 - critical_pct)))
            critical_threshold = d_scores.topk(k_critical).values[-1].item()
            n_critical[d] = int((d_scores >= critical_threshold).sum().item())

            # Domain-unnecessary: bottom unnecessary_pct AND below per-layer median
            k_unnecessary = int(n_neurons * unnecessary_pct)
            sorted_scores, _ = d_scores.sort()
            unnecessary_threshold = sorted_scores[min(k_unnecessary, n_neurons - 1)].item()
            is_unnecessary = (d_scores <= unnecessary_threshold) & (d_scores < per_layer_median)
            n_unnecessary[d] = int(is_unnecessary.sum().item())

        report = DomainWandaReport(
            layer_idx=layer_idx,
            domains=available,
            domain_wanda_scores=scores,
            global_mean_wanda=global_mean,
            n_domain_critical=n_critical,
            n_domain_unnecessary=n_unnecessary,
        )
        reports.append(report)

    return reports


def run_domain_conditional_importance(
    inspector: ModelInspector,
    batch_size: int = 4,
    max_batches: int = 16,
    samples_per_domain: int = 64,
    custom_domain_datasets: Optional[Dict[str, TextDataset]] = None,
    prior_results: Optional[Dict] = None,
) -> Dict:
    """
    Study 22: Domain-Conditional Wanda Importance.

    Entry point. Loads standard domains, optionally merges custom ones,
    runs streaming Wanda computation, and prints summary table.

    Args:
        inspector: ModelInspector instance.
        batch_size: Calibration batch size.
        max_batches: Max batches per domain per layer.
        samples_per_domain: Number of text chunks per domain.
        custom_domain_datasets: Optional dict of custom domain TextDatasets
            to merge with the standard set (english, math, code, italian).

    Returns:
        dict with "domain_wanda_reports", "domain_wanda_scores", "domains"
    """
    print("\n" + "=" * 80)
    print("STUDY 22: Domain-Conditional Wanda Importance")
    print("=" * 80)

    # Load standard domain datasets
    domain_datasets = load_domain_datasets(
        inspector.tokenizer, max_seq_len=512, samples_per_domain=samples_per_domain,
    )

    # Merge custom domain datasets if provided
    if custom_domain_datasets:
        for name, ds in custom_domain_datasets.items():
            if name not in domain_datasets:
                domain_datasets[name] = ds
                print(f"    Added custom domain: {name} ({len(ds)} samples)")
            else:
                print(f"    Custom domain '{name}' overrides standard domain")
                domain_datasets[name] = ds

    # Compute per-domain Wanda scores
    print("\n  Computing domain-conditional Wanda scores (streaming)...")
    domain_wanda = compute_domain_wanda_scores_streaming(
        inspector, domain_datasets,
        batch_size=batch_size, max_batches=max_batches,
        prior_results=prior_results,
    )

    # Compute necessity metrics
    print("\n  Computing domain necessity metrics...")
    reports = compute_domain_necessity(
        domain_wanda, inspector.num_layers,
    )

    # Print summary table
    domains = sorted(domain_wanda.keys())
    print(f"\n  {'Layer':>5} | ", end="")
    for d in domains:
        print(f"{'Crit(' + d[:3] + ')':>10} | {'Unnec(' + d[:3] + ')':>11} | ", end="")
    print(f"{'Mean Wanda':>10}")
    print("  " + "-" * (16 + 24 * len(domains)))

    total_critical = {d: 0 for d in domains}
    total_unnecessary = {d: 0 for d in domains}

    for r in reports:
        print(f"  {r.layer_idx:>5} | ", end="")
        for d in domains:
            crit = r.n_domain_critical.get(d, 0)
            unnec = r.n_domain_unnecessary.get(d, 0)
            total_critical[d] += crit
            total_unnecessary[d] += unnec
            print(f"{crit:>10} | {unnec:>11} | ", end="")
        print(f"{r.global_mean_wanda.mean():.4f}")

    # Print totals
    print(f"\n  Total domain-critical neurons (summed across layers):")
    for d in domains:
        print(f"    {d}: {total_critical[d]:,}")
    print(f"\n  Total domain-unnecessary neurons (summed across layers):")
    for d in domains:
        print(f"    {d}: {total_unnecessary[d]:,}")

    # Cross-domain analysis
    print(f"\n  Cross-domain importance divergence:")
    for r in reports[:5]:  # Show first 5 layers
        if len(domains) >= 2:
            d1, d2 = domains[0], domains[1]
            s1 = r.domain_wanda_scores.get(d1)
            s2 = r.domain_wanda_scores.get(d2)
            if s1 is not None and s2 is not None:
                cosine = (s1 @ s2) / (s1.norm() * s2.norm() + 1e-10)
                print(f"    Layer {r.layer_idx}: {d1} vs {d2} importance cosine = {cosine:.4f}")

    return {
        "domain_wanda_reports": reports,
        "domain_wanda_scores": domain_wanda,
        "domains": domains,
    }

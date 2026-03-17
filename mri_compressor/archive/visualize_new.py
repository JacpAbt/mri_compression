#!/usr/bin/env python3
"""
Visualizations for Studies 17-21.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def plot_study17_alignment(reports, model_name: str, output_dir: str):
    """Plot cross-layer activation subspace alignment."""
    if not reports:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Study 17: Cross-Layer Subspace Alignment — {model_name}", fontsize=14)

    layers_a = [r.layer_a for r in reports]
    cka_vals = [r.cka_linear for r in reports]
    overlap_vals = [r.top_subspace_overlap for r in reports]
    cosine_vals = [r.residual_delta_cosine for r in reports]
    merge_vals = [r.merge_score for r in reports]

    # CKA
    ax = axes[0, 0]
    ax.bar(layers_a, cka_vals, color="steelblue", alpha=0.8)
    ax.set_xlabel("Layer pair (a → a+1)")
    ax.set_ylabel("Linear CKA")
    ax.set_title("Linear CKA Between Consecutive Layers")
    ax.axhline(y=0.8, color="red", linestyle="--", alpha=0.5, label="High similarity")
    ax.legend()

    # Subspace overlap
    ax = axes[0, 1]
    ax.bar(layers_a, overlap_vals, color="coral", alpha=0.8)
    ax.set_xlabel("Layer pair")
    ax.set_ylabel("Top-k Subspace Overlap")
    ax.set_title("Principal Subspace Overlap")

    # Cosine similarity
    ax = axes[1, 0]
    ax.bar(layers_a, cosine_vals, color="mediumseagreen", alpha=0.8)
    ax.set_xlabel("Layer pair")
    ax.set_ylabel("Mean Cosine Similarity")
    ax.set_title("Residual Delta Cosine Similarity")

    # Merge score
    ax = axes[1, 1]
    colors = ["red" if v > 0.5 else "steelblue" for v in merge_vals]
    ax.bar(layers_a, merge_vals, color=colors, alpha=0.8)
    ax.set_xlabel("Layer pair")
    ax.set_ylabel("Merge Score")
    ax.set_title("Composite Merge Score (>0.5 = candidate)")
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5)

    plt.tight_layout()
    path = Path(output_dir) / "study17_alignment.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_study18_rank(reports, model_name: str, output_dir: str):
    """Plot effective weight matrix rank analysis."""
    if not reports:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Study 18: Effective Weight Matrix Rank — {model_name}", fontsize=14)

    layers = [r.layer_idx for r in reports]

    # Rank ratios at 95%
    ax = axes[0, 0]
    ax.plot(layers, [r.gate_proj_ratio95 for r in reports], 'o-', label="gate_proj", markersize=3)
    ax.plot(layers, [r.up_proj_ratio95 for r in reports], 's-', label="up_proj", markersize=3)
    ax.plot(layers, [r.down_proj_ratio95 for r in reports], '^-', label="down_proj", markersize=3)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Rank / Full Rank")
    ax.set_title("Rank Ratio at 95% Energy")
    ax.legend()
    ax.set_ylim(0, 1.05)

    # Rank ratios at 99%
    ax = axes[0, 1]
    ax.plot(layers, [r.gate_proj_ratio95 for r in reports], 'o-', label="rank@95%", alpha=0.7, markersize=3)
    ax.plot(layers, [r.avg_ratio99 for r in reports], 's-', label="rank@99%", alpha=0.7, markersize=3)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Avg Rank Ratio")
    ax.set_title("Average Rank Ratio: 95% vs 99%")
    ax.legend()

    # Estimated param saving
    ax = axes[1, 0]
    savings = [r.estimated_param_saving_95 for r in reports]
    colors = ["red" if s > 0.3 else "steelblue" for s in savings]
    ax.bar(layers, savings, color=colors, alpha=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Param Saving Fraction")
    ax.set_title("Estimated Param Saving from Rank-95% Factorization")
    ax.axhline(y=0.3, color="red", linestyle="--", alpha=0.5, label=">30% saving")
    ax.legend()

    # Absolute ranks
    ax = axes[1, 1]
    ax.plot(layers, [r.gate_proj_rank95 for r in reports], 'o-', label="gate_proj r95", markersize=3)
    ax.plot(layers, [r.gate_proj_full_rank for r in reports], '--', label="full rank", alpha=0.5)
    ax.plot(layers, [r.down_proj_rank95 for r in reports], '^-', label="down_proj r95", markersize=3)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Rank")
    ax.set_title("Absolute Rank vs Full Rank")
    ax.legend()

    plt.tight_layout()
    path = Path(output_dir) / "study18_rank.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_study19_head_clusters(report, model_name: str, output_dir: str):
    """Plot attention head clustering results."""
    if report.num_heads_total == 0:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Study 19: Attention Head Functional Clustering — {model_name}", fontsize=14)

    # Cluster size distribution
    ax = axes[0]
    sizes = report.cluster_sizes
    if sizes:
        max_sz = max(sizes)
        bins = range(1, max_sz + 2)
        ax.hist(sizes, bins=bins, color="steelblue", alpha=0.8, edgecolor="black")
    ax.set_xlabel("Cluster Size")
    ax.set_ylabel("Count")
    ax.set_title(f"Cluster Size Distribution\n({report.num_clusters} clusters, "
                 f"{report.num_redundant_heads} redundant heads)")

    # NN similarity distribution
    ax = axes[1]
    nn_sims = list(report.head_nn_similarity.values())
    if nn_sims:
        ax.hist(nn_sims, bins=50, color="coral", alpha=0.8, edgecolor="black")
        ax.axvline(x=0.9, color="red", linestyle="--", label="Clustering threshold")
        ax.legend()
    ax.set_xlabel("Nearest-Neighbor Cosine Similarity")
    ax.set_ylabel("Count")
    ax.set_title("Head-to-Head Similarity Distribution")

    # Per-layer prunable heads
    ax = axes[2]
    if report.prunable_heads:
        from collections import Counter
        layer_counts = Counter(l for l, h in report.prunable_heads)
        all_layers = sorted(layer_counts.keys())
        counts = [layer_counts.get(l, 0) for l in all_layers]
        ax.bar(all_layers, counts, color="mediumseagreen", alpha=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Prunable Heads")
    ax.set_title("Prunable Heads per Layer")

    plt.tight_layout()
    path = Path(output_dir) / "study19_head_clusters.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_study20_static_dynamic(reports, model_name: str, output_dir: str):
    """Plot static vs dynamic activation decomposition."""
    if not reports:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Study 20: Static vs Dynamic Activation Decomposition — {model_name}", fontsize=14)

    layers = [r.layer_idx for r in reports]

    # Static fraction per layer
    ax = axes[0, 0]
    static = [r.static_fraction for r in reports]
    ax.bar(layers, static, color="steelblue", alpha=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Static Fraction")
    ax.set_title("Static Fraction of Activation Energy")
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5)

    # Foldable neurons per layer
    ax = axes[0, 1]
    foldable = [r.foldable_neuron_count for r in reports]
    ax.bar(layers, foldable, color="coral", alpha=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Foldable Neurons (>95% static)")
    ax.set_title("Neurons Replaceable by Bias Terms")

    # Static vs dynamic neuron counts (stacked)
    ax = axes[1, 0]
    mostly_static = [r.n_mostly_static for r in reports]
    mixed = [r.n_mixed for r in reports]
    mostly_dynamic = [r.n_mostly_dynamic for r in reports]
    ax.bar(layers, mostly_static, label="Mostly static (>90%)", color="steelblue", alpha=0.8)
    ax.bar(layers, mixed, bottom=mostly_static, label="Mixed", color="gold", alpha=0.8)
    ax.bar(layers, mostly_dynamic,
           bottom=[s + m for s, m in zip(mostly_static, mixed)],
           label="Mostly dynamic (>90%)", color="coral", alpha=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Neuron Count")
    ax.set_title("Neuron Classification by Static/Dynamic Balance")
    ax.legend(fontsize=8)

    # Param saving potential
    ax = axes[1, 1]
    savings = [r.foldable_param_saving for r in reports]
    ax.bar(layers, savings, color="mediumseagreen", alpha=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Fraction of MLP Params Saveable")
    ax.set_title("Estimated MLP Param Savings from Static Folding")

    plt.tight_layout()
    path = Path(output_dir) / "study20_static_dynamic.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_study21_magnitude_divergence(reports, model_name: str, output_dir: str):
    """Plot token-conditional magnitude divergence."""
    if not reports:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Study 21: Token-Conditional Magnitude Divergence — {model_name}", fontsize=14)

    layers = [r.layer_idx for r in reports]
    domains = reports[0].domains if reports else []

    # Domain-sensitive neuron count
    ax = axes[0, 0]
    sensitive = [r.n_domain_sensitive_neurons for r in reports]
    invariant = [r.n_domain_invariant_neurons for r in reports]
    ax.plot(layers, sensitive, 'o-', label="Sensitive (>2x ratio)", color="coral", markersize=3)
    ax.plot(layers, invariant, 's-', label="Invariant (<1.2x ratio)", color="steelblue", markersize=3)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Neuron Count")
    ax.set_title("Domain-Sensitive vs Invariant Neurons")
    ax.legend()

    # Coefficient of variation
    ax = axes[0, 1]
    cv = [r.magnitude_cv for r in reports]
    ax.bar(layers, cv, color="mediumseagreen", alpha=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("CV of Magnitude Across Domains")
    ax.set_title("Magnitude Coefficient of Variation")

    # Mean magnitudes per domain
    ax = axes[1, 0]
    for dom in domains:
        mags = [r.domain_mean_magnitudes.get(dom, 0) for r in reports]
        ax.plot(layers, mags, 'o-', label=dom, markersize=3)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean |Activation|")
    ax.set_title("Per-Domain Mean Activation Magnitude")
    ax.legend(fontsize=8)

    # Domain sensitivity score
    ax = axes[1, 1]
    scores = [r.domain_sensitivity_score for r in reports]
    colors = ["red" if s > 0.3 else "steelblue" for s in scores]
    ax.bar(layers, scores, color=colors, alpha=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Sensitivity Score")
    ax.set_title("Domain Sensitivity Score (0-1)")

    plt.tight_layout()
    path = Path(output_dir) / "study21_magnitude_divergence.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from pathlib import Path
from collections import defaultdict


def set_style():
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': '#f8f9fa',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
    })


def plot_study12_motifs(reports, model_name: str, output_dir: str):
    """Plot Study 12: Cross-layer motif structure (lift-based)."""
    set_style()
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Study 12: Feed-Forward Loop Motifs — {model_name}", fontsize=14, fontweight='bold')

    layers = [r.source_layer for r in reports]

    # Strong connections
    axes[0, 0].bar(layers, [r.n_strong_connections for r in reports], color='#2196F3', alpha=0.7)
    axes[0, 0].set_xlabel('Source Layer')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Strong Connections (lift > threshold)')

    # Bypass vs bottleneck
    axes[0, 1].bar(layers, [r.n_bypass_neurons for r in reports], color='#4CAF50', alpha=0.7, label='Bypass (safe)')
    axes[0, 1].bar(layers, [-r.n_bottleneck_neurons for r in reports], color='#F44336', alpha=0.7, label='Bottleneck (critical)')
    axes[0, 1].set_xlabel('Source Layer')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Bypass vs Bottleneck Neurons')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].axhline(y=0, color='black', linewidth=0.5)

    # Relay density
    axes[0, 2].plot(layers, [r.relay_density for r in reports], 'o-', color='#9C27B0', markersize=4)
    axes[0, 2].set_xlabel('Source Layer')
    axes[0, 2].set_ylabel('Density')
    axes[0, 2].set_title('Relay Density (fraction strong)')

    # Mean lift
    axes[1, 0].plot(layers, [r.mean_lift for r in reports], 'o-', color='#FF9800', markersize=4)
    axes[1, 0].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='No association')
    axes[1, 0].set_xlabel('Source Layer')
    axes[1, 0].set_ylabel('Mean Lift')
    axes[1, 0].set_title('Mean Lift (strong connections)')
    axes[1, 0].legend(fontsize=8)

    # Inhibitory
    axes[1, 1].bar(layers, [r.n_inhibitory for r in reports], color='#F44336', alpha=0.7)
    axes[1, 1].set_xlabel('Source Layer')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Inhibitory Connections (lift < 1/threshold)')

    # Max lift
    axes[1, 2].plot(layers, [r.max_lift for r in reports], 'o-', color='#E91E63', markersize=4)
    axes[1, 2].set_xlabel('Source Layer')
    axes[1, 2].set_ylabel('Max Lift')
    axes[1, 2].set_title('Strongest Single Connection')

    plt.tight_layout()
    path = Path(output_dir) / "study12_motifs.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


def plot_study13_bottleneck(reports, model_name: str, output_dir: str):
    """Plot Study 13: Information Bottleneck profile."""
    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Study 13: Information Bottleneck — {model_name}", fontsize=14, fontweight='bold')

    layers = [r.layer_idx for r in reports]

    # Compression ratio (I(X;Z) proxy)
    axes[0, 0].plot(layers, [r.compression_ratio for r in reports], 'o-', color='#E91E63', markersize=4)
    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('Reconstruction MSE / Baseline')
    axes[0, 0].set_title('Input Information Compression')
    axes[0, 0].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='baseline')
    axes[0, 0].legend(fontsize=8)

    # Partial PPL (I(Z;Y) proxy)
    ppls = [r.partial_lm_loss for r in reports]
    max_finite = max([p for p in ppls if p < float('inf')] or [100])
    ppls_clipped = [min(p, max_finite * 1.2) for p in ppls]
    axes[0, 1].plot(layers, ppls_clipped, 'o-', color='#2196F3', markersize=4)
    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('PPL (layers 0..L only)')
    axes[0, 1].set_title('Output Information: PPL with only layers 0..L')
    axes[0, 1].set_yscale('log')

    # Information retained
    axes[1, 0].plot(layers, [r.information_retained for r in reports], 'o-', color='#4CAF50', markersize=4)
    axes[1, 0].set_xlabel('Layer')
    axes[1, 0].set_ylabel('Information Retained')
    axes[1, 0].set_title('Fraction of Output Info Available at Layer L')
    axes[1, 0].set_ylim(-0.05, 1.05)
    axes[1, 0].axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)

    # Information plane: compression vs PPL
    scatter = axes[1, 1].scatter(
        [r.compression_ratio for r in reports],
        ppls_clipped,
        c=layers, cmap='viridis', s=60, zorder=5)
    for r in reports:
        axes[1, 1].annotate(str(r.layer_idx),
                           (r.compression_ratio, min(r.partial_lm_loss, max_finite * 1.2)),
                           fontsize=6, ha='center', va='bottom')
    axes[1, 1].set_xlabel('Compression (higher = more compressed)')
    axes[1, 1].set_ylabel('Partial PPL (lower = more output info)')
    axes[1, 1].set_title('Information Plane')
    axes[1, 1].set_yscale('log')
    plt.colorbar(scatter, ax=axes[1, 1], label='Layer')

    plt.tight_layout()
    path = Path(output_dir) / "study13_bottleneck.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


def plot_study14_redundancy(reports, model_name: str, output_dir: str):
    """Plot Study 14: Functional Redundancy Census."""
    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Study 14: Functional Redundancy — {model_name}", fontsize=14, fontweight='bold')

    layers = [r.layer_idx for r in reports]

    # Mean max similarity
    axes[0, 0].plot(layers, [r.mean_max_similarity for r in reports], 'o-', color='#2196F3', label='Mean', markersize=4)
    axes[0, 0].plot(layers, [r.median_max_similarity for r in reports], 's--', color='#FF9800', label='Median', markersize=4)
    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('Max Cosine Similarity')
    axes[0, 0].set_title('Neuron Redundancy Level')
    axes[0, 0].legend(fontsize=8)

    # Redundant vs keystone counts
    axes[0, 1].bar(layers, [r.n_highly_redundant for r in reports], color='#F44336', alpha=0.7, label='Redundant (>0.9)')
    axes[0, 1].bar(layers, [-r.n_keystone for r in reports], color='#4CAF50', alpha=0.7, label='Keystone (<0.3)')
    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Redundant vs Keystone Neurons')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].axhline(y=0, color='black', linewidth=0.5)

    # Safe to prune
    total = [r.n_neurons for r in reports]
    safe = [r.safe_to_prune_count for r in reports]
    pct_safe = [s / t * 100 for s, t in zip(safe, total)]
    axes[1, 0].bar(layers, pct_safe, color='#9C27B0', alpha=0.7)
    axes[1, 0].set_xlabel('Layer')
    axes[1, 0].set_ylabel('% Safe to Prune')
    axes[1, 0].set_title('Pruning Opportunity (max_sim > 0.8)')

    # Distribution of max similarities for example layers
    example_layers = [0, len(reports)//4, len(reports)//2, 3*len(reports)//4, len(reports)-1]
    example_layers = sorted(set(l for l in example_layers if l < len(reports)))
    colors = plt.cm.viridis(np.linspace(0, 1, len(example_layers)))
    for el, color in zip(example_layers, colors):
        r = reports[el]
        vals = r.max_similarity_distribution.numpy()
        axes[1, 1].hist(vals, bins=50, alpha=0.4, color=color, label=f'Layer {r.layer_idx}', density=True)
    axes[1, 1].set_xlabel('Max Similarity to Nearest Neighbor')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Redundancy Distribution (select layers)')
    axes[1, 1].legend(fontsize=7)

    plt.tight_layout()
    path = Path(output_dir) / "study14_redundancy.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


def plot_study15_cascade(reports, model_name: str, output_dir: str):
    """Plot Study 15: Perturbation Cascade (fixed metrics)."""
    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Study 15: Perturbation Cascade — {model_name}", fontsize=14, fontweight='bold')

    by_layer = defaultdict(list)
    for r in reports:
        by_layer[r.layer_idx].append(r)

    source_layers = sorted(by_layer.keys())

    # Mean damping ratio per source layer
    avg_damping = [np.mean([r.mean_damping_ratio for r in by_layer[l]]) for l in source_layers]
    colors = ['#F44336' if d > 1 else '#4CAF50' for d in avg_damping]
    axes[0, 0].bar(source_layers, avg_damping, color=colors, alpha=0.7)
    axes[0, 0].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='neutral')
    axes[0, 0].set_xlabel('Source Layer')
    axes[0, 0].set_ylabel('Mean Damping Ratio')
    axes[0, 0].set_title('Local Damping (< 1 = decays, > 1 = amplifies)')
    axes[0, 0].legend(fontsize=8)

    # Max amplification
    avg_amps = [np.mean([r.max_amplification for r in by_layer[l]]) for l in source_layers]
    axes[0, 1].bar(source_layers, avg_amps, color='#FF9800', alpha=0.7)
    axes[0, 1].set_xlabel('Source Layer')
    axes[0, 1].set_ylabel('Max Amplification (x)')
    axes[0, 1].set_title('Peak Perturbation / Initial')

    # Peak offset
    avg_peaks = [np.mean([r.peak_layer_offset for r in by_layer[l]]) for l in source_layers]
    axes[1, 0].bar(source_layers, avg_peaks, color='#2196F3', alpha=0.7)
    axes[1, 0].set_xlabel('Source Layer')
    axes[1, 0].set_ylabel('Peak Offset (layers)')
    axes[1, 0].set_title('Where Perturbation Peaks')

    # Normalized cascade curves for example neurons
    colors_cascade = plt.cm.Set2(np.linspace(0, 1, min(8, len(reports))))
    for i, r in enumerate(reports[:8]):
        if r.cascade_normalized:
            x = list(range(r.layer_idx + 1, r.layer_idx + 1 + len(r.cascade_normalized)))
            axes[1, 1].plot(x, r.cascade_normalized, '-o', markersize=3,
                          color=colors_cascade[i],
                          label=f'L{r.layer_idx}:N{r.neuron_idx}', alpha=0.7)
    axes[1, 1].set_xlabel('Layer')
    axes[1, 1].set_ylabel('Perturbation / Baseline Norm')
    axes[1, 1].set_title('Normalized Cascade Curves')
    axes[1, 1].legend(fontsize=6, ncol=2)

    plt.tight_layout()
    path = Path(output_dir) / "study15_cascade.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


def plot_study16_phase(reports, model_name: str, output_dir: str):
    """Plot Study 16: Phase Transition."""
    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Study 16: Activation Phase Transition — {model_name}", fontsize=14, fontweight='bold')

    layers = [r.layer_idx for r in reports]
    alphas = [r.power_law_alpha for r in reports]

    # Power-law exponent across layers
    ax = axes[0, 0]
    colors = ['#F44336' if r.is_heavy_tailed else '#4CAF50' for r in reports]
    ax.bar(layers, alphas, color=colors, alpha=0.7)
    ax.axhline(y=2.5, color='orange', linestyle='--', label='Heavy tail boundary (α=2.5)')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Power-law α')
    ax.set_title('Tail Exponent (red = heavy-tailed)')
    ax.legend(fontsize=8)

    # Tail fraction
    axes[0, 1].bar(layers, [r.tail_fraction for r in reports], color='#9C27B0', alpha=0.7)
    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('Fraction > 10× median')
    axes[0, 1].set_title('Tail Fraction')

    # KS statistic (non-Gaussianity)
    axes[1, 0].plot(layers, [r.gaussian_fit_ks for r in reports], 'o-', color='#FF9800', markersize=4)
    axes[1, 0].set_xlabel('Layer')
    axes[1, 0].set_ylabel('KS Statistic')
    axes[1, 0].set_title('Non-Gaussianity (higher = less Gaussian)')

    # Activation entropy
    axes[1, 1].plot(layers, [r.activation_entropy for r in reports], 'o-', color='#2196F3', markersize=4)
    axes[1, 1].set_xlabel('Layer')
    axes[1, 1].set_ylabel('Entropy (nats)')
    axes[1, 1].set_title('Activation Distribution Entropy')

    plt.tight_layout()
    path = Path(output_dir) / "study16_phase.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")

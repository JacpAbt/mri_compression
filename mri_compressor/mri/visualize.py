"""
Visualization module for all studies (1-21).
Generates a comprehensive set of plots saved as PNG files.
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List, Optional
from pathlib import Path
from collections import defaultdict


def set_style():
    """Set consistent plot style."""
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


def plot_activation_profiles(profiles, model_name: str, output_dir: str):
    """Plot Study 1: Activation distribution profiles across layers."""
    set_style()

    n_layers = len(profiles)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Study 1: Activation Profiles — {model_name}", fontsize=14, fontweight='bold')

    layers = [p.layer_idx for p in profiles]

    # 1a. Natural sparsity by layer
    ax = axes[0, 0]
    ax.bar(layers, [p.pct_near_zero for p in profiles], color='#2196F3', alpha=0.7, label='Near-zero (<1% of max)')
    ax.bar(layers, [p.pct_exactly_zero for p in profiles], color='#F44336', alpha=0.7, label='Exactly zero')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Fraction')
    ax.set_title('Natural Sparsity by Layer')
    ax.legend(fontsize=8)

    # 1b. Kurtosis (outlier tendency) by layer
    ax = axes[0, 1]
    colors = ['#F44336' if p.kurtosis > 50 else '#2196F3' for p in profiles]
    ax.bar(layers, [p.kurtosis for p in profiles], color=colors, alpha=0.7)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Excess Kurtosis')
    ax.set_title('Kurtosis by Layer (red = extreme outliers)')
    ax.set_yscale('symlog')

    # 1c. Top-1 ratio (massive activation indicator)
    ax = axes[0, 2]
    ax.bar(layers, [p.top1_ratio for p in profiles], color='#9C27B0', alpha=0.7)
    ax.set_xlabel('Layer')
    ax.set_ylabel('max / mean ratio')
    ax.set_title('Top-1 Activation Ratio (Massive Activation Indicator)')

    # 1d. Gini coefficient (activation inequality)
    ax = axes[1, 0]
    ax.plot(layers, [p.gini_coefficient for p in profiles], 'o-', color='#FF9800', markersize=4)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Gini Coefficient')
    ax.set_title('Activation Inequality (higher = more unequal)')
    ax.axhline(y=0.5, linestyle='--', color='gray', alpha=0.5, label='Moderate inequality')
    ax.legend(fontsize=8)

    # 1e. Fraction negative (relevant for SiLU)
    ax = axes[1, 1]
    ax.bar(layers, [p.pct_negative for p in profiles], color='#009688', alpha=0.7)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Fraction Negative')
    ax.set_title('Fraction of Negative Activations')

    # 1f. Mean and std
    ax = axes[1, 2]
    means = [p.mean for p in profiles]
    stds = [p.std for p in profiles]
    ax.errorbar(layers, means, yerr=stds, fmt='o-', color='#3F51B5', capsize=3, markersize=4)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Activation Value')
    ax.set_title('Mean +/- Std of Activations')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/study1_activation_profiles.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: study1_activation_profiles.png")


def plot_gate_training(gate_results, model_name: str, output_dir: str):
    """Plot Study 2: Gate training dynamics and learned patterns."""
    set_style()

    sparsities = sorted(gate_results.keys())
    n_sp = len(sparsities)

    fig, axes = plt.subplots(2, max(n_sp, 2), figsize=(6 * max(n_sp, 2), 10))
    if n_sp == 1:
        axes = axes.reshape(2, -1)
    fig.suptitle(f"Study 2: Learned Gate Patterns — {model_name}", fontsize=14, fontweight='bold')

    for col, sp in enumerate(sparsities):
        patterns, metrics = gate_results[sp]
        n_layers = len(patterns)

        # Top row: Training loss curves
        ax = axes[0, col]
        ax.plot(metrics["lm_loss"], label='LM Loss', color='#2196F3', alpha=0.7)
        ax.plot(metrics["actual_sparsity"], label='Actual Sparsity', color='#F44336', alpha=0.7)
        ax.axhline(y=sp, linestyle='--', color='#F44336', alpha=0.3, label=f'Target ({sp:.0%})')
        ax.set_title(f'Training @ {sp:.0%} target sparsity')
        ax.set_xlabel('Step')
        ax.legend(fontsize=7)

        # Bottom row: Heatmap of gate values across layers x neurons
        ax = axes[1, col]
        # Create matrix: (n_layers, intermediate_size) - subsample neurons for visibility
        max_neurons_to_show = 200
        gate_matrix = []
        for layer_idx in sorted(patterns.keys()):
            vals = patterns[layer_idx]
            if len(vals) > max_neurons_to_show:
                # Sort neurons by gate value, take evenly spaced sample
                sorted_vals, _ = vals.sort()
                indices = torch.linspace(0, len(sorted_vals)-1, max_neurons_to_show).long()
                vals = sorted_vals[indices]
            gate_matrix.append(vals.numpy())

        gate_matrix = np.array(gate_matrix)
        im = ax.imshow(gate_matrix, aspect='auto', cmap='RdYlBu_r', vmin=0, vmax=1)
        ax.set_xlabel('Neuron (sorted by gate value)')
        ax.set_ylabel('Layer')
        ax.set_title(f'Gate Values @ {sp:.0%} sparsity')
        plt.colorbar(im, ax=ax, shrink=0.8)

    # Fill unused columns
    for col in range(n_sp, axes.shape[1]):
        axes[0, col].set_visible(False)
        axes[1, col].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/study2_gate_training.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: study2_gate_training.png")


def plot_wanda_scores(wanda_scores, model_name: str, output_dir: str):
    """Plot Study 3: Wanda importance score distributions."""
    set_style()

    n_layers = len(wanda_scores)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Study 3: Wanda Importance Scores — {model_name}", fontsize=14, fontweight='bold')

    # 3a. Distribution of scores per layer (box plot)
    ax = axes[0]
    data = [wanda_scores[i].numpy() for i in range(n_layers)]
    bp = ax.boxplot(data, positions=range(n_layers), widths=0.6,
                     patch_artist=True, showfliers=False)
    for patch in bp['boxes']:
        patch.set_facecolor('#2196F3')
        patch.set_alpha(0.5)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Wanda Score')
    ax.set_title('Wanda Score Distribution by Layer')

    # 3b. Heatmap of sorted scores
    ax = axes[1]
    score_matrix = []
    for i in range(n_layers):
        sorted_scores, _ = wanda_scores[i].sort(descending=True)
        # Subsample to 200 for visibility
        n = len(sorted_scores)
        idx = torch.linspace(0, n-1, min(200, n)).long()
        score_matrix.append(sorted_scores[idx].numpy())

    score_matrix = np.array(score_matrix)
    im = ax.imshow(score_matrix, aspect='auto', cmap='viridis')
    ax.set_xlabel('Neuron (sorted by importance)')
    ax.set_ylabel('Layer')
    ax.set_title('Wanda Scores (sorted, per layer)')
    plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/study3_wanda_scores.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: study3_wanda_scores.png")


def plot_massive_activations(reports, model_name: str, output_dir: str):
    """Plot Study 4: Massive activation scan results."""
    set_style()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Study 4: Massive Activation Scan — {model_name}", fontsize=14, fontweight='bold')

    layers = [r.layer_idx for r in reports]

    ax = axes[0]
    ax.bar(layers, [len(r.massive_neuron_indices) for r in reports],
           color='#F44336', alpha=0.7, label='Massive')
    ax.bar(layers, [len(r.input_agnostic_indices) for r in reports],
           color='#FF9800', alpha=0.7, label='Input-agnostic')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Count')
    ax.set_title('Massive & Input-Agnostic Neurons')
    ax.legend()

    ax = axes[1]
    max_ratios = [max(r.massive_neuron_ratios) if r.massive_neuron_ratios else 0 for r in reports]
    ax.bar(layers, max_ratios, color='#9C27B0', alpha=0.7)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Max ratio to layer median')
    ax.set_title('Largest Massive Activation Ratio')
    ax.set_yscale('symlog')

    # Neuron-level view for the layer with most massive activations
    ax = axes[2]
    max_layer = max(reports, key=lambda r: len(r.massive_neuron_indices))
    mean_act = max_layer.neuron_mean_activation.numpy()
    sorted_idx = np.argsort(mean_act)[::-1]
    ax.semilogy(mean_act[sorted_idx], color='#2196F3', alpha=0.7)
    ax.axhline(y=np.median(mean_act) * 50, color='red', linestyle='--', alpha=0.5,
               label=f'50x median threshold')
    ax.set_xlabel('Neuron (sorted by magnitude)')
    ax.set_ylabel('Mean |activation|')
    ax.set_title(f'Neuron Magnitudes — Layer {max_layer.layer_idx}')
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/study4_massive_activations.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: study4_massive_activations.png")


def plot_dead_neurons(reports, model_name: str, output_dir: str):
    """Plot Study 5: Dead/dormant neuron census."""
    set_style()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Study 5: Dead/Dormant Neuron Census — {model_name}", fontsize=14, fontweight='bold')

    layers = [r.layer_idx for r in reports]
    total = reports[0].total_neurons

    ax = axes[0]
    ax.bar(layers, [r.dead_count/total*100 for r in reports], label='Dead (never fire)',
           color='#F44336', alpha=0.7)
    ax.bar(layers, [r.dormant_count/total*100 for r in reports],
           bottom=[r.dead_count/total*100 for r in reports],
           label='Dormant (<1%)', color='#FF9800', alpha=0.7)
    ax.bar(layers, [r.rare_count/total*100 for r in reports],
           bottom=[(r.dead_count+r.dormant_count)/total*100 for r in reports],
           label='Rare (<10%)', color='#FFC107', alpha=0.7)
    ax.set_xlabel('Layer')
    ax.set_ylabel('% of neurons')
    ax.set_title('Inactive Neuron Breakdown')
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.bar(layers, [r.hyperactive_count/total*100 for r in reports],
           color='#4CAF50', alpha=0.7)
    ax.set_xlabel('Layer')
    ax.set_ylabel('% of neurons')
    ax.set_title('Hyperactive Neurons (fire >99% of time)')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/study5_dead_neurons.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: study5_dead_neurons.png")


def plot_attention_heads(reports, num_layers: int, model_name: str, output_dir: str):
    """Plot Study 6: Attention head analysis."""
    set_style()

    if not reports:
        print("  No attention data to plot.")
        return

    # Study 6 may return a mix of HeadImportanceReport (standard softmax-attention)
    # and LinearAttentionChannelReport (DeltaNet / linear-attention layers).
    # Only HeadImportanceReport objects have head_idx / entropy fields; filter to those.
    standard_reports = [r for r in reports if hasattr(r, 'head_idx')]
    if not standard_reports:
        print("  No standard-attention head data to plot (model may be fully linear-attention).")
        return

    num_heads = max(r.head_idx for r in standard_reports) + 1

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Study 6: Attention Head Analysis — {model_name}", fontsize=14, fontweight='bold')

    # Heatmap: entropy per head (standard softmax-attention layers only)
    entropy_matrix = np.zeros((num_layers, num_heads))
    first_token_matrix = np.zeros((num_layers, num_heads))
    concentration_matrix = np.zeros((num_layers, num_heads))

    for r in standard_reports:
        entropy_matrix[r.layer_idx, r.head_idx] = r.mean_entropy
        first_token_matrix[r.layer_idx, r.head_idx] = r.first_token_attention
        concentration_matrix[r.layer_idx, r.head_idx] = r.max_attention_concentration

    ax = axes[0]
    im = ax.imshow(entropy_matrix, aspect='auto', cmap='viridis')
    ax.set_xlabel('Head')
    ax.set_ylabel('Layer')
    ax.set_title('Attention Entropy (low = focused)')
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[1]
    im = ax.imshow(first_token_matrix, aspect='auto', cmap='Reds')
    ax.set_xlabel('Head')
    ax.set_ylabel('Layer')
    ax.set_title('First Token Attention (Attention Sink)')
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[2]
    im = ax.imshow(concentration_matrix, aspect='auto', cmap='magma')
    ax.set_xlabel('Head')
    ax.set_ylabel('Layer')
    ax.set_title('Max Attention Concentration')
    plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/study6_attention_heads.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: study6_attention_heads.png")


def plot_gate_wanda_correlation(correlation_reports, gate_results, wanda_scores,
                                 model_name: str, output_dir: str):
    """Plot Study 7: Gate vs Wanda correlation."""
    set_style()

    if not correlation_reports:
        print("  No correlation data to plot.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Study 7: Learned Gates vs Wanda Scores — {model_name}", fontsize=14, fontweight='bold')

    layers = [r.layer_idx for r in correlation_reports]

    ax = axes[0]
    ax.plot(layers, [r.pearson_r for r in correlation_reports], 'o-',
            color='#2196F3', label='Pearson r', markersize=4)
    ax.plot(layers, [r.spearman_rho for r in correlation_reports], 's-',
            color='#F44336', label='Spearman rho', markersize=4)
    ax.axhline(y=0, linestyle='--', color='gray', alpha=0.3)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Correlation')
    ax.set_title('Rank Correlation by Layer')
    ax.legend()
    ax.set_ylim(-0.3, 1.0)

    ax = axes[1]
    for frac, color in [(0.10, '#F44336'), (0.25, '#FF9800'), (0.50, '#4CAF50')]:
        overlaps = [r.top_k_overlap[frac] for r in correlation_reports]
        ax.plot(layers, overlaps, 'o-', color=color, label=f'Top-{frac:.0%}', markersize=4)
    ax.axhline(y=1.0, linestyle='--', color='gray', alpha=0.3)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Overlap Fraction')
    ax.set_title('Top-K Neuron Overlap (Gate vs Wanda)')
    ax.legend()

    # Scatter plot for one example layer
    ax = axes[2]
    mid_layer = len(layers) // 2
    sp = list(gate_results.keys())[len(gate_results)//2]  # middle sparsity
    patterns = gate_results[sp][0]
    if mid_layer in patterns and mid_layer in wanda_scores:
        gate_vals = patterns[mid_layer].numpy()
        wanda_vals = wanda_scores[mid_layer].numpy()
        # Subsample for visibility
        n = len(gate_vals)
        idx = np.random.choice(n, min(500, n), replace=False)
        ax.scatter(wanda_vals[idx], gate_vals[idx], alpha=0.3, s=10, color='#2196F3')
        ax.set_xlabel('Wanda Score')
        ax.set_ylabel('Gate Value (sigmoid)')
        ax.set_title(f'Layer {mid_layer}: Gate vs Wanda')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/study7_gate_wanda_correlation.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: study7_gate_wanda_correlation.png")


def plot_layer_redundancy(reports, model_name: str, output_dir: str):
    """Plot Study 10: Layer redundancy."""
    set_style()

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    fig.suptitle(f"Study 10: Layer Redundancy — {model_name}", fontsize=14, fontweight='bold')

    mlp_reports = sorted([r for r in reports if r.component == "mlp"], key=lambda r: r.layer_idx)
    attn_reports = sorted([r for r in reports if r.component == "attention"], key=lambda r: r.layer_idx)

    width = 0.35

    if mlp_reports:
        x_mlp = np.array([r.layer_idx for r in mlp_reports])
        ax.bar(x_mlp - width/2, [r.ppl_delta for r in mlp_reports], width,
               label='MLP removed', color='#2196F3', alpha=0.7)
    if attn_reports:
        x_attn = np.array([r.layer_idx for r in attn_reports])
        ax.bar(x_attn + width/2, [r.ppl_delta for r in attn_reports], width,
               label='Attention removed', color='#F44336', alpha=0.7)

    ax.set_xlabel('Layer')
    ax.set_ylabel('PPL Increase (higher = more important)')
    ax.set_title('Perplexity Impact of Removing Each Component')
    ax.legend()
    ax.set_yscale('symlog')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/study10_layer_redundancy.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: study10_layer_redundancy.png")


def plot_domain_divergence(domain_results: dict, model_name: str, output_dir: str):
    """Plot Study 11: Domain-specific activation divergence."""
    set_style()

    overview_reports = domain_results["overview_reports"]
    pairwise_reports = domain_results["pairwise_reports"]
    domains = domain_results["domains"]

    n_layers = len(overview_reports)
    layers = [r.layer_idx for r in overview_reports]

    # Get all unique domain pairs
    pairs = sorted(set((r.domain_a, r.domain_b) for r in pairwise_reports))

    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(f"Study 11: Domain-Specific Activation Divergence — {model_name}",
                 fontsize=14, fontweight='bold')

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    # ---- 11a: Specialization depth profile (THE key plot) ----
    ax = fig.add_subplot(gs[0, :2])
    specificities = [r.domain_specificity_score for r in overview_reports]
    ax.plot(layers, specificities, 'o-', color='#E91E63', markersize=5, linewidth=2)
    ax.fill_between(layers, specificities, alpha=0.15, color='#E91E63')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Domain Specificity Score')
    ax.set_title('Specialization Depth Profile\n(higher = more domain-specific activation patterns)')

    # Annotate the most and least specialized layers
    max_idx = np.argmax(specificities)
    min_idx = np.argmin(specificities)
    ax.annotate(f'Most specialized\n(layer {layers[max_idx]})',
                xy=(layers[max_idx], specificities[max_idx]),
                xytext=(layers[max_idx]+1, specificities[max_idx]+0.02),
                arrowprops=dict(arrowstyle='->', color='red'), fontsize=8, color='red')
    ax.annotate(f'Most shared\n(layer {layers[min_idx]})',
                xy=(layers[min_idx], specificities[min_idx]),
                xytext=(layers[min_idx]+1, specificities[min_idx]-0.02),
                arrowprops=dict(arrowstyle='->', color='blue'), fontsize=8, color='blue')

    # ---- 11b: Universal vs dead neurons by layer ----
    ax = fig.add_subplot(gs[0, 2])
    intermediate_size = sum(r.per_domain_active_count[domains[0]] for r in overview_reports[:1])
    # Get actual intermediate size from first layer's counts
    # (approximate: universal + domain-specific + dead should ~ total)
    ax.bar(layers, [r.n_universal_neurons for r in overview_reports],
           color='#4CAF50', alpha=0.7, label='Universal (all domains)')
    ax.bar(layers, [r.n_dead_across_all for r in overview_reports],
           bottom=[r.n_universal_neurons for r in overview_reports],
           color='#9E9E9E', alpha=0.7, label='Dead (no domain)')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Neuron Count')
    ax.set_title('Universal vs Dead Neurons')
    ax.legend(fontsize=7)

    # ---- 11c: Pairwise Jaccard heatmap per layer (one row per pair, one col per layer) ----
    ax = fig.add_subplot(gs[1, :2])
    jaccard_matrix = np.zeros((len(pairs), n_layers))
    pair_labels = []
    for pi, (da, db) in enumerate(pairs):
        pair_labels.append(f"{da[:3]}-{db[:3]}")
        for r in pairwise_reports:
            if r.domain_a == da and r.domain_b == db:
                jaccard_matrix[pi, r.layer_idx] = r.jaccard_similarity

    im = ax.imshow(jaccard_matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels(layers, fontsize=7)
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels(pair_labels, fontsize=8)
    ax.set_xlabel('Layer')
    ax.set_title('Pairwise Jaccard Similarity (green=shared, red=divergent)')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # ---- 11d: Per-domain active neuron counts ----
    ax = fig.add_subplot(gs[1, 2])
    colors = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0']
    for di, d in enumerate(domains):
        counts = [r.per_domain_active_count[d] for r in overview_reports]
        ax.plot(layers, counts, 'o-', color=colors[di % len(colors)],
                label=d, markersize=3, linewidth=1.5)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Active Neurons')
    ax.set_title('Active Neuron Count per Domain')
    ax.legend(fontsize=7)

    # ---- 11e: Domain-specific neuron counts for each pair at most specialized layer ----
    ax = fig.add_subplot(gs[2, 0])
    most_spec_layer = overview_reports[max_idx].layer_idx
    spec_layer_reports = [r for r in pairwise_reports if r.layer_idx == most_spec_layer]

    pair_names = [f"{r.domain_a[:3]}v{r.domain_b[:3]}" for r in spec_layer_reports]
    a_specific = [r.n_domain_a_specific for r in spec_layer_reports]
    b_specific = [r.n_domain_b_specific for r in spec_layer_reports]
    shared = [r.n_shared_active for r in spec_layer_reports]

    x = np.arange(len(pair_names))
    w = 0.25
    ax.bar(x - w, a_specific, w, label='Domain A specific', color='#2196F3', alpha=0.7)
    ax.bar(x, shared, w, label='Shared', color='#4CAF50', alpha=0.7)
    ax.bar(x + w, b_specific, w, label='Domain B specific', color='#F44336', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(pair_names, fontsize=7, rotation=30)
    ax.set_ylabel('Neuron Count')
    ax.set_title(f'Breakdown @ Layer {most_spec_layer} (most specialized)')
    ax.legend(fontsize=7)

    # ---- 11f: Pearson correlation of firing rates ----
    ax = fig.add_subplot(gs[2, 1])
    for pi, (da, db) in enumerate(pairs):
        correlations = []
        for r in pairwise_reports:
            if r.domain_a == da and r.domain_b == db:
                correlations.append(r.magnitude_correlation)
        label = f"{da[:3]}-{db[:3]}"
        ax.plot(layers, correlations, 'o-', color=colors[pi % len(colors)],
                label=label, markersize=3, linewidth=1.5)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Pearson r')
    ax.set_title('Firing Rate Correlation\n(how similarly neurons respond across domains)')
    ax.legend(fontsize=6, ncol=2)
    ax.axhline(y=0.5, linestyle='--', color='gray', alpha=0.3)

    # ---- 11g: Pruning opportunity score ----
    ax = fig.add_subplot(gs[2, 2])
    # For each layer, compute: how many neurons could you safely prune
    # if you specialize for each domain?
    # = neurons dead in target domain + neurons only active in OTHER domains
    for di, target_domain in enumerate(domains):
        prunable = []
        for ov in overview_reports:
            # Neurons not needed by this domain = total - active_in_domain
            # But this overcounts, so use active count as a proxy
            total_active = max(ov.per_domain_active_count.values())
            domain_active = ov.per_domain_active_count[target_domain]
            # "Extra" neurons other domains use but this one doesn't
            prunable.append(total_active - domain_active)
        ax.plot(layers, prunable, 'o-', color=colors[di % len(colors)],
                label=f"Prunable for {target_domain}", markersize=3, linewidth=1.5)

    ax.set_xlabel('Layer')
    ax.set_ylabel('Potentially Prunable Neurons')
    ax.set_title('Domain-Specific Pruning Opportunity\n(neurons other domains use but target doesn\'t)')
    ax.legend(fontsize=6)

    plt.savefig(f"{output_dir}/study11_domain_divergence.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: study11_domain_divergence.png")


# ---------------------------------------------------------------------------
# Studies 12-16 (from visualize_novel.py)
# ---------------------------------------------------------------------------

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
    ax.axhline(y=2.5, color='orange', linestyle='--', label='Heavy tail boundary (alpha=2.5)')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Power-law alpha')
    ax.set_title('Tail Exponent (red = heavy-tailed)')
    ax.legend(fontsize=8)

    # Tail fraction
    axes[0, 1].bar(layers, [r.tail_fraction for r in reports], color='#9C27B0', alpha=0.7)
    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('Fraction > 10x median')
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


# ---------------------------------------------------------------------------
# Studies 17-21 (from visualize_new.py)
# ---------------------------------------------------------------------------

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
    ax.set_xlabel("Layer pair (a -> a+1)")
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


def plot_sparsity_structure(reports, model_name: str, output_dir: str):
    """Study 8: Visualize sparsity structure analysis."""
    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Study 8: Sparsity Structure — {model_name}", fontsize=14, fontweight='bold')

    layers = [r.layer_idx for r in reports]

    # Token sparsity variance
    ax = axes[0, 0]
    ax.bar(layers, [r.token_sparsity_variance for r in reports], color='steelblue', alpha=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Token Sparsity Variance")
    ax.set_title("Token-level Sparsity Variance")

    # Neuron specialization
    ax = axes[0, 1]
    ax.bar(layers, [r.neuron_specialization_score for r in reports], color='coral', alpha=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Specialization Score")
    ax.set_title("Neuron Specialization")

    # Co-activation clusters
    ax = axes[1, 0]
    ax.bar(layers, [r.co_activation_clusters for r in reports], color='seagreen', alpha=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Number of Clusters")
    ax.set_title("Co-activation Clusters")

    # Activation consistency
    ax = axes[1, 1]
    ax.bar(layers, [r.activation_consistency for r in reports], color='mediumpurple', alpha=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Consistency")
    ax.set_title("Activation Consistency")
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/study8_sparsity_structure.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Study 8] Saved sparsity_structure plot")


def plot_critical_neurons(reports, model_name: str, output_dir: str):
    """Study 9: Visualize critical neuron search results."""
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Study 9: Critical Neurons — {model_name}", fontsize=14, fontweight='bold')

    layers = [r.layer_idx for r in reports]

    # PPL increase per critical neuron
    ax = axes[0]
    colors = plt.cm.Reds(np.linspace(0.3, 1.0, len(reports)))
    ax.barh(range(len(reports)), [r.ppl_increase_single for r in reports], color=colors)
    ax.set_yticks(range(len(reports)))
    ax.set_yticklabels([f"L{r.layer_idx}:N{r.neuron_idx}" for r in reports], fontsize=7)
    ax.set_xlabel("PPL Increase")
    ax.set_title("PPL Impact (single neuron zeroed)")

    # Weight norm vs activation mean
    ax = axes[1]
    scatter = ax.scatter(
        [r.weight_norm for r in reports],
        [r.activation_mean for r in reports],
        c=[r.ppl_increase_single for r in reports],
        cmap='Reds', s=40, alpha=0.8, edgecolors='black', linewidth=0.5
    )
    ax.set_xlabel("Weight Norm")
    ax.set_ylabel("Mean Activation")
    ax.set_title("Weight Norm vs Activation")
    plt.colorbar(scatter, ax=ax, label="PPL Impact")

    # Critical neurons per layer
    ax = axes[2]
    from collections import Counter
    layer_counts = Counter(layers)
    sorted_layers = sorted(layer_counts.keys())
    ax.bar(sorted_layers, [layer_counts[l] for l in sorted_layers], color='crimson', alpha=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Critical Neuron Count")
    ax.set_title("Critical Neurons per Layer")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/study9_critical_neurons.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Study 9] Saved critical_neurons plot")


# ---------------------------------------------------------------------------
# Master dispatch
# ---------------------------------------------------------------------------

def generate_all_plots(results: dict, model_name: str, output_dir: str):
    """Generate all available plots from results dictionary."""
    print("\nGenerating visualizations...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Study 1
    if "activation_profiles" in results:
        try:
            plot_activation_profiles(results["activation_profiles"], model_name, output_dir)
        except Exception as e:
            print(f"  [Study 1] plot_activation_profiles failed: {e}")

    # Study 2
    if "gate_training" in results:
        try:
            plot_gate_training(results["gate_training"], model_name, output_dir)
        except Exception as e:
            print(f"  [Study 2] plot_gate_training failed: {e}")

    # Study 3
    if "wanda_scores" in results:
        try:
            plot_wanda_scores(results["wanda_scores"], model_name, output_dir)
        except Exception as e:
            print(f"  [Study 3] plot_wanda_scores failed: {e}")

    # Study 4
    if "massive_activations" in results:
        try:
            plot_massive_activations(results["massive_activations"], model_name, output_dir)
        except Exception as e:
            print(f"  [Study 4] plot_massive_activations failed: {e}")

    # Study 5
    if "dead_neurons" in results:
        try:
            plot_dead_neurons(results["dead_neurons"], model_name, output_dir)
        except Exception as e:
            print(f"  [Study 5] plot_dead_neurons failed: {e}")

    # Study 6
    if "attention_heads" in results:
        try:
            n_layers = len(results.get("activation_profiles", [None] * 12))
            plot_attention_heads(results["attention_heads"], n_layers, model_name, output_dir)
        except Exception as e:
            print(f"  [Study 6] plot_attention_heads failed: {e}")

    # Study 7
    if "gate_wanda_correlation" in results and "gate_training" in results and "wanda_scores" in results:
        try:
            plot_gate_wanda_correlation(
                results["gate_wanda_correlation"],
                results["gate_training"],
                results["wanda_scores"],
                model_name, output_dir
            )
        except Exception as e:
            print(f"  [Study 7] plot_gate_wanda_correlation failed: {e}")

    # Study 8
    if "sparsity_structure" in results:
        try:
            plot_sparsity_structure(results["sparsity_structure"], model_name, output_dir)
        except Exception as e:
            print(f"  [Study 8] plot_sparsity_structure failed: {e}")

    # Study 9
    if "critical_neurons" in results:
        try:
            plot_critical_neurons(results["critical_neurons"], model_name, output_dir)
        except Exception as e:
            print(f"  [Study 9] plot_critical_neurons failed: {e}")

    # Study 10
    if "layer_redundancy" in results:
        try:
            plot_layer_redundancy(results["layer_redundancy"], model_name, output_dir)
        except Exception as e:
            print(f"  [Study 10] plot_layer_redundancy failed: {e}")

    # Study 11
    if "domain_divergence" in results:
        try:
            plot_domain_divergence(results["domain_divergence"], model_name, output_dir)
        except Exception as e:
            print(f"  [Study 11] plot_domain_divergence failed: {e}")

    # Study 12
    if "cross_layer_motifs" in results:
        try:
            plot_study12_motifs(results["cross_layer_motifs"], model_name, output_dir)
        except Exception as e:
            print(f"  [Study 12] plot_study12_motifs failed: {e}")

    # Study 13
    if "information_bottleneck" in results:
        try:
            plot_study13_bottleneck(results["information_bottleneck"], model_name, output_dir)
        except Exception as e:
            print(f"  [Study 13] plot_study13_bottleneck failed: {e}")

    # Study 14
    if "functional_redundancy" in results:
        try:
            plot_study14_redundancy(results["functional_redundancy"], model_name, output_dir)
        except Exception as e:
            print(f"  [Study 14] plot_study14_redundancy failed: {e}")

    # Study 15
    if "perturbation_cascade" in results:
        try:
            plot_study15_cascade(results["perturbation_cascade"], model_name, output_dir)
        except Exception as e:
            print(f"  [Study 15] plot_study15_cascade failed: {e}")

    # Study 16
    if "phase_transition" in results:
        try:
            plot_study16_phase(results["phase_transition"], model_name, output_dir)
        except Exception as e:
            print(f"  [Study 16] plot_study16_phase failed: {e}")

    # Study 17
    if "cross_layer_alignment" in results:
        try:
            plot_study17_alignment(results["cross_layer_alignment"], model_name, output_dir)
        except Exception as e:
            print(f"  [Study 17] plot_study17_alignment failed: {e}")

    # Study 18
    if "weight_rank" in results:
        try:
            plot_study18_rank(results["weight_rank"], model_name, output_dir)
        except Exception as e:
            print(f"  [Study 18] plot_study18_rank failed: {e}")

    # Study 19
    if "head_clustering" in results:
        try:
            plot_study19_head_clusters(results["head_clustering"], model_name, output_dir)
        except Exception as e:
            print(f"  [Study 19] plot_study19_head_clusters failed: {e}")

    # Study 20
    if "static_dynamic" in results:
        try:
            plot_study20_static_dynamic(results["static_dynamic"], model_name, output_dir)
        except Exception as e:
            print(f"  [Study 20] plot_study20_static_dynamic failed: {e}")

    # Study 21
    if "magnitude_divergence" in results:
        try:
            plot_study21_magnitude_divergence(results["magnitude_divergence"], model_name, output_dir)
        except Exception as e:
            print(f"  [Study 21] plot_study21_magnitude_divergence failed: {e}")

    print("Visualization complete!")

"""
Visualization module for all studies.
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
    ax.set_title('Mean ± Std of Activations')
    
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
    
    num_heads = max(r.head_idx for r in reports) + 1
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Study 6: Attention Head Analysis — {model_name}", fontsize=14, fontweight='bold')
    
    # Heatmap: entropy per head
    entropy_matrix = np.zeros((num_layers, num_heads))
    first_token_matrix = np.zeros((num_layers, num_heads))
    concentration_matrix = np.zeros((num_layers, num_heads))
    
    for r in reports:
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
            color='#F44336', label='Spearman ρ', markersize=4)
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
    
    x = np.arange(len(mlp_reports))
    width = 0.35
    
    if mlp_reports:
        ax.bar(x - width/2, [r.ppl_delta for r in mlp_reports], width, 
               label='MLP removed', color='#2196F3', alpha=0.7)
    if attn_reports:
        ax.bar(x + width/2, [r.ppl_delta for r in attn_reports], width, 
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


def generate_all_plots(results: dict, model_name: str, output_dir: str):
    """Generate all available plots from results dictionary."""
    print("\nGenerating visualizations...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    if "activation_profiles" in results:
        plot_activation_profiles(results["activation_profiles"], model_name, output_dir)
    
    if "gate_training" in results:
        plot_gate_training(results["gate_training"], model_name, output_dir)
    
    if "wanda_scores" in results:
        plot_wanda_scores(results["wanda_scores"], model_name, output_dir)
    
    if "massive_activations" in results:
        plot_massive_activations(results["massive_activations"], model_name, output_dir)
    
    if "dead_neurons" in results:
        plot_dead_neurons(results["dead_neurons"], model_name, output_dir)
    
    if "attention_heads" in results:
        n_layers = len(results.get("activation_profiles", [None] * 12))
        plot_attention_heads(results["attention_heads"], n_layers, model_name, output_dir)
    
    if "gate_wanda_correlation" in results and "gate_training" in results and "wanda_scores" in results:
        plot_gate_wanda_correlation(
            results["gate_wanda_correlation"], 
            results["gate_training"],
            results["wanda_scores"],
            model_name, output_dir
        )
    
    if "layer_redundancy" in results:
        plot_layer_redundancy(results["layer_redundancy"], model_name, output_dir)
    
    if "domain_divergence" in results:
        plot_domain_divergence(results["domain_divergence"], model_name, output_dir)
    
    print("Visualization complete!")


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
    # (approximate: universal + domain-specific + dead should ≈ total)
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

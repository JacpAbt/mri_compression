"""
Studies 4-7: Extended analyses

  4. Massive Activation Scan: Find input-agnostic outlier activations (Sun et al., 2024)
  5. Dead Neuron Analysis: Census of never/rarely-firing neurons (Voita et al., 2023)
  6. Attention Head Importance: Which heads matter via ablation
  7. Gate-Wanda Correlation: Do learned gates agree with Wanda importance?
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

from model_utils import ModelInspector, ActivationCollector, AttentionPatternCollector
from data_utils import TextDataset, get_dataloader, evaluate_perplexity


# =============================================================================
# Study 4: Massive Activation Scan
# =============================================================================

@dataclass
class MassiveActivationReport:
    """Report on massive/outlier activations per layer."""
    layer_idx: int
    # Per-neuron statistics across all inputs
    neuron_mean_activation: torch.Tensor      # (intermediate_size,) - mean |activation| per neuron
    neuron_variance: torch.Tensor             # (intermediate_size,) - how much each neuron varies with input
    # Massive activation candidates
    massive_neuron_indices: List[int]          # neurons with activation >> median
    massive_neuron_ratios: List[float]         # ratio of massive neuron's mean to layer mean
    # Input-agnostic check: neurons that fire at roughly the same value regardless of input
    input_agnostic_indices: List[int]          # neurons with low variance / high mean ratio
    input_agnostic_values: List[float]         # their mean activation values


def run_massive_activation_scan(
    inspector: ModelInspector,
    dataset: TextDataset,
    batch_size: int = 4,
    max_batches: int = 16,
    massive_threshold: float = 50.0,  # ratio threshold from Sun et al.
) -> List[MassiveActivationReport]:
    """
    Study 4: Scan for massive activations.
    
    From Sun et al. (2024, COLM): "very few activations exhibit significantly 
    larger values than others (e.g., 100,000x larger). They function as 
    indispensable bias terms in LLMs."
    
    We check:
    - Which neurons have disproportionately large mean activations?
    - Which neurons are input-agnostic (low variance relative to mean)?
    - Do these correlate with what our gates learn to keep?
    """
    print("\n" + "="*80)
    print("STUDY 4: Massive Activation Scan")
    print("="*80)
    
    collector = ActivationCollector(inspector)
    collector.register_hooks()
    
    dataloader = get_dataloader(dataset, batch_size=batch_size)
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            input_ids = batch["input_ids"].to(inspector.device)
            inspector.model(input_ids=input_ids)
    
    reports = []
    for layer_idx in range(inspector.num_layers):
        act = collector.get_concatenated(layer_idx).float()  # (N, intermediate_size)
        
        # Per-neuron statistics
        neuron_mean = act.abs().mean(dim=0)   # (intermediate_size,)
        neuron_var = act.var(dim=0)            # (intermediate_size,)
        
        layer_median = neuron_mean.median().item()
        
        # Massive neurons: mean activation >> layer median
        massive_mask = neuron_mean > (massive_threshold * layer_median)
        massive_indices = massive_mask.nonzero(as_tuple=True)[0].tolist()
        massive_ratios = (neuron_mean[massive_mask] / (layer_median + 1e-10)).tolist()
        
        # Input-agnostic neurons: low coefficient of variation (std/mean)
        cv = neuron_var.sqrt() / (neuron_mean + 1e-10)  # coefficient of variation
        # Low CV + high mean = input-agnostic massive activation
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
    
    collector.remove_hooks()
    collector.clear()
    
    return reports


# =============================================================================
# Study 5: Dead Neuron Analysis
# =============================================================================

@dataclass
class DeadNeuronReport:
    """Report on dead/dormant neurons per layer."""
    layer_idx: int
    total_neurons: int
    dead_count: int           # never fire (|activation| < eps on all inputs)
    dormant_count: int        # fire < 1% of the time
    rare_count: int           # fire < 10% of the time
    hyperactive_count: int    # fire > 99% of the time
    activation_rate: torch.Tensor  # (intermediate_size,) - fraction of inputs where neuron fires


def run_dead_neuron_analysis(
    inspector: ModelInspector,
    dataset: TextDataset,
    batch_size: int = 4,
    max_batches: int = 16,
    fire_threshold: float = 0.01,  # activation magnitude to count as "firing"
) -> List[DeadNeuronReport]:
    """
    Study 5: Census of dead/dormant/hyperactive neurons.
    
    From Voita et al. (2023) and ACL 2024 Findings: "Many neurons are dead. 
    Only the first half of the model is sparse."
    
    We categorize each neuron by how often it fires across all inputs:
    - Dead: never fires
    - Dormant: fires < 1% of the time  
    - Rare: fires < 10% of the time
    - Hyperactive: fires > 99% of the time (always on)
    
    This tells us which neurons are candidates for free pruning (dead/dormant)
    vs which are structural (hyperactive = possibly massive activations).
    """
    print("\n" + "="*80)
    print("STUDY 5: Dead/Dormant Neuron Analysis")
    print("="*80)
    
    collector = ActivationCollector(inspector)
    collector.register_hooks()
    
    dataloader = get_dataloader(dataset, batch_size=batch_size)
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            input_ids = batch["input_ids"].to(inspector.device)
            inspector.model(input_ids=input_ids)
    
    reports = []
    for layer_idx in range(inspector.num_layers):
        act = collector.get_concatenated(layer_idx).float()  # (N, intermediate_size)
        N = act.shape[0]
        
        # Binary fire/not-fire per token per neuron
        firing = (act.abs() > fire_threshold).float()
        activation_rate = firing.mean(dim=0)  # (intermediate_size,)
        
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
    
    collector.remove_hooks()
    collector.clear()
    
    return reports


# =============================================================================
# Study 6: Attention Head Importance
# =============================================================================

@dataclass
class HeadImportanceReport:
    """Importance scores for attention heads."""
    layer_idx: int
    head_idx: int
    # Metrics
    mean_entropy: float              # average entropy of attention distribution
    first_token_attention: float     # fraction of attention going to first token (attention sink)
    max_attention_concentration: float  # max attention to any single position (averaged)
    ablation_ppl_delta: Optional[float]  # perplexity change when this head is zeroed out


def run_attention_head_importance(
    inspector: ModelInspector,
    dataset: TextDataset,
    batch_size: int = 4,
    max_batches: int = 16,
    do_ablation: bool = False,  # Set True for full ablation (slow but definitive)
) -> List[HeadImportanceReport]:
    """
    Study 6: Analyze attention head specialization and importance.
    
    From Voita et al. (2019): "Specialized heads do the heavy lifting, 
    the rest can be pruned."
    
    We measure:
    - Entropy: Low entropy = head is focused, likely specialized
    - First-token attention: High = attention sink behavior (Xiao et al., 2023)
    - Concentration: How peaked is the attention distribution?
    
    Note: For GPT-2 we can get attention weights directly.
    For newer models we may need output_attentions=True.
    """
    print("\n" + "="*80)
    print("STUDY 6: Attention Head Importance")
    print("="*80)
    
    # Collect attention patterns
    dataloader = get_dataloader(dataset, batch_size=batch_size)
    
    # We'll accumulate statistics across batches
    head_stats = defaultdict(lambda: {
        "entropy_sum": 0.0, "first_token_sum": 0.0, "max_attn_sum": 0.0, "count": 0
    })
    
    with torch.no_grad():
        # Force eager attention implementation to get actual attention weights
        # SDPA (the default in newer transformers) doesn't return weights
        original_attn_impl = getattr(inspector.model.config, '_attn_implementation', None)
        try:
            inspector.model.config._attn_implementation = "eager"
        except Exception:
            pass
        
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            input_ids = batch["input_ids"].to(inspector.device)
            
            try:
                outputs = inspector.model(input_ids=input_ids, output_attentions=True)
            except Exception as e:
                print(f"  WARNING: output_attentions failed: {e}")
                print("  Skipping Study 6.")
                if original_attn_impl is not None:
                    inspector.model.config._attn_implementation = original_attn_impl
                return []
            
            # outputs.attentions is a tuple of (batch, num_heads, seq, seq) per layer
            if outputs.attentions is None:
                print("  WARNING: Model did not return attention weights. Skipping.")
                if original_attn_impl is not None:
                    inspector.model.config._attn_implementation = original_attn_impl
                return []
            
            for layer_idx, attn_weights in enumerate(outputs.attentions):
                # Some layers may return None (e.g. SDPA fallback)
                if attn_weights is None:
                    continue
                
                # attn_weights: (batch, num_heads, seq, seq)
                B, H, S, _ = attn_weights.shape
                
                for h in range(H):
                    head_attn = attn_weights[:, h, :, :]  # (B, S, S)
                    
                    # Entropy of attention distribution (per query position, averaged)
                    # Clamp to avoid log(0)
                    log_attn = torch.log(head_attn.clamp(min=1e-10))
                    entropy = -(head_attn * log_attn).sum(dim=-1).mean().item()
                    
                    # Attention to first token (attention sink)
                    first_token_attn = head_attn[:, :, 0].mean().item()
                    
                    # Max attention concentration
                    max_attn = head_attn.max(dim=-1).values.mean().item()
                    
                    key = (layer_idx, h)
                    head_stats[key]["entropy_sum"] += entropy
                    head_stats[key]["first_token_sum"] += first_token_attn
                    head_stats[key]["max_attn_sum"] += max_attn
                    head_stats[key]["count"] += 1
    
    # Restore original attention implementation
    if original_attn_impl is not None:
        inspector.model.config._attn_implementation = original_attn_impl
    
    if not head_stats:
        print("  WARNING: No attention weights were collected. Skipping.")
        return []
    
    reports = []
    for (layer_idx, head_idx), stats in sorted(head_stats.items()):
        n = stats["count"]
        if n == 0:
            continue
        report = HeadImportanceReport(
            layer_idx=layer_idx,
            head_idx=head_idx,
            mean_entropy=stats["entropy_sum"] / n,
            first_token_attention=stats["first_token_sum"] / n,
            max_attention_concentration=stats["max_attn_sum"] / n,
            ablation_ppl_delta=None,
        )
        reports.append(report)
    
    # Print summary per layer
    for layer_idx in range(inspector.num_layers):
        layer_reports = [r for r in reports if r.layer_idx == layer_idx]
        avg_entropy = np.mean([r.mean_entropy for r in layer_reports])
        avg_first = np.mean([r.first_token_attention for r in layer_reports])
        max_first = max([r.first_token_attention for r in layer_reports])
        min_entropy_head = min(layer_reports, key=lambda r: r.mean_entropy)
        print(f"  Layer {layer_idx:2d}: avg_entropy={avg_entropy:.2f}, "
              f"avg_first_token_attn={avg_first:.3f}, "
              f"max_first_token_attn={max_first:.3f}, "
              f"most_focused_head={min_entropy_head.head_idx}")
    
    return reports


# =============================================================================
# Study 7: Gate-Wanda Correlation
# =============================================================================

@dataclass  
class CorrelationReport:
    """Correlation between learned gate values and Wanda importance scores."""
    layer_idx: int
    pearson_r: float
    spearman_rho: float
    kendall_tau: float
    # Agreement at various sparsity levels
    top_k_overlap: Dict  # {k: overlap_fraction} for k = 10%, 25%, 50%


def compute_rank_correlation(x: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
    """Compute Pearson and Spearman correlation."""
    x = x.float().numpy()
    y = y.float().numpy()
    
    from scipy import stats
    pearson_r, _ = stats.pearsonr(x, y)
    spearman_rho, _ = stats.spearmanr(x, y)
    
    return float(pearson_r), float(spearman_rho)


def run_gate_wanda_correlation(
    gate_patterns: Dict[int, torch.Tensor],    # from Study 2
    wanda_scores: Dict[int, torch.Tensor],     # from Study 3
    num_layers: int,
) -> List[CorrelationReport]:
    """
    Study 7: Compare what learned gates think is important vs Wanda metric.
    
    Key question: Does a simple weight*activation heuristic (Wanda) agree
    with what gradient-based gate training discovers?
    
    If yes: Wanda is a good proxy and expensive gate training is unnecessary.
    If no: There's structure that only gradient-based learning can find,
           which would explain why CATS outperforms static pruning.
    """
    print("\n" + "="*80)
    print("STUDY 7: Gate-Wanda Correlation Analysis")
    print("="*80)
    
    reports = []
    for layer_idx in range(num_layers):
        if layer_idx not in gate_patterns or layer_idx not in wanda_scores:
            continue
        
        gates = gate_patterns[layer_idx]  # sigmoid values, higher = more important
        wanda = wanda_scores[layer_idx]   # higher = more important
        
        pearson_r, spearman_rho = compute_rank_correlation(gates, wanda)
        
        # Top-k overlap at various thresholds
        top_k_overlap = {}
        for frac in [0.10, 0.25, 0.50]:
            k = int(frac * len(gates))
            gate_topk = set(gates.topk(k).indices.tolist())
            wanda_topk = set(wanda.topk(k).indices.tolist())
            overlap = len(gate_topk & wanda_topk) / k
            top_k_overlap[frac] = overlap
        
        report = CorrelationReport(
            layer_idx=layer_idx,
            pearson_r=pearson_r,
            spearman_rho=spearman_rho,
            kendall_tau=0.0,  # skip for speed
            top_k_overlap=top_k_overlap,
        )
        reports.append(report)
        
        print(f"  Layer {layer_idx:2d}: Pearson={pearson_r:.3f}, "
              f"Spearman={spearman_rho:.3f}, "
              f"Top-10% overlap={top_k_overlap[0.10]:.1%}, "
              f"Top-25% overlap={top_k_overlap[0.25]:.1%}")
    
    # Overall summary
    avg_pearson = np.mean([r.pearson_r for r in reports])
    avg_spearman = np.mean([r.spearman_rho for r in reports])
    avg_overlap_10 = np.mean([r.top_k_overlap[0.10] for r in reports])
    print(f"\n  OVERALL: avg Pearson={avg_pearson:.3f}, avg Spearman={avg_spearman:.3f}, "
          f"avg Top-10% overlap={avg_overlap_10:.1%}")
    
    return reports
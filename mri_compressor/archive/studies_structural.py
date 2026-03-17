"""
Studies 8-10: Structural and vulnerability analyses

  8. Sparsity Structure Analysis: Spatial patterns in learned masks
  9. Critical Neuron Search: Find "super weights" and critical neurons
  10. Layer Redundancy: Layer contribution via residual stream knockout
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

from model_utils import ModelInspector
from data_utils import TextDataset, get_dataloader, evaluate_perplexity


# =============================================================================
# Study 8: Sparsity Structure Analysis
# =============================================================================

@dataclass
class SparsityStructureReport:
    """Analysis of spatial patterns in activation sparsity."""
    layer_idx: int
    # Token-level patterns
    token_sparsity_variance: float     # how much sparsity varies across tokens
    position_dependent_sparsity: torch.Tensor  # (seq_len,) - avg sparsity by position
    # Neuron-level patterns  
    neuron_specialization_score: float  # entropy of neuron activation patterns
    co_activation_clusters: int        # number of clusters of co-activating neurons
    # Cross-layer patterns
    activation_consistency: float      # do same neurons fire across different inputs?


def run_sparsity_structure_analysis(
    inspector: ModelInspector,
    dataset: TextDataset,
    batch_size: int = 4,
    max_batches: int = 16,
) -> List[SparsityStructureReport]:
    """
    Study 8: Analyze the structure/pattern of which neurons fire and when.
    
    Key questions:
    - Is sparsity uniform across token positions, or do certain positions
      activate more neurons? (e.g., first token, punctuation)
    - Do neurons fire in correlated groups (clusters)?
    - Is the sparsity pattern consistent across different inputs,
      or is it truly input-adaptive?
    
    This is important because it tells us whether we can use static masks
    (same mask for all inputs) or need dynamic/input-dependent masks (CATS).
    """
    print("\n" + "="*80)
    print("STUDY 8: Sparsity Structure Analysis")
    print("="*80)
    
    from model_utils import ActivationCollector
    
    collector = ActivationCollector(inspector)
    collector.register_hooks()
    
    dataloader = get_dataloader(dataset, batch_size=batch_size)
    
    # Collect raw activations preserving batch/seq structure
    raw_activations: Dict[int, List[torch.Tensor]] = defaultdict(list)
    
    # We need a different collector that preserves shape
    hooks = []
    for layer_idx in range(inspector.num_layers):
        mlp_info = inspector.mlp_layers[layer_idx]
        
        def make_hook(idx):
            def hook_fn(module, input, output):
                x = input[0] if isinstance(input, tuple) else input
                raw_activations[idx].append(x.detach().cpu())
            return hook_fn
        
        hook = mlp_info.down_proj.register_forward_hook(make_hook(layer_idx))
        hooks.append(hook)
    
    # Remove the collector hooks since we're using our own
    collector.remove_hooks()
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            input_ids = batch["input_ids"].to(inspector.device)
            inspector.model(input_ids=input_ids)
    
    for h in hooks:
        h.remove()
    
    reports = []
    for layer_idx in range(inspector.num_layers):
        # Stack all activations: (total_batch, seq_len, intermediate_size)
        all_act = torch.cat(raw_activations[layer_idx], dim=0).float()
        B, S, D = all_act.shape
        
        # Binary firing patterns
        firing = (all_act.abs() > 0.01).float()
        
        # 1. Token-position-dependent sparsity
        # Average sparsity at each position across all batches
        position_sparsity = 1.0 - firing.mean(dim=(0, 2))  # (seq_len,)
        token_sparsity_var = position_sparsity.var().item()
        
        # 2. Neuron activation consistency across inputs
        # For each neuron, what fraction of inputs activate it?
        neuron_activation_rate = firing.reshape(-1, D).mean(dim=0)  # (D,)
        # Consistency: if a neuron fires 50% of the time, it's maximally inconsistent
        # If it fires 0% or 100%, it's maximally consistent
        consistency_per_neuron = (neuron_activation_rate - 0.5).abs() * 2  # 0=inconsistent, 1=consistent
        activation_consistency = consistency_per_neuron.mean().item()
        
        # 3. Neuron specialization: entropy of activation rate distribution
        rates = neuron_activation_rate.clamp(0.001, 0.999)
        entropy = -(rates * rates.log() + (1-rates) * (1-rates).log()).mean().item()
        
        # 4. Co-activation clustering (simplified: correlation of firing patterns)
        # Sample neurons to avoid O(D^2) computation
        sample_size = min(256, D)
        sampled_idx = torch.randperm(D)[:sample_size]
        sampled_firing = firing.reshape(-1, D)[:, sampled_idx]  # (N, sample_size)
        
        # Correlation matrix
        if sampled_firing.shape[0] > 100:
            # Subsample tokens too
            token_sample = torch.randperm(sampled_firing.shape[0])[:1000]
            sampled_firing = sampled_firing[token_sample]
        
        corr = torch.corrcoef(sampled_firing.T)
        # Count clusters: neurons with correlation > 0.5
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
    
    return reports


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


# =============================================================================
# Study 10: Layer Redundancy Analysis
# =============================================================================

@dataclass
class LayerRedundancyReport:
    """How much each layer contributes to the model's output."""
    layer_idx: int
    component: str         # "mlp" or "attention" or "full"
    ppl_without: float     # perplexity with this component zeroed
    ppl_delta: float       # change from baseline
    residual_norm: float   # L2 norm of this layer's contribution to residual stream


def run_layer_redundancy(
    inspector: ModelInspector,
    dataset: TextDataset,
    batch_size: int = 4,
    max_eval_batches: int = 8,
) -> List[LayerRedundancyReport]:
    """
    Study 10: Measure each layer's contribution by removing it.
    
    Method: For each layer, zero out its MLP and/or attention output
    (effectively making it a no-op in the residual stream).
    
    This reveals:
    - Which layers are "load-bearing" vs redundant
    - Whether attention or MLP matters more at each depth
    - Safe targets for layer pruning/skipping
    """
    print("\n" + "="*80)
    print("STUDY 10: Layer Redundancy Analysis")
    print("="*80)
    
    eval_loader = get_dataloader(dataset, batch_size=batch_size)
    baseline_ppl = evaluate_perplexity(inspector.model, eval_loader,
                                        inspector.device, max_batches=max_eval_batches)
    print(f"  Baseline perplexity: {baseline_ppl:.2f}")
    
    reports = []
    layers = inspector.get_layers()
    
    for layer_idx in range(inspector.num_layers):
        layer = layers[layer_idx]
        
        # --- Zero out MLP ---
        mlp_info = inspector.mlp_layers[layer_idx]
        
        # Save and zero the down_proj bias and weight contribution
        # Simplest approach: hook that zeros MLP output
        mlp_zeroed = [False]
        
        def make_mlp_zero_hook(flag):
            def hook_fn(module, input, output):
                if flag[0]:
                    return torch.zeros_like(output)
                return output
            return hook_fn
        
        # Find the MLP module
        mlp_module = None
        for attr in ['mlp', 'feed_forward']:
            if hasattr(layer, attr):
                mlp_module = getattr(layer, attr)
                break
        
        if mlp_module is not None:
            hook = mlp_module.register_forward_hook(make_mlp_zero_hook(mlp_zeroed))
            mlp_zeroed[0] = True
            ppl_no_mlp = evaluate_perplexity(inspector.model, eval_loader,
                                              inspector.device, max_batches=max_eval_batches)
            mlp_zeroed[0] = False
            hook.remove()
            
            reports.append(LayerRedundancyReport(
                layer_idx=layer_idx, component="mlp",
                ppl_without=ppl_no_mlp, ppl_delta=ppl_no_mlp - baseline_ppl,
                residual_norm=0.0,
            ))
        
        # --- Zero out Attention ---
        attn_zeroed = [False]
        attn_module = None
        for attr in ['self_attn', 'attn', 'attention']:
            if hasattr(layer, attr):
                attn_module = getattr(layer, attr)
                break
        
        if attn_module is not None:
            hook = attn_module.register_forward_hook(
                lambda m, i, o, flag=attn_zeroed: (
                    (torch.zeros_like(o[0]),) + o[1:] if flag[0] and isinstance(o, tuple) 
                    else (torch.zeros_like(o) if flag[0] else o)
                )
            )
            attn_zeroed[0] = True
            ppl_no_attn = evaluate_perplexity(inspector.model, eval_loader,
                                               inspector.device, max_batches=max_eval_batches)
            attn_zeroed[0] = False
            hook.remove()
            
            reports.append(LayerRedundancyReport(
                layer_idx=layer_idx, component="attention",
                ppl_without=ppl_no_attn, ppl_delta=ppl_no_attn - baseline_ppl,
                residual_norm=0.0,
            ))
    
    # Print summary
    for layer_idx in range(inspector.num_layers):
        layer_reports = [r for r in reports if r.layer_idx == layer_idx]
        mlp_r = next((r for r in layer_reports if r.component == "mlp"), None)
        attn_r = next((r for r in layer_reports if r.component == "attention"), None)
        
        mlp_delta = f"{mlp_r.ppl_delta:+.2f}" if mlp_r else "N/A"
        attn_delta = f"{attn_r.ppl_delta:+.2f}" if attn_r else "N/A"
        print(f"  Layer {layer_idx:2d}: MLP PPL delta={mlp_delta}, "
              f"Attn PPL delta={attn_delta}")
    
    return reports
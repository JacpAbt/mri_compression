"""
Memory-optimized versions of Studies 1, 3, 4, 5, 8, 11.

KEY CHANGE: Instead of registering hooks on ALL layers simultaneously and
accumulating activations across all of them, we process ONE LAYER AT A TIME.

For a model with 36 layers and intermediate_size=11008 (Qwen 3B):
  - OLD: 36 layers × 32 batches × 4 × 512 × 11008 × 4 bytes ≈ 103 GB on CPU
  - NEW: 1 layer  × 16 batches × 4 × 512 × 11008 × 4 bytes ≈ 1.4 GB on CPU (per iteration)

This makes the difference between OOM and fitting on a single 3090.
"""

import torch
import torch.nn as nn
import numpy as np
import gc
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

from model_utils import ModelInspector, collect_single_layer
from data_utils import TextDataset, get_dataloader


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
# Study 2: CATS-style Learned Gates
# =============================================================================

class LearnedSparsityGate(nn.Module):
    """
    Per-neuron sigmoid gate: gate(x) = sigmoid(w * x + b)
    Output = gate(x) * activation(x)
    """
    def __init__(self, intermediate_size: int, init_bias: float = 2.0):
        super().__init__()
        self.gate_weight = nn.Parameter(torch.ones(intermediate_size) * 0.1)
        self.gate_bias = nn.Parameter(torch.ones(intermediate_size) * init_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_logits = self.gate_weight * x + self.gate_bias
        gate_values = torch.sigmoid(gate_logits)
        return gate_values * x

    def get_gate_openness(self, x: torch.Tensor) -> torch.Tensor:
        gate_logits = self.gate_weight * x + self.gate_bias
        return torch.sigmoid(gate_logits)

    def get_expected_sparsity(self, x: torch.Tensor, threshold: float = 0.5) -> float:
        gate_vals = self.get_gate_openness(x)
        return (gate_vals < threshold).float().mean().item()


class GatedModelWrapper(nn.Module):
    """
    Wraps a model with learned sparsity gates on each MLP layer.
    Only the gate parameters are trainable.
    
    Memory optimizations:
    - Enables gradient checkpointing on the base model
    - Gates are float32 even when model is bf16
    """
    def __init__(self, inspector: ModelInspector, use_gradient_checkpointing: bool = True):
        super().__init__()
        self.inspector = inspector
        self.model = inspector.model

        # Freeze all model parameters
        for p in self.model.parameters():
            p.requires_grad = False

        # Enable gradient checkpointing to save memory during backward
        if use_gradient_checkpointing:
            try:
                self.model.gradient_checkpointing_enable()
                print("    Gradient checkpointing: ENABLED")
            except Exception as e:
                print(f"    Gradient checkpointing: FAILED ({e}), continuing without")

        # Create learnable gates (float32 for training stability)
        self.gates = nn.ModuleList([
            LearnedSparsityGate(mlp.intermediate_size)
            for mlp in inspector.mlp_layers
        ])
        self.gates = self.gates.to(inspector.device)

        self._hooks = []
        self._register_gate_hooks()

    def _register_gate_hooks(self):
        """Insert gates between activation function and down_proj."""
        for layer_idx, (gate, mlp_info) in enumerate(
            zip(self.gates, self.inspector.mlp_layers)
        ):
            if mlp_info.is_gated:
                # Gated MLP: hook on down_proj input
                def make_pre_hook(g):
                    def hook_fn(module, input):
                        if isinstance(input, tuple):
                            x = input[0]
                            # Cast to float32 for gate, then back to model dtype
                            x_f32 = x.float()
                            gated = g(x_f32).to(x.dtype)
                            return (gated,) + input[1:]
                        else:
                            x_f32 = input.float()
                            return g(x_f32).to(input.dtype)
                    return hook_fn

                hook = mlp_info.down_proj.register_forward_pre_hook(make_pre_hook(gate))
                self._hooks.append(hook)
            else:
                # Standard MLP: same approach
                def make_pre_hook(g):
                    def hook_fn(module, input):
                        if isinstance(input, tuple):
                            x = input[0]
                            x_f32 = x.float()
                            gated = g(x_f32).to(x.dtype)
                            return (gated,) + input[1:]
                        else:
                            x_f32 = input.float()
                            return g(x_f32).to(input.dtype)
                    return hook_fn

                hook = mlp_info.down_proj.register_forward_pre_hook(make_pre_hook(gate))
                self._hooks.append(hook)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def disable_gradient_checkpointing(self):
        """Restore normal mode after training."""
        try:
            self.model.gradient_checkpointing_disable()
        except Exception:
            pass

    def forward(self, input_ids, labels=None, **kwargs):
        return self.model(input_ids=input_ids, labels=labels, **kwargs)

    def compute_sparsity_loss(self, target_sparsity: float) -> torch.Tensor:
        """L1-style loss encouraging gates to reach target sparsity."""
        total_openness = 0.0
        for gate in self.gates:
            avg_gate = torch.sigmoid(gate.gate_bias).mean()
            total_openness = total_openness + avg_gate

        avg_openness = total_openness / len(self.gates)
        target_openness = 1.0 - target_sparsity
        return (avg_openness - target_openness).pow(2)



def train_gates(
    inspector: ModelInspector,
    dataset: TextDataset,
    target_sparsity: float = 0.5,
    lr: float = 1e-2,
    num_steps: int = 500,
    batch_size: int = 2,           # reduced from 4 for 3B models
    sparsity_weight: float = 2.0,
    warmup_steps: int = 50,
    use_amp: bool = True,          # automatic mixed precision
    use_gradient_checkpointing: bool = True,
) -> Tuple[GatedModelWrapper, Dict]:
    """
    Study 2: Train CATS-style learned gates at a target sparsity level.

    Memory optimizations:
    - AMP: forward in bf16, gate gradients in float32
    - Gradient checkpointing: recompute activations during backward
    - Reduced batch size
    """
    print(f"\n  Training gates for target sparsity = {target_sparsity:.0%}")
    print(f"    batch_size={batch_size}, amp={use_amp}, "
          f"grad_ckpt={use_gradient_checkpointing}")

    gated = GatedModelWrapper(inspector, use_gradient_checkpointing=use_gradient_checkpointing)
    optimizer = torch.optim.Adam(gated.gates.parameters(), lr=lr)
    dataloader = get_dataloader(dataset, batch_size=batch_size, shuffle=True)

    # AMP scaler (only for CUDA)
    use_amp = use_amp and inspector.device == "cuda"
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    metrics = {"lm_loss": [], "sparsity_loss": [], "total_loss": [], "actual_sparsity": []}
    data_iter = iter(dataloader)

    for step in range(num_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(inspector.device)

        # Forward pass with AMP
        with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.bfloat16):
            outputs = gated(input_ids=input_ids, labels=input_ids)
            lm_loss = outputs.loss

        # Sparsity loss (computed in float32 — gates are float32)
        sp_loss = gated.compute_sparsity_loss(target_sparsity)

        if step < warmup_steps:
            effective_weight = sparsity_weight * (step / warmup_steps)
        else:
            effective_weight = sparsity_weight

        total_loss = lm_loss + effective_weight * sp_loss

        # Backward with AMP scaling
        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(gated.gates.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        # Track metrics
        with torch.no_grad():
            avg_openness = torch.stack([
                torch.sigmoid(g.gate_bias).mean() for g in gated.gates
            ]).mean().item()

        metrics["lm_loss"].append(lm_loss.item())
        metrics["sparsity_loss"].append(sp_loss.item())
        metrics["total_loss"].append(total_loss.item())
        metrics["actual_sparsity"].append(1.0 - avg_openness)

        if step % 100 == 0 or step == num_steps - 1:
            print(f"    Step {step:4d}: LM={lm_loss.item():.3f}, "
                  f"sparsity={1-avg_openness:.1%} (target={target_sparsity:.0%})")

    return gated, metrics


def extract_gate_patterns(gated_model: GatedModelWrapper) -> Dict[int, torch.Tensor]:
    """Extract learned gate bias values (proxy for which neurons to keep)."""
    patterns = {}
    with torch.no_grad():
        for layer_idx, gate in enumerate(gated_model.gates):
            patterns[layer_idx] = torch.sigmoid(gate.gate_bias).cpu()
    return patterns


def run_gate_training(
    inspector: ModelInspector,
    dataset: TextDataset,
    target_sparsities: List[float] = [0.25, 0.50, 0.75],
    batch_size: int = 2,
    **kwargs,
) -> Dict[float, Tuple[Dict[int, torch.Tensor], Dict]]:
    """
    Study 2: Train gates at multiple sparsity levels.
    Returns: dict of sparsity -> (gate_patterns, training_metrics)
    """
    print("\n" + "="*80)
    print("STUDY 2: CATS-style Gate Training (memory-optimized)")
    print("="*80)

    results = {}
    for sp in target_sparsities:
        gated, metrics = train_gates(
            inspector, dataset, target_sparsity=sp,
            batch_size=batch_size, **kwargs
        )
        patterns = extract_gate_patterns(gated)

        # Clean up
        gated.remove_hooks()
        gated.disable_gradient_checkpointing()
        results[sp] = (patterns, metrics)

        del gated
        gc.collect()
        torch.cuda.empty_cache()

    return results


# =============================================================================
# Study 3: Wanda Importance Scores (streaming version)
# =============================================================================

def compute_wanda_scores(
    inspector: ModelInspector,
    dataset: TextDataset,
    batch_size: int = 4,
    max_batches: int = 16,
) -> Dict[int, torch.Tensor]:
    """Study 3: STREAMING Wanda scores — one layer at a time."""
    print("\n" + "="*80)
    print("STUDY 3: Wanda Importance Scores")
    print("="*80)
    
    wanda_scores = {}
    for layer_idx in range(inspector.num_layers):
        act = collect_single_layer(inspector, dataset, layer_idx,
                                   batch_size=batch_size, max_batches=max_batches)
        
        # Activation norm per neuron
        act_norm = act.float().norm(p=2, dim=0)  # (intermediate_size,)
        
        # Weight magnitude of down_proj
        down_w = inspector.mlp_layers[layer_idx].down_proj.weight.data.cpu().float()
        intermediate_size = act_norm.shape[0]
        if down_w.shape[0] == intermediate_size:
            weight_norm = down_w.abs().mean(dim=1)
        else:
            weight_norm = down_w.abs().mean(dim=0)
        
        score = weight_norm * act_norm
        wanda_scores[layer_idx] = score
        
        top5 = score.topk(5).indices.tolist()
        bot5 = score.topk(5, largest=False).indices.tolist()
        print(f"  Layer {layer_idx:2d}: mean={score.mean():.4f}, "
              f"max={score.max():.4f}, top5={top5}, bot5={bot5}")
        
        del act, act_norm, down_w, weight_norm
        gc.collect()
    
    return wanda_scores


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
"""
Studies 2 and 7: CATS-style Learned Gates and Gate-Wanda Correlation.

Study 2 - Gate Training: Train learned sparsity gates on each MLP layer
    to discover which neurons can be safely deactivated at target sparsity
    levels, following the CATS approach.

Study 7 - Gate-Wanda Correlation: Compare what learned gates think is
    important vs the Wanda metric (weight * activation norm). Tests whether
    expensive gate training finds structure beyond simple heuristics.
"""

import torch
import torch.nn as nn
import numpy as np
import gc
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

from ..model_utils import ModelInspector, collect_single_layer
from ..data_utils import TextDataset, get_dataloader


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

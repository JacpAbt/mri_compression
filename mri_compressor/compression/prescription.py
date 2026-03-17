"""
Compression Prescription Data Classes
=======================================
Pure data classes for compression prescriptions.
Extracted from compression_v2/diagnostics.py.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional


class CompressionStrategy(Enum):
    DEAD_REMOVAL_AND_MERGE = auto()
    DORMANT_REMOVAL = auto()
    STRUCTURED_PRUNE = auto()
    QUANTIZE_AGGRESSIVE = auto()
    QUANTIZE_MIXED_PRECISION = auto()
    LIGHT_TOUCH = auto()
    DEPTH_PRUNE = auto()
    ATTENTION_PRUNE = auto()
    LOW_RANK_FACTORIZE = auto()
    DOMAIN_SPECIALIZE = auto()


@dataclass
class LayerPrescription:
    layer_idx: int
    strategy: CompressionStrategy
    target_sparsity: float = 0.0
    quant_bits: int = 16
    outlier_bits: int = 16
    outlier_fraction: float = 0.0
    merge_target_width: Optional[int] = None
    dead_neuron_count: int = 0
    dormant_neuron_count: int = 0
    attn_heads_to_prune: int = 0
    attn_ppl_delta: float = 0.0
    low_rank_target: Optional[int] = None
    depth_prune: bool = False
    reconstruction_priority: int = 0
    reconstruction_iterations: int = 100
    alpha: float = 0.0
    redundancy_frac: float = 0.0
    cascade_amplification: float = 1.0
    mlp_ppl_delta: float = 0.0
    cka_with_next: float = 0.0

    # Protection and precomputed data (from Studies 1-11)
    protected_neuron_indices: Optional[set] = None  # Never prune these (from Studies 4, 9)
    precomputed_wanda_scores: Optional[Any] = None  # From Study 3 if available
    head_importance_data: Optional[list] = None  # From Study 6 if available
    pruning_approach: str = "structured"  # "structured", "uniform_or_quant", "gate_guided"
    static_mask_viable: bool = True  # From Study 8
    merge_worthwhile: bool = False  # From Study 8

    # Studies 12-21 integrations
    information_retained: float = 1.0  # Study 13: 0-1, how much info this layer preserves
    cka_merge_score: float = 0.0  # Study 17: 0-1, similarity to next layer
    low_rank_ranks: Optional[dict] = None  # Study 18: {"gate_proj": rank95, "up_proj": ..., "down_proj": ...}
    cluster_prunable_heads: Optional[list] = None  # Study 19: head indices redundant by clustering
    foldable_neuron_count: int = 0  # Study 20: neurons that can be folded to biases
    foldable_neuron_indices: Optional[list] = None  # Study 20: specific indices
    n_domain_sensitive_neurons: int = 0  # Study 21: neurons with domain-specific behavior
    domain_sensitive_indices: Optional[list] = None  # Study 21: specific indices

    # Study 22: Domain-conditional importance
    domain_critical_indices: Optional[set] = None    # Neurons critical for target domain (protect)
    domain_unnecessary_count: int = 0                # Number of domain-unnecessary neurons to remove
    domain_unnecessary_indices: Optional[list] = None  # Specific indices to remove
    target_domain: Optional[str] = None              # Target domain for this layer


@dataclass
class CompressionPrescription:
    model_name: str
    baseline_ppl: float
    num_layers: int
    intermediate_size: int
    num_attention_heads: int = 0
    layers: list[LayerPrescription] = field(default_factory=list)
    depth_prune_candidates: list[int] = field(default_factory=list)
    depth_prune_applied: list[int] = field(default_factory=list)
    weight_sharing_pairs: list[tuple[int, int]] = field(default_factory=list)
    estimated_param_reduction: float = 0.0
    estimated_ppl_budget_used: float = 0.0
    total_dead_neurons: int = 0
    total_dormant_neurons: int = 0
    total_merged_neurons: int = 0
    total_attn_heads_pruned: int = 0
    total_low_rank_layers: int = 0
    total_folded_neurons: int = 0
    total_weight_shared_pairs: int = 0
    total_domain_unnecessary_removed: int = 0
    target_domain: Optional[str] = None

    def summary(self) -> str:
        lines = [
            f"{'='*72}",
            f"  MRI-Compress v2.1 Prescription for {self.model_name}",
            f"  Baseline PPL: {self.baseline_ppl:.2f}",
            f"  Layers: {self.num_layers}, Intermediate size: {self.intermediate_size}",
            f"  Estimated PPL budget used: {self.estimated_ppl_budget_used:.2f}",
        ]
        if self.target_domain:
            lines.append(f"  Target domain: {self.target_domain}")
        lines.extend([f"{'='*72}", ""])
        strategy_counts: dict[CompressionStrategy, int] = {}
        for lp in self.layers:
            strategy_counts[lp.strategy] = strategy_counts.get(lp.strategy, 0) + 1

        lines.append("  Strategy distribution:")
        for s, c in sorted(strategy_counts.items(), key=lambda x: -x[1]):
            lines.append(f"    {s.name:30s}: {c} layers")
        lines.append("")

        lines.append(f"  Total dead neurons to remove:    {self.total_dead_neurons:,}")
        lines.append(f"  Total dormant neurons to remove: {self.total_dormant_neurons:,}")
        lines.append(f"  Total neurons to merge:          {self.total_merged_neurons:,}")
        lines.append(f"  Total attn heads to prune:       {self.total_attn_heads_pruned}")
        if self.depth_prune_applied:
            lines.append(f"  Depth-pruned layers:             {self.depth_prune_applied}")
        if self.weight_sharing_pairs:
            lines.append(f"  Weight-sharing pairs:            {self.weight_sharing_pairs}")
        lines.append(f"  Low-rank factorized layers:      {self.total_low_rank_layers}")
        lines.append(f"  Static neurons to fold:          {self.total_folded_neurons:,}")
        if self.total_domain_unnecessary_removed > 0:
            lines.append(f"  Domain-unnecessary removed:      {self.total_domain_unnecessary_removed:,}")
        lines.append("")

        lines.append(f"  {'Layer':>5} | {'Strategy':>30} | {'Dead':>6} | {'Dorm':>6} | "
                      f"{'AttnPr':>6} | {'LowRk':>5} | {'Fold':>5} | {'MLP_Δ':>6} | {'Attn_Δ':>6}")
        lines.append(f"  {'-'*5}-+-{'-'*30}-+-{'-'*6}-+-{'-'*6}-+-"
                      f"{'-'*6}-+-{'-'*5}-+-{'-'*5}-+-{'-'*6}-+-{'-'*6}")

        for lp in self.layers:
            lr_str = str(lp.low_rank_target) if lp.low_rank_target else "-"
            fold_str = str(lp.foldable_neuron_count) if lp.foldable_neuron_count > 0 else "-"
            lines.append(
                f"  {lp.layer_idx:>5} | {lp.strategy.name:>30} | "
                f"{lp.dead_neuron_count:>6} | {lp.dormant_neuron_count:>6} | "
                f"{lp.attn_heads_to_prune:>6} | {lr_str:>5} | {fold_str:>5} | "
                f"{lp.mlp_ppl_delta:>+5.2f} | {lp.attn_ppl_delta:>+5.2f}"
            )
        lines.append("")
        return "\n".join(lines)

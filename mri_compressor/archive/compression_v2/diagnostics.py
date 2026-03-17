"""
MRI-Compress Diagnostic Module v2.1
=====================================
Post-mortem fixes from v2 full run (PPL +230%):

ROOT CAUSES:
1. Depth pruning 2 layers → cascading error (individual deltas don't compose)
2. 104 attention heads pruned → Frobenius norm is a poor importance proxy
3. No validation between strategies → compound damage undetected

FIXES in v2.1:
- Depth pruning: OFF by default. When enabled, max 1 layer, and only if
  combined delta (mlp+attn) < 0.15 (not 0.30+0.10 separately)
- Attention pruning: max 2 heads/layer (not 4), only in layers with
  attn_ppl_delta < 0.10 (not 0.40), skip layers near depth-pruned ones
- NEW: ablation modes to test strategies independently
- NEW: cumulative budget — track estimated total PPL impact
"""

from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


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

    def summary(self) -> str:
        lines = [
            f"{'='*72}",
            f"  MRI-Compress v2.1 Prescription for {self.model_name}",
            f"  Baseline PPL: {self.baseline_ppl:.2f}",
            f"  Layers: {self.num_layers}, Intermediate size: {self.intermediate_size}",
            f"  Estimated PPL budget used: {self.estimated_ppl_budget_used:.2f}",
            f"{'='*72}", "",
        ]
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
        lines.append("")

        lines.append(f"  {'Layer':>5} | {'Strategy':>30} | {'Dead':>6} | {'Dorm':>6} | "
                      f"{'AttnPr':>6} | {'LowRk':>5} | {'MLP_Δ':>6} | {'Attn_Δ':>6}")
        lines.append(f"  {'-'*5}-+-{'-'*30}-+-{'-'*6}-+-{'-'*6}-+-"
                      f"{'-'*6}-+-{'-'*5}-+-{'-'*6}-+-{'-'*6}")

        for lp in self.layers:
            lr_str = str(lp.low_rank_target) if lp.low_rank_target else "-"
            lines.append(
                f"  {lp.layer_idx:>5} | {lp.strategy.name:>30} | "
                f"{lp.dead_neuron_count:>6} | {lp.dormant_neuron_count:>6} | "
                f"{lp.attn_heads_to_prune:>6} | {lr_str:>5} | "
                f"{lp.mlp_ppl_delta:>+5.2f} | {lp.attn_ppl_delta:>+5.2f}"
            )
        lines.append("")
        return "\n".join(lines)


class MRIDiagnostician:
    """
    v2.1: Conservative, budget-aware prescription generation.

    Key principle: individual PPL deltas DO NOT compose linearly.
    A safety multiplier of 3x is applied when estimating combined impact.
    Total estimated PPL impact is capped by a budget.
    """

    def __init__(
        self,
        # Phase transition
        heavy_tail_alpha: float = 2.5,
        high_redundancy_frac: float = 0.15,
        # Dead/dormant
        dead_frac_threshold: float = 0.05,
        dormant_frac_threshold: float = 0.10,
        # Criticality
        critical_mlp_delta: float = 2.0,
        critical_attn_delta: float = 2.0,
        # Depth pruning — VERY conservative in v2.1
        enable_depth_pruning: bool = False,        # OFF by default
        depth_prune_combined_delta: float = 0.15,  # mlp+attn combined must be < this
        max_depth_prune_layers: int = 1,            # Max 1 layer (was 2)
        depth_prune_min_gap: int = 5,               # Min 5 layers between depth prunes
        # Attention head pruning — conservative in v2.1
        enable_attn_pruning: bool = True,
        attn_prune_delta_threshold: float = 0.10,   # Was 0.40 — now 4x stricter
        attn_heads_to_prune_per_layer: int = 2,     # Was 4 — now 2 (12.5%)
        attn_prune_guard_layers: int = 2,            # Don't prune near depth-pruned layers
        max_total_attn_heads: int = 32,              # Global budget: max 32 heads total
        # Merge settings — RE-ENABLED in v2.1 (was disabled by accident)
        enable_merging: bool = True,
        merge_redundancy_threshold: float = 0.15,    # Study 14 safe_to_prune_frac > this
        merge_max_reduction: float = 0.25,           # Max 25% reduction per layer via merge
        # Reconstruction
        base_recon_iterations: int = 100,
        high_cascade_recon_multiplier: float = 3.0,
        cascade_threshold: float = 50.0,
        # PPL budget
        ppl_budget: float = 1.0,                    # Max allowed estimated PPL increase
        ppl_safety_multiplier: float = 3.0,          # Assume 3x worse than individual deltas
    ):
        self.heavy_tail_alpha = heavy_tail_alpha
        self.high_redundancy_frac = high_redundancy_frac
        self.dead_frac_threshold = dead_frac_threshold
        self.dormant_frac_threshold = dormant_frac_threshold
        self.critical_mlp_delta = critical_mlp_delta
        self.critical_attn_delta = critical_attn_delta
        self.enable_depth_pruning = enable_depth_pruning
        self.depth_prune_combined_delta = depth_prune_combined_delta
        self.max_depth_prune_layers = max_depth_prune_layers
        self.depth_prune_min_gap = depth_prune_min_gap
        self.enable_attn_pruning = enable_attn_pruning
        self.attn_prune_delta_threshold = attn_prune_delta_threshold
        self.attn_heads_to_prune_per_layer = attn_heads_to_prune_per_layer
        self.attn_prune_guard_layers = attn_prune_guard_layers
        self.max_total_attn_heads = max_total_attn_heads
        self.enable_merging = enable_merging
        self.merge_redundancy_threshold = merge_redundancy_threshold
        self.merge_max_reduction = merge_max_reduction
        self.base_recon_iterations = base_recon_iterations
        self.high_cascade_recon_multiplier = high_cascade_recon_multiplier
        self.cascade_threshold = cascade_threshold
        self.ppl_budget = ppl_budget
        self.ppl_safety_multiplier = ppl_safety_multiplier

    def diagnose(
        self,
        summary_json_path: str,
        study5_data: Optional[list[dict]] = None,
        study10_data: Optional[list[dict]] = None,
        study17_data: Optional[list[dict]] = None,
        study18_data: Optional[list[dict]] = None,
    ) -> CompressionPrescription:
        with open(summary_json_path) as f:
            summary = json.load(f)

        model_name = summary["model"]
        baseline_ppl = summary["baseline_ppl"]
        num_layers = summary["num_layers"]
        intermediate_size = summary["intermediate_size"]
        num_attention_heads = summary.get("num_attention_heads", 16)
        findings = summary["findings"]

        # ---- Extract per-layer diagnostics ----
        s16 = findings.get("study16_phase_transition", {})
        alpha_by_layer = {}
        for item in s16.get("per_layer", []):
            alpha_by_layer[item["layer"]] = {
                "alpha": item["alpha"],
                "heavy_tailed": item.get("heavy_tailed", item["alpha"] < self.heavy_tail_alpha),
            }

        s14 = findings.get("study14_functional_redundancy", {})
        redundancy_by_layer = {}
        for item in s14.get("per_layer", []):
            safe = item.get("safe_to_prune", 0)
            redundancy_by_layer[item["layer"]] = {
                "safe_to_prune": safe,
                "safe_to_prune_frac": safe / intermediate_size if intermediate_size > 0 else 0,
                "redundant": item.get("redundant", 0),
            }

        s15 = findings.get("study15_perturbation_cascade", {})
        cascade_by_layer = {}
        for src_str, data in s15.get("per_source_layer", {}).items():
            cascade_by_layer[int(src_str)] = {"max_amplification": data.get("avg_max_amplification", 1.0)}

        dead_by_layer = {}
        if study5_data:
            for item in study5_data:
                dead_by_layer[item["layer"]] = {
                    "dead": item.get("dead", 0),
                    "dormant": item.get("dormant", 0),
                    "dead_frac": item.get("dead", 0) / intermediate_size,
                    "dormant_frac": item.get("dormant", 0) / intermediate_size,
                }

        criticality_by_layer = {}
        if study10_data:
            for item in study10_data:
                criticality_by_layer[item["layer"]] = {
                    "mlp_ppl_delta": item.get("mlp_ppl_delta", 0.0),
                    "attn_ppl_delta": item.get("attn_ppl_delta", 0.0),
                }

        # ---- Phase 1: Depth pruning (only if enabled) ----
        depth_prune_applied = []
        if self.enable_depth_pruning:
            candidates = []
            for layer_idx in range(2, num_layers - 2):  # Never first/last 2
                crit = criticality_by_layer.get(layer_idx, {"mlp_ppl_delta": 1.0, "attn_ppl_delta": 1.0})
                combined = crit["mlp_ppl_delta"] + crit["attn_ppl_delta"]
                if combined < self.depth_prune_combined_delta:
                    candidates.append((combined, layer_idx))
            candidates.sort()

            for _, li in candidates:
                if len(depth_prune_applied) >= self.max_depth_prune_layers:
                    break
                # Check min gap
                if any(abs(li - existing) < self.depth_prune_min_gap for existing in depth_prune_applied):
                    continue
                depth_prune_applied.append(li)

        # ---- Phase 2: Per-layer prescriptions ----
        prescriptions = []
        total_dead = 0
        total_dormant = 0
        total_merged = 0
        total_attn_pruned = 0
        ppl_budget_used = 0.0

        # Guard zones around depth-pruned layers (no attn pruning there)
        depth_guard_set = set()
        for dl in depth_prune_applied:
            for g in range(-self.attn_prune_guard_layers, self.attn_prune_guard_layers + 1):
                depth_guard_set.add(dl + g)

        for layer_idx in range(num_layers):
            alpha_data = alpha_by_layer.get(layer_idx, {"alpha": 3.0, "heavy_tailed": False})
            redund_data = redundancy_by_layer.get(layer_idx, {"safe_to_prune_frac": 0, "safe_to_prune": 0, "redundant": 0})
            dead_data = dead_by_layer.get(layer_idx, {"dead": 0, "dormant": 0, "dead_frac": 0, "dormant_frac": 0})
            crit_data = criticality_by_layer.get(layer_idx, {"mlp_ppl_delta": 1.0, "attn_ppl_delta": 1.0})
            cascade_data = self._interpolate_cascade(layer_idx, cascade_by_layer, num_layers)

            alpha = alpha_data["alpha"]
            is_heavy_tail = alpha < self.heavy_tail_alpha
            safe_to_prune_frac = redund_data["safe_to_prune_frac"]
            safe_to_prune = redund_data["safe_to_prune"]
            redundant_count = redund_data["redundant"]
            is_high_redundancy = safe_to_prune_frac > self.high_redundancy_frac
            dead_count = dead_data["dead"]
            dormant_count = dead_data["dormant"]
            has_significant_dead = dead_data["dead_frac"] > self.dead_frac_threshold
            has_significant_dormant = dead_data["dormant_frac"] > self.dormant_frac_threshold
            mlp_delta = crit_data["mlp_ppl_delta"]
            attn_delta = crit_data["attn_ppl_delta"]
            is_critical = mlp_delta > self.critical_mlp_delta or attn_delta > self.critical_attn_delta
            max_amp = cascade_data["max_amplification"]

            lp = LayerPrescription(
                layer_idx=layer_idx,
                strategy=CompressionStrategy.LIGHT_TOUCH,
                alpha=alpha,
                redundancy_frac=safe_to_prune_frac,
                cascade_amplification=max_amp,
                mlp_ppl_delta=mlp_delta,
                attn_ppl_delta=attn_delta,
                dead_neuron_count=dead_count,
                dormant_neuron_count=dormant_count,
            )

            # ---- Depth pruning ----
            if layer_idx in depth_prune_applied:
                lp.strategy = CompressionStrategy.DEPTH_PRUNE
                lp.depth_prune = True
                est_cost = (mlp_delta + attn_delta) * self.ppl_safety_multiplier
                ppl_budget_used += est_cost
                prescriptions.append(lp)
                continue

            # ---- MLP compression decision tree ----
            if is_critical:
                if has_significant_dead and mlp_delta <= self.critical_mlp_delta:
                    lp.strategy = CompressionStrategy.DEAD_REMOVAL_AND_MERGE
                    alive = intermediate_size - dead_count
                    lp.merge_target_width = alive  # No merge, just dead removal
                    total_dead += dead_count
                else:
                    lp.strategy = CompressionStrategy.LIGHT_TOUCH

            elif is_heavy_tail and (is_high_redundancy or has_significant_dead):
                lp.strategy = CompressionStrategy.DEAD_REMOVAL_AND_MERGE
                alive_neurons = intermediate_size - dead_count

                # Also remove dormant
                if has_significant_dormant:
                    lp.dormant_neuron_count = dormant_count
                    alive_neurons -= dormant_count
                    total_dormant += dormant_count

                total_dead += dead_count

                # Merge redundant neurons (re-enabled in v2.1)
                if (self.enable_merging and is_high_redundancy
                    and redundant_count > 0 and alive_neurons > 0):
                    # Cap merge at merge_max_reduction of alive neurons
                    effective_redundant = max(0, safe_to_prune - dead_count - lp.dormant_neuron_count)
                    max_removable = int(alive_neurons * self.merge_max_reduction)
                    to_merge = min(effective_redundant, max_removable)
                    lp.merge_target_width = alive_neurons - to_merge
                    total_merged += to_merge
                else:
                    lp.merge_target_width = alive_neurons

            elif has_significant_dead or has_significant_dormant:
                lp.strategy = CompressionStrategy.DORMANT_REMOVAL
                if has_significant_dead:
                    total_dead += dead_count
                if has_significant_dormant and not is_critical:
                    total_dormant += dormant_count
                else:
                    lp.dormant_neuron_count = 0

            else:
                lp.strategy = CompressionStrategy.LIGHT_TOUCH

            # ---- Attention head pruning (budget-aware) ----
            if (self.enable_attn_pruning
                and attn_delta < self.attn_prune_delta_threshold
                and not is_critical
                and layer_idx not in (0, 1)
                and layer_idx not in depth_guard_set
                and total_attn_pruned < self.max_total_attn_heads
                and lp.strategy != CompressionStrategy.DEPTH_PRUNE):

                # Check PPL budget
                # Estimate: pruning N/16 heads ≈ (N/16) * attn_delta * safety_multiplier
                n_heads = min(
                    self.attn_heads_to_prune_per_layer,
                    self.max_total_attn_heads - total_attn_pruned,
                )
                est_cost = (n_heads / 16) * attn_delta * self.ppl_safety_multiplier
                if ppl_budget_used + est_cost <= self.ppl_budget:
                    lp.attn_heads_to_prune = n_heads
                    total_attn_pruned += n_heads
                    ppl_budget_used += est_cost

            # ---- Reconstruction iterations ----
            lp.reconstruction_priority = num_layers - 1 - layer_idx
            if max_amp > self.cascade_threshold:
                lp.reconstruction_iterations = int(
                    self.base_recon_iterations * self.high_cascade_recon_multiplier)
            else:
                lp.reconstruction_iterations = self.base_recon_iterations

            prescriptions.append(lp)

        return CompressionPrescription(
            model_name=model_name,
            baseline_ppl=baseline_ppl,
            num_layers=num_layers,
            intermediate_size=intermediate_size,
            num_attention_heads=num_attention_heads,
            layers=prescriptions,
            depth_prune_candidates=[li for _, li in sorted(
                [(criticality_by_layer.get(i, {"mlp_ppl_delta": 1, "attn_ppl_delta": 1})["mlp_ppl_delta"] +
                  criticality_by_layer.get(i, {"mlp_ppl_delta": 1, "attn_ppl_delta": 1})["attn_ppl_delta"], i)
                 for i in range(2, num_layers - 2)
                 if (criticality_by_layer.get(i, {"mlp_ppl_delta": 1, "attn_ppl_delta": 1})["mlp_ppl_delta"] +
                     criticality_by_layer.get(i, {"mlp_ppl_delta": 1, "attn_ppl_delta": 1})["attn_ppl_delta"]) < 0.5]
            )],
            depth_prune_applied=depth_prune_applied,
            estimated_ppl_budget_used=ppl_budget_used,
            total_dead_neurons=total_dead,
            total_dormant_neurons=total_dormant,
            total_merged_neurons=total_merged,
            total_attn_heads_pruned=total_attn_pruned,
        )

    def _interpolate_cascade(self, layer_idx, cascade_by_layer, num_layers):
        if layer_idx in cascade_by_layer:
            return cascade_by_layer[layer_idx]
        measured = sorted(cascade_by_layer.keys())
        if not measured:
            return {"max_amplification": 1.0}
        lower = max((m for m in measured if m <= layer_idx), default=measured[0])
        upper = min((m for m in measured if m >= layer_idx), default=measured[-1])
        if lower == upper:
            return cascade_by_layer[lower]
        t = (layer_idx - lower) / (upper - lower)
        d_low = cascade_by_layer[lower]
        d_up = cascade_by_layer[upper]
        return {"max_amplification": d_low["max_amplification"] * (1 - t) + d_up["max_amplification"] * t}


# ---- Parsers ----

def parse_study5_from_terminal(text: str) -> list[dict]:
    import re
    results = []
    pattern = re.compile(
        r"Layer\s+(\d+):\s+dead=\s*(\d+)\s+\([\d.]+%\),\s+dormant=\s*(\d+)\s+\([\d.]+%\),\s+"
        r"rare=\s*(\d+),\s+hyperactive=\s*(\d+)"
    )
    for line in text.strip().split("\n"):
        m = pattern.search(line)
        if m:
            results.append({
                "layer": int(m.group(1)),
                "dead": int(m.group(2)),
                "dormant": int(m.group(3)),
                "rare": int(m.group(4)),
                "hyperactive": int(m.group(5)),
            })
    return results


def parse_study10_from_terminal(text: str) -> list[dict]:
    import re
    results = []
    pattern = re.compile(
        r"Layer\s+(\d+):\s+MLP PPL delta=([+-]?[\d.]+),\s+Attn PPL delta=([+-]?[\d.]+)"
    )
    for line in text.strip().split("\n"):
        m = pattern.search(line)
        if m:
            results.append({
                "layer": int(m.group(1)),
                "mlp_ppl_delta": float(m.group(2)),
                "attn_ppl_delta": float(m.group(3)),
            })
    return results
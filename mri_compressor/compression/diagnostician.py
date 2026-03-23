"""
MRI-Compress Diagnostic Module v2.1
=====================================
Post-mortem fixes from v2 full run (PPL +230%):

ROOT CAUSES:
1. Depth pruning 2 layers -> cascading error (individual deltas don't compose)
2. 104 attention heads pruned -> Frobenius norm is a poor importance proxy
3. No validation between strategies -> compound damage undetected

FIXES in v2.1:
- Depth pruning: OFF by default. When enabled, max 1 layer, and only if
  combined delta (mlp+attn) < 0.15 (not 0.30+0.10 separately)
- Attention pruning: max 2 heads/layer (not 4), only in layers with
  attn_ppl_delta < 0.10 (not 0.40), skip layers near depth-pruned ones
- NEW: ablation modes to test strategies independently
- NEW: cumulative budget -- track estimated total PPL impact
"""

from __future__ import annotations
import json
import logging
import os
from pathlib import Path
from typing import Optional


def _layer_position_scale(layer_idx: int, num_layers: int) -> float:
    """
    Return a [0, 1] scaling factor that discounts the compression budget for
    accuracy-critical boundary layers, based on their position in the network.

    Motivation
    ----------
    Study 24 shows that PPL-based safe sparsity is 70%+ for every layer, but
    task accuracy crashes -20pp at 15.4% global compression. The "maybe" class
    collapses entirely. This divergence is because:

      - Early layers  parse task structure (question, context boundary).
      - Middle layers compose features redundantly — most compressible.
      - Late layers   crystallise predictions into final token probabilities.
        Layer (N-1) has 5× higher PPL sensitivity than the model average.

    By scaling DOWN the compression budget for boundary layers, we preserve the
    reasoning circuits that live there while still compressing middle layers
    aggressively.

    Scale table (approximate, 24-layer model)
    ------------------------------------------
    Position              Scale   Rationale
    Last layer (N-1)      0.10    Output crystallisation; extremely sensitive
    Top 20% (excl. last)  0.30    Rising sensitivity from Study 24
    Middle 50–80%         1.00    Most robust; full PPL-safe budget
    First 25%             0.50    Task-structure encoding; moderately sensitive
    """
    if num_layers <= 1:
        return 1.0
    last = num_layers - 1
    if layer_idx == last:
        return 0.10
    boundary_start = int(num_layers * 0.80)
    if layer_idx >= boundary_start:
        return 0.30
    middle_start = int(num_layers * 0.25)
    if layer_idx >= middle_start:
        return 1.00
    return 0.50

from .prescription import (
    CompressionStrategy,
    LayerPrescription,
    CompressionPrescription,
)

logger = logging.getLogger(__name__)


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
        # Depth pruning -- VERY conservative in v2.1
        enable_depth_pruning: bool = False,        # OFF by default
        depth_prune_combined_delta: float = 0.15,  # mlp+attn combined must be < this
        max_depth_prune_layers: int = 1,            # Max 1 layer (was 2)
        depth_prune_min_gap: int = 5,               # Min 5 layers between depth prunes
        # Attention head pruning -- zeroing only (no param reduction yet)
        enable_attn_pruning: bool = True,
        attn_prune_delta_threshold: float = 0.30,   # Only prune near-free heads (zeroing doesn't save params)
        attn_heads_to_prune_per_layer: int = 2,     # Was 4 -- now 2 (12.5%)
        attn_prune_guard_layers: int = 2,            # Don't prune near depth-pruned layers
        max_total_attn_heads: int = 32,              # Global budget: max 32 heads total
        # Merge settings -- RE-ENABLED in v2.1 (was disabled by accident)
        enable_merging: bool = True,
        merge_redundancy_threshold: float = 0.15,    # Study 14 safe_to_prune_frac > this
        merge_max_reduction: float = 0.25,           # Max 25% reduction per layer via merge
        # Reconstruction
        base_recon_iterations: int = 100,
        high_cascade_recon_multiplier: float = 3.0,
        cascade_threshold: float = 50.0,
        # PPL budget
        ppl_budget: float = 2.0,                    # Max allowed estimated PPL increase (was 1.0)
        ppl_safety_multiplier: float = 2.0,          # Safety multiplier (was 3.0, too conservative)
        # Domain specialization (Study 22)
        target_domain: Optional[str] = None,
        domain_unnecessary_removal_frac: float = 0.05,   # Remove up to 5% of intermediate_size per layer
        domain_critical_protection_frac: float = 0.10,    # Protect top 10% domain-critical neurons
        # Proportional layer budget (Study 22 post-pass)
        proportional_budget_margin: float = 2.0,  # No layer compressed > margin × global-average fraction
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
        self.target_domain = target_domain
        self.domain_unnecessary_removal_frac = domain_unnecessary_removal_frac
        self.domain_critical_protection_frac = domain_critical_protection_frac
        self.proportional_budget_margin = proportional_budget_margin

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
                # Estimate: pruning N/16 heads ~ (N/16) * attn_delta * safety_multiplier
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

    def diagnose_from_summary(self, summary: dict) -> CompressionPrescription:
        """
        Generate a CompressionPrescription from the enriched summary.json format
        produced by mri.summary.build_summary().

        Reads per-layer data directly from the summary dict. Integrates:
        - Study 1: kurtosis/gini for pruning approach selection
        - Study 5: dead/dormant neuron counts
        - Study 6: head importance data (entropy + sink)
        - Study 7: pearson correlation for gate-guided recommendation
        - Study 8: co-activation clusters and consistency for merge/mask viability
        - Study 10: layer criticality (mlp/attn PPL deltas)
        - Study 14: functional redundancy
        - Study 15: perturbation cascade amplification
        - Study 16: phase transition (power law alpha)

        Expected summary format (from build_summary()):
            {
                "architecture": {"num_layers": ..., "intermediate_size": ..., ...},
                "per_layer": {
                    "0": {
                        "study1_activation_profile": {"kurtosis": ..., "gini_coefficient": ...},
                        "study5_neuron_health": {"dead_count": ..., "dormant_count": ...},
                        "study6_attention_heads": [{"head_idx": 0, "mean_entropy": ..., ...}],
                        "study7_gate_wanda_correlation": {"pearson_r": ..., ...},
                        "study8_sparsity_structure": {"n_coactivation_clusters": ..., ...},
                        "study10_layer_redundancy": {"mlp_ppl_delta": ..., "attn_ppl_delta": ...},
                        "study14_functional_redundancy": {"safe_to_prune_count": ..., ...},
                        "study15_perturbation_cascade": {"max_amplification": ...},
                        "study16_phase_transition": {"power_law_alpha": ..., ...},
                    },
                    ...
                },
                "protection_lists": {
                    "never_prune_neurons": [{"layer": 0, "neuron": 5, ...}, ...],
                    "never_prune_heads": [{"layer": 0, "head": 2, ...}, ...],
                },
                "compression_hints": {
                    "per_layer_strategy_hints": {"0": {...}, ...},
                },
                "model": "...",
                "baseline_ppl": ...,
            }
        """
        # ---- Read architecture ----
        arch = summary.get("architecture", {})
        num_layers = arch.get("num_layers", summary.get("num_layers", 0))
        intermediate_size = arch.get("intermediate_size", summary.get("intermediate_size", 0))
        num_attention_heads = arch.get("num_attention_heads", summary.get("num_attention_heads", 16))
        model_name = summary.get("model", "unknown")
        baseline_ppl = summary.get("baseline_ppl", 0.0)

        # per_layer is a dict keyed by string layer index: {"0": {...}, "1": {...}}
        per_layer_raw = summary.get("per_layer", {})

        # Build protection lists indexed by layer
        raw_protection = summary.get("protection_lists", {})
        protection_by_layer = {}  # {layer_idx: set of neuron indices}
        for entry in raw_protection.get("never_prune_neurons", []):
            li = entry.get("layer", -1)
            ni = entry.get("neuron", -1)
            if li >= 0 and ni >= 0:
                protection_by_layer.setdefault(li, set()).add(ni)

        # Compression hints are nested under per_layer_strategy_hints
        raw_hints = summary.get("compression_hints", {})
        compression_hints = raw_hints.get("per_layer_strategy_hints", raw_hints)

        # ---- Phase 1: Depth pruning (only if enabled) ----
        depth_prune_applied = []
        if self.enable_depth_pruning:
            candidates = []
            for layer_idx in range(2, num_layers - 2):
                ld = per_layer_raw.get(str(layer_idx), {})
                s10 = ld.get("study10_layer_redundancy", {})
                mlp_delta = s10.get("mlp_ppl_delta", 1.0)
                attn_delta = s10.get("attn_ppl_delta", 1.0)
                combined = mlp_delta + attn_delta
                if combined < self.depth_prune_combined_delta:
                    candidates.append((combined, layer_idx))
            candidates.sort()

            for _, li in candidates:
                if len(depth_prune_applied) >= self.max_depth_prune_layers:
                    break
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

        depth_guard_set = set()
        for dl in depth_prune_applied:
            for g in range(-self.attn_prune_guard_layers, self.attn_prune_guard_layers + 1):
                depth_guard_set.add(dl + g)

        for layer_idx in range(num_layers):
            layer_key = str(layer_idx)
            ld = per_layer_raw.get(layer_key, {})

            # Study 16: phase transition (alpha)
            s16 = ld.get("study16_phase_transition", {})
            alpha = s16.get("power_law_alpha", 3.0)
            is_heavy_tail = alpha < self.heavy_tail_alpha

            # Study 14: redundancy
            s14 = ld.get("study14_functional_redundancy", {})
            safe_to_prune = s14.get("safe_to_prune_count", 0)
            safe_to_prune_frac = safe_to_prune / intermediate_size if intermediate_size > 0 else 0
            redundant_count = s14.get("n_highly_redundant", 0)
            is_high_redundancy = safe_to_prune_frac > self.high_redundancy_frac

            # Study 5: dead/dormant
            s5 = ld.get("study5_neuron_health", {})
            dead_count = s5.get("dead_count", 0)
            dormant_count = s5.get("dormant_count", 0)

            # Study 11: cross-domain dead neurons — neurons confirmed dead across ALL domains
            # are strictly more reliable than Study 5's single-calibration-set dead count.
            # Take the max to avoid double-counting while still capturing neurons Study 5
            # may have missed (only run on the narrow calibration distribution).
            s11 = ld.get("study11_domain_divergence", {})
            n_dead_all_domains = s11.get("n_dead_across_all", 0)
            if n_dead_all_domains > dead_count:
                dead_count = n_dead_all_domains

            dead_frac = dead_count / intermediate_size if intermediate_size > 0 else 0
            dormant_frac = dormant_count / intermediate_size if intermediate_size > 0 else 0
            has_significant_dead = dead_frac > self.dead_frac_threshold
            has_significant_dormant = dormant_frac > self.dormant_frac_threshold

            # Study 25: geometric importance — neurons with near-zero write contribution
            # are geometrically inert even if their activation value is non-zero.
            # Augment dead_count only if Study 25 evidence exceeds Studies 5+11 count.
            s25 = ld.get("study25_write_vector_geometry", {})
            geom_path = s25.get("geom_importance_path")
            if geom_path:
                _output_dir = summary.get("output_dir", "")
                _full_gp = (
                    os.path.join(_output_dir, geom_path)
                    if _output_dir
                    else geom_path
                )
                if os.path.exists(_full_gp):
                    try:
                        import torch as _torch
                        _geom_imp = _torch.load(_full_gp, weights_only=True).float()
                        # Neurons below 1% of median are geometrically inert
                        _threshold = 0.01 * _geom_imp.median().item()
                        _n_geom_dead = int((_geom_imp < _threshold).sum().item())
                        if _n_geom_dead > dead_count:
                            dead_count = _n_geom_dead
                            dead_frac = dead_count / intermediate_size if intermediate_size > 0 else 0
                            has_significant_dead = dead_frac > self.dead_frac_threshold
                    except Exception:
                        pass

            # Study 10: criticality
            s10 = ld.get("study10_layer_redundancy", {})
            mlp_delta = s10.get("mlp_ppl_delta", 1.0)
            attn_delta = s10.get("attn_ppl_delta", 1.0)
            is_critical = mlp_delta > self.critical_mlp_delta or attn_delta > self.critical_attn_delta

            # Study 15: cascade
            s15 = ld.get("study15_perturbation_cascade", {})
            max_amp = s15.get("max_amplification", 1.0)

            # Study 1: kurtosis/gini for pruning approach
            s1 = ld.get("study1_activation_profile", {})
            kurtosis = s1.get("kurtosis", 5.0)
            gini = s1.get("gini_coefficient", 0.5)

            # Determine pruning approach from Study 1
            if kurtosis > 10:
                pruning_approach = "structured"
            elif kurtosis < 3:
                pruning_approach = "uniform_or_quant"
            else:
                pruning_approach = "structured"

            # Study 7: pearson correlation → gate-guided pruning when low correlation
            s7 = ld.get("study7_gate_wanda_correlation", {})
            pearson = s7.get("pearson_r", 1.0)
            if pearson < 0.5:
                pruning_approach = "gate_guided"

            # Study 2: gate importance scores for gate_guided pruning
            gate_importance_scores = None
            if pruning_approach == "gate_guided":
                s2 = ld.get("study2_gate_patterns", {})
                if s2:
                    gate_importance_scores = s2.get("gate_scores", None)
                else:
                    logger.warning(
                        "Layer %d: gate_guided pruning requested but no study2_gate_patterns "
                        "found in summary — falling back to Wanda.", layer_idx
                    )

            # Study 8: co-activation and consistency
            s8 = ld.get("study8_sparsity_structure", {})
            co_activation_clusters = s8.get("n_coactivation_clusters", 0)
            consistency = s8.get("activation_consistency", 0.0)
            merge_worthwhile = co_activation_clusters > 5
            static_mask_viable = consistency > 0.7

            # Study 6: head importance data
            # summary stores a list of per-head dicts with mean_entropy and first_token_attention
            s6_heads = ld.get("study6_attention_heads", None)
            head_importance_data = None
            if s6_heads is not None and isinstance(s6_heads, list):
                head_importance_data = []
                for h_data in s6_heads:
                    entry = {"head_idx": h_data.get("head_idx", 0)}
                    if "mean_entropy" in h_data:
                        entry["entropy"] = h_data["mean_entropy"]
                    if "first_token_attention" in h_data:
                        entry["sink_score"] = h_data["first_token_attention"]
                    if "max_attention_concentration" in h_data:
                        entry["concentration"] = h_data["max_attention_concentration"]
                    head_importance_data.append(entry)

            # Study 3: precomputed wanda scores (tensor path reference)
            s3 = ld.get("study3_wanda_scores", {})
            precomputed_wanda_scores = s3.get("tensor_path", None)

            # Study 12: cross-layer motifs (bottleneck neurons → protect)
            s12 = ld.get("study12_cross_layer_motifs", {})
            n_bottleneck = s12.get("n_bottleneck_neurons", 0)
            # If layer has many bottleneck neurons, protect them by adding to protection
            if n_bottleneck > 0 and protection_by_layer.get(layer_idx) is not None:
                # Bottleneck neurons are important relay points — don't over-prune this layer
                pass  # Protection is already handled; just note the count for budget scaling

            # Study 13: information bottleneck
            s13 = ld.get("study13_information_bottleneck", {})
            information_retained = s13.get("information_retained", 1.0)

            # Study 14 additional: n_keystone for protection
            n_keystone = s14.get("n_keystone", 0)

            # Study 15 additional: mean_damping_ratio for self-healing detection
            mean_damping = s15.get("mean_damping_ratio", 1.0)
            is_self_healing = mean_damping > 1.5  # high damping = errors dissipate

            # Study 16 additional: tail_fraction and activation_entropy
            tail_fraction = s16.get("tail_fraction", 0.0)
            activation_entropy = s16.get("activation_entropy", 5.0)

            # Study 17: cross-layer alignment (weight sharing candidate)
            s17 = ld.get("study17_cross_layer_alignment", {})
            cka_linear = s17.get("cka_linear", 0.0)
            merge_score_17 = s17.get("merge_score", 0.0)

            # Study 18: weight rank (low-rank factorization target)
            # A ratio95 of 0.85 means 85% of full rank captures 95% of spectral
            # energy -- still significant savings via SVD decomposition.
            s18 = ld.get("study18_weight_rank", {})
            avg_ratio95 = s18.get("avg_ratio95", 1.0)
            low_rank_ranks = None
            if avg_ratio95 < 0.90:
                # Per-projection selective ranks: only include projections with a clear
                # low-rank structure (ratio95 < 0.50 → needs < 50% of full rank, i.e.
                # ≥ 2× effective compression in that dimension). Projections absent from
                # the dict will be SKIPPED by factorize_mlp (no loss on near-full-rank ones).
                _PER_PROJ_LR_THRESH = 0.50
                per_proj: dict = {}
                for _pname, _pkey in [
                    ("gate_proj", "gate_proj_ratio95"),
                    ("up_proj",   "up_proj_ratio95"),
                    ("down_proj", "down_proj_ratio95"),
                ]:
                    _ratio = s18.get(_pkey, 1.0)
                    if _ratio < _PER_PROJ_LR_THRESH:
                        per_proj[_pname] = int(_ratio * hidden_size)
                if per_proj:
                    low_rank_ranks = per_proj

            # Study 19: head clustering (prunable heads from cluster analysis)
            cluster_prunable_heads = ld.get("study19_prunable_heads", None)

            # Study 20: static/dynamic decomposition (foldable neurons)
            s20 = ld.get("study20_static_dynamic", {})
            foldable_neuron_count = s20.get("foldable_neuron_count", 0)
            foldable_indices_path = s20.get("foldable_indices_path", None)
            # Load foldable indices from tensor file if available
            foldable_neuron_indices = None
            if foldable_indices_path is not None:
                try:
                    import torch
                    output_dir = summary.get("output_dir", "")
                    full_path = os.path.join(output_dir, foldable_indices_path) if output_dir else foldable_indices_path
                    if os.path.exists(full_path):
                        foldable_neuron_indices = torch.load(full_path, weights_only=True).tolist()
                except Exception:
                    pass

            # Study 21: magnitude divergence (domain sensitivity)
            s21 = ld.get("study21_magnitude_divergence", {})
            n_domain_sensitive = s21.get("n_domain_sensitive_neurons", 0)
            domain_sensitivity_score = s21.get("domain_sensitivity_score", 0.0)

            # Studies 6+19: DeltaNet channel groups for output-proj zeroing
            # study6_linear_low_magnitude_groups: [[s0,e0], [s1,e1], ...]
            # study19_linear_prunable_groups:     [[s0,e0], [s1,e1], ...]
            # Both sources are merged (dedup) into attn_zero_channel_groups.
            s6_low_mag = ld.get("study6_linear_low_magnitude_groups", None)
            s19_linear = ld.get("study19_linear_prunable_groups", None)
            attn_zero_groups = None
            if self.enable_attn_pruning:
                combined = []
                if s6_low_mag:
                    combined.extend([tuple(g) for g in s6_low_mag])
                if s19_linear:
                    for g in s19_linear:
                        t = tuple(g)
                        if t not in combined:
                            combined.append(t)
                if combined:
                    attn_zero_groups = combined

            # Study 23: Attention output projection low-rank factorization
            # Threshold: compression_ratio (rank_95 / d_model) < 0.35
            s23 = ld.get("study23_attn_rank", {})
            attn_out_low_rank = False
            attn_out_target_rank = None
            if (s23 and self.enable_attn_pruning
                    and s23.get("compression_ratio", 1.0) < 0.35):
                attn_out_low_rank = True
                attn_out_target_rank = s23.get("rank_95")

            # Protection lists — built per-layer from the restructured data
            protected_indices = protection_by_layer.get(layer_idx, None)

            # Reconstruction budget adjustment from Study 7
            extra_recon_budget = 0
            if pearson < 0.5:
                extra_recon_budget = 100  # higher reconstruction budget for low-correlation layers

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
                # New fields
                protected_neuron_indices=protected_indices,
                precomputed_wanda_scores=precomputed_wanda_scores,
                head_importance_data=head_importance_data,
                pruning_approach=pruning_approach,
                gate_importance_scores=gate_importance_scores,
                static_mask_viable=static_mask_viable,
                merge_worthwhile=merge_worthwhile,
                # Studies 12-21 integrations
                information_retained=information_retained,
                cka_merge_score=merge_score_17,
                cka_with_next=cka_linear,
                low_rank_ranks=low_rank_ranks,
                cluster_prunable_heads=cluster_prunable_heads,
                foldable_neuron_count=foldable_neuron_count,
                foldable_neuron_indices=foldable_neuron_indices,
                n_domain_sensitive_neurons=n_domain_sensitive,
                # Studies 6+19+23: hybrid attention compression
                attn_zero_channel_groups=attn_zero_groups,
                attn_output_proj_low_rank=attn_out_low_rank,
                attn_output_proj_target_rank=attn_out_target_rank,
            )

            # ---- Depth pruning ----
            if layer_idx in depth_prune_applied:
                lp.strategy = CompressionStrategy.DEPTH_PRUNE
                lp.depth_prune = True
                est_cost = (mlp_delta + attn_delta) * self.ppl_safety_multiplier
                ppl_budget_used += est_cost
                prescriptions.append(lp)
                continue

            # ---- Use compression_hints if available ----
            hint = compression_hints.get(layer_key, {})
            # hint may contain: pruning_method, recommended_importance, static_mask_viable, etc.

            # ---- MLP compression decision tree (same as diagnose()) ----
            if is_critical:
                if has_significant_dead and mlp_delta <= self.critical_mlp_delta:
                    lp.strategy = CompressionStrategy.DEAD_REMOVAL_AND_MERGE
                    alive = intermediate_size - dead_count
                    lp.merge_target_width = alive
                    total_dead += dead_count
                else:
                    lp.strategy = CompressionStrategy.LIGHT_TOUCH

            elif is_heavy_tail and (is_high_redundancy or has_significant_dead):
                lp.strategy = CompressionStrategy.DEAD_REMOVAL_AND_MERGE
                alive_neurons = intermediate_size - dead_count

                if has_significant_dormant:
                    lp.dormant_neuron_count = dormant_count
                    alive_neurons -= dormant_count
                    total_dormant += dormant_count

                total_dead += dead_count

                # Merge: use Study 8 merge_worthwhile signal
                if (self.enable_merging and (is_high_redundancy or merge_worthwhile)
                    and redundant_count > 0 and alive_neurons > 0):
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

            # ---- Study 13: Information bottleneck scaling ----
            # Layers with low information retention can be pruned harder
            if information_retained < 0.5 and lp.strategy == CompressionStrategy.LIGHT_TOUCH:
                if has_significant_dead or has_significant_dormant:
                    lp.strategy = CompressionStrategy.DORMANT_REMOVAL
                    if has_significant_dead:
                        total_dead += dead_count
                    if has_significant_dormant:
                        total_dormant += dormant_count

            # ---- Study 15: Self-healing layers tolerate more aggressive pruning ----
            if is_self_healing and lp.merge_target_width is not None:
                cur_width = lp.merge_target_width
                extra_merge = int(cur_width * 0.10)
                lp.merge_target_width = max(cur_width - extra_merge, cur_width // 2)
                total_merged += extra_merge

            # ---- Attention head pruning (budget-aware) ----
            # NOTE: Current attention pruning zeros head weights but does NOT
            # reduce parameter count (shapes stay the same). This means the
            # only benefit is inference speedup, not compression. Until we
            # implement actual head removal (reshaping the attention matrices),
            # we must be very conservative: only prune heads where the cost
            # is near-zero. Use attn_delta < 0.30 as threshold.
            if (self.enable_attn_pruning
                and attn_delta < self.attn_prune_delta_threshold
                and not is_critical
                and layer_idx not in (0, 1)
                and layer_idx not in depth_guard_set
                and total_attn_pruned < self.max_total_attn_heads
                and lp.strategy != CompressionStrategy.DEPTH_PRUNE):

                n_heads = min(
                    self.attn_heads_to_prune_per_layer,
                    self.max_total_attn_heads - total_attn_pruned,
                )
                est_cost = (n_heads / num_attention_heads) * attn_delta * self.ppl_safety_multiplier
                if ppl_budget_used + est_cost <= self.ppl_budget:
                    lp.attn_heads_to_prune = n_heads
                    total_attn_pruned += n_heads
                    ppl_budget_used += est_cost

            # ---- Reconstruction iterations ----
            lp.reconstruction_priority = num_layers - 1 - layer_idx
            base_iters = self.base_recon_iterations + extra_recon_budget
            if max_amp > self.cascade_threshold:
                lp.reconstruction_iterations = int(
                    base_iters * self.high_cascade_recon_multiplier)
            else:
                lp.reconstruction_iterations = base_iters

            # ---- Study 18: Low-rank factorization target ----
            # Apply low-rank to any non-critical LIGHT_TOUCH layer where
            # Study 18 identified rank deficiency.
            #
            # NOTE: The target_rank must be in terms of the SMALLER matrix
            # dimension (hidden_size), not intermediate_size. For SwiGLU:
            #   gate_proj/up_proj: [intermediate, hidden] -> min dim = hidden
            #   down_proj:         [hidden, intermediate] -> min dim = hidden
            # So the rank target should be ratio * hidden_size.
            #
            # Profitability check: factorization saves params only when
            #   rank * (m + n) < m * n, i.e. rank < m*n/(m+n).
            # We require at least 5% param savings per matrix to justify
            # the quality loss. With hidden=h, intermediate=I:
            #   max_useful_rank = h*I/(h+I)
            #   min_saving_rank = 0.95 * max_useful_rank
            hidden_size = arch.get("hidden_size", intermediate_size)
            if (low_rank_ranks is not None
                    and lp.strategy == CompressionStrategy.LIGHT_TOUCH
                    and not is_critical):
                target_rank = int(avg_ratio95 * hidden_size)
                max_useful_rank = (intermediate_size * hidden_size) // (intermediate_size + hidden_size)
                min_saving_rank = int(0.95 * max_useful_rank)

                # Only factorize if the target rank is low enough to save ≥5% params
                if target_rank <= min_saving_rank:
                    lp.low_rank_target = target_rank
                    lp.low_rank_ranks = low_rank_ranks

            # ---- Study 25: Dynamic rank refinement ----
            # intrinsic_dim_95 measures the actual PCA rank of write vectors —
            # i.e. how many dimensions the MLP actually uses during inference.
            # If this is smaller than the static SVD estimate from Study 18,
            # use the tighter (empirical) bound as the low_rank_target.
            # Guard: only override when dim95 < 80% of hidden_size to avoid
            # over-trusting small sample estimates.
            intrinsic_dim_25 = s25.get("intrinsic_dim_95", 0)
            if (
                intrinsic_dim_25 > 0
                and intrinsic_dim_25 < int(hidden_size * 0.80)
                and lp.low_rank_target is not None
                and intrinsic_dim_25 < lp.low_rank_target
            ):
                lp.low_rank_target = intrinsic_dim_25

            # ---- Study 20: Static neuron folding on LIGHT_TOUCH layers ----
            # The compressor only applies folding inside DEAD_REMOVAL_AND_MERGE
            # or DORMANT_REMOVAL. For LIGHT_TOUCH layers with foldable neurons,
            # we still want folding. This is handled in the compressor fix.

            prescriptions.append(lp)

        # Build depth prune candidates list using Study 10 data
        depth_prune_candidates = []
        for i in range(2, num_layers - 2):
            ld_i = per_layer_raw.get(str(i), {})
            s10_i = ld_i.get("study10_layer_redundancy", {})
            combined_i = s10_i.get("mlp_ppl_delta", 1.0) + s10_i.get("attn_ppl_delta", 1.0)
            if combined_i < 0.5:
                depth_prune_candidates.append((combined_i, i))
        depth_prune_candidates.sort()

        # ---- Study 17: Weight sharing pairs ----
        weight_sharing_pairs = []
        for ws_idx in range(num_layers - 1):
            ws_key = str(ws_idx)
            ws_ld = per_layer_raw.get(ws_key, {})
            ws_s17 = ws_ld.get("study17_cross_layer_alignment", {})
            ws_cka = ws_s17.get("cka_linear", 0.0)
            ws_ms = ws_s17.get("merge_score", 0.0)
            if ws_cka > 0.9 and ws_ms > 0.7:
                # Neither layer should be depth-pruned
                if ws_idx not in depth_prune_applied and (ws_idx + 1) not in depth_prune_applied:
                    weight_sharing_pairs.append((ws_idx, ws_idx + 1))

        # ---- Study 22: Domain Specialization Overlay ----
        # When target_domain is set, read Study 22 per-layer data and:
        # 1. Protect domain-critical neurons (add to protected_neuron_indices)
        # 2. Upgrade LIGHT_TOUCH -> DOMAIN_SPECIALIZE with domain-unnecessary removal
        # 3. Enhance DEAD_REMOVAL layers with additional domain-unnecessary removal
        total_domain_unnecessary = 0
        if self.target_domain is not None:
            import torch
            output_dir = summary.get("output_dir", "")
            logger.info(f"  Domain specialization overlay for domain: {self.target_domain}")

            # Study 25: pre-fetch per-layer domain divergence angle list once.
            # domain_direction_divergences[domain][layer_idx] = angle in degrees.
            # Used only for re-ranking candidates, never for budget scaling.
            s25_agg_dom_div = (
                summary.get("aggregated", {})
                       .get("study25", {})
                       .get("domain_direction_divergences", {})
                       .get(self.target_domain, [])
            )

            # Model-average geom_importance_concentration across all layers (Study 25).
            # Used for model-relative pool expansion: each layer is compared to its own
            # model's average, not to any absolute threshold.
            _s25_conc_vals = [
                per_layer_raw.get(str(li), {})
                             .get("study25_write_vector_geometry", {})
                             .get("geom_importance_concentration")
                for li in range(num_layers)
            ]
            _s25_conc_vals = [v for v in _s25_conc_vals if v is not None]
            s25_avg_geom_conc = sum(_s25_conc_vals) / len(_s25_conc_vals) if _s25_conc_vals else 0.0

            for lp in prescriptions:
                layer_key = str(lp.layer_idx)
                ld = per_layer_raw.get(layer_key, {})
                s22 = ld.get("study22_domain_conditional_importance", {})

                if not s22:
                    continue

                domain_tensor_paths = s22.get("domain_tensor_paths", {})
                target_tensor_path = domain_tensor_paths.get(self.target_domain)

                if target_tensor_path is None:
                    continue

                # Load domain-specific Wanda tensor
                full_path = os.path.join(output_dir, target_tensor_path) if output_dir else target_tensor_path
                if not os.path.exists(full_path):
                    logger.warning(f"    Layer {lp.layer_idx}: tensor not found at {full_path}")
                    continue

                try:
                    domain_scores = torch.load(full_path, weights_only=True)
                except Exception as e:
                    logger.warning(f"    Layer {lp.layer_idx}: failed to load tensor: {e}")
                    continue

                n_neurons = domain_scores.shape[0]

                # --- Protect domain-critical neurons (top K) ---
                k_critical = max(1, int(n_neurons * self.domain_critical_protection_frac))
                critical_indices = set(domain_scores.topk(k_critical).indices.tolist())

                # Merge with existing protection
                if lp.protected_neuron_indices is not None:
                    lp.protected_neuron_indices = lp.protected_neuron_indices | critical_indices
                else:
                    lp.protected_neuron_indices = critical_indices
                lp.domain_critical_indices = critical_indices

                # --- Compute domain-unnecessary indices ---
                # Load global mean for safety gate
                global_path_key = s22.get("global_mean_wanda_path")
                global_scores = None
                if global_path_key:
                    gp = os.path.join(output_dir, global_path_key) if output_dir else global_path_key
                    if os.path.exists(gp):
                        try:
                            global_scores = torch.load(gp, weights_only=True)
                        except Exception:
                            pass

                # Study 24: use per-layer evidence-based safe sparsity when available.
                # Study 24 iteratively measures domain PPL at multiple sparsity levels
                # and reports the highest level where PPL increase stays < 15%.
                # This replaces the fixed domain_unnecessary_removal_frac (5%) cap
                # with a per-layer budget grounded in actual measured PPL impact.
                s24 = ld.get("study24_domain_compression_curve", {})
                ppl_safe_sparsity = s24.get(
                    "safe_domain_sparsity",
                    self.domain_unnecessary_removal_frac,   # fallback: 5% fixed cap
                )

                # Non-uniform layer scaling: protect boundary layers from accuracy
                # degradation. PPL alone is too optimistic for early/late layers —
                # task reasoning circuits live there and are not captured by PPL.
                # Middle layers (25-80% depth) get the full PPL-safe budget;
                # boundary layers get a position-scaled fraction of it.
                pos_scale = _layer_position_scale(lp.layer_idx, num_layers)
                layer_safe_sparsity = ppl_safe_sparsity * pos_scale
                max_removable = int(n_neurons * layer_safe_sparsity)

                if pos_scale < 1.0:
                    logger.debug(
                        "  Layer %d: PPL-safe=%.1f%%  pos_scale=%.2f  "
                        "effective_budget=%.1f%%",
                        lp.layer_idx,
                        ppl_safe_sparsity * 100,
                        pos_scale,
                        layer_safe_sparsity * 100,
                    )

                # Study 25: read domain divergence angle — used only for re-ranking, not budget.
                # Low angle → layer is domain-agnostic → domain-Wanda less reliable → geometry
                # gets more weight in the combined ranking below.
                div_angle = (
                    s25_agg_dom_div[lp.layer_idx]
                    if lp.layer_idx < len(s25_agg_dom_div) else None
                )

                sorted_scores, sorted_indices = domain_scores.sort()

                # Study 25: adaptive candidate pool — model-relative, always expands.
                # The pool determines WHICH neurons are candidates; actual removal count
                # is capped by max_removable (Study 24 budget). Pool must never be smaller
                # than the budget or the pool itself becomes the bottleneck.
                #
                # base_pool = 1.5× the Study 24 budget, floored at 25%.
                # Expansion: if this layer's concentration is below the model average,
                # importance is more concentrated (few neurons dominate) → safe to expand
                # the pool further. Pool never shrinks below base_pool.
                _s25_dom = ld.get("study25_write_vector_geometry", {})
                geom_conc = _s25_dom.get("geom_importance_concentration", None)
                base_pool = max(layer_safe_sparsity * 1.5, 0.25)
                if geom_conc is not None and s25_avg_geom_conc > 0:
                    relative_deficit = max(0.0, s25_avg_geom_conc - geom_conc) / s25_avg_geom_conc
                    candidate_frac = min(1.0, base_pool + 0.10 * relative_deficit)
                else:
                    candidate_frac = base_pool
                bottom_k_indices = sorted_indices[:int(n_neurons * candidate_frac)].tolist()

                # Study 25: domain geometric importance — re-rank removal candidates
                # by combined Wanda+Geom score so neurons that are doubly unimportant
                # (low input-side Wanda AND low output-side geometric contribution)
                # rise to the top of the removal list. Neurons with low Wanda but
                # non-trivial geometric importance are de-prioritised.
                s25_dom_key = f"study25_domain_{self.target_domain}"
                dom_geom_rel = ld.get(s25_dom_key, {}).get("geom_importance_path")
                dom_geom_scores = None
                if dom_geom_rel:
                    dom_geom_full = (
                        os.path.join(output_dir, dom_geom_rel)
                        if output_dir else dom_geom_rel
                    )
                    if os.path.exists(dom_geom_full):
                        try:
                            import torch as _torch_s25
                            dom_geom_scores = _torch_s25.load(
                                dom_geom_full, weights_only=True).float()
                        except Exception:
                            pass

                if dom_geom_scores is not None and dom_geom_scores.shape[0] == n_neurons:
                    # Study 25 Add 3: direction_coherence → weight the geometry signal.
                    # Rank-based combination (avoids scale sensitivity between metrics):
                    #   wanda_rank[j] = ascending position in domain-Wanda sort
                    #   geom_rank[j]  = ascending position in domain-geom sort
                    #   combined[j]   = wanda_rank + geom_weight * geom_rank
                    #
                    # geom_weight combines two model-relative signals:
                    #   coherence: high → write vectors aligned → geometry reliable → weight up
                    #   div_angle: low → layer domain-agnostic → domain-Wanda unreliable → weight up
                    # Both signals push geometry weight higher when Wanda is less trustworthy.
                    # Range: [0.5 × 1.0, 1.5 × 1.5] = [0.5, 2.25]
                    coherence = _s25_dom.get("direction_coherence", 0.5)
                    div_factor = min(div_angle / 90.0, 1.0) if div_angle is not None else 0.5
                    geom_weight = (0.5 + coherence) * (1.0 + (1.0 - div_factor) * 0.5)

                    wanda_ranks = {int(idx): r
                                   for r, idx in enumerate(sorted_indices.tolist())}
                    geom_sorted = dom_geom_scores.argsort()  # ascending
                    geom_ranks = {int(idx): r
                                  for r, idx in enumerate(geom_sorted.tolist())}
                    bottom_k_indices = sorted(
                        bottom_k_indices,
                        key=lambda idx: (wanda_ranks.get(idx, n_neurons)
                                         + geom_weight * geom_ranks.get(idx, n_neurons)),
                    )
                    logger.debug(
                        f"    Layer {lp.layer_idx}: Wanda+Geom re-ranking "
                        f"(coherence={coherence:.3f}, div_angle={div_angle}, "
                        f"geom_weight={geom_weight:.2f}, pool={candidate_frac:.2f}, "
                        f"{len(bottom_k_indices)} candidates)"
                    )

                if global_scores is not None:
                    global_median = global_scores.median().item()
                    unnecessary = [
                        idx for idx in bottom_k_indices
                        if domain_scores[idx].item() < global_median
                        and idx not in lp.protected_neuron_indices
                    ]
                else:
                    unnecessary = [
                        idx for idx in bottom_k_indices
                        if idx not in lp.protected_neuron_indices
                    ]

                # Cap at per-layer safe sparsity (from Study 24) or fixed fallback
                unnecessary = unnecessary[:max_removable]

                if unnecessary:
                    lp.domain_unnecessary_indices = unnecessary
                    lp.domain_unnecessary_count = len(unnecessary)
                    lp.target_domain = self.target_domain
                    total_domain_unnecessary += len(unnecessary)

                    # Upgrade LIGHT_TOUCH -> DOMAIN_SPECIALIZE
                    if lp.strategy == CompressionStrategy.LIGHT_TOUCH:
                        lp.strategy = CompressionStrategy.DOMAIN_SPECIALIZE
                        logger.info(
                            f"    Layer {lp.layer_idx}: LIGHT_TOUCH -> DOMAIN_SPECIALIZE "
                            f"({len(unnecessary)} unnecessary, {len(critical_indices)} protected)")

                    # For DEAD_REMOVAL layers, the compressor will handle
                    # domain-unnecessary removal as an additional step
                    elif lp.strategy in (CompressionStrategy.DEAD_REMOVAL_AND_MERGE,
                                         CompressionStrategy.DORMANT_REMOVAL):
                        logger.info(
                            f"    Layer {lp.layer_idx}: +{len(unnecessary)} domain-unnecessary "
                            f"neurons on top of {lp.strategy.name}")

        # ---- Proportional layer budget: prevent information bottlenecks ----
        #
        # After computing per-layer removal counts, ensure no single layer is
        # compressed more than (proportional_budget_margin × global_avg_fraction).
        #
        # Motivation
        # ----------
        # With non-uniform PPL-safe budgets a middle layer might lose 35% of its
        # neurons while boundary layers lose only 3%.  While each decision is
        # individually PPL-safe (from Study 24), the resulting SHAPE asymmetry
        # means a thin middle layer processes the full residual stream after a
        # wide preceding layer — creating a computational bottleneck analogous
        # to a narrow pipe in a flow system.
        #
        # The proportional cap preserves the relative width distribution of the
        # network.  It does NOT override the position-scaling (pos_scale) from
        # _layer_position_scale — that runs first and limits boundary layers.
        # The proportional cap then catches any middle layer that nevertheless
        # remains far above the global average.
        #
        # Example: global avg = 10%, margin = 2.0 → ceiling = 20%.
        # A layer budgeted at 35% is capped to 20%; one at 15% is untouched.
        if self.target_domain is not None and self.proportional_budget_margin > 0:
            _layers_with_removals = [
                lp for lp in prescriptions
                if lp.domain_unnecessary_indices
            ]
            if _layers_with_removals:
                _total_neurons = len(_layers_with_removals) * intermediate_size
                _total_removed = sum(
                    len(lp.domain_unnecessary_indices)
                    for lp in _layers_with_removals
                )
                _global_avg_frac = (
                    _total_removed / _total_neurons if _total_neurons > 0 else 0.0
                )
                _ceiling_frac = _global_avg_frac * self.proportional_budget_margin

                n_capped = 0
                for lp in _layers_with_removals:
                    _current_n     = len(lp.domain_unnecessary_indices)
                    _proportional_max = int(intermediate_size * _ceiling_frac)
                    if _current_n > _proportional_max:
                        _old_n = _current_n
                        lp.domain_unnecessary_indices = (
                            lp.domain_unnecessary_indices[:_proportional_max]
                        )
                        lp.domain_unnecessary_count = _proportional_max
                        total_domain_unnecessary -= (_old_n - _proportional_max)
                        n_capped += 1
                        logger.info(
                            "  Layer %d: proportional cap: %d → %d neurons "
                            "(global_avg=%.1f%%, ceiling=%.1f%%)",
                            lp.layer_idx, _old_n, _proportional_max,
                            _global_avg_frac * 100, _ceiling_frac * 100,
                        )

                if n_capped > 0:
                    logger.info(
                        "  Proportional budget: %d layers capped "
                        "(margin=%.1f×, global_avg=%.1f%%, ceiling=%.1f%%)",
                        n_capped,
                        self.proportional_budget_margin,
                        _global_avg_frac * 100,
                        _ceiling_frac * 100,
                    )

        # Count totals for new operations
        total_low_rank = sum(1 for lp in prescriptions if lp.low_rank_target is not None)
        total_folded = sum(lp.foldable_neuron_count for lp in prescriptions)

        return CompressionPrescription(
            model_name=model_name,
            baseline_ppl=baseline_ppl,
            num_layers=num_layers,
            intermediate_size=intermediate_size,
            num_attention_heads=num_attention_heads,
            layers=prescriptions,
            depth_prune_candidates=[li for _, li in depth_prune_candidates],
            depth_prune_applied=depth_prune_applied,
            estimated_ppl_budget_used=ppl_budget_used,
            total_dead_neurons=total_dead,
            total_dormant_neurons=total_dormant,
            total_merged_neurons=total_merged,
            total_attn_heads_pruned=total_attn_pruned,
            weight_sharing_pairs=weight_sharing_pairs,
            total_low_rank_layers=total_low_rank,
            total_folded_neurons=total_folded,
            total_weight_shared_pairs=len(weight_sharing_pairs),
            total_domain_unnecessary_removed=total_domain_unnecessary,
            target_domain=self.target_domain,
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

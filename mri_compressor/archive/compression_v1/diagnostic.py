"""
MRI-Compress Diagnostic Module
================================
Reads Sparsity MRI study outputs and generates a per-layer compression prescription.

Each layer gets assigned a compression strategy based on measured properties:
  - Phase (heavy-tail vs Gaussian via power-law alpha)
  - Redundancy level (functional similarity census)
  - Dead/dormant neuron fraction
  - Perturbation cascade amplification
  - Layer criticality (MLP/Attn PPL delta)

Output: a CompessionPrescription object that the compressor module consumes.
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
    """Per-layer compression strategy menu."""
    DEAD_REMOVAL_AND_MERGE = auto()   # Remove dead neurons, merge redundant clusters
    STRUCTURED_PRUNE = auto()          # Wanda-guided structured width pruning
    QUANTIZE_AGGRESSIVE = auto()       # 4-bit weight quantization (Gaussian core)
    QUANTIZE_MIXED_PRECISION = auto()  # Mixed precision: outlier channels at higher bits
    LIGHT_TOUCH = auto()               # Minimal compression — critical layers
    DEPTH_PRUNE = auto()               # Entire layer removal candidate


@dataclass
class LayerPrescription:
    """Compression prescription for a single layer."""
    layer_idx: int
    strategy: CompressionStrategy
    # Strategy-specific parameters
    target_sparsity: float = 0.0           # For pruning strategies
    quant_bits: int = 16                   # For quantization strategies
    outlier_bits: int = 16                 # For mixed-precision quant
    outlier_fraction: float = 0.0          # Fraction of channels kept at higher bits
    merge_target_width: Optional[int] = None  # For merge strategy: target neuron count
    dead_neuron_count: int = 0             # Neurons to remove outright (free compression)
    dormant_neuron_count: int = 0
    # Reconstruction parameters (cascade-aware)
    reconstruction_priority: int = 0       # Lower = reconstruct first (late layers first)
    reconstruction_iterations: int = 100   # More iterations for high-cascade layers
    # Diagnostic data that motivated the choice
    alpha: float = 0.0                     # Power-law exponent
    redundancy_frac: float = 0.0           # Fraction of redundant neurons
    cascade_amplification: float = 1.0     # Max perturbation amplification
    mlp_ppl_delta: float = 0.0
    attn_ppl_delta: float = 0.0


@dataclass
class CompressionPrescription:
    """Full model compression prescription."""
    model_name: str
    baseline_ppl: float
    num_layers: int
    intermediate_size: int
    layers: list[LayerPrescription] = field(default_factory=list)
    # Depth pruning candidates (entire layers to remove)
    depth_prune_candidates: list[int] = field(default_factory=list)
    # Summary statistics
    estimated_param_reduction: float = 0.0
    total_dead_neurons: int = 0
    total_merged_neurons: int = 0

    def summary(self) -> str:
        lines = [
            f"{'='*72}",
            f"  MRI-Compress Prescription for {self.model_name}",
            f"  Baseline PPL: {self.baseline_ppl:.2f}",
            f"  Layers: {self.num_layers}, Intermediate size: {self.intermediate_size}",
            f"{'='*72}",
            "",
        ]
        strategy_counts: dict[CompressionStrategy, int] = {}
        for lp in self.layers:
            strategy_counts[lp.strategy] = strategy_counts.get(lp.strategy, 0) + 1

        lines.append("  Strategy distribution:")
        for s, c in sorted(strategy_counts.items(), key=lambda x: -x[1]):
            lines.append(f"    {s.name:30s}: {c} layers")
        lines.append("")

        lines.append(f"  Total dead neurons to remove:  {self.total_dead_neurons:,}")
        lines.append(f"  Total neurons to merge:        {self.total_merged_neurons:,}")
        if self.depth_prune_candidates:
            lines.append(f"  Depth prune candidates:        {self.depth_prune_candidates}")
        lines.append("")

        lines.append(f"  {'Layer':>5} | {'Strategy':>30} | {'Sparsity':>8} | {'Quant':>5} | "
                      f"{'Dead':>6} | {'Alpha':>5} | {'Redund':>6} | {'Cascade':>7} | {'ReconPri':>8}")
        lines.append(f"  {'-'*5}-+-{'-'*30}-+-{'-'*8}-+-{'-'*5}-+-"
                      f"{'-'*6}-+-{'-'*5}-+-{'-'*6}-+-{'-'*7}-+-{'-'*8}")

        for lp in self.layers:
            lines.append(
                f"  {lp.layer_idx:>5} | {lp.strategy.name:>30} | "
                f"{lp.target_sparsity:>7.1%} | {lp.quant_bits:>4}b | "
                f"{lp.dead_neuron_count:>6} | {lp.alpha:>5.2f} | "
                f"{lp.redundancy_frac:>5.1%} | {lp.cascade_amplification:>6.1f}x | "
                f"{lp.reconstruction_priority:>8}"
            )
        lines.append("")
        return "\n".join(lines)


class MRIDiagnostician:
    """
    Reads MRI study outputs and generates compression prescriptions.

    The prescription is based on four diagnostic axes:
    1. Activation distribution shape (Study 16 alpha) → selects technique class
    2. Functional redundancy (Study 14) → determines merge vs prune
    3. Dead/dormant census (Study 5) → identifies free compression
    4. Perturbation cascade (Study 15) → determines reconstruction order
    5. Layer criticality (Study 10) → guards against over-compression

    Thresholds are configurable but defaults are derived from empirical
    analysis of Qwen2.5-0.5B and Qwen2.5-3B MRI runs.
    """

    def __init__(
        self,
        # Phase classification thresholds
        heavy_tail_alpha: float = 2.5,     # Below this → heavy-tailed
        # Redundancy thresholds
        high_redundancy_frac: float = 0.15,  # >15% neurons safe-to-prune → high redundancy
        # Dead neuron thresholds (from Study 5 terminal output)
        dead_frac_threshold: float = 0.05,   # >5% dead → worth removing
        # Criticality thresholds (PPL delta)
        critical_mlp_delta: float = 2.0,     # MLP PPL delta above this → critical
        critical_attn_delta: float = 2.0,
        # Depth pruning thresholds
        depth_prune_mlp_delta: float = 0.35,   # Both MLP and Attn below these → removable
        depth_prune_attn_delta: float = 0.15,
        # Quantization settings
        gaussian_quant_bits: int = 4,
        mixed_prec_bulk_bits: int = 4,
        mixed_prec_outlier_bits: int = 8,
        outlier_tail_fraction: float = 0.02,  # Top 2% of neurons at higher precision
        # Target sparsity for pruning strategies
        prune_sparsity_heavy_redundant: float = 0.50,
        prune_sparsity_structured: float = 0.30,
        # Merge safety: don't merge survivors when dead fraction exceeds this
        # Learned from testing: layers with >50% dead have precious survivors
        high_dead_merge_cutoff: float = 0.50,
        # Reconstruction settings
        base_recon_iterations: int = 100,
        high_cascade_recon_multiplier: float = 3.0,
        cascade_threshold: float = 50.0,  # Amplification above this → extra reconstruction
        # Quantization backend flag
        # Set to True only if using a proper quantization backend (GPTQ/AWQ/bitsandbytes).
        # RTN (round-to-nearest) is too destructive — baseline shows 4-bit RTN → PPL=1.1M.
        enable_quantization: bool = False,
    ):
        self.heavy_tail_alpha = heavy_tail_alpha
        self.high_redundancy_frac = high_redundancy_frac
        self.dead_frac_threshold = dead_frac_threshold
        self.critical_mlp_delta = critical_mlp_delta
        self.critical_attn_delta = critical_attn_delta
        self.depth_prune_mlp_delta = depth_prune_mlp_delta
        self.depth_prune_attn_delta = depth_prune_attn_delta
        self.gaussian_quant_bits = gaussian_quant_bits
        self.mixed_prec_bulk_bits = mixed_prec_bulk_bits
        self.mixed_prec_outlier_bits = mixed_prec_outlier_bits
        self.outlier_tail_fraction = outlier_tail_fraction
        self.prune_sparsity_heavy_redundant = prune_sparsity_heavy_redundant
        self.prune_sparsity_structured = prune_sparsity_structured
        self.high_dead_merge_cutoff = high_dead_merge_cutoff
        self.base_recon_iterations = base_recon_iterations
        self.high_cascade_recon_multiplier = high_cascade_recon_multiplier
        self.cascade_threshold = cascade_threshold
        self.enable_quantization = enable_quantization

    def diagnose(
        self,
        summary_json_path: str,
        study5_data: Optional[list[dict]] = None,
        study10_data: Optional[list[dict]] = None,
    ) -> CompressionPrescription:
        """
        Generate a compression prescription from MRI study outputs.

        Args:
            summary_json_path: Path to the results/summary.json from MRI runs.
            study5_data: Parsed Study 5 dead/dormant data (list of dicts per layer).
                         If None, will try to parse from terminal output or use defaults.
            study10_data: Parsed Study 10 layer redundancy data (list of dicts per layer).
                          If None, will try to parse from terminal output or use defaults.
        """
        with open(summary_json_path) as f:
            summary = json.load(f)

        model_name = summary["model"]
        baseline_ppl = summary["baseline_ppl"]
        num_layers = summary["num_layers"]
        intermediate_size = summary["intermediate_size"]

        findings = summary["findings"]

        # ---- Extract per-layer diagnostics ----

        # Study 16: Phase transition (alpha values)
        s16 = findings.get("study16_phase_transition", {})
        alpha_by_layer = {}
        for item in s16.get("per_layer", []):
            alpha_by_layer[item["layer"]] = {
                "alpha": item["alpha"],
                "tail_fraction": item["tail_fraction"],
                "heavy_tailed": item.get("heavy_tailed", item["alpha"] < self.heavy_tail_alpha),
            }

        # Study 14: Functional redundancy
        s14 = findings.get("study14_functional_redundancy", {})
        redundancy_by_layer = {}
        for item in s14.get("per_layer", []):
            safe = item.get("safe_to_prune", 0)
            redundancy_by_layer[item["layer"]] = {
                "mean_max_sim": item["mean_max_sim"],
                "redundant": item.get("redundant", 0),
                "keystone": item.get("keystone", 0),
                "safe_to_prune": safe,
                "safe_to_prune_frac": safe / intermediate_size if intermediate_size > 0 else 0,
            }

        # Study 15: Perturbation cascade
        s15 = findings.get("study15_perturbation_cascade", {})
        cascade_by_layer = {}
        source_layers = s15.get("per_source_layer", {})
        for src_str, data in source_layers.items():
            src = int(src_str)
            cascade_by_layer[src] = {
                "max_amplification": data.get("avg_max_amplification", 1.0),
                "cascade_depth": data.get("avg_cascade_depth", 0),
                "decay_rate": data.get("avg_decay_rate", 0),
            }

        # Study 13: Information bottleneck (for context, not direct strategy selection)
        s13 = findings.get("study13_information_bottleneck", {})
        bottleneck_by_layer = {}
        for item in s13.get("compression_profile", []):
            bottleneck_by_layer[item["layer"]] = {
                "compression_ratio": item["compression_ratio"],
                "partial_ppl": item["partial_ppl"],
            }

        # Study 5: Dead/dormant neurons (from parsed terminal output)
        dead_by_layer = {}
        if study5_data:
            for item in study5_data:
                dead_by_layer[item["layer"]] = {
                    "dead": item.get("dead", 0),
                    "dormant": item.get("dormant", 0),
                    "dead_frac": item.get("dead", 0) / intermediate_size,
                    "dormant_frac": item.get("dormant", 0) / intermediate_size,
                    "hyperactive": item.get("hyperactive", 0),
                }

        # Study 10: Layer redundancy / criticality (from parsed terminal output)
        criticality_by_layer = {}
        if study10_data:
            for item in study10_data:
                criticality_by_layer[item["layer"]] = {
                    "mlp_ppl_delta": item.get("mlp_ppl_delta", 0.0),
                    "attn_ppl_delta": item.get("attn_ppl_delta", 0.0),
                }

        # ---- Generate prescriptions ----
        prescriptions = []
        depth_prune_candidates = []
        total_dead = 0
        total_merged = 0

        for layer_idx in range(num_layers):
            # Gather diagnostics for this layer
            alpha_data = alpha_by_layer.get(layer_idx, {"alpha": 3.0, "heavy_tailed": False, "tail_fraction": 0})
            redund_data = redundancy_by_layer.get(layer_idx, {"safe_to_prune_frac": 0, "safe_to_prune": 0, "redundant": 0, "mean_max_sim": 0})
            dead_data = dead_by_layer.get(layer_idx, {"dead": 0, "dormant": 0, "dead_frac": 0, "dormant_frac": 0, "hyperactive": 0})
            crit_data = criticality_by_layer.get(layer_idx, {"mlp_ppl_delta": 1.0, "attn_ppl_delta": 1.0})

            # Interpolate cascade amplification for layers between measured source layers
            cascade_data = self._interpolate_cascade(layer_idx, cascade_by_layer, num_layers)

            alpha = alpha_data["alpha"]
            is_heavy_tail = alpha_data["heavy_tailed"] or alpha < self.heavy_tail_alpha
            tail_frac = alpha_data["tail_fraction"]
            safe_to_prune_frac = redund_data["safe_to_prune_frac"]
            safe_to_prune = redund_data["safe_to_prune"]
            redundant_count = redund_data["redundant"]
            is_high_redundancy = safe_to_prune_frac > self.high_redundancy_frac
            dead_count = dead_data["dead"]
            dormant_count = dead_data["dormant"]
            has_significant_dead = dead_data["dead_frac"] > self.dead_frac_threshold
            mlp_delta = crit_data["mlp_ppl_delta"]
            attn_delta = crit_data["attn_ppl_delta"]
            is_critical = mlp_delta > self.critical_mlp_delta or attn_delta > self.critical_attn_delta
            max_amp = cascade_data["max_amplification"]

            # ---- Decision tree ----
            lp = LayerPrescription(
                layer_idx=layer_idx,
                strategy=CompressionStrategy.LIGHT_TOUCH,  # default
                alpha=alpha,
                redundancy_frac=safe_to_prune_frac,
                cascade_amplification=max_amp,
                mlp_ppl_delta=mlp_delta,
                attn_ppl_delta=attn_delta,
                dead_neuron_count=dead_count,
                dormant_neuron_count=dormant_count,
            )

            if is_critical:
                # Critical layers: minimal compression, but still remove dead neurons
                # Attention criticality shouldn't block MLP dead neuron removal
                is_attn_critical = attn_delta > self.critical_attn_delta
                is_mlp_critical = mlp_delta > self.critical_mlp_delta

                if has_significant_dead and not is_mlp_critical:
                    # MLP is safe to prune, only attention is critical
                    lp.strategy = CompressionStrategy.DEAD_REMOVAL_AND_MERGE
                    # But be conservative: no merging, only dead removal
                    alive_neurons = intermediate_size - dead_count
                    lp.merge_target_width = alive_neurons  # No merging
                    lp.target_sparsity = 0.0
                    total_dead += dead_count
                elif is_heavy_tail and self.enable_quantization:
                    lp.strategy = CompressionStrategy.QUANTIZE_MIXED_PRECISION
                    lp.quant_bits = self.mixed_prec_bulk_bits
                    lp.outlier_bits = self.mixed_prec_outlier_bits
                    lp.outlier_fraction = max(tail_frac, self.outlier_tail_fraction)
                else:
                    lp.strategy = CompressionStrategy.LIGHT_TOUCH
                    lp.quant_bits = 8  # Conservative quantization only

            elif is_heavy_tail and (is_high_redundancy or has_significant_dead):
                # Heavy-tail + lots of dead/redundant neurons → merge & remove
                lp.strategy = CompressionStrategy.DEAD_REMOVAL_AND_MERGE
                lp.target_sparsity = self.prune_sparsity_heavy_redundant
                alive_neurons = intermediate_size - dead_count - dormant_count

                # Guard: if dead fraction is very high (>50%), the surviving neurons
                # are disproportionately important. Don't merge them — the dead removal
                # alone is already massive compression. Learned from testing: layer 2
                # (78% dead) had reconstruction MSE=5.0 after merging 18% of survivors,
                # while layer 5 (9% dead) had MSE=0.00003 after a similar merge ratio.
                dead_fraction = dead_count / intermediate_size if intermediate_size > 0 else 0
                if dead_fraction > self.high_dead_merge_cutoff:
                    lp.merge_target_width = None  # No merging at all
                elif redundant_count > 0 and alive_neurons > 0:
                    # Redundancy metrics (Study 14) were measured on ALL neurons
                    # including dead ones. After dead removal, survivors are more
                    # important per-neuron than raw redundancy suggests.
                    effective_redundant_alive = max(0, safe_to_prune - dead_count)
                    non_redundant_alive = alive_neurons - effective_redundant_alive
                    # Keep at least 50% of alive neurons as floor
                    min_keep = max(alive_neurons // 2, non_redundant_alive)
                    lp.merge_target_width = max(min_keep, 1)
                else:
                    lp.merge_target_width = alive_neurons
                total_dead += dead_count
                total_merged += max(0, alive_neurons - (lp.merge_target_width or alive_neurons))

            elif is_heavy_tail and not is_high_redundancy:
                # Heavy-tail but neurons are all important
                # Note: RTN quantization is too destructive for these layers.
                # With a proper quantization backend (GPTQ/AWQ), this would use
                # mixed-precision quantization. For now, light touch.
                if self.enable_quantization:
                    lp.strategy = CompressionStrategy.QUANTIZE_MIXED_PRECISION
                    lp.quant_bits = self.mixed_prec_bulk_bits
                    lp.outlier_bits = self.mixed_prec_outlier_bits
                    lp.outlier_fraction = max(tail_frac, self.outlier_tail_fraction)
                else:
                    lp.strategy = CompressionStrategy.LIGHT_TOUCH

            elif not is_heavy_tail and is_high_redundancy:
                # Gaussian + redundant → structured pruning works well here
                lp.strategy = CompressionStrategy.STRUCTURED_PRUNE
                lp.target_sparsity = self.prune_sparsity_structured

            elif not is_heavy_tail and not is_high_redundancy:
                # Gaussian + all neurons unique → no structural compression possible
                # Note: With proper quantization backend (GPTQ/AWQ), this would use
                # aggressive 4-bit quantization (Gaussian distributions quantize cleanly).
                # For now, light touch.
                if self.enable_quantization:
                    lp.strategy = CompressionStrategy.QUANTIZE_AGGRESSIVE
                    lp.quant_bits = self.gaussian_quant_bits
                else:
                    lp.strategy = CompressionStrategy.LIGHT_TOUCH

            # Check depth pruning eligibility (very low impact layers)
            if (mlp_delta < self.depth_prune_mlp_delta
                and attn_delta < self.depth_prune_attn_delta
                and not is_critical
                and layer_idx not in (0, num_layers - 1)):  # Never remove first/last
                depth_prune_candidates.append(layer_idx)

            # Dead neuron removal is always applied on top of any strategy
            if dead_count > 0 and lp.strategy != CompressionStrategy.DEAD_REMOVAL_AND_MERGE:
                total_dead += dead_count

            # ---- Cascade-aware reconstruction priority ----
            # Lower priority number = reconstruct first
            # Late layers have low amplification → reconstruct first (they're safer)
            # Early layers have high amplification → reconstruct last (they benefit from
            #   already-reconstructed downstream layers)
            lp.reconstruction_priority = num_layers - 1 - layer_idx  # Base: reverse order
            if max_amp > self.cascade_threshold:
                lp.reconstruction_iterations = int(
                    self.base_recon_iterations * self.high_cascade_recon_multiplier
                )
            else:
                lp.reconstruction_iterations = self.base_recon_iterations

            prescriptions.append(lp)

        return CompressionPrescription(
            model_name=model_name,
            baseline_ppl=baseline_ppl,
            num_layers=num_layers,
            intermediate_size=intermediate_size,
            layers=prescriptions,
            depth_prune_candidates=depth_prune_candidates,
            total_dead_neurons=total_dead,
            total_merged_neurons=total_merged,
        )

    def _interpolate_cascade(
        self,
        layer_idx: int,
        cascade_by_layer: dict[int, dict],
        num_layers: int,
    ) -> dict:
        """Interpolate cascade data for layers not directly measured in Study 15."""
        if layer_idx in cascade_by_layer:
            return cascade_by_layer[layer_idx]

        # Find nearest measured layers
        measured = sorted(cascade_by_layer.keys())
        if not measured:
            return {"max_amplification": 1.0, "cascade_depth": 0, "decay_rate": 0}

        # Linear interpolation between nearest measured points
        lower = max((m for m in measured if m <= layer_idx), default=measured[0])
        upper = min((m for m in measured if m >= layer_idx), default=measured[-1])

        if lower == upper:
            return cascade_by_layer[lower]

        t = (layer_idx - lower) / (upper - lower)
        d_low = cascade_by_layer[lower]
        d_up = cascade_by_layer[upper]

        return {
            "max_amplification": d_low["max_amplification"] * (1 - t) + d_up["max_amplification"] * t,
            "cascade_depth": d_low["cascade_depth"] * (1 - t) + d_up["cascade_depth"] * t,
            "decay_rate": d_low["decay_rate"] * (1 - t) + d_up["decay_rate"] * t,
        }


def parse_study5_from_terminal(text: str) -> list[dict]:
    """
    Parse Study 5 output from terminal logs.
    Expected format per line:
      Layer  N: dead=X (Y%), dormant=Z (W%), rare=R, hyperactive=H (P%)
    """
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
    """
    Parse Study 10 output from terminal logs.
    Expected format per line:
      Layer  N: MLP PPL delta=+X.XX, Attn PPL delta=+Y.YY
    """
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
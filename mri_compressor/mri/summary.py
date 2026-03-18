"""
Enriched summary writer for MRI study results.

Produces summary.json with per-layer data from all studies,
plus .pt files for large tensor data (Wanda scores, neuron indices).
"""

import json
import os
import torch
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime


def build_summary(
    model_name: str,
    baseline_ppl: float,
    architecture: dict,
    results: dict,
    output_dir: str,
) -> dict:
    """
    Build enriched summary from MRI study results.

    Args:
        model_name: HuggingFace model name
        baseline_ppl: Baseline perplexity of the unmodified model
        architecture: Dict with num_layers, intermediate_size, hidden_size,
                      num_attention_heads, num_kv_heads, is_gated, activation_fn
        results: Dict of study results keyed by study name
        output_dir: Where to save summary.json and tensor files

    Returns:
        The summary dict (also saved to disk)
    """
    tensor_dir = os.path.join(output_dir, "summary_tensors")
    os.makedirs(tensor_dir, exist_ok=True)

    num_layers = architecture.get("num_layers", 0)

    summary = {
        "model": model_name,
        "baseline_ppl": baseline_ppl,
        "timestamp": datetime.now().isoformat(),
        "architecture": architecture,
        "per_layer": {},
        "aggregated": {},
        "protection_lists": {
            "never_prune_neurons": [],
            "never_prune_heads": [],
        },
        "compression_hints": {
            "per_layer_strategy_hints": {},
        },
    }

    # Initialize per-layer dicts
    for i in range(num_layers):
        summary["per_layer"][str(i)] = {}

    # ---- Study 1: Activation Profiling ----
    if "activation_profiles" in results:
        profiles = results["activation_profiles"]
        for p in profiles:
            layer_key = str(p.layer_idx)
            if layer_key not in summary["per_layer"]:
                summary["per_layer"][layer_key] = {}
            summary["per_layer"][layer_key]["study1_activation_profile"] = {
                "pct_near_zero": p.pct_near_zero,
                "kurtosis": p.kurtosis,
                "gini_coefficient": p.gini_coefficient,
                "skewness": p.skewness,
                "top1_ratio": p.top1_ratio,
                "mean": p.mean,
                "std": p.std,
                "pct_negative": p.pct_negative,
                "pct_exactly_zero": p.pct_exactly_zero,
            }
        summary["aggregated"]["study1"] = {
            "avg_natural_sparsity": sum(p.pct_near_zero for p in profiles) / len(profiles) if profiles else 0,
            "avg_kurtosis": sum(p.kurtosis for p in profiles) / len(profiles) if profiles else 0,
            "max_kurtosis_layer": max(profiles, key=lambda p: p.kurtosis).layer_idx if profiles else -1,
        }

    # ---- Study 2: Gate Patterns (canonical sparsity level) ----
    # results["gate_training"] = Dict[float, Tuple[Dict[int, Tensor], metrics_dict]]
    # We serialise gate scores at the sparsity level closest to 0.5 so the
    # diagnostician can use them for gate-guided pruning (Study 7).
    if "gate_training" in results:
        gate_training = results["gate_training"]
        sparsities = sorted(gate_training.keys())
        canonical_sp = min(sparsities, key=lambda s: abs(s - 0.5))
        patterns, _ = gate_training[canonical_sp]  # patterns: Dict[int, Tensor]
        for layer_idx, gate_scores in patterns.items():
            layer_key = str(layer_idx)
            if layer_key not in summary["per_layer"]:
                summary["per_layer"][layer_key] = {}
            summary["per_layer"][layer_key]["study2_gate_patterns"] = {
                "gate_scores": gate_scores.cpu().tolist(),  # list[float], len=intermediate_size
                "sparsity_level": canonical_sp,
            }

    # ---- Study 3: Wanda Scores ----
    if "wanda_scores" in results:
        wanda = results["wanda_scores"]  # dict: layer_idx -> tensor of scores
        for layer_idx, scores in wanda.items():
            layer_key = str(layer_idx)
            if layer_key not in summary["per_layer"]:
                summary["per_layer"][layer_key] = {}

            # Save full tensor to .pt file
            tensor_path = os.path.join(tensor_dir, f"wanda_layer_{layer_idx}.pt")
            torch.save(scores.cpu(), tensor_path)

            # Store summary statistics in JSON
            top5_vals, top5_idx = scores.topk(min(5, len(scores)))
            bot5_vals, bot5_idx = scores.topk(min(5, len(scores)), largest=False)

            summary["per_layer"][layer_key]["study3_wanda_scores"] = {
                "mean": float(scores.mean()),
                "max": float(scores.max()),
                "std": float(scores.std()),
                "top5_indices": top5_idx.tolist(),
                "bottom5_indices": bot5_idx.tolist(),
                "tensor_path": f"summary_tensors/wanda_layer_{layer_idx}.pt",
            }

    # ---- Study 4: Massive Activations ----
    if "massive_activations" in results:
        massive = results["massive_activations"]
        for r in massive:
            layer_key = str(r.layer_idx)
            if layer_key not in summary["per_layer"]:
                summary["per_layer"][layer_key] = {}
            summary["per_layer"][layer_key]["study4_massive_activations"] = {
                "massive_neuron_indices": r.massive_neuron_indices,
                "massive_neuron_ratios": r.massive_neuron_ratios,
                "input_agnostic_indices": r.input_agnostic_indices,
            }
            # Add to protection list
            for idx, ratio in zip(r.massive_neuron_indices, r.massive_neuron_ratios):
                summary["protection_lists"]["never_prune_neurons"].append({
                    "layer": r.layer_idx,
                    "neuron": idx,
                    "reason": "massive_activation",
                    "ratio": ratio,
                })
            for idx in r.input_agnostic_indices:
                summary["protection_lists"]["never_prune_neurons"].append({
                    "layer": r.layer_idx,
                    "neuron": idx,
                    "reason": "input_agnostic",
                })

    # ---- Study 5: Dead/Dormant Neurons ----
    if "dead_neurons" in results:
        dead = results["dead_neurons"]
        for r in dead:
            layer_key = str(r.layer_idx)
            if layer_key not in summary["per_layer"]:
                summary["per_layer"][layer_key] = {}

            neuron_health = {
                "total_neurons": r.total_neurons,
                "dead_count": r.dead_count,
                "dormant_count": r.dormant_count,
                "rare_count": r.rare_count,
                "hyperactive_count": r.hyperactive_count,
            }

            # Save firing rates tensor if available
            if hasattr(r, 'activation_rate') and r.activation_rate is not None:
                tensor_path = os.path.join(tensor_dir, f"firing_rates_layer_{r.layer_idx}.pt")
                torch.save(r.activation_rate.cpu(), tensor_path)
                neuron_health["firing_rates_path"] = f"summary_tensors/firing_rates_layer_{r.layer_idx}.pt"

            summary["per_layer"][layer_key]["study5_neuron_health"] = neuron_health

        summary["aggregated"]["study5"] = {
            "total_dead": sum(r.dead_count for r in dead),
            "total_dormant": sum(r.dormant_count for r in dead),
            "layers_with_dead": [r.layer_idx for r in dead if r.dead_count > 0],
        }

    # ---- Study 6: Attention Head Importance ----
    if "attention_heads" in results:
        heads = results["attention_heads"]
        by_layer = {}
        for r in heads:
            if r.layer_idx not in by_layer:
                by_layer[r.layer_idx] = []
            by_layer[r.layer_idx].append({
                "head_idx": r.head_idx,
                "mean_entropy": r.mean_entropy,
                "first_token_attention": r.first_token_attention,
                "max_attention_concentration": r.max_attention_concentration,
            })
            # Add attention sinks to protection list
            if r.first_token_attention > 0.5:
                summary["protection_lists"]["never_prune_heads"].append({
                    "layer": r.layer_idx,
                    "head": r.head_idx,
                    "reason": "attention_sink",
                    "first_token_attn": r.first_token_attention,
                })

        for layer_idx, head_data in by_layer.items():
            layer_key = str(layer_idx)
            if layer_key not in summary["per_layer"]:
                summary["per_layer"][layer_key] = {}
            summary["per_layer"][layer_key]["study6_attention_heads"] = head_data

    # ---- Study 7: Gate-Wanda Correlation ----
    if "gate_wanda_correlation" in results:
        corrs = results["gate_wanda_correlation"]
        for r in corrs:
            layer_key = str(r.layer_idx)
            if layer_key not in summary["per_layer"]:
                summary["per_layer"][layer_key] = {}

            pearson = r.pearson_r
            top10_overlap = r.top_k_overlap.get(0.10, 1.0) if hasattr(r, 'top_k_overlap') else 1.0

            recommendation = "wanda"
            if pearson < 0.5 or top10_overlap < 0.5:
                recommendation = "cats_gates"

            summary["per_layer"][layer_key]["study7_gate_wanda_correlation"] = {
                "pearson_r": pearson,
                "spearman_rho": r.spearman_rho,
                "top10_overlap": top10_overlap,
                "recommendation": recommendation,
            }

    # ---- Study 8: Sparsity Structure ----
    if "sparsity_structure" in results:
        structs = results["sparsity_structure"]
        for r in structs:
            layer_key = str(r.layer_idx)
            if layer_key not in summary["per_layer"]:
                summary["per_layer"][layer_key] = {}
            summary["per_layer"][layer_key]["study8_sparsity_structure"] = {
                "token_sparsity_variance": r.token_sparsity_variance,
                "activation_consistency": r.activation_consistency,
                "neuron_specialization_score": r.neuron_specialization_score,
                "n_coactivation_clusters": r.co_activation_clusters,
            }

    # ---- Study 9: Critical Neurons ----
    # CriticalNeuronReport has one report per neuron (not per layer):
    #   layer_idx, neuron_idx, ppl_increase_single, ppl_increase_rank, weight_norm, activation_mean
    if "critical_neurons" in results:
        crits = results["critical_neurons"]
        # Group individual neuron reports by layer
        by_layer = {}
        for r in crits:
            by_layer.setdefault(r.layer_idx, []).append(r)

        for layer_idx, neuron_reports in by_layer.items():
            layer_key = str(layer_idx)
            if layer_key not in summary["per_layer"]:
                summary["per_layer"][layer_key] = {}

            critical_list = []
            for r in neuron_reports:
                critical_list.append({
                    "neuron_idx": r.neuron_idx,
                    "ppl_increase": r.ppl_increase_single,
                    "ppl_increase_rank": r.ppl_increase_rank,
                    "weight_norm": r.weight_norm,
                    "activation_mean": r.activation_mean,
                })
                # Add to protection list if significant
                if r.ppl_increase_single > 0.5:
                    summary["protection_lists"]["never_prune_neurons"].append({
                        "layer": r.layer_idx,
                        "neuron": r.neuron_idx,
                        "reason": "critical_neuron",
                        "ppl_increase": r.ppl_increase_single,
                    })

            summary["per_layer"][layer_key]["study9_critical_neurons"] = critical_list

        summary["aggregated"]["study9"] = {
            "top_10_critical": sorted(
                [e for e in summary["protection_lists"]["never_prune_neurons"]
                 if e["reason"] == "critical_neuron"],
                key=lambda x: x.get("ppl_increase", 0),
                reverse=True,
            )[:10],
        }

    # ---- Study 10: Layer Redundancy ----
    if "layer_redundancy" in results:
        redundancy = results["layer_redundancy"]
        for r in redundancy:
            layer_key = str(r.layer_idx)
            if layer_key not in summary["per_layer"]:
                summary["per_layer"][layer_key] = {}

            existing = summary["per_layer"][layer_key].get("study10_layer_redundancy", {})
            if r.component == "mlp":
                existing["mlp_ppl_delta"] = r.ppl_delta
            elif r.component == "attention":
                existing["attn_ppl_delta"] = r.ppl_delta
            summary["per_layer"][layer_key]["study10_layer_redundancy"] = existing

        mlp_reports = [r for r in redundancy if r.component == "mlp"]
        if mlp_reports:
            summary["aggregated"]["study10"] = {
                "most_critical_mlp_layer": max(mlp_reports, key=lambda r: r.ppl_delta).layer_idx,
                "most_redundant_mlp_layer": min(mlp_reports, key=lambda r: r.ppl_delta).layer_idx,
            }

    # ---- Study 11: Domain Divergence ----
    if "domain_divergence" in results:
        dom = results["domain_divergence"]
        overview = dom.get("overview_reports", [])
        for r in overview:
            layer_key = str(r.layer_idx)
            if layer_key not in summary["per_layer"]:
                summary["per_layer"][layer_key] = {}
            summary["per_layer"][layer_key]["study11_domain_divergence"] = {
                "domain_specificity_score": r.domain_specificity_score,
                "per_domain_active_count": r.per_domain_active_count,
                "n_universal_neurons": r.n_universal_neurons,
                "n_dead_across_all": r.n_dead_across_all,
            }

    # ---- Studies 12-16 (Novel) ----
    if "cross_layer_motifs" in results:
        for r in results["cross_layer_motifs"]:
            layer_key = str(r.source_layer)
            if layer_key not in summary["per_layer"]:
                summary["per_layer"][layer_key] = {}
            summary["per_layer"][layer_key]["study12_cross_layer_motifs"] = {
                "target_layer": r.target_layer,
                "n_strong_connections": r.n_strong_connections,
                "n_bypass_neurons": r.n_bypass_neurons,
                "n_bottleneck_neurons": r.n_bottleneck_neurons,
                "relay_density": r.relay_density,
                "mean_lift": r.mean_lift,
            }

    if "information_bottleneck" in results:
        for r in results["information_bottleneck"]:
            layer_key = str(r.layer_idx)
            if layer_key not in summary["per_layer"]:
                summary["per_layer"][layer_key] = {}
            summary["per_layer"][layer_key]["study13_information_bottleneck"] = {
                "reconstruction_mse": r.reconstruction_mse,
                "partial_lm_loss": r.partial_lm_loss,
                "compression_ratio": r.compression_ratio,
                "information_retained": r.information_retained,
            }

    if "functional_redundancy" in results:
        for r in results["functional_redundancy"]:
            layer_key = str(r.layer_idx)
            if layer_key not in summary["per_layer"]:
                summary["per_layer"][layer_key] = {}
            summary["per_layer"][layer_key]["study14_functional_redundancy"] = {
                "mean_max_similarity": r.mean_max_similarity,
                "n_highly_redundant": r.n_highly_redundant,
                "n_keystone": r.n_keystone,
                "safe_to_prune_count": r.safe_to_prune_count,
            }

    if "perturbation_cascade" in results:
        from collections import defaultdict
        by_layer = defaultdict(list)
        for r in results["perturbation_cascade"]:
            by_layer[r.layer_idx].append(r)
        for layer_idx, reports in by_layer.items():
            layer_key = str(layer_idx)
            if layer_key not in summary["per_layer"]:
                summary["per_layer"][layer_key] = {}
            import numpy as np
            summary["per_layer"][layer_key]["study15_perturbation_cascade"] = {
                "max_amplification": float(np.mean([r.max_amplification for r in reports])),
                "mean_damping_ratio": float(np.mean([r.mean_damping_ratio for r in reports])),
                "peak_layer_offset": float(np.mean([r.peak_layer_offset for r in reports])),
            }

    if "phase_transition" in results:
        for r in results["phase_transition"]:
            layer_key = str(r.layer_idx)
            if layer_key not in summary["per_layer"]:
                summary["per_layer"][layer_key] = {}
            summary["per_layer"][layer_key]["study16_phase_transition"] = {
                "power_law_alpha": r.power_law_alpha,
                "is_heavy_tailed": r.is_heavy_tailed,
                "tail_fraction": r.tail_fraction,
                "gaussian_fit_ks": r.gaussian_fit_ks,
                "activation_entropy": r.activation_entropy,
            }

    # ---- Studies 17-21 (Next-gen) ----
    if "cross_layer_alignment" in results:
        for r in results["cross_layer_alignment"]:
            layer_key = str(r.layer_a)
            if layer_key not in summary["per_layer"]:
                summary["per_layer"][layer_key] = {}
            summary["per_layer"][layer_key]["study17_cross_layer_alignment"] = {
                "layer_b": r.layer_b,
                "cka_linear": r.cka_linear,
                "top_subspace_overlap": r.top_subspace_overlap,
                "residual_delta_cosine": r.residual_delta_cosine,
                "merge_score": r.merge_score,
            }

    if "weight_rank" in results:
        for r in results["weight_rank"]:
            layer_key = str(r.layer_idx)
            if layer_key not in summary["per_layer"]:
                summary["per_layer"][layer_key] = {}
            summary["per_layer"][layer_key]["study18_weight_rank"] = {
                "avg_ratio95": r.avg_ratio95,
                "avg_ratio99": r.avg_ratio99,
                "estimated_param_saving_95": r.estimated_param_saving_95,
                "gate_proj_ratio95": r.gate_proj_ratio95,
                "up_proj_ratio95": r.up_proj_ratio95,
                "down_proj_ratio95": r.down_proj_ratio95,
            }

    if "head_clustering" in results:
        report = results["head_clustering"]
        summary["aggregated"]["study19"] = {
            "num_heads_total": report.num_heads_total,
            "num_clusters": report.num_clusters,
            "num_redundant_heads": report.num_redundant_heads,
            "prunable_heads": [(l, h) for l, h in report.prunable_heads],
        }
        # Build per-layer prunable head sets
        prunable_by_layer = {}
        for l, h in report.prunable_heads:
            prunable_by_layer.setdefault(str(l), []).append(h)
        for layer_key, heads in prunable_by_layer.items():
            if layer_key not in summary["per_layer"]:
                summary["per_layer"][layer_key] = {}
            summary["per_layer"][layer_key]["study19_prunable_heads"] = heads

    if "static_dynamic" in results:
        for r in results["static_dynamic"]:
            layer_key = str(r.layer_idx)
            if layer_key not in summary["per_layer"]:
                summary["per_layer"][layer_key] = {}
            summary["per_layer"][layer_key]["study20_static_dynamic"] = {
                "static_fraction": r.static_fraction,
                "foldable_neuron_count": r.foldable_neuron_count,
                "foldable_param_saving": r.foldable_param_saving,
                "n_mostly_static": r.n_mostly_static,
                "n_mostly_dynamic": r.n_mostly_dynamic,
            }
            # If the report has foldable indices, save them
            if hasattr(r, 'foldable_neuron_indices') and r.foldable_neuron_indices is not None:
                tensor_path = os.path.join(tensor_dir, f"foldable_indices_layer_{r.layer_idx}.pt")
                torch.save(torch.tensor(r.foldable_neuron_indices), tensor_path)
                summary["per_layer"][layer_key]["study20_static_dynamic"]["foldable_indices_path"] = (
                    f"summary_tensors/foldable_indices_layer_{r.layer_idx}.pt"
                )

    if "magnitude_divergence" in results:
        for r in results["magnitude_divergence"]:
            layer_key = str(r.layer_idx)
            if layer_key not in summary["per_layer"]:
                summary["per_layer"][layer_key] = {}
            summary["per_layer"][layer_key]["study21_magnitude_divergence"] = {
                "n_domain_sensitive_neurons": r.n_domain_sensitive_neurons,
                "n_domain_invariant_neurons": r.n_domain_invariant_neurons,
                "domain_sensitivity_score": r.domain_sensitivity_score,
                "magnitude_cv": r.magnitude_cv,
            }

    # ---- Study 22: Domain-Conditional Wanda Importance ----
    if "domain_conditional_importance" in results:
        dci = results["domain_conditional_importance"]
        reports = dci.get("domain_wanda_reports", [])
        domains = dci.get("domains", [])

        for r in reports:
            layer_key = str(r.layer_idx)
            if layer_key not in summary["per_layer"]:
                summary["per_layer"][layer_key] = {}

            # Save per-domain wanda tensors to .pt files
            domain_tensor_paths = {}
            for d in r.domains:
                if d in r.domain_wanda_scores:
                    tensor_path = os.path.join(
                        tensor_dir,
                        f"domain_wanda_{d}_layer_{r.layer_idx}.pt"
                    )
                    torch.save(r.domain_wanda_scores[d].cpu(), tensor_path)
                    domain_tensor_paths[d] = (
                        f"summary_tensors/domain_wanda_{d}_layer_{r.layer_idx}.pt"
                    )

            # Save global mean wanda
            global_path = os.path.join(
                tensor_dir, f"domain_wanda_global_layer_{r.layer_idx}.pt"
            )
            torch.save(r.global_mean_wanda.cpu(), global_path)

            summary["per_layer"][layer_key]["study22_domain_conditional_importance"] = {
                "domains": r.domains,
                "n_domain_critical": r.n_domain_critical,
                "n_domain_unnecessary": r.n_domain_unnecessary,
                "domain_tensor_paths": domain_tensor_paths,
                "global_mean_wanda_path": (
                    f"summary_tensors/domain_wanda_global_layer_{r.layer_idx}.pt"
                ),
            }

        # Aggregated stats
        if reports:
            summary["aggregated"]["study22"] = {
                "domains": domains,
                "total_critical_per_domain": {
                    d: sum(r.n_domain_critical.get(d, 0) for r in reports)
                    for d in domains
                },
                "total_unnecessary_per_domain": {
                    d: sum(r.n_domain_unnecessary.get(d, 0) for r in reports)
                    for d in domains
                },
            }

    # ---- Build compression hints from study data ----
    for layer_key, layer_data in summary["per_layer"].items():
        hints = {}

        # Study 1: kurtosis/gini -> pruning approach
        s1 = layer_data.get("study1_activation_profile", {})
        kurtosis = s1.get("kurtosis", 0)
        gini = s1.get("gini_coefficient", 0)
        if kurtosis > 10.0:
            hints["pruning_method"] = "structured"
            hints["importance_concentration"] = "high" if gini > 0.8 else "moderate"
        elif kurtosis < 3.0:
            hints["pruning_method"] = "uniform_or_quant"
            hints["importance_concentration"] = "low"
        else:
            hints["pruning_method"] = "structured"
            hints["importance_concentration"] = "moderate"

        # Study 7: importance method recommendation
        s7 = layer_data.get("study7_gate_wanda_correlation", {})
        hints["recommended_importance"] = s7.get("recommendation", "wanda")

        # Study 8: mask and merge decisions
        s8 = layer_data.get("study8_sparsity_structure", {})
        consistency = s8.get("activation_consistency", 1.0)
        clusters = s8.get("n_coactivation_clusters", 0)
        hints["static_mask_viable"] = consistency > 0.7
        hints["merge_candidates_exist"] = clusters > 5

        # Study 11: domain conditional possibility
        s11 = layer_data.get("study11_domain_divergence", {})
        hints["domain_conditional_possible"] = s11.get("domain_specificity_score", 0) > 0.3

        summary["compression_hints"]["per_layer_strategy_hints"][layer_key] = hints

    # ---- Deduplicate protection lists ----
    seen_neurons = set()
    deduped_neurons = []
    for entry in summary["protection_lists"]["never_prune_neurons"]:
        key = (entry["layer"], entry["neuron"])
        if key not in seen_neurons:
            seen_neurons.add(key)
            deduped_neurons.append(entry)
    summary["protection_lists"]["never_prune_neurons"] = deduped_neurons

    seen_heads = set()
    deduped_heads = []
    for entry in summary["protection_lists"]["never_prune_heads"]:
        key = (entry["layer"], entry["head"])
        if key not in seen_heads:
            seen_heads.add(key)
            deduped_heads.append(entry)
    summary["protection_lists"]["never_prune_heads"] = deduped_heads

    # ---- Save ----
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nSummary saved to {summary_path}")
    print(f"  Per-layer data for {len(summary['per_layer'])} layers")
    print(f"  Protection list: {len(summary['protection_lists']['never_prune_neurons'])} neurons, "
          f"{len(summary['protection_lists']['never_prune_heads'])} heads")

    tensor_files = [f for f in os.listdir(tensor_dir) if f.endswith('.pt')] if os.path.exists(tensor_dir) else []
    if tensor_files:
        print(f"  Tensor files: {len(tensor_files)} saved to {tensor_dir}")

    return summary


def load_summary(output_dir: str) -> dict:
    """Load a previously saved summary."""
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "r") as f:
        return json.load(f)

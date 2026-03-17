#!/usr/bin/env python3
"""
Sparsity MRI v3: Full Diagnostic Suite (Studies 1-21)
=====================================================

Usage:
    # Run new studies only (17-21)
    python main_novel.py --model Qwen/Qwen2.5-3B --studies 17,18,19,20,21

    # Run all novel studies (12-21)
    python main_novel.py --model Qwen/Qwen2.5-3B --studies 12,13,14,15,16,17,18,19,20,21

    # Quick mode
    python main_novel.py --model Qwen/Qwen2.5-3B --studies 17,18,20 --quick

    # Full suite
    python main_novel.py --model Qwen/Qwen2.5-3B --all

Studies:
    Core (1-11):
      1  Activation Profiling              2  Gate Training (CATS-style)
      3  Wanda Importance Scores           4  Massive Activation Scan
      5  Dead/Dormant Neuron Analysis      6  Attention Head Importance
      7  Gate-Wanda Correlation            8  Sparsity Structure Analysis
      9  Critical Neuron Search           10  Layer Redundancy
     11  Domain-Specific Activation Divergence

    Cross-Disciplinary (12-16):
     12  Feed-Forward Loop Motifs          13  Information Bottleneck Profile
     14  Functional Redundancy Census      15  Perturbation Cascade Depth
     16  Activation Phase Transition

    Next-Gen Compression Diagnostics (17-21):
     17  Cross-Layer Subspace Alignment    18  Effective Weight Matrix Rank
     19  Attention Head Clustering         20  Static/Dynamic Decomposition
     21  Magnitude Divergence Across Domains
"""

import argparse
import json
import time
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from model_utils import ModelInspector
from data_utils import load_wikitext_data, get_dataloader, evaluate_perplexity


def run_experiment(
    model_name: str,
    output_dir: str,
    device: str = "cuda",
    dtype: str = "auto",
    batch_size: int = 4,
    max_batches: int = 16,
    num_samples: int = 256,
    gate_steps: int = 500,
    sparsities: list = None,
    selected_studies: set = None,
):
    if sparsities is None:
        sparsities = [0.25, 0.50, 0.75]
    if selected_studies is None:
        selected_studies = {17, 18, 19, 20, 21}

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  SPARSITY MRI v3 — Full Diagnostic Suite")
    print(f"  Model: {model_name}")
    print(f"  Studies: {sorted(selected_studies)}")
    print(f"  Output: {output_dir}")
    print(f"  Dtype: {dtype} | Batch: {batch_size} | Max batches: {max_batches}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    inspector = ModelInspector(model_name, device, dtype=dtype)

    dataset = load_wikitext_data(
        inspector.tokenizer, split="validation",
        max_seq_len=512, num_samples=num_samples,
    )

    eval_loader = get_dataloader(dataset, batch_size=batch_size)
    baseline_ppl = evaluate_perplexity(
        inspector.model, eval_loader, device, max_batches=max_batches,
    )
    print(f"\nBaseline Perplexity: {baseline_ppl:.2f}")

    results = {
        "baseline_ppl": baseline_ppl,
        "model_name": model_name,
        "dtype": str(inspector.dtype),
        "num_layers": inspector.num_layers,
        "intermediate_size": inspector.mlp_layers[0].intermediate_size,
        "is_gated": inspector.mlp_layers[0].is_gated,
        "activation_fn": inspector.mlp_layers[0].activation_fn,
    }

    # ================================================================
    # Studies 1-11 (core) — unchanged
    # ================================================================
    if 1 in selected_studies:
        from studies_core import run_activation_profiling
        t0 = time.time()
        results["activation_profiles"] = run_activation_profiling(
            inspector, dataset, batch_size=batch_size, max_batches=max_batches)
        print(f"  Study 1 completed in {time.time()-t0:.1f}s\n")

    if 2 in selected_studies:
        from studies_core import run_gate_training
        t0 = time.time()
        results["gate_training"] = run_gate_training(
            inspector, dataset, target_sparsities=sparsities,
            batch_size=max(1, batch_size // 2), num_steps=gate_steps)
        print(f"  Study 2 completed in {time.time()-t0:.1f}s\n")

    if 3 in selected_studies:
        from studies_core import compute_wanda_scores
        t0 = time.time()
        results["wanda_scores"] = compute_wanda_scores(
            inspector, dataset, batch_size=batch_size, max_batches=max_batches)
        print(f"  Study 3 completed in {time.time()-t0:.1f}s\n")

    if 4 in selected_studies:
        from studies_core import run_massive_activation_scan
        t0 = time.time()
        results["massive_activations"] = run_massive_activation_scan(
            inspector, dataset, batch_size=batch_size, max_batches=max_batches)
        print(f"  Study 4 completed in {time.time()-t0:.1f}s\n")

    if 5 in selected_studies:
        from studies_core import run_dead_neuron_analysis
        t0 = time.time()
        results["dead_neurons"] = run_dead_neuron_analysis(
            inspector, dataset, batch_size=batch_size, max_batches=max_batches)
        print(f"  Study 5 completed in {time.time()-t0:.1f}s\n")

    if 6 in selected_studies:
        from studies_extended import run_attention_head_importance
        t0 = time.time()
        results["attention_heads"] = run_attention_head_importance(
            inspector, dataset, batch_size=batch_size, max_batches=max_batches)
        print(f"  Study 6 completed in {time.time()-t0:.1f}s\n")

    if 7 in selected_studies:
        if "gate_training" in results and "wanda_scores" in results:
            from studies_extended import run_gate_wanda_correlation
            t0 = time.time()
            mid_sp = sorted(results["gate_training"].keys())[len(results["gate_training"])//2]
            gate_patterns = results["gate_training"][mid_sp][0]
            results["gate_wanda_correlation"] = run_gate_wanda_correlation(
                gate_patterns, results["wanda_scores"], inspector.num_layers)
            print(f"  Study 7 completed in {time.time()-t0:.1f}s\n")
        else:
            print("\n  Study 7 SKIPPED: requires Studies 2 and 3\n")

    if 8 in selected_studies:
        from studies_core import run_sparsity_structure_analysis
        t0 = time.time()
        results["sparsity_structure"] = run_sparsity_structure_analysis(
            inspector, dataset, batch_size=batch_size, max_batches=max_batches)
        print(f"  Study 8 completed in {time.time()-t0:.1f}s\n")

    if 9 in selected_studies:
        from studies_structural import run_critical_neuron_search
        t0 = time.time()
        results["critical_neurons"] = run_critical_neuron_search(
            inspector, dataset, batch_size=batch_size, top_k_per_layer=3)
        print(f"  Study 9 completed in {time.time()-t0:.1f}s\n")

    if 10 in selected_studies:
        from studies_structural import run_layer_redundancy
        t0 = time.time()
        results["layer_redundancy"] = run_layer_redundancy(
            inspector, dataset, batch_size=batch_size)
        print(f"  Study 10 completed in {time.time()-t0:.1f}s\n")

    if 11 in selected_studies:
        from study_domains import run_domain_divergence_study
        t0 = time.time()
        results["domain_divergence"] = run_domain_divergence_study(
            inspector, batch_size=batch_size, max_batches=max_batches,
            samples_per_domain=min(64, num_samples // 4))
        print(f"  Study 11 completed in {time.time()-t0:.1f}s\n")

    # ================================================================
    # Studies 12-16 (cross-disciplinary) — unchanged
    # ================================================================
    if 12 in selected_studies:
        from studies_novel_part1 import run_cross_layer_motif_analysis
        t0 = time.time()
        results["cross_layer_motifs"] = run_cross_layer_motif_analysis(
            inspector, dataset, batch_size=batch_size, max_batches=max_batches, top_k_neurons=200)
        print(f"  Study 12 completed in {time.time()-t0:.1f}s\n")

    if 13 in selected_studies:
        from studies_novel_part1 import run_information_bottleneck_profile
        t0 = time.time()
        results["information_bottleneck"] = run_information_bottleneck_profile(
            inspector, dataset, batch_size=batch_size, max_eval_batches=min(8, max_batches))
        print(f"  Study 13 completed in {time.time()-t0:.1f}s\n")

    if 14 in selected_studies:
        from studies_novel_part2 import run_functional_redundancy_census
        t0 = time.time()
        results["functional_redundancy"] = run_functional_redundancy_census(
            inspector, dataset, batch_size=batch_size, max_batches=max_batches, sample_neurons=1024)
        print(f"  Study 14 completed in {time.time()-t0:.1f}s\n")

    if 15 in selected_studies:
        from studies_novel_part2 import run_perturbation_cascade_analysis
        t0 = time.time()
        results["perturbation_cascade"] = run_perturbation_cascade_analysis(
            inspector, dataset, batch_size=max(1, batch_size // 2),
            max_eval_batches=4, neurons_per_layer=3, sample_layers=8)
        print(f"  Study 15 completed in {time.time()-t0:.1f}s\n")

    if 16 in selected_studies:
        from studies_novel_part2 import run_phase_transition_analysis
        t0 = time.time()
        results["phase_transition"] = run_phase_transition_analysis(
            inspector, dataset, batch_size=batch_size, max_batches=max_batches)
        print(f"  Study 16 completed in {time.time()-t0:.1f}s\n")

    # ================================================================
    # Studies 17-21 (next-gen compression diagnostics) — NEW
    # ================================================================
    if 17 in selected_studies:
        from studies_new import run_cross_layer_alignment
        t0 = time.time()
        results["cross_layer_alignment"] = run_cross_layer_alignment(
            inspector, dataset,
            batch_size=max(1, batch_size // 2),
            max_batches=min(8, max_batches),
            top_k_subspace=64)
        print(f"  Study 17 completed in {time.time()-t0:.1f}s\n")

    if 18 in selected_studies:
        from studies_new import run_weight_rank_analysis
        t0 = time.time()
        results["weight_rank"] = run_weight_rank_analysis(
            inspector, dataset,
            batch_size=batch_size, max_batches=max_batches)
        print(f"  Study 18 completed in {time.time()-t0:.1f}s\n")

    if 19 in selected_studies:
        from studies_new import run_attention_head_clustering
        t0 = time.time()
        results["attention_head_clusters"] = run_attention_head_clustering(
            inspector, dataset,
            batch_size=max(1, batch_size // 2),
            max_batches=min(4, max_batches),
            similarity_threshold=0.90)
        print(f"  Study 19 completed in {time.time()-t0:.1f}s\n")

    if 20 in selected_studies:
        from studies_new import run_static_dynamic_decomposition
        t0 = time.time()
        results["static_dynamic"] = run_static_dynamic_decomposition(
            inspector, dataset,
            batch_size=batch_size, max_batches=max_batches,
            static_threshold=0.90, foldable_threshold=0.95)
        print(f"  Study 20 completed in {time.time()-t0:.1f}s\n")

    if 21 in selected_studies:
        from studies_new import run_magnitude_divergence
        t0 = time.time()
        results["magnitude_divergence"] = run_magnitude_divergence(
            inspector, dataset,
            batch_size=max(1, batch_size // 2),
            max_batches=min(8, max_batches),
            samples_per_domain=32)
        print(f"  Study 21 completed in {time.time()-t0:.1f}s\n")

    # ================================================================
    # Visualizations
    # ================================================================
    print("\n" + "=" * 80)
    print("  GENERATING VISUALIZATIONS")
    print("=" * 80)

    # Studies 1-11
    if any(s in selected_studies for s in range(1, 12)):
        try:
            from visualize import generate_all_plots
            generate_all_plots(results, model_name, output_dir)
        except Exception as e:
            print(f"  Original visualizations skipped: {e}")

    # Studies 12-16
    try:
        from visualize_novel import (
            plot_study12_motifs, plot_study13_bottleneck,
            plot_study14_redundancy, plot_study15_cascade, plot_study16_phase,
        )
        if "cross_layer_motifs" in results and results["cross_layer_motifs"]:
            plot_study12_motifs(results["cross_layer_motifs"], model_name, output_dir)
        if "information_bottleneck" in results and results["information_bottleneck"]:
            plot_study13_bottleneck(results["information_bottleneck"], model_name, output_dir)
        if "functional_redundancy" in results and results["functional_redundancy"]:
            plot_study14_redundancy(results["functional_redundancy"], model_name, output_dir)
        if "perturbation_cascade" in results and results["perturbation_cascade"]:
            plot_study15_cascade(results["perturbation_cascade"], model_name, output_dir)
        if "phase_transition" in results and results["phase_transition"]:
            plot_study16_phase(results["phase_transition"], model_name, output_dir)
    except Exception as e:
        print(f"  Novel (12-16) visualizations skipped: {e}")

    # Studies 17-21
    try:
        from visualize_new import (
            plot_study17_alignment, plot_study18_rank,
            plot_study19_head_clusters, plot_study20_static_dynamic,
            plot_study21_magnitude_divergence,
        )
        if "cross_layer_alignment" in results and results["cross_layer_alignment"]:
            plot_study17_alignment(results["cross_layer_alignment"], model_name, output_dir)
        if "weight_rank" in results and results["weight_rank"]:
            plot_study18_rank(results["weight_rank"], model_name, output_dir)
        if "attention_head_clusters" in results:
            r = results["attention_head_clusters"]
            if hasattr(r, 'num_heads_total') and r.num_heads_total > 0:
                plot_study19_head_clusters(r, model_name, output_dir)
        if "static_dynamic" in results and results["static_dynamic"]:
            plot_study20_static_dynamic(results["static_dynamic"], model_name, output_dir)
        if "magnitude_divergence" in results and results["magnitude_divergence"]:
            plot_study21_magnitude_divergence(results["magnitude_divergence"], model_name, output_dir)
    except Exception as e:
        print(f"  New (17-21) visualizations error: {e}")
        import traceback; traceback.print_exc()

    # ================================================================
    # Save summary JSON
    # ================================================================
    save_summary(results, model_name, output_dir)
    return results


def save_summary(results: dict, model_name: str, output_dir: str):
    """Save machine-readable summary of all findings."""
    summary = {
        "model": model_name,
        "baseline_ppl": results["baseline_ppl"],
        "num_layers": results.get("num_layers"),
        "intermediate_size": results.get("intermediate_size"),
        "is_gated": results.get("is_gated"),
        "activation_fn": results.get("activation_fn"),
        "dtype": results.get("dtype"),
        "timestamp": datetime.now().isoformat(),
        "findings": {},
    }

    # ---- Studies 1-16 (same as v2) ----
    if "activation_profiles" in results:
        profiles = results["activation_profiles"]
        summary["findings"]["study01_activation_profiling"] = {
            "avg_natural_sparsity": float(np.mean([p.pct_near_zero for p in profiles])),
            "avg_kurtosis": float(np.mean([p.kurtosis for p in profiles])),
            "max_kurtosis_layer": int(max(profiles, key=lambda p: p.kurtosis).layer_idx),
            "avg_gini": float(np.mean([p.gini_coefficient for p in profiles])),
        }

    if "massive_activations" in results:
        reports = results["massive_activations"]
        all_ratios = [r for rep in reports for r in rep.massive_neuron_ratios]
        summary["findings"]["study04_massive_activations"] = {
            "total_massive_neurons": int(sum(len(r.massive_neuron_indices) for r in reports)),
            "total_input_agnostic": int(sum(len(r.input_agnostic_indices) for r in reports)),
            "max_ratio": float(max(all_ratios)) if all_ratios else 0.0,
        }

    if "dead_neurons" in results:
        reports = results["dead_neurons"]
        summary["findings"]["study05_dead_neurons"] = {
            "total_dead": int(sum(r.dead_count for r in reports)),
            "total_dormant": int(sum(r.dormant_count for r in reports)),
            "total_hyperactive": int(sum(r.hyperactive_count for r in reports)),
        }

    if "cross_layer_motifs" in results:
        reports = results["cross_layer_motifs"]
        summary["findings"]["study12_cross_layer_motifs"] = {
            "total_strong_connections": int(sum(r.n_strong_connections for r in reports)),
            "avg_relay_density": float(np.mean([r.relay_density for r in reports])),
            "per_layer": [
                {"source": r.source_layer, "target": r.target_layer,
                 "strong": r.n_strong_connections, "density": round(r.relay_density, 5)}
                for r in reports
            ],
        }

    if "information_bottleneck" in results:
        reports = results["information_bottleneck"]
        summary["findings"]["study13_information_bottleneck"] = {
            "compression_profile": [
                {"layer": r.layer_idx, "compression_ratio": round(r.compression_ratio, 4),
                 "partial_ppl": round(r.partial_lm_loss, 2) if r.partial_lm_loss < 1e6 else "inf",
                 "recon_mse": round(r.reconstruction_mse, 4)}
                for r in reports
            ],
        }

    if "functional_redundancy" in results:
        reports = results["functional_redundancy"]
        summary["findings"]["study14_functional_redundancy"] = {
            "avg_mean_max_similarity": float(np.mean([r.mean_max_similarity for r in reports])),
            "total_safe_to_prune": int(sum(r.safe_to_prune_count for r in reports)),
            "per_layer": [
                {"layer": r.layer_idx, "mean_max_sim": round(r.mean_max_similarity, 3),
                 "redundant": r.n_highly_redundant, "keystone": r.n_keystone,
                 "safe_to_prune": r.safe_to_prune_count}
                for r in reports
            ],
        }

    if "perturbation_cascade" in results:
        reports = results["perturbation_cascade"]
        by_layer = defaultdict(list)
        for r in reports:
            by_layer[r.layer_idx].append(r)
        summary["findings"]["study15_perturbation_cascade"] = {
            "per_source_layer": {
                int(l): {
                    "avg_cascade_depth": round(float(np.mean([r.cascade_depth for r in rs])), 1),
                    "avg_max_amplification": round(float(np.mean([r.max_amplification for r in rs])), 2),
                    "avg_decay_rate": round(float(np.mean([r.decay_rate for r in rs])), 4),
                }
                for l, rs in sorted(by_layer.items())
            },
        }

    if "phase_transition" in results:
        reports = results["phase_transition"]
        alphas = [r.power_law_alpha for r in reports if r.power_law_alpha > 0]
        heavy_layers = [r.layer_idx for r in reports if r.is_heavy_tailed]
        summary["findings"]["study16_phase_transition"] = {
            "alpha_range": [round(min(alphas), 2), round(max(alphas), 2)] if alphas else [0, 0],
            "heavy_tailed_layers": heavy_layers,
            "n_heavy_tailed": len(heavy_layers),
            "per_layer": [
                {"layer": r.layer_idx, "alpha": round(r.power_law_alpha, 2),
                 "tail_fraction": round(r.tail_fraction, 4),
                 "ks_stat": round(r.gaussian_fit_ks, 3),
                 "entropy": round(r.activation_entropy, 2),
                 "heavy_tailed": r.is_heavy_tailed}
                for r in reports
            ],
        }

    if "domain_divergence" in results:
        dd = results["domain_divergence"]
        overview = dd["overview_reports"]
        most_spec = max(overview, key=lambda r: r.domain_specificity_score)
        summary["findings"]["study11_domain_divergence"] = {
            "domains": dd["domains"],
            "most_specialized_layer": int(most_spec.layer_idx),
            "most_specialized_score": float(most_spec.domain_specificity_score),
        }

    # ---- Studies 17-21 (NEW) ----

    if "cross_layer_alignment" in results:
        reports = results["cross_layer_alignment"]
        if reports:
            top_merge = sorted(reports, key=lambda r: -r.merge_score)[:10]
            summary["findings"]["study17_cross_layer_alignment"] = {
                "avg_cka": round(float(np.mean([r.cka_linear for r in reports])), 4),
                "avg_merge_score": round(float(np.mean([r.merge_score for r in reports])), 4),
                "max_merge_score": round(float(max(r.merge_score for r in reports)), 4),
                "n_high_similarity_pairs": int(sum(1 for r in reports if r.cka_linear > 0.8)),
                "top_merge_candidates": [
                    {"layer_a": r.layer_a, "layer_b": r.layer_b,
                     "cka": round(r.cka_linear, 4),
                     "subspace_overlap": round(r.top_subspace_overlap, 4),
                     "cosine": round(r.residual_delta_cosine, 4),
                     "merge_score": round(r.merge_score, 4)}
                    for r in top_merge
                ],
                "per_pair": [
                    {"layer_a": r.layer_a, "layer_b": r.layer_b,
                     "cka": round(r.cka_linear, 4),
                     "subspace_overlap": round(r.top_subspace_overlap, 4),
                     "cosine": round(r.residual_delta_cosine, 4),
                     "merge_score": round(r.merge_score, 4)}
                    for r in reports
                ],
            }

    if "weight_rank" in results:
        reports = results["weight_rank"]
        if reports:
            avg_saving = float(np.mean([r.estimated_param_saving_95 for r in reports]))
            total_saving = sum(r.estimated_param_saving_95 for r in reports) / len(reports)
            summary["findings"]["study18_weight_rank"] = {
                "avg_rank_ratio_95": round(float(np.mean([r.avg_ratio95 for r in reports])), 3),
                "avg_rank_ratio_99": round(float(np.mean([r.avg_ratio99 for r in reports])), 3),
                "avg_estimated_saving_95": round(avg_saving, 3),
                "layers_with_high_saving": [
                    r.layer_idx for r in reports if r.estimated_param_saving_95 > 0.3
                ],
                "per_layer": [
                    {"layer": r.layer_idx,
                     "gate_ratio95": r.gate_proj_ratio95,
                     "up_ratio95": r.up_proj_ratio95,
                     "down_ratio95": r.down_proj_ratio95,
                     "avg_ratio95": r.avg_ratio95,
                     "avg_ratio99": r.avg_ratio99,
                     "estimated_saving_95": r.estimated_param_saving_95}
                    for r in reports
                ],
            }

    if "attention_head_clusters" in results:
        r = results["attention_head_clusters"]
        if hasattr(r, 'num_heads_total'):
            summary["findings"]["study19_attention_head_clusters"] = {
                "num_heads_analyzed": r.num_heads_total,
                "num_clusters": r.num_clusters,
                "num_singleton_heads": r.num_singleton_heads,
                "num_redundant_heads": r.num_redundant_heads,
                "num_prunable_heads": len(r.prunable_heads),
                "cluster_size_distribution": r.cluster_sizes[:20],
                "prunable_heads": [
                    {"layer": l, "head": h} for l, h in r.prunable_heads[:50]
                ],
            }

    if "static_dynamic" in results:
        reports = results["static_dynamic"]
        if reports:
            total_foldable = sum(r.foldable_neuron_count for r in reports)
            summary["findings"]["study20_static_dynamic"] = {
                "avg_static_fraction": round(float(np.mean([r.static_fraction for r in reports])), 4),
                "total_foldable_neurons": total_foldable,
                "total_mostly_static": int(sum(r.n_mostly_static for r in reports)),
                "total_mostly_dynamic": int(sum(r.n_mostly_dynamic for r in reports)),
                "per_layer": [
                    {"layer": r.layer_idx,
                     "static_fraction": round(r.static_fraction, 4),
                     "n_mostly_static": r.n_mostly_static,
                     "n_mostly_dynamic": r.n_mostly_dynamic,
                     "foldable": r.foldable_neuron_count,
                     "foldable_saving": round(r.foldable_param_saving, 4)}
                    for r in reports
                ],
            }

    if "magnitude_divergence" in results:
        reports = results["magnitude_divergence"]
        if reports:
            total_sensitive = sum(r.n_domain_sensitive_neurons for r in reports)
            summary["findings"]["study21_magnitude_divergence"] = {
                "domains": reports[0].domains,
                "avg_sensitivity_score": round(float(np.mean([r.domain_sensitivity_score for r in reports])), 4),
                "total_domain_sensitive_neurons": total_sensitive,
                "avg_magnitude_cv": round(float(np.mean([r.magnitude_cv for r in reports])), 4),
                "per_layer": [
                    {"layer": r.layer_idx,
                     "n_sensitive": r.n_domain_sensitive_neurons,
                     "n_invariant": r.n_domain_invariant_neurons,
                     "magnitude_cv": round(r.magnitude_cv, 4),
                     "sensitivity_score": round(r.domain_sensitivity_score, 4),
                     "domain_magnitudes": {k: round(v, 4) for k, v in r.domain_mean_magnitudes.items()}}
                    for r in reports
                ],
            }

    # Save
    path = Path(output_dir) / "summary.json"
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Sparsity MRI v3: Full Diagnostic Suite (Studies 1-21)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_novel.py --model Qwen/Qwen2.5-3B --studies 17,18,19,20,21
  python main_novel.py --model Qwen/Qwen2.5-3B --studies 17,18,20 --quick
  python main_novel.py --model Qwen/Qwen2.5-3B --all
        """,
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="auto",
                        choices=["auto", "float32", "bfloat16"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=256)
    parser.add_argument("--gate_steps", type=int, default=500)
    parser.add_argument("--studies", type=str, default=None,
                        help="Comma-separated study numbers (e.g., '17,18,20')")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--all", action="store_true", help="Run all studies 1-21")
    parser.add_argument("--sparsities", type=str, default="0.25,0.50,0.75")

    args = parser.parse_args()

    if args.studies:
        selected_studies = set(int(s.strip()) for s in args.studies.split(','))
    elif args.all:
        selected_studies = set(range(1, 22))
    else:
        selected_studies = {17, 18, 19, 20, 21}

    if args.quick:
        args.num_samples = 64
        args.max_batches = 8
        args.gate_steps = 200
        args.batch_size = min(args.batch_size, 2)

    sparsities = [float(s) for s in args.sparsities.split(',')]

    model_lower = args.model.lower()
    if any(tag in model_lower for tag in ['3b', '7b', '8b', '13b', '70b']):
        if args.dtype == "auto":
            print(f"[INFO] Large model detected, using bfloat16")
        if args.batch_size > 2 and not args.quick:
            print(f"[INFO] Large model: reducing batch_size {args.batch_size} -> 2")
            args.batch_size = 2

    results = run_experiment(
        model_name=args.model,
        output_dir=args.output_dir,
        device=args.device,
        dtype=args.dtype,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        num_samples=args.num_samples,
        gate_steps=args.gate_steps,
        sparsities=sparsities,
        selected_studies=selected_studies,
    )

    print("\n" + "=" * 80)
    print("  EXPERIMENT COMPLETE")
    print(f"  Results: {args.output_dir}/")
    print(f"  Studies run: {sorted(selected_studies)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
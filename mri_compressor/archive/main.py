#!/usr/bin/env python3
"""
Sparsity MRI: Comprehensive analysis of LLM internal structure during activation sparsification.

Usage:
    python main.py --model gpt2 --quick                 # Fast: ~10 min
    python main.py --model gpt2                          # Full: ~30-60 min
    python main.py --model Qwen/Qwen2.5-0.5B            # Modern gated MLP
    python main.py --model gpt2 --studies 1,2,3,7        # Selective
"""

import argparse
import json
import time
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

from config import ExperimentConfig
from model_utils import ModelInspector
from data_utils import load_wikitext_data, get_dataloader, evaluate_perplexity


def run_experiment(config: ExperimentConfig, selected_studies: set = None):
    all_studies = set(range(1, 12))
    if selected_studies is None:
        selected_studies = all_studies
    
    print("=" * 80)
    print(f"  SPARSITY MRI — Comprehensive LLM Internal Analysis")
    print(f"  Model: {config.model_name}")
    print(f"  Studies: {sorted(selected_studies)}")
    print(f"  Output: {config.output_dir}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    inspector = ModelInspector(config.model_name, config.device)
    
    dataset = load_wikitext_data(
        inspector.tokenizer, split="validation",
        max_seq_len=config.max_seq_len,
        num_samples=config.num_calibration_samples,
        dataset_name=config.dataset, dataset_config=config.dataset_config,
    )
    
    eval_loader = get_dataloader(dataset, batch_size=config.batch_size)
    baseline_ppl = evaluate_perplexity(
        inspector.model, eval_loader, config.device,
        max_batches=config.num_eval_samples // config.batch_size
    )
    print(f"\nBaseline Perplexity: {baseline_ppl:.2f}")
    
    results = {"baseline_ppl": baseline_ppl, "model_name": config.model_name}
    
    # Study 1: Activation Profiling
    if 1 in selected_studies:
        from studies_core import run_activation_profiling
        t0 = time.time()
        results["activation_profiles"] = run_activation_profiling(
            inspector, dataset, config.batch_size)
        print(f"  Study 1 completed in {time.time()-t0:.1f}s")
    
    # Study 2: Gate Training
    if 2 in selected_studies:
        from studies_core import run_gate_training
        t0 = time.time()
        results["gate_training"] = run_gate_training(
            inspector, dataset,
            target_sparsities=config.target_sparsities,
            lr=config.gate_lr, num_steps=config.gate_training_steps,
            batch_size=config.batch_size,
            sparsity_weight=config.sparsity_loss_weight,
            warmup_steps=config.gate_warmup_steps,
        )
        print(f"  Study 2 completed in {time.time()-t0:.1f}s")
    
    # Study 3: Wanda Scores
    if 3 in selected_studies:
        from studies_core import compute_wanda_scores
        t0 = time.time()
        results["wanda_scores"] = compute_wanda_scores(
            inspector, dataset, config.batch_size)
        print(f"  Study 3 completed in {time.time()-t0:.1f}s")
    
    # Study 4: Massive Activation Scan
    if 4 in selected_studies:
        from studies_extended import run_massive_activation_scan
        t0 = time.time()
        results["massive_activations"] = run_massive_activation_scan(
            inspector, dataset, config.batch_size)
        print(f"  Study 4 completed in {time.time()-t0:.1f}s")
    
    # Study 5: Dead Neuron Analysis
    if 5 in selected_studies:
        from studies_extended import run_dead_neuron_analysis
        t0 = time.time()
        results["dead_neurons"] = run_dead_neuron_analysis(
            inspector, dataset, config.batch_size)
        print(f"  Study 5 completed in {time.time()-t0:.1f}s")
    
    # Study 6: Attention Head Importance
    if 6 in selected_studies:
        from studies_extended import run_attention_head_importance
        t0 = time.time()
        results["attention_heads"] = run_attention_head_importance(
            inspector, dataset, config.batch_size)
        print(f"  Study 6 completed in {time.time()-t0:.1f}s")
    
    # Study 7: Gate-Wanda Correlation (requires studies 2 and 3)
    if 7 in selected_studies:
        if "gate_training" in results and "wanda_scores" in results:
            from studies_extended import run_gate_wanda_correlation
            t0 = time.time()
            mid_sp = sorted(results["gate_training"].keys())[len(results["gate_training"])//2]
            gate_patterns = results["gate_training"][mid_sp][0]
            results["gate_wanda_correlation"] = run_gate_wanda_correlation(
                gate_patterns, results["wanda_scores"], inspector.num_layers)
            print(f"  Study 7 completed in {time.time()-t0:.1f}s")
        else:
            print("\n  Study 7 SKIPPED: requires Studies 2 and 3")
    
    # Study 8: Sparsity Structure
    if 8 in selected_studies:
        from studies_structural import run_sparsity_structure_analysis
        t0 = time.time()
        results["sparsity_structure"] = run_sparsity_structure_analysis(
            inspector, dataset, config.batch_size)
        print(f"  Study 8 completed in {time.time()-t0:.1f}s")
    
    # Study 9: Critical Neuron Search
    if 9 in selected_studies:
        from studies_structural import run_critical_neuron_search
        t0 = time.time()
        results["critical_neurons"] = run_critical_neuron_search(
            inspector, dataset, config.batch_size, top_k_per_layer=3)
        print(f"  Study 9 completed in {time.time()-t0:.1f}s")
    
    # Study 10: Layer Redundancy
    if 10 in selected_studies:
        from studies_structural import run_layer_redundancy
        t0 = time.time()
        results["layer_redundancy"] = run_layer_redundancy(
            inspector, dataset, config.batch_size)
        print(f"  Study 10 completed in {time.time()-t0:.1f}s")
    
    # Study 11: Domain-Specific Activation Divergence
    if 11 in selected_studies:
        from study_domains import run_domain_divergence_study
        t0 = time.time()
        results["domain_divergence"] = run_domain_divergence_study(
            inspector, batch_size=config.batch_size,
            samples_per_domain=min(64, config.num_calibration_samples // 4),
        )
        print(f"  Study 11 completed in {time.time()-t0:.1f}s")
    
    # Generate Visualizations
    from visualize import generate_all_plots
    generate_all_plots(results, config.model_name, config.output_dir)
    
    # Save Summary
    save_summary(results, config)
    
    return results


def save_summary(results: dict, config: ExperimentConfig):
    summary = {
        "model": config.model_name,
        "baseline_ppl": results["baseline_ppl"],
        "timestamp": datetime.now().isoformat(),
        "architecture": {
            "is_gated": None,
            "activation_fn": None,
        },
        "findings": {},
    }
    
    if "activation_profiles" in results:
        profiles = results["activation_profiles"]
        summary["findings"]["study1_activation_profiling"] = {
            "avg_natural_sparsity": float(np.mean([p.pct_near_zero for p in profiles])),
            "avg_kurtosis": float(np.mean([p.kurtosis for p in profiles])),
            "max_kurtosis_layer": int(max(profiles, key=lambda p: p.kurtosis).layer_idx),
            "avg_gini": float(np.mean([p.gini_coefficient for p in profiles])),
            "avg_pct_negative": float(np.mean([p.pct_negative for p in profiles])),
        }
    
    if "dead_neurons" in results:
        reports = results["dead_neurons"]
        total = reports[0].total_neurons
        summary["findings"]["study5_dead_neurons"] = {
            "total_dead": int(sum(r.dead_count for r in reports)),
            "total_dormant": int(sum(r.dormant_count for r in reports)),
            "total_hyperactive": int(sum(r.hyperactive_count for r in reports)),
            "pct_dead_avg": float(np.mean([r.dead_count/total for r in reports])),
        }
    
    if "massive_activations" in results:
        reports = results["massive_activations"]
        all_ratios = [r for rep in reports for r in rep.massive_neuron_ratios]
        summary["findings"]["study4_massive_activations"] = {
            "total_massive_neurons": int(sum(len(r.massive_neuron_indices) for r in reports)),
            "total_input_agnostic": int(sum(len(r.input_agnostic_indices) for r in reports)),
            "max_ratio": float(max(all_ratios)) if all_ratios else 0.0,
        }
    
    if "gate_wanda_correlation" in results:
        reports = results["gate_wanda_correlation"]
        summary["findings"]["study7_gate_wanda_correlation"] = {
            "avg_pearson": float(np.mean([r.pearson_r for r in reports])),
            "avg_spearman": float(np.mean([r.spearman_rho for r in reports])),
            "avg_top10_overlap": float(np.mean([r.top_k_overlap[0.10] for r in reports])),
            "avg_top25_overlap": float(np.mean([r.top_k_overlap[0.25] for r in reports])),
        }
    
    if "critical_neurons" in results:
        reports = results["critical_neurons"]
        top3 = sorted(reports, key=lambda r: r.ppl_increase_single, reverse=True)[:3]
        summary["findings"]["study9_critical_neurons"] = {
            "top_3_most_critical": [
                {"layer": r.layer_idx, "neuron": r.neuron_idx, 
                 "ppl_increase": float(r.ppl_increase_single)}
                for r in top3
            ]
        }
    
    if "layer_redundancy" in results:
        reports = results["layer_redundancy"]
        mlp_deltas = [r.ppl_delta for r in reports if r.component == "mlp"]
        attn_deltas = [r.ppl_delta for r in reports if r.component == "attention"]
        summary["findings"]["study10_layer_redundancy"] = {
            "most_critical_mlp_layer": int(
                max([r for r in reports if r.component == "mlp"], 
                    key=lambda r: r.ppl_delta).layer_idx
            ) if mlp_deltas else None,
            "most_redundant_mlp_layer": int(
                min([r for r in reports if r.component == "mlp"],
                    key=lambda r: r.ppl_delta).layer_idx
            ) if mlp_deltas else None,
        }
    
    if "domain_divergence" in results:
        dd = results["domain_divergence"]
        overview = dd["overview_reports"]
        specificities = [r.domain_specificity_score for r in overview]
        most_spec = max(overview, key=lambda r: r.domain_specificity_score)
        least_spec = min(overview, key=lambda r: r.domain_specificity_score)
        summary["findings"]["study11_domain_divergence"] = {
            "domains": dd["domains"],
            "most_specialized_layer": int(most_spec.layer_idx),
            "most_specialized_score": float(most_spec.domain_specificity_score),
            "least_specialized_layer": int(least_spec.layer_idx),
            "least_specialized_score": float(least_spec.domain_specificity_score),
            "mean_specificity": float(np.mean(specificities)),
            "universal_neurons_total": int(sum(r.n_universal_neurons for r in overview)),
            "per_layer_specificity": {
                int(r.layer_idx): float(r.domain_specificity_score) for r in overview
            },
        }
    
    path = Path(config.output_dir) / "summary.json"
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Sparsity MRI: LLM Internal Analysis")
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--device", type=str, 
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=256)
    parser.add_argument("--gate_steps", type=int, default=500)
    parser.add_argument("--studies", type=str, default=None,
                        help="Comma-separated study numbers (e.g., '1,2,3,7')")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: fewer samples, fewer steps")
    parser.add_argument("--sparsities", type=str, default="0.25,0.50,0.75")
    
    args = parser.parse_args()
    
    selected_studies = None
    if args.studies:
        selected_studies = set(int(s) for s in args.studies.split(','))
    elif args.quick:
        selected_studies = {1, 2, 3, 5, 7, 11}
    
    sparsities = [float(s) for s in args.sparsities.split(',')]
    
    config = ExperimentConfig(
        model_name=args.model,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        num_calibration_samples=args.num_samples if not args.quick else 64,
        num_eval_samples=args.num_samples // 2 if not args.quick else 32,
        gate_training_steps=args.gate_steps if not args.quick else 200,
        target_sparsities=sparsities,
    )
    
    results = run_experiment(config, selected_studies)
    
    print("\n" + "=" * 80)
    print("  EXPERIMENT COMPLETE")
    print(f"  Results saved to: {config.output_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    main()

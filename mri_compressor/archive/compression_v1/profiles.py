#!/usr/bin/env python3
"""
MRI-Compress Targeted Profiles
================================

Creates compression prescriptions targeting specific param reduction percentages,
for direct comparison with uniform Wanda pruning baselines.

The key insight: MRI data tells us WHERE to apply compression, so at the same
overall reduction rate, we should get better PPL than uniform methods.

Usage:
  # 20% reduction (compare with Wanda-25%)
  python profiles.py --model Qwen/Qwen2.5-3B \
    --summary results/summary.json \
    --target-reduction 0.20 \
    --output ./results_mri_20pct

  # 40% reduction (compare with Wanda-50%)
  python profiles.py --model Qwen/Qwen2.5-3B \
    --summary results/summary.json \
    --target-reduction 0.40 \
    --output ./results_mri_40pct
"""

from __future__ import annotations
import argparse
import copy
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from diagnostic import (
    CompressionPrescription,
    CompressionStrategy,
    LayerPrescription,
    MRIDiagnostician,
)
from run import (
    QWEN25_3B_STUDY5,
    QWEN25_3B_STUDY10,
    get_study_data,
    build_calibration_dataloader,
    save_prescription_json,
)


def estimate_param_reduction(
    prescription: CompressionPrescription,
) -> dict:
    """
    Estimate actual parameter reduction from a prescription.
    
    For Qwen2.5-3B with SwiGLU MLP:
    - Each MLP neuron contributes: 3 * hidden_size params (gate_proj, up_proj rows + down_proj col)
    - hidden_size = 2048 for Qwen2.5-3B
    - Each full layer: MLP + Attention + LayerNorms
    
    Returns dict with per-layer and total reduction estimates.
    """
    hidden_size = 2048  # Qwen2.5-3B
    intermediate_size = prescription.intermediate_size
    
    # Per-layer MLP params: gate_proj + up_proj + down_proj
    # gate_proj: [intermediate, hidden]
    # up_proj:   [intermediate, hidden]  
    # down_proj: [hidden, intermediate]
    # Total MLP params per layer: 3 * intermediate * hidden
    mlp_params_per_layer = 3 * intermediate_size * hidden_size
    
    # Attention params per layer (q, k, v, o projections)
    # For Qwen2.5-3B: num_heads=16, head_dim=128, num_kv_heads=2
    # q_proj: [hidden, hidden] = 2048*2048
    # k_proj: [num_kv_heads*head_dim, hidden] = 256*2048
    # v_proj: [num_kv_heads*head_dim, hidden] = 256*2048
    # o_proj: [hidden, hidden] = 2048*2048
    attn_params = 2048*2048 + 256*2048 + 256*2048 + 2048*2048
    
    # LayerNorm params (negligible but count them)
    ln_params = 2 * hidden_size  # input_layernorm + post_attention_layernorm
    
    total_per_layer = mlp_params_per_layer + attn_params + ln_params
    
    # Embedding + final norm (not compressed)
    # vocab_size=151936, hidden=2048 → embed_tokens + lm_head
    fixed_params = 151936 * 2048 * 2  # embed + lm_head (tied or not)
    fixed_params += hidden_size  # final norm
    
    total_original = fixed_params + total_per_layer * prescription.num_layers
    total_compressed = fixed_params
    
    per_layer_info = []
    for lp in prescription.layers:
        if lp.strategy == CompressionStrategy.DEPTH_PRUNE:
            # Entire layer removed
            saved = total_per_layer
            remaining = 0
        else:
            # MLP width reduction
            if lp.strategy in (CompressionStrategy.DEAD_REMOVAL_AND_MERGE, 
                             CompressionStrategy.STRUCTURED_PRUNE):
                if lp.merge_target_width is not None:
                    new_width = lp.merge_target_width
                elif lp.target_sparsity > 0:
                    alive = intermediate_size - lp.dead_neuron_count
                    new_width = int(alive * (1 - lp.target_sparsity))
                else:
                    new_width = intermediate_size - lp.dead_neuron_count
                new_mlp = 3 * new_width * hidden_size
            else:
                new_mlp = mlp_params_per_layer
            
            remaining = new_mlp + attn_params + ln_params
            saved = total_per_layer - remaining
        
        total_compressed += remaining
        per_layer_info.append({
            "layer": lp.layer_idx,
            "strategy": lp.strategy.name,
            "params_saved": saved,
            "params_remaining": remaining,
        })
    
    return {
        "total_original": total_original,
        "total_compressed": total_compressed,
        "reduction_pct": (1 - total_compressed / total_original) * 100,
        "per_layer": per_layer_info,
    }


def build_mri_profile(
    summary_path: str,
    target_reduction: float,
    study5_data=None,
    study10_data=None,
) -> CompressionPrescription:
    """
    Build a compression prescription targeting a specific param reduction.
    
    Strategy: Start from the MRI diagnostic baseline, then progressively add
    more aggressive compression to layers that can tolerate it, guided by:
    - Study 5: dead neurons (free compression)
    - Study 14: redundancy (merge candidates)
    - Study 10: layer criticality (which layers to avoid)
    - Study 15: cascade amplification (compression order preference)
    - Study 16: distribution shape (technique selection)
    
    The key differentiator from uniform pruning: we compress MORE where the
    MRI data says it's safe, and LESS where it says it's dangerous.
    """
    # Start with the standard MRI diagnostic
    diagnostician = MRIDiagnostician()
    base_prescription = diagnostician.diagnose(summary_path, study5_data, study10_data)
    
    # Load the raw data for finer-grained decisions
    with open(summary_path) as f:
        summary = json.load(f)
    
    findings = summary["findings"]
    intermediate_size = summary["intermediate_size"]
    num_layers = summary["num_layers"]
    hidden_size = 2048  # Qwen2.5-3B
    
    # Build per-layer "compression budget" based on MRI safety
    # Lower score = safer to compress more
    layer_safety = []
    
    s10 = study10_data or QWEN25_3B_STUDY10
    s5 = study5_data or QWEN25_3B_STUDY5
    
    s14_data = {item["layer"]: item for item in findings.get("study14_functional_redundancy", {}).get("per_layer", [])}
    s16_data = {item["layer"]: item for item in findings.get("study16_phase_transition", {}).get("per_layer", [])}
    
    for layer_idx in range(num_layers):
        crit = next((d for d in s10 if d["layer"] == layer_idx), {"mlp_ppl_delta": 1.0, "attn_ppl_delta": 1.0})
        dead_info = next((d for d in s5 if d["layer"] == layer_idx), {"dead": 0, "dormant": 0})
        redund = s14_data.get(layer_idx, {"safe_to_prune": 0, "mean_max_sim": 0})
        phase = s16_data.get(layer_idx, {"alpha": 3.0})
        
        # Criticality score: higher = more dangerous to compress
        mlp_crit = crit["mlp_ppl_delta"]
        attn_crit = crit["attn_ppl_delta"]
        
        # "Compressibility" score: higher = safer to compress
        dead_frac = dead_info["dead"] / intermediate_size
        redundancy_frac = redund["safe_to_prune"] / intermediate_size
        
        layer_safety.append({
            "layer": layer_idx,
            "mlp_crit": mlp_crit,
            "attn_crit": attn_crit,
            "total_crit": mlp_crit + attn_crit * 0.1,  # Weight attention less for MLP compression
            "dead_frac": dead_frac,
            "redundancy_frac": redundancy_frac,
            "compressibility": dead_frac + redundancy_frac * 0.5 - mlp_crit * 0.1,
            "dead": dead_info["dead"],
            "dormant": dead_info.get("dormant", 0),
            "safe_to_prune": redund["safe_to_prune"],
            "alpha": phase["alpha"],
        })
    
    # Sort layers by compressibility (most compressible first)
    layers_by_compress = sorted(layer_safety, key=lambda x: -x["compressibility"])
    
    # Build prescription iteratively until we hit target reduction
    prescription = CompressionPrescription(
        model_name=summary["model"],
        baseline_ppl=summary["baseline_ppl"],
        num_layers=num_layers,
        intermediate_size=intermediate_size,
    )
    
    # Initialize all layers as LIGHT_TOUCH (no compression)
    for layer_idx in range(num_layers):
        prescription.layers.append(LayerPrescription(
            layer_idx=layer_idx,
            strategy=CompressionStrategy.LIGHT_TOUCH,
        ))
    
    # Phase 1: Apply dead neuron removal (always free)
    for info in layer_safety:
        if info["dead"] > 0:
            lp = prescription.layers[info["layer"]]
            lp.strategy = CompressionStrategy.DEAD_REMOVAL_AND_MERGE
            lp.dead_neuron_count = info["dead"]
            lp.dormant_neuron_count = info["dormant"]
            alive = intermediate_size - info["dead"] - info["dormant"]
            lp.merge_target_width = alive  # No merging yet, just dead removal
    
    current_reduction = estimate_param_reduction(prescription)["reduction_pct"] / 100
    logger.info(f"After dead removal: {current_reduction:.1%} reduction")
    
    if current_reduction >= target_reduction:
        _finalize_prescription(prescription, layer_safety, base_prescription)
        return prescription
    
    # Phase 2: Add merging on high-redundancy layers (guided by MRI)
    # Sort by redundancy (most redundant first)
    merge_candidates = [
        info for info in layers_by_compress
        if info["redundancy_frac"] > 0.10 and info["layer"] not in (0, num_layers - 1)
    ]
    
    for info in merge_candidates:
        if current_reduction >= target_reduction:
            break
        
        lp = prescription.layers[info["layer"]]
        alive = intermediate_size - info["dead"] - info["dormant"]
        if alive <= 0:
            continue
        
        # Determine merge aggressiveness based on how much more reduction we need
        remaining_budget = target_reduction - current_reduction
        
        # More aggressive merging for higher redundancy
        effective_redundant = max(0, info["safe_to_prune"] - info["dead"])
        if alive > 0:
            merge_frac = min(0.5, effective_redundant / alive)  # Cap at 50% of alive
            # Scale up if we need more reduction
            if remaining_budget > 0.20:
                merge_frac = min(0.65, merge_frac * 1.3)
        else:
            merge_frac = 0
        
        new_width = max(int(alive * (1 - merge_frac)), alive // 3)
        
        if lp.strategy == CompressionStrategy.DEAD_REMOVAL_AND_MERGE:
            lp.merge_target_width = new_width
        else:
            lp.strategy = CompressionStrategy.DEAD_REMOVAL_AND_MERGE
            lp.dead_neuron_count = info["dead"]
            lp.dormant_neuron_count = info["dormant"]
            lp.merge_target_width = new_width
        
        current_reduction = estimate_param_reduction(prescription)["reduction_pct"] / 100
        logger.info(f"  Layer {info['layer']}: merge to {new_width} "
                    f"(from {alive} alive) → cumulative {current_reduction:.1%}")
    
    if current_reduction >= target_reduction:
        _finalize_prescription(prescription, layer_safety, base_prescription)
        return prescription
    
    # Phase 3: Add structured pruning on non-critical middle layers
    prune_candidates = [
        info for info in layers_by_compress
        if info["mlp_crit"] < 1.0
        and info["layer"] not in (0, num_layers - 1)
        and prescription.layers[info["layer"]].strategy == CompressionStrategy.LIGHT_TOUCH
    ]
    
    for info in prune_candidates:
        if current_reduction >= target_reduction:
            break
        
        lp = prescription.layers[info["layer"]]
        
        # Sparsity based on criticality: less critical → more pruning
        remaining_budget = target_reduction - current_reduction
        base_sparsity = 0.20
        if info["mlp_crit"] < 0.3:
            base_sparsity = 0.35
        if remaining_budget > 0.15:
            base_sparsity = min(0.50, base_sparsity + 0.15)
        
        lp.strategy = CompressionStrategy.STRUCTURED_PRUNE
        lp.target_sparsity = base_sparsity
        lp.dead_neuron_count = info["dead"]
        
        current_reduction = estimate_param_reduction(prescription)["reduction_pct"] / 100
        logger.info(f"  Layer {info['layer']}: structured prune {base_sparsity:.0%} "
                    f"→ cumulative {current_reduction:.1%}")
    
    if current_reduction >= target_reduction:
        _finalize_prescription(prescription, layer_safety, base_prescription)
        return prescription
    
    # Phase 4: Depth pruning (remove entire layers) — most aggressive
    # Only for layers with very low PPL delta
    depth_candidates = [
        info for info in layer_safety
        if info["mlp_crit"] < 0.35
        and info["attn_crit"] < 0.15
        and info["layer"] not in (0, 1, num_layers - 1, num_layers - 2)
    ]
    depth_candidates.sort(key=lambda x: x["total_crit"])
    
    for info in depth_candidates:
        if current_reduction >= target_reduction:
            break
        
        lp = prescription.layers[info["layer"]]
        lp.strategy = CompressionStrategy.DEPTH_PRUNE
        prescription.depth_prune_candidates.append(info["layer"])
        
        current_reduction = estimate_param_reduction(prescription)["reduction_pct"] / 100
        logger.info(f"  Layer {info['layer']}: DEPTH PRUNE → cumulative {current_reduction:.1%}")
    
    _finalize_prescription(prescription, layer_safety, base_prescription)
    
    final = estimate_param_reduction(prescription)
    logger.info(f"Final estimated reduction: {final['reduction_pct']:.1f}% "
                f"(target: {target_reduction:.0%})")
    
    return prescription


def _finalize_prescription(
    prescription: CompressionPrescription,
    layer_safety: list[dict],
    base_prescription: CompressionPrescription,
):
    """Add reconstruction parameters and cascade ordering to prescription."""
    num_layers = prescription.num_layers
    
    # Copy cascade/alpha data from base prescription
    for i, lp in enumerate(prescription.layers):
        base_lp = base_prescription.layers[i]
        lp.alpha = base_lp.alpha
        lp.redundancy_frac = base_lp.redundancy_frac
        lp.cascade_amplification = base_lp.cascade_amplification
        lp.mlp_ppl_delta = base_lp.mlp_ppl_delta
        lp.attn_ppl_delta = base_lp.attn_ppl_delta
        
        # Cascade-aware reconstruction priority
        lp.reconstruction_priority = num_layers - 1 - lp.layer_idx
        
        # More iterations for high-cascade layers
        if lp.cascade_amplification > 50:
            lp.reconstruction_iterations = 300
        else:
            lp.reconstruction_iterations = 100
    
    # Count totals
    prescription.total_dead_neurons = sum(lp.dead_neuron_count for lp in prescription.layers)
    prescription.total_merged_neurons = 0  # Will be computed during compression


def cmd_profile(args):
    """Generate and optionally run a targeted compression profile."""
    study5, study10 = get_study_data(args.summary, args.study5, args.study10)
    
    prescription = build_mri_profile(
        args.summary,
        target_reduction=args.target_reduction,
        study5_data=study5,
        study10_data=study10,
    )
    
    est = estimate_param_reduction(prescription)
    print(prescription.summary())
    print(f"\n  Estimated param reduction: {est['reduction_pct']:.1f}%")
    print(f"  Original params:   {est['total_original']:,}")
    print(f"  Compressed params: {est['total_compressed']:,}")
    
    # Save prescription
    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_prescription_json(prescription, out_dir / "prescription.json")
        logger.info(f"Prescription saved to {out_dir / 'prescription.json'}")
    
    if args.run:
        # Actually run the compression
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from compressor import MRICompressor
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name = args.model or prescription.model_name
        dtype = torch.bfloat16
        
        logger.info(f"Loading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype, device_map="auto", trust_remote_code=True,
        )
        model.eval()
        
        logger.info("Loading calibration data...")
        cal_dataloader = build_calibration_dataloader(
            tokenizer,
            seq_len=args.seq_len,
            num_samples=args.num_calibration_samples,
            batch_size=args.batch_size,
        )
        
        compressor = MRICompressor(
            model=model,
            tokenizer=tokenizer,
            prescription=prescription,
            calibration_dataloader=cal_dataloader,
            device=device,
            max_calibration_batches=args.max_calibration_batches,
            do_reconstruction=not args.no_reconstruction,
            reconstruction_lr=args.reconstruction_lr,
        )
        
        result = compressor.compress()
        
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        report = {
            "target_reduction": args.target_reduction,
            "model": model_name,
            "original_ppl": result.original_ppl,
            "compressed_ppl": result.compressed_ppl,
            "ppl_change_abs": result.compressed_ppl - result.original_ppl,
            "params_original": result.total_params_original,
            "params_compressed": result.total_params_compressed,
            "param_reduction_pct": (1 - result.total_params_compressed / max(result.total_params_original, 1)) * 100,
            "neurons_removed": result.total_neurons_removed,
            "neurons_merged": result.total_neurons_merged,
            "elapsed_seconds": result.elapsed_seconds,
            "per_layer": result.per_layer_results,
        }
        
        with open(out_dir / "compression_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        if args.save_model:
            model.save_pretrained(out_dir / "model")
            tokenizer.save_pretrained(out_dir / "model")
        
        print(result.summary())


def main():
    parser = argparse.ArgumentParser(
        description="MRI-Compress targeted profiles for baseline comparison"
    )
    parser.add_argument("--summary", required=True, help="Path to summary.json")
    parser.add_argument("--model", default=None, help="HF model name")
    parser.add_argument("--study5", default=None)
    parser.add_argument("--study10", default=None)
    parser.add_argument("--target-reduction", type=float, required=True,
                        help="Target param reduction fraction (0.20 = 20%%)")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--run", action="store_true", help="Actually run compression")
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--no-reconstruction", action="store_true")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--num-calibration-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-calibration-batches", type=int, default=16)
    parser.add_argument("--reconstruction-lr", type=float, default=1e-4)
    
    args = parser.parse_args()
    cmd_profile(args)


if __name__ == "__main__":
    main()
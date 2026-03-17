#!/usr/bin/env python3
"""
MRI-Compress v2.1: Conservative + Ablation Modes
=================================================

Lessons from v2:
  - dead-only:  7.3% reduction, +0.37% PPL ✓ (excellent)
  - full v2:    7.2% reduction, +230% PPL  ✗ (catastrophic)
  - Cause: depth pruning + aggressive attn pruning don't compose safely

v2.1 strategy: start from the solid dead-only baseline and add
ONE strategy at a time via ablation flags.

Usage:
  # Baseline (same as v2 dead-only, the proven safe mode)
  python run_compress_v2_1.py compress --model Qwen/Qwen2.5-3B \
    --summary results/summary.json --output ./v2.1_baseline --save-model

  # Add conservative attention pruning (2 heads × eligible layers)
  python run_compress_v2_1.py compress --model Qwen/Qwen2.5-3B \
    --summary results/summary.json --output ./v2.1_attn \
    --enable-attn --save-model

  # Add depth pruning (1 layer only, safest candidate)
  python run_compress_v2_1.py compress --model Qwen/Qwen2.5-3B \
    --summary results/summary.json --output ./v2.1_depth \
    --enable-depth --save-model

  # Add both (still conservative)
  python run_compress_v2_1.py compress --model Qwen/Qwen2.5-3B \
    --summary results/summary.json --output ./v2.1_both \
    --enable-attn --enable-depth --save-model

  # Prescribe only (see what it would do)
  python run_compress_v2_1.py prescribe --summary results/summary.json
  python run_compress_v2_1.py prescribe --summary results/summary.json --enable-attn --enable-depth
"""

from __future__ import annotations
import argparse
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


# ============================================================================
# Hardcoded Study 5 & 10 data for Qwen2.5-3B
# ============================================================================

QWEN25_3B_STUDY5 = [
    {"layer": 0, "dead": 0, "dormant": 0},
    {"layer": 1, "dead": 8924, "dormant": 723},
    {"layer": 2, "dead": 8642, "dormant": 381},
    {"layer": 3, "dead": 6328, "dormant": 368},
    {"layer": 4, "dead": 2605, "dormant": 3456},
    {"layer": 5, "dead": 973, "dormant": 1746},
    {"layer": 6, "dead": 1505, "dormant": 257},
    {"layer": 7, "dead": 1443, "dormant": 275},
    {"layer": 8, "dead": 81, "dormant": 77},
    {"layer": 9, "dead": 5, "dormant": 204},
] + [{"layer": i, "dead": 0, "dormant": 0} for i in range(10, 36)]

QWEN25_3B_STUDY10 = [
    {"layer": 0, "mlp_ppl_delta": 19.00, "attn_ppl_delta": 10573.66},
    {"layer": 1, "mlp_ppl_delta": 0.27, "attn_ppl_delta": 5067.43},
    {"layer": 2, "mlp_ppl_delta": 0.60, "attn_ppl_delta": 0.17},
    {"layer": 3, "mlp_ppl_delta": 0.25, "attn_ppl_delta": 0.15},
    {"layer": 4, "mlp_ppl_delta": 0.24, "attn_ppl_delta": 0.31},
    {"layer": 5, "mlp_ppl_delta": 0.44, "attn_ppl_delta": 0.18},
    {"layer": 6, "mlp_ppl_delta": 0.47, "attn_ppl_delta": 0.24},
    {"layer": 7, "mlp_ppl_delta": 0.46, "attn_ppl_delta": 0.13},
    {"layer": 8, "mlp_ppl_delta": 0.31, "attn_ppl_delta": 0.04},
    {"layer": 9, "mlp_ppl_delta": 0.42, "attn_ppl_delta": 0.09},
    {"layer": 10, "mlp_ppl_delta": 0.44, "attn_ppl_delta": 0.13},
    {"layer": 11, "mlp_ppl_delta": 0.50, "attn_ppl_delta": 0.05},
    {"layer": 12, "mlp_ppl_delta": 0.81, "attn_ppl_delta": 0.19},
    {"layer": 13, "mlp_ppl_delta": 0.33, "attn_ppl_delta": 0.19},
    {"layer": 14, "mlp_ppl_delta": 0.37, "attn_ppl_delta": 0.12},
    {"layer": 15, "mlp_ppl_delta": 0.33, "attn_ppl_delta": 0.24},
    {"layer": 16, "mlp_ppl_delta": 0.38, "attn_ppl_delta": 0.16},
    {"layer": 17, "mlp_ppl_delta": 0.22, "attn_ppl_delta": 0.07},
    {"layer": 18, "mlp_ppl_delta": 0.28, "attn_ppl_delta": 0.18},
    {"layer": 19, "mlp_ppl_delta": 0.22, "attn_ppl_delta": 0.18},
    {"layer": 20, "mlp_ppl_delta": 0.30, "attn_ppl_delta": 0.18},
    {"layer": 21, "mlp_ppl_delta": 0.26, "attn_ppl_delta": 0.06},
    {"layer": 22, "mlp_ppl_delta": 0.28, "attn_ppl_delta": 0.09},
    {"layer": 23, "mlp_ppl_delta": 0.36, "attn_ppl_delta": 0.13},
    {"layer": 24, "mlp_ppl_delta": 0.32, "attn_ppl_delta": 0.24},
    {"layer": 25, "mlp_ppl_delta": 0.36, "attn_ppl_delta": 0.38},
    {"layer": 26, "mlp_ppl_delta": 0.46, "attn_ppl_delta": 0.17},
    {"layer": 27, "mlp_ppl_delta": 0.51, "attn_ppl_delta": 0.58},
    {"layer": 28, "mlp_ppl_delta": 0.53, "attn_ppl_delta": 0.17},
    {"layer": 29, "mlp_ppl_delta": 0.66, "attn_ppl_delta": 0.20},
    {"layer": 30, "mlp_ppl_delta": 1.14, "attn_ppl_delta": 0.14},
    {"layer": 31, "mlp_ppl_delta": 1.67, "attn_ppl_delta": 0.89},
    {"layer": 32, "mlp_ppl_delta": 1.67, "attn_ppl_delta": 1.02},
    {"layer": 33, "mlp_ppl_delta": 1.97, "attn_ppl_delta": 0.63},
    {"layer": 34, "mlp_ppl_delta": 2.55, "attn_ppl_delta": 0.64},
    {"layer": 35, "mlp_ppl_delta": 4.98, "attn_ppl_delta": 0.38},
]


def get_study_data(summary_path, study5_path=None, study10_path=None):
    from diagnostics import parse_study5_from_terminal, parse_study10_from_terminal
    with open(summary_path) as f:
        summary = json.load(f)
    model_name = summary.get("model", "")

    study5_data = None
    if study5_path and Path(study5_path).exists():
        with open(study5_path) as f:
            study5_data = parse_study5_from_terminal(f.read())
    elif "3B" in model_name:
        study5_data = QWEN25_3B_STUDY5
        logger.info("Using hardcoded Study 5 data for Qwen2.5-3B")

    study10_data = None
    if study10_path and Path(study10_path).exists():
        with open(study10_path) as f:
            study10_data = parse_study10_from_terminal(f.read())
    elif "3B" in model_name:
        study10_data = QWEN25_3B_STUDY10
        logger.info("Using hardcoded Study 10 data for Qwen2.5-3B")

    return study5_data, study10_data


def cmd_prescribe(args):
    from diagnostics import MRIDiagnostician

    study5, study10 = get_study_data(args.summary, args.study5, args.study10)

    diagnostician = MRIDiagnostician(
        enable_depth_pruning=args.enable_depth,
        enable_attn_pruning=args.enable_attn,
        enable_merging=args.enable_merge,
        max_depth_prune_layers=args.max_depth_prune,
        attn_heads_to_prune_per_layer=args.attn_heads_per_layer,
        max_total_attn_heads=args.max_attn_heads,
        ppl_budget=args.ppl_budget,
    )
    prescription = diagnostician.diagnose(args.summary, study5, study10)
    print(prescription.summary())

    # Show what eligible layers look like for attn pruning
    if args.enable_attn:
        eligible = [lp for lp in prescription.layers if lp.attn_heads_to_prune > 0]
        print(f"\n  Attention pruning: {len(eligible)} layers × "
              f"{args.attn_heads_per_layer} heads = {sum(lp.attn_heads_to_prune for lp in eligible)} total heads")
        for lp in eligible:
            print(f"    Layer {lp.layer_idx}: prune {lp.attn_heads_to_prune} heads (attn_Δ={lp.attn_ppl_delta:.3f})")

    if args.output:
        out_path = Path(args.output) / "prescription.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_prescription_json(prescription, out_path)
        logger.info(f"Prescription saved to {out_path}")


def cmd_compress(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from diagnostics import MRIDiagnostician, CompressionStrategy
    from compressor import MRICompressor

    # ---- Build prescription ----
    study5, study10 = get_study_data(args.summary, args.study5, args.study10)

    diagnostician = MRIDiagnostician(
        enable_depth_pruning=args.enable_depth,
        enable_attn_pruning=args.enable_attn,
        enable_merging=args.enable_merge,
        max_depth_prune_layers=args.max_depth_prune,
        attn_heads_to_prune_per_layer=args.attn_heads_per_layer,
        max_total_attn_heads=args.max_attn_heads,
        ppl_budget=args.ppl_budget,
    )
    prescription = diagnostician.diagnose(args.summary, study5, study10)
    print(prescription.summary())

    # ---- Load model ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = args.model or prescription.model_name
    dtype = torch.bfloat16 if prescription.num_layers > 24 else torch.float32

    logger.info(f"Loading model: {model_name} (dtype={dtype})")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map="auto", trust_remote_code=True)
    model.eval()

    # ---- Calibration data ----
    logger.info("Preparing calibration data...")
    cal_dataloader = build_calibration_dataloader(
        tokenizer, seq_len=args.seq_len,
        num_samples=args.num_calibration_samples,
        batch_size=args.batch_size,
    )

    # ---- Compress ----
    compressor = MRICompressor(
        model=model,
        tokenizer=tokenizer,
        prescription=prescription,
        calibration_dataloader=cal_dataloader,
        device=device,
        max_calibration_batches=args.max_calibration_batches,
        do_reconstruction=not args.no_reconstruction,
        reconstruction_lr=args.reconstruction_lr,
        enable_low_rank=False,           # Disabled in v2.1 (Study 18: no benefit)
        enable_attn_pruning=args.enable_attn,
        enable_depth_pruning=args.enable_depth,
    )

    result = compressor.compress()

    # ---- Save results ----
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "version": "v2.1",
        "model": model_name,
        "mode": _describe_mode(args),
        "original_ppl": result.original_ppl,
        "compressed_ppl": result.compressed_ppl,
        "ppl_change_abs": result.compressed_ppl - result.original_ppl,
        "ppl_change_pct": (result.compressed_ppl / max(result.original_ppl, 1e-8) - 1) * 100,
        "params_original": result.total_params_original,
        "params_compressed": result.total_params_compressed,
        "param_reduction_pct": (1 - result.total_params_compressed / max(result.total_params_original, 1)) * 100,
        "neurons_removed": result.total_neurons_removed,
        "dormant_removed": result.total_dormant_removed,
        "neurons_merged": result.total_neurons_merged,
        "attn_heads_pruned": result.total_attn_heads_pruned,
        "depth_pruned_layers": result.total_depth_pruned_layers,
        "elapsed_seconds": result.elapsed_seconds,
        "per_layer": result.per_layer_results,
    }

    with open(output_dir / "compression_report.json", "w") as f:
        json.dump(report, f, indent=2)
    save_prescription_json(prescription, output_dir / "prescription.json")

    if args.save_model:
        logger.info(f"Saving compressed model to {output_dir / 'model'}")
        model.save_pretrained(output_dir / "model")
        tokenizer.save_pretrained(output_dir / "model")

    print(result.summary())
    return result


def _describe_mode(args):
    parts = ["dead+dormant"]
    if args.enable_merge:
        parts.append("merge")
    if args.enable_attn:
        parts.append(f"attn({args.attn_heads_per_layer}h/layer,max{args.max_attn_heads})")
    if args.enable_depth:
        parts.append(f"depth(max{args.max_depth_prune})")
    return "+".join(parts)


def build_calibration_dataloader(tokenizer, seq_len=2048, num_samples=128, batch_size=4):
    import torch
    from torch.utils.data import DataLoader, Dataset
    from datasets import load_dataset

    dataset = None
    for ds_name in ["Salesforce/wikitext", "wikitext"]:
        try:
            dataset = load_dataset(ds_name, "wikitext-103-raw-v1", split="validation")
            break
        except Exception:
            continue
    if dataset is None:
        raise RuntimeError("Could not load calibration dataset")

    all_token_ids = []
    target_total = num_samples * seq_len + seq_len
    for sample in dataset:
        text = sample["text"]
        if len(text.strip()) < 10:
            continue
        tokens = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        all_token_ids.append(tokens)
        if sum(len(t) for t in all_token_ids) >= target_total:
            break

    full_stream = torch.cat(all_token_ids, dim=0)
    n_chunks = min(len(full_stream) // seq_len, num_samples)
    all_ids = [full_stream[i * seq_len : (i + 1) * seq_len] for i in range(n_chunks)]
    logger.info(f"Calibration: {len(all_ids)} samples × {seq_len} tokens")

    class TokenDataset(Dataset):
        def __init__(self, tokens_list):
            self.tokens = tokens_list
        def __len__(self):
            return len(self.tokens)
        def __getitem__(self, idx):
            t = self.tokens[idx]
            return {"input_ids": t, "attention_mask": torch.ones_like(t)}

    return DataLoader(TokenDataset(all_ids), batch_size=batch_size, shuffle=False)


def save_prescription_json(prescription, path):
    data = {
        "model_name": prescription.model_name,
        "baseline_ppl": prescription.baseline_ppl,
        "num_layers": prescription.num_layers,
        "intermediate_size": prescription.intermediate_size,
        "depth_prune_candidates": prescription.depth_prune_candidates,
        "depth_prune_applied": prescription.depth_prune_applied,
        "estimated_ppl_budget_used": prescription.estimated_ppl_budget_used,
        "total_dead_neurons": prescription.total_dead_neurons,
        "total_dormant_neurons": prescription.total_dormant_neurons,
        "total_merged_neurons": prescription.total_merged_neurons,
        "total_attn_heads_pruned": prescription.total_attn_heads_pruned,
        "layers": [{
            "layer_idx": lp.layer_idx,
            "strategy": lp.strategy.name,
            "dead_neuron_count": lp.dead_neuron_count,
            "dormant_neuron_count": lp.dormant_neuron_count,
            "attn_heads_to_prune": lp.attn_heads_to_prune,
            "depth_prune": lp.depth_prune,
            "mlp_ppl_delta": lp.mlp_ppl_delta,
            "attn_ppl_delta": lp.attn_ppl_delta,
        } for lp in prescription.layers],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="MRI-Compress v2.1: Conservative + Ablation Modes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- Common args shared between prescribe and compress ----
    def add_common_args(p):
        p.add_argument("--summary", required=True, help="Path to summary.json")
        p.add_argument("--study5", default=None)
        p.add_argument("--study10", default=None)
        p.add_argument("--output", default=None)
        # Strategy toggles (all OFF by default for safety)
        p.add_argument("--enable-attn", action="store_true",
                        help="Enable conservative attention head pruning")
        p.add_argument("--enable-depth", action="store_true",
                        help="Enable depth pruning (max 1 layer)")
        p.add_argument("--enable-merge", action="store_true",
                        help="Enable neuron merging for redundant layers")
        # Tuning
        p.add_argument("--max-depth-prune", type=int, default=1)
        p.add_argument("--attn-heads-per-layer", type=int, default=2,
                        help="Heads to prune per eligible layer (default: 2 of 16)")
        p.add_argument("--max-attn-heads", type=int, default=32,
                        help="Global budget for total attention heads to prune")
        p.add_argument("--ppl-budget", type=float, default=1.0,
                        help="Max estimated PPL increase budget (default: 1.0)")

    # Prescribe
    p = subparsers.add_parser("prescribe", help="Generate prescription (no model needed)")
    add_common_args(p)

    # Compress
    p = subparsers.add_parser("compress", help="Run compression pipeline")
    add_common_args(p)
    p.add_argument("--model", default=None)
    p.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16"])
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--num-calibration-samples", type=int, default=128)
    p.add_argument("--max-calibration-batches", type=int, default=16)
    p.add_argument("--no-reconstruction", action="store_true")
    p.add_argument("--reconstruction-lr", type=float, default=1e-4)
    p.add_argument("--save-model", action="store_true")

    args = parser.parse_args()

    if args.command == "prescribe":
        cmd_prescribe(args)
    elif args.command == "compress":
        if not args.output:
            parser.error("--output is required for compress")
        cmd_compress(args)


if __name__ == "__main__":
    main()
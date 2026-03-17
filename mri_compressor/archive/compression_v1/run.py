#!/usr/bin/env python3
"""
MRI-Compress: Diagnostic-Guided Heterogeneous LLM Compression
=============================================================

Usage:
  # Step 1: Generate prescription only (no GPU needed)
  python run_compress.py prescribe \
    --summary results/summary.json

  # Step 2: Apply compression to model (needs GPU)
  python run_compress.py compress \
    --model Qwen/Qwen2.5-3B \
    --summary results/summary.json \
    --output ./compressed_model

  # Quick test: dead neuron removal only (fastest, safest)
  python run_compress.py compress \
    --model Qwen/Qwen2.5-3B \
    --summary results/summary.json \
    --output ./compressed_model \
    --dead-only
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
# Parsed from terminal output — avoids requiring separate text files
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


def get_study_data(summary_path: str, study5_path=None, study10_path=None):
    """Load study data from files or use hardcoded defaults for known models."""
    from diagnostic import parse_study5_from_terminal, parse_study10_from_terminal

    with open(summary_path) as f:
        summary = json.load(f)
    model_name = summary.get("model", "")

    # Study 5
    study5_data = None
    if study5_path and Path(study5_path).exists():
        with open(study5_path) as f:
            study5_data = parse_study5_from_terminal(f.read())
        logger.info(f"Loaded Study 5 from file: {len(study5_data)} layers")
    elif "3B" in model_name:
        study5_data = QWEN25_3B_STUDY5
        logger.info("Using hardcoded Study 5 data for Qwen2.5-3B")

    # Study 10
    study10_data = None
    if study10_path and Path(study10_path).exists():
        with open(study10_path) as f:
            study10_data = parse_study10_from_terminal(f.read())
        logger.info(f"Loaded Study 10 from file: {len(study10_data)} layers")
    elif "3B" in model_name:
        study10_data = QWEN25_3B_STUDY10
        logger.info("Using hardcoded Study 10 data for Qwen2.5-3B")

    return study5_data, study10_data


def cmd_prescribe(args):
    """Generate and display the compression prescription."""
    from diagnostic import MRIDiagnostician

    study5, study10 = get_study_data(args.summary, args.study5, args.study10)
    diagnostician = MRIDiagnostician()
    prescription = diagnostician.diagnose(args.summary, study5, study10)
    print(prescription.summary())

    # Save prescription as JSON for inspection
    if args.output:
        out_path = Path(args.output) / "prescription.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_prescription_json(prescription, out_path)
        logger.info(f"Prescription saved to {out_path}")


def cmd_compress(args):
    """Load model and run the full compression pipeline."""
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from diagnostic import MRIDiagnostician, CompressionStrategy
    from compressor import MRICompressor

    # Step 1: Build prescription
    study5, study10 = get_study_data(args.summary, args.study5, args.study10)
    diagnostician = MRIDiagnostician()
    prescription = diagnostician.diagnose(args.summary, study5, study10)

    # If --dead-only, override all strategies to LIGHT_TOUCH except dead removal
    if args.dead_only:
        logger.info("--dead-only mode: only removing dead neurons")
        for lp in prescription.layers:
            if lp.dead_neuron_count > 0:
                lp.strategy = CompressionStrategy.DEAD_REMOVAL_AND_MERGE
                lp.merge_target_width = None  # Skip merging
                lp.target_sparsity = 0.0
            else:
                lp.strategy = CompressionStrategy.LIGHT_TOUCH

    print(prescription.summary())

    # Step 2: Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logger.warning("No CUDA device found! Running on CPU will be extremely slow.")

    model_name = args.model or prescription.model_name
    dtype = torch.bfloat16 if prescription.num_layers > 24 else torch.float32
    logger.info(f"Loading model: {model_name} (dtype={dtype})")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Step 3: Prepare calibration data
    logger.info("Preparing calibration data...")
    cal_dataloader = build_calibration_dataloader(
        tokenizer,
        seq_len=args.seq_len,
        num_samples=args.num_calibration_samples,
        batch_size=args.batch_size,
        dataset_name=args.calibration_dataset,
    )

    # Step 4: Run compression
    compressor = MRICompressor(
        model=model,
        tokenizer=tokenizer,
        prescription=prescription,
        calibration_dataloader=cal_dataloader,
        device=device,
        max_calibration_batches=args.max_calibration_batches,
        do_reconstruction=not args.no_reconstruction and not args.dead_only,
        reconstruction_lr=args.reconstruction_lr,
    )

    result = compressor.compress()

    # Step 5: Save outputs
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save report
    report = {
        "model": model_name,
        "original_ppl": result.original_ppl,
        "compressed_ppl": result.compressed_ppl,
        "ppl_change_abs": result.compressed_ppl - result.original_ppl,
        "ppl_change_pct": (result.compressed_ppl / max(result.original_ppl, 1e-8) - 1) * 100,
        "params_original": result.total_params_original,
        "params_compressed": result.total_params_compressed,
        "param_reduction_pct": (1 - result.total_params_compressed / max(result.total_params_original, 1)) * 100,
        "neurons_removed": result.total_neurons_removed,
        "neurons_merged": result.total_neurons_merged,
        "elapsed_seconds": result.elapsed_seconds,
        "per_layer": result.per_layer_results,
        "args": vars(args),
    }
    report_path = output_dir / "compression_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report saved to {report_path}")

    # Save prescription
    save_prescription_json(prescription, output_dir / "prescription.json")

    # Save model if requested
    if args.save_model:
        logger.info(f"Saving compressed model to {output_dir / 'model'}")
        model_dir = output_dir / "model"
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        logger.info("Model saved.")

    print(result.summary())
    return result


def build_calibration_dataloader(
    tokenizer,
    seq_len: int = 2048,
    num_samples: int = 128,
    batch_size: int = 4,
    dataset_name: str = "wikitext",
):
    """
    Build a calibration DataLoader from wikitext or c4.

    Concatenates all text into one long token stream, then chunks it into
    non-overlapping windows of seq_len. This ensures we always get enough
    samples regardless of individual text length.
    """
    import torch
    from torch.utils.data import DataLoader, Dataset
    from datasets import load_dataset

    # Try loading dataset with fallbacks
    dataset = None
    text_key = "text"

    if dataset_name == "wikitext":
        # Try multiple dataset identifiers (HF redirects change over time)
        for ds_name in ["Salesforce/wikitext", "wikitext"]:
            try:
                dataset = load_dataset(ds_name, "wikitext-103-raw-v1", split="validation")
                logger.info(f"Loaded dataset: {ds_name}")
                break
            except Exception as e:
                logger.warning(f"Failed to load {ds_name}: {e}")
                continue

    elif dataset_name == "c4":
        try:
            dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        except Exception as e:
            logger.warning(f"Failed to load c4: {e}")

    if dataset is None:
        raise RuntimeError(
            "Could not load calibration dataset. Try:\n"
            "  pip install datasets\n"
            "  huggingface-cli login\n"
            "Or use --calibration-dataset c4"
        )

    # Concatenate all text, tokenize as one long stream, then chunk
    logger.info("Tokenizing calibration text...")
    all_token_ids = []
    target_total = num_samples * seq_len + seq_len  # A bit extra

    for sample in dataset:
        text = sample[text_key]
        if len(text.strip()) < 10:
            continue
        tokens = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        all_token_ids.append(tokens)
        total_so_far = sum(len(t) for t in all_token_ids)
        if total_so_far >= target_total:
            break

    # Concatenate into one long tensor and chunk
    full_stream = torch.cat(all_token_ids, dim=0)
    n_chunks = min(len(full_stream) // seq_len, num_samples)
    all_ids = [full_stream[i * seq_len : (i + 1) * seq_len] for i in range(n_chunks)]

    if len(all_ids) == 0:
        raise RuntimeError(
            f"Could not create any calibration samples. "
            f"Got {len(full_stream)} tokens total, need at least {seq_len}."
        )

    logger.info(f"Calibration: {len(all_ids)} samples × {seq_len} tokens "
                f"(from {len(full_stream):,} total tokens)")

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
    """Serialize a CompressionPrescription to JSON."""
    data = {
        "model_name": prescription.model_name,
        "baseline_ppl": prescription.baseline_ppl,
        "num_layers": prescription.num_layers,
        "intermediate_size": prescription.intermediate_size,
        "depth_prune_candidates": prescription.depth_prune_candidates,
        "total_dead_neurons": prescription.total_dead_neurons,
        "total_merged_neurons": prescription.total_merged_neurons,
        "layers": [],
    }
    for lp in prescription.layers:
        data["layers"].append({
            "layer_idx": lp.layer_idx,
            "strategy": lp.strategy.name,
            "target_sparsity": lp.target_sparsity,
            "quant_bits": lp.quant_bits,
            "outlier_bits": lp.outlier_bits,
            "outlier_fraction": lp.outlier_fraction,
            "merge_target_width": lp.merge_target_width,
            "dead_neuron_count": lp.dead_neuron_count,
            "dormant_neuron_count": lp.dormant_neuron_count,
            "reconstruction_priority": lp.reconstruction_priority,
            "reconstruction_iterations": lp.reconstruction_iterations,
            "alpha": lp.alpha,
            "redundancy_frac": lp.redundancy_frac,
            "cascade_amplification": lp.cascade_amplification,
            "mlp_ppl_delta": lp.mlp_ppl_delta,
            "attn_ppl_delta": lp.attn_ppl_delta,
        })
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="MRI-Compress: Diagnostic-Guided LLM Compression",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # ---- Prescribe subcommand ----
    p_prescribe = subparsers.add_parser(
        "prescribe", help="Generate compression prescription from MRI data"
    )
    p_prescribe.add_argument("--summary", required=True, help="Path to summary.json")
    p_prescribe.add_argument("--study5", default=None, help="Path to Study 5 terminal output")
    p_prescribe.add_argument("--study10", default=None, help="Path to Study 10 terminal output")
    p_prescribe.add_argument("--output", default=None, help="Output directory for prescription JSON")

    # ---- Compress subcommand ----
    p_compress = subparsers.add_parser(
        "compress", help="Apply compression to a model"
    )
    p_compress.add_argument("--model", default=None, help="HF model name (default: from summary)")
    p_compress.add_argument("--summary", required=True, help="Path to summary.json")
    p_compress.add_argument("--study5", default=None, help="Path to Study 5 terminal output")
    p_compress.add_argument("--study10", default=None, help="Path to Study 10 terminal output")
    p_compress.add_argument("--output", required=True, help="Output directory")

    # Compression options
    p_compress.add_argument("--dead-only", action="store_true",
                            help="Only remove dead neurons (fastest, safest test)")
    p_compress.add_argument("--no-reconstruction", action="store_true",
                            help="Skip local reconstruction phase")
    p_compress.add_argument("--save-model", action="store_true",
                            help="Save compressed model weights")

    # Calibration options
    p_compress.add_argument("--calibration-dataset", default="wikitext",
                            choices=["wikitext", "c4"])
    p_compress.add_argument("--num-calibration-samples", type=int, default=128)
    p_compress.add_argument("--seq-len", type=int, default=2048)
    p_compress.add_argument("--batch-size", type=int, default=4)
    p_compress.add_argument("--max-calibration-batches", type=int, default=16)

    # Reconstruction options
    p_compress.add_argument("--reconstruction-lr", type=float, default=1e-4)

    args = parser.parse_args()

    if args.command == "prescribe":
        cmd_prescribe(args)
    elif args.command == "compress":
        cmd_compress(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
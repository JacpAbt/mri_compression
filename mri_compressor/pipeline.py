#!/usr/bin/env python3
"""
Unified MRI-to-Compression Pipeline.

Single entry point that chains:
  1. MRI Scan: Run diagnostic studies on the model
  2. Diagnosis: Analyze results, build per-layer compression prescription
  3. Compression: Apply the prescription to the model
  4. Evaluation: Measure compressed model quality

Usage:
    # MRI only
    python pipeline.py --model Qwen/Qwen2.5-3B --studies 1,3,4,5,6,8,9,10 --output ./results

    # Full pipeline: MRI + Compression
    python pipeline.py --model Qwen/Qwen2.5-3B --studies 1,3,4,5,6,8,9,10,11 \
        --compress --enable-attn --output ./results --save-model

    # Compression from existing MRI results
    python pipeline.py --model Qwen/Qwen2.5-3B --from-summary ./results/summary.json \
        --compress --save-model
"""

import argparse
import json
import os
import time
import torch
from pathlib import Path

from .config import ExperimentConfig
from .model_utils import ModelInspector
from .data_utils import load_wikitext_data, get_dataloader, evaluate_perplexity


def run_mri_stage(config: ExperimentConfig, inspector: ModelInspector, studies: list[int]) -> dict:
    """Stage 1: Run MRI studies and save enriched summary."""
    from .mri.runner import MRIRunner

    print("\n" + "=" * 80)
    print("STAGE 1: MRI SCAN")
    print("=" * 80)

    runner = MRIRunner(config, inspector=inspector)
    runner.run_studies(studies)
    summary = runner.save(config.output_dir)
    return summary


def run_diagnosis_stage(summary: dict, config: ExperimentConfig) -> object:
    """Stage 2: Analyze MRI results and build compression prescription."""
    from .compression.diagnostician import MRIDiagnostician

    print("\n" + "=" * 80)
    print("STAGE 2: DIAGNOSIS")
    print("=" * 80)

    diagnostician = MRIDiagnostician(
        enable_attn_pruning=config.enable_attn_pruning,
        enable_depth_pruning=config.enable_depth_pruning,
        enable_merging=config.enable_merge,
        target_domain=config.target_domain,
        domain_unnecessary_removal_frac=config.domain_unnecessary_frac,
        domain_critical_protection_frac=config.domain_critical_frac,
    )

    prescription = diagnostician.diagnose_from_summary(summary)
    print(prescription.summary())
    return prescription


def run_compression_stage(
    inspector: ModelInspector,
    prescription: object,
    config: ExperimentConfig,
    summary: dict,
) -> object:
    """Stage 3: Apply compression prescription to the model."""
    from .compression.compressor import MRICompressor

    print("\n" + "=" * 80)
    print("STAGE 3: COMPRESSION")
    print("=" * 80)

    dataset = load_wikitext_data(
        inspector.tokenizer,
        max_seq_len=config.max_length,
        num_samples=config.max_samples,
    )
    dataloader = get_dataloader(dataset, batch_size=config.batch_size)

    # Create domain-specific calibration dataloader when target_domain is set
    domain_dataloader = None
    if config.target_domain:
        domain_dataloader = _create_domain_dataloader(
            config.target_domain, inspector.tokenizer,
            max_seq_len=config.max_length, batch_size=config.batch_size,
            custom_path=config.custom_domain_path,
            custom_name=config.custom_domain_name,
        )
        if domain_dataloader is not None:
            print(f"  Using domain-specific reconstruction data: {config.target_domain}")

    compressor = MRICompressor(
        model=inspector.model,
        tokenizer=inspector.tokenizer,
        prescription=prescription,
        device=inspector.device,
        calibration_dataloader=dataloader,
        reconstruction_iterations=config.reconstruction_steps,
        reconstruction_lr=config.reconstruction_lr,
        enable_low_rank=config.enable_low_rank,
        enable_attn_pruning=config.enable_attn_pruning,
        enable_depth_pruning=config.enable_depth_pruning,
        enable_static_fold=config.enable_static_fold,
        enable_weight_sharing=config.enable_weight_sharing,
        domain_calibration_dataloader=domain_dataloader,
    )

    result = compressor.compress()
    return result


def _create_domain_dataloader(
    domain_name: str,
    tokenizer,
    max_seq_len: int = 512,
    batch_size: int = 4,
    custom_path: str = None,
    custom_name: str = None,
):
    """Create a dataloader from domain-specific data for reconstruction."""
    from .data_utils import TextDataset

    # If the target is a custom domain with a path, load from that
    if custom_path and custom_name and custom_name == domain_name:
        try:
            if os.path.isfile(custom_path):
                with open(custom_path, "r", encoding="utf-8") as f:
                    text = f.read()
            else:
                from datasets import load_dataset
                ds = load_dataset(custom_path, split="train", trust_remote_code=True)
                for field_name in ["text", "content", "question", "prompt"]:
                    if field_name in ds.column_names:
                        text = "\n".join(str(row[field_name]) for row in ds if len(str(row[field_name])) > 50)
                        break
                else:
                    return None

            tokens = tokenizer.encode(text, return_tensors="pt")[0]
            n_chunks = min(128, len(tokens) // max_seq_len)
            if n_chunks < 4:
                return None
            chunks = tokens[:n_chunks * max_seq_len].reshape(n_chunks, max_seq_len)
            dataset = TextDataset(chunks)
            return get_dataloader(dataset, batch_size=batch_size)
        except Exception as e:
            print(f"  Warning: Failed to load custom domain data: {e}")
            return None

    # Standard domain: load via the study domain system
    try:
        from .mri.studies_domain import load_domain_datasets
        domain_datasets = load_domain_datasets(
            tokenizer, max_seq_len=max_seq_len, samples_per_domain=128,
        )
        if domain_name in domain_datasets:
            return get_dataloader(domain_datasets[domain_name], batch_size=batch_size)
        else:
            print(f"  Warning: Domain '{domain_name}' not found in standard domains "
                  f"({list(domain_datasets.keys())})")
            return None
    except Exception as e:
        print(f"  Warning: Failed to load domain datasets: {e}")
        return None


def run_evaluation_stage(
    inspector: ModelInspector,
    config: ExperimentConfig,
    baseline_ppl: float,
) -> dict:
    """Stage 4: Evaluate compressed model."""
    print("\n" + "=" * 80)
    print("STAGE 4: EVALUATION")
    print("=" * 80)

    dataset = load_wikitext_data(
        inspector.tokenizer,
        max_seq_len=config.max_length,
        num_samples=config.max_samples,
    )
    loader = get_dataloader(dataset, batch_size=config.batch_size)

    compressed_ppl = evaluate_perplexity(
        inspector.model, loader, inspector.device,
        max_batches=config.max_eval_batches,
    )

    ppl_increase = compressed_ppl - baseline_ppl
    ppl_increase_pct = (ppl_increase / baseline_ppl) * 100

    print(f"\n  Baseline PPL:    {baseline_ppl:.2f}")
    print(f"  Compressed PPL:  {compressed_ppl:.2f}")
    print(f"  PPL increase:    {ppl_increase:.2f} ({ppl_increase_pct:+.1f}%)")

    # Count parameters
    total_params = sum(p.numel() for p in inspector.model.parameters())
    print(f"  Total parameters: {total_params:,}")

    eval_report = {
        "baseline_ppl": baseline_ppl,
        "compressed_ppl": compressed_ppl,
        "ppl_increase": ppl_increase,
        "ppl_increase_pct": ppl_increase_pct,
        "total_parameters": total_params,
    }

    # Save evaluation report
    report_path = os.path.join(config.output_dir, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(eval_report, f, indent=2)
    print(f"\n  Evaluation report saved to {report_path}")

    return eval_report


def run_pipeline(args):
    """Run the full or partial pipeline."""
    t_start = time.time()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    config = ExperimentConfig(
        model_name=args.model,
        device=device,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        max_samples=args.max_samples,
        max_length=args.max_length,
        max_eval_batches=min(args.max_batches, 8),
        output_dir=args.output,
        enable_compression=args.compress,
        enable_attn_pruning=args.enable_attn,
        enable_depth_pruning=args.enable_depth,
        enable_merge=not args.disable_merge,
        reconstruction_steps=args.reconstruction_steps,
        reconstruction_lr=args.reconstruction_lr,
        save_compressed_model=args.save_model,
        target_domain=args.target_domain,
        custom_domain_path=getattr(args, 'custom_domain_path', None),
        custom_domain_name=getattr(args, 'custom_domain_name', None),
        domain_unnecessary_frac=getattr(args, 'domain_unnecessary_frac', 0.05),
        domain_critical_frac=getattr(args, 'domain_critical_frac', 0.10),
        enable_low_rank=not args.disable_low_rank,
        enable_static_fold=not args.disable_static_fold,
        enable_weight_sharing=args.enable_weight_sharing,
    )

    # Load model
    print(f"Loading model: {args.model}")
    inspector = ModelInspector(args.model, device)
    print(f"  Architecture: {'gated (SwiGLU)' if inspector.is_gated else 'standard (GELU)'}")
    print(f"  Intermediate size: {inspector.mlp_layers[0].intermediate_size}")
    print(f"  Layers: {inspector.num_layers}")
    print(f"  Device: {device}")

    # Determine what to run
    summary = None

    if args.from_summary:
        # Skip MRI, load existing summary
        print(f"\nLoading existing summary from {args.from_summary}")
        with open(args.from_summary, "r") as f:
            summary = json.load(f)
        # Store the directory containing the summary so diagnostician
        # can find tensor files (e.g., domain wanda scores from Study 22)
        summary_dir = str(Path(args.from_summary).parent)
        summary["output_dir"] = summary_dir
        baseline_ppl = summary.get("baseline_ppl", 0)
    else:
        # Parse study list
        studies = [int(s.strip()) for s in args.studies.split(",")]

        # Stage 1: MRI
        summary = run_mri_stage(config, inspector, studies)
        summary["output_dir"] = config.output_dir
        baseline_ppl = summary.get("baseline_ppl", 0)

    # Stage 2-4: Compression pipeline (if requested)
    if args.compress:
        if summary is None:
            print("ERROR: No MRI summary available. Run studies or provide --from-summary.")
            return

        # Stage 2: Diagnosis
        prescription = run_diagnosis_stage(summary, config)

        # Stage 3: Compression
        result = run_compression_stage(inspector, prescription, config, summary)

        # Stage 4: Evaluation
        eval_report = run_evaluation_stage(inspector, config, baseline_ppl)

        # Save compressed model
        if args.save_model:
            model_dir = os.path.join(config.output_dir, "compressed_model")
            print(f"\nSaving compressed model to {model_dir}")
            inspector.model.save_pretrained(model_dir)
            inspector.tokenizer.save_pretrained(model_dir)
            print("  Model saved.")

    elapsed = time.time() - t_start
    print(f"\nPipeline complete. Total time: {elapsed:.0f}s")


def main():
    parser = argparse.ArgumentParser(
        description="MRI-to-Compression Pipeline for LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # MRI scan only
  python pipeline.py --model gpt2 --studies 1,5,10

  # Full pipeline
  python pipeline.py --model Qwen/Qwen2.5-3B --studies 1,3,4,5,6,8,9,10,11 --compress

  # Compression from existing MRI
  python pipeline.py --model Qwen/Qwen2.5-3B --from-summary ./results/summary.json --compress
        """,
    )

    # Model
    parser.add_argument("--model", type=str, default="gpt2", help="Model name or path")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cuda/cpu)")

    # MRI
    parser.add_argument("--studies", type=str, default="1,3,4,5,6,8,9,10",
                        help="Comma-separated study numbers to run")
    parser.add_argument("--from-summary", type=str, default=None,
                        help="Skip MRI, load existing summary.json")

    # Data
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--max-length", type=int, default=512)

    # Compression
    parser.add_argument("--compress", action="store_true",
                        help="Run compression after MRI")
    parser.add_argument("--enable-attn", action="store_true",
                        help="Enable attention head pruning")
    parser.add_argument("--enable-depth", action="store_true",
                        help="Enable depth pruning (remove entire layers)")
    parser.add_argument("--disable-merge", action="store_true",
                        help="Disable neuron merging")
    parser.add_argument("--reconstruction-steps", type=int, default=200,
                        help="Local reconstruction fine-tuning steps per layer")
    parser.add_argument("--reconstruction-lr", type=float, default=1e-4,
                        help="Learning rate for local reconstruction")
    parser.add_argument("--target-domain", type=str, default=None,
                        help="Target domain for domain-conditional compression "
                             "(e.g., english, math, code, italian, or a custom name)")
    parser.add_argument("--custom-domain-path", type=str, default=None,
                        help="Path to custom domain text file or HuggingFace dataset")
    parser.add_argument("--custom-domain-name", type=str, default=None,
                        help="Name for the custom domain (e.g., cybersecurity)")
    parser.add_argument("--domain-unnecessary-frac", type=float, default=0.05,
                        help="Max fraction of neurons to remove per layer for domain specialization")
    parser.add_argument("--domain-critical-frac", type=float, default=0.10,
                        help="Fraction of top domain-critical neurons to protect per layer")
    parser.add_argument("--enable-low-rank", action="store_true", default=True,
                        help="Enable low-rank factorization (default: on)")
    parser.add_argument("--disable-low-rank", action="store_true",
                        help="Disable low-rank factorization")
    parser.add_argument("--enable-static-fold", action="store_true", default=True,
                        help="Enable static neuron folding (default: on)")
    parser.add_argument("--disable-static-fold", action="store_true",
                        help="Disable static neuron folding")
    parser.add_argument("--enable-weight-sharing", action="store_true",
                        help="Enable weight sharing between similar layers (experimental)")

    # Output
    parser.add_argument("--output", type=str, default="./results",
                        help="Output directory")
    parser.add_argument("--save-model", action="store_true",
                        help="Save the compressed model")

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()

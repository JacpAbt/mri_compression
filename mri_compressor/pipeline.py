#!/usr/bin/env python3
"""
Unified MRI-to-Compression Pipeline.

Single entry point that chains:
  1. MRI Scan: Run diagnostic studies on the model
  2. Diagnosis: Analyze results, build per-layer compression prescription
  3. Compression: Apply the prescription to the model
  4. Evaluation: Measure compressed model quality

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PRESET MODES  (recommended — no need to memorise study numbers)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  general          – General MRI scan (studies 1,3,4,5,6,8,9,10,15,16,17,18,20)
                     No compression. Good first look at any model.

  compress         – General-purpose compression (same studies + compress stage)
                     Produces a smaller model preserving general capability.

  domain_scan      – Domain analysis: which neurons serve the target domain?
                     Requires --domain.  Studies: 1,5,11,22,24.

  domain_compress  – Domain-specialized compression (requires --domain)
                     Aggressively removes neurons irrelevant to the domain.
                     Studies: 1,3,4,5,6,8,9,10,11,22,24 + compress stage.

  full_scan        – All MRI studies (no compression). Research / exploration.

Examples:
  # General compression
  python -m mri_compressor.pipeline --model Qwen/Qwen3.5-0.8B-Base --mode compress

  # Biomedical specialisation
  python -m mri_compressor.pipeline --model Qwen/Qwen3.5-0.8B-Base \\
      --mode domain_compress --domain biomedical --save-model

  # Custom domain from a text file
  python -m mri_compressor.pipeline --model Qwen/Qwen3.5-0.8B-Base \\
      --mode domain_compress --domain cybersecurity \\
      --domain-dataset /path/to/cybersecurity.txt --save-model

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MANUAL MODE  (advanced — specify studies yourself)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  python -m mri_compressor.pipeline --model gpt2 --studies 1,5,10
  python -m mri_compressor.pipeline --model Qwen/Qwen2.5-3B \\
      --studies 1,3,4,5,6,8,9,10,11 --compress
  python -m mri_compressor.pipeline --model Qwen/Qwen2.5-3B \\
      --from-summary ./results/summary.json --compress
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


# ---------------------------------------------------------------------------
# Preset modes — study lists and defaults
# ---------------------------------------------------------------------------

# Built-in domain names that have their own data loaders (no path required).
_BUILTIN_DOMAINS = {"biomedical"}

# Studies that are always meaningful regardless of model/domain.
_CORE_STUDIES         = [1, 3, 4, 5, 6, 8, 9, 10]
_EXTENDED_STUDIES     = [15, 16, 17, 18, 20, 25]       # geometry / rank / cascade / write-vector
_DOMAIN_STUDIES       = [11, 22, 24, 25]               # need domain data (25 adds domain divergence)
_STRUCTURAL_STUDIES   = [12, 13, 14, 19, 21, 23]       # advanced / hybrid

PRESET_MODES = {
    "general": {
        "description": (
            "General MRI scan — identify compression opportunities "
            "without domain specialisation"
        ),
        "studies": _CORE_STUDIES + _EXTENDED_STUDIES,
        "compress": False,
        "requires_domain": False,
    },
    "compress": {
        "description": (
            "General-purpose MRI + compression — produces a smaller model "
            "that preserves general capability"
        ),
        "studies": _CORE_STUDIES + _EXTENDED_STUDIES,
        "compress": True,
        "enable_attn": True,
        "requires_domain": False,
    },
    "domain_scan": {
        "description": (
            "Domain-specific analysis — map which neurons serve a target domain, "
            "find safe per-layer pruning levels (requires --domain)"
        ),
        "studies": [1, 5, 11, 22, 24, 25],
        "compress": False,
        "requires_domain": True,
    },
    "domain_compress": {
        "description": (
            "Domain-specialised compression — aggressively removes neurons "
            "irrelevant to the target domain (requires --domain)"
        ),
        "studies": _CORE_STUDIES + _DOMAIN_STUDIES,
        "compress": True,
        "enable_attn": True,
        "requires_domain": True,
    },
    "full_scan": {
        "description": (
            "All MRI studies — comprehensive research scan, no compression"
        ),
        "studies": (
            _CORE_STUDIES
            + _EXTENDED_STUDIES
            + _STRUCTURAL_STUDIES
            + [2, 7, 11, 22]   # 24 added dynamically when --domain is given
        ),
        "compress": False,
        "requires_domain": False,
    },
}


def apply_preset_mode(args) -> None:
    """
    Expand a preset mode into concrete study list and flag defaults.
    Called before ExperimentConfig is built so the expanded values are
    available via `args.*`.

    If ``args.mode`` is None (legacy / manual mode), nothing is changed.
    """
    if args.mode is None:
        return

    mode = PRESET_MODES.get(args.mode)
    if mode is None:
        known = ", ".join(PRESET_MODES)
        raise SystemExit(
            f"Unknown --mode '{args.mode}'.  Known modes: {known}"
        )

    if mode.get("requires_domain") and not args.domain:
        raise SystemExit(
            f"--mode {args.mode} requires --domain <name>.  "
            f"Built-in choices: {', '.join(sorted(_BUILTIN_DOMAINS))}.  "
            f"Or supply any name with --domain-dataset."
        )

    # Expand study list — merge with any explicit --studies override
    if not args.studies:
        studies = list(mode["studies"])
        # full_scan: add Study 24 automatically when a domain is provided
        if args.mode == "full_scan" and args.domain:
            studies.append(24)
        args.studies = ",".join(str(s) for s in sorted(set(studies)))
    else:
        print(
            f"  [mode={args.mode}] --studies override is present; "
            f"using supplied study list instead of preset defaults."
        )

    # Set compression flag
    if mode.get("compress") and not args.compress:
        args.compress = True

    # Set attention pruning flag
    if mode.get("enable_attn") and not args.enable_attn:
        args.enable_attn = True

    # Wire --domain into the target_domain / custom_domain_name fields
    if args.domain:
        if not args.target_domain:
            args.target_domain = args.domain
        if not args.custom_domain_name:
            args.custom_domain_name = args.domain
        # If there's an explicit dataset path, wire it in
        if args.domain_dataset and not args.custom_domain_path:
            args.custom_domain_path = args.domain_dataset


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


def run_imprinting_stage(
    inspector: ModelInspector,
    config: ExperimentConfig,
    domain_dataloader=None,
    pre_computed_centroids=None,
    pre_computed_class_centroids=None,
    class_dataloaders=None,
) -> dict:
    """
    Stage 3.5 (optional): Domain Imprinting.

    Injects per-layer domain activation centroids into each down_proj bias.
    No retraining.  Partial recovery of domain PPL degradation from compression.

    Two modes (auto-selected based on available inputs):

    1. Class-conditional (preferred, biomedical domain):
       When *pre_computed_class_centroids* and *class_dataloaders* are both
       provided, computes per-class centroids on the compressed model and
       injects the importance-weighted mean delta.  Each class (yes/no/maybe/
       general) contributes proportionally to its weight — "maybe" at 3×
       because it is the most fragile under pruning.

    2. Global (fallback):
       Uses *pre_computed_centroids* (output-space centroids from the original
       model) or recomputes from the compressed model.  Corrects the global
       domain manifold displacement but not per-class geometry.

    Returns the imprinting report dict (pre/post PPL, recovery %, per-layer).
    """
    from .compression.compressor import MRICompressor

    print("\n" + "=" * 80)
    print("STAGE 3.5: DOMAIN IMPRINTING")
    if pre_computed_class_centroids is not None and class_dataloaders is not None:
        imprint_mode = "class-conditional"
        centroid_src = f"original model ({len(pre_computed_class_centroids)} classes)"
    elif pre_computed_centroids is not None:
        imprint_mode = "global"
        centroid_src = "original model (global)"
    else:
        imprint_mode = "global"
        centroid_src = "compressed model (fresh)"
    print(f"  Mode: {imprint_mode}  |  Scale: {config.imprinting_scale:.3f}"
          f"  |  Domain: {config.target_domain}  |  Centroids from: {centroid_src}")
    print("=" * 80)

    if domain_dataloader is None:
        domain_dataloader = _create_domain_dataloader(
            config.target_domain, inspector.tokenizer,
            max_seq_len=config.max_length, batch_size=config.batch_size,
            custom_path=config.custom_domain_path,
            custom_name=config.custom_domain_name,
        )

    # Class-conditional mode: domain_dataloader is not strictly required
    # (class_dataloaders replaces it for PPL measurement).
    # Fall back to loading a regular domain loader only for global mode.
    if class_dataloaders is not None and pre_computed_class_centroids is not None:
        # Class-conditional path: domain_dataloader may be None — use first class DL for PPL
        _dl_for_ppl = next(iter(class_dataloaders.values()), None)
        if _dl_for_ppl is None:
            print("  Warning: class dataloaders empty — imprinting skipped.")
            return {}
    else:
        if domain_dataloader is None:
            print("  Warning: domain dataloader unavailable — imprinting skipped.")
            return {}

    # Build a minimal compressor wrapper
    from .data_utils import load_wikitext_data, get_dataloader
    dataset = load_wikitext_data(
        inspector.tokenizer,
        max_seq_len=config.max_length,
        num_samples=config.max_samples,
    )
    dataloader = get_dataloader(dataset, batch_size=config.batch_size)

    # Use any available domain dataloader for the compressor's internal reference
    _any_domain_dl = domain_dataloader or (
        next(iter(class_dataloaders.values())) if class_dataloaders else None
    )

    from .compression.prescription import CompressionPrescription
    dummy_prescription = CompressionPrescription(
        model_name=config.model_name,
        baseline_ppl=0.0,
        num_layers=inspector.num_layers,
        intermediate_size=inspector.mlp_layers[0].intermediate_size,
        layers=[],
    )

    compressor = MRICompressor(
        model=inspector.model,
        tokenizer=inspector.tokenizer,
        prescription=dummy_prescription,
        calibration_dataloader=dataloader,
        device=inspector.device,
        domain_calibration_dataloader=_any_domain_dl,
        inspector=inspector,
        enable_imprinting=True,
        imprinting_scale=config.imprinting_scale,
    )

    # Choose imprinting mode
    if class_dataloaders is not None and pre_computed_class_centroids is not None:
        # Preferred: class-conditional geometry restoration
        report = compressor.apply_class_conditional_imprinting(
            original_class_centroids=pre_computed_class_centroids,
            class_dataloaders=class_dataloaders,
            scale=config.imprinting_scale,
        )
    else:
        # Fallback: global domain centroid imprinting
        report = compressor.apply_domain_imprinting(
            domain_dataloader=domain_dataloader,
            scale=config.imprinting_scale,
            pre_computed_centroids=pre_computed_centroids,
        )
        report["mode"] = "global"

    return report


def run_compression_stage(
    inspector: ModelInspector,
    prescription: object,
    config: ExperimentConfig,
    summary: dict,
) -> tuple:
    """
    Stage 3: Apply compression prescription to the model.

    Returns
    -------
    (result, original_centroids)
        result              : CompressionResult from MRICompressor.compress()
        original_centroids  : dict[layer_idx -> Tensor(hidden_size,)] computed
                              from the original model BEFORE compression, or
                              None if imprinting is disabled / no domain set.
                              These are passed to run_imprinting_stage() so the
                              cleaner, undamaged domain signal is used.
    """
    from .compression.compressor import MRICompressor
    from .compression.operations.imprinting import DomainImprinter

    print("\n" + "=" * 80)
    print("STAGE 3: COMPRESSION")
    print("=" * 80)

    dataset = load_wikitext_data(
        inspector.tokenizer,
        max_seq_len=config.max_length,
        num_samples=config.max_samples,
    )
    dataloader = get_dataloader(dataset, batch_size=config.batch_size)

    # Domain dataloader (used for both reconstruction and centroid capture)
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

    # ------------------------------------------------------------------ #
    #  Capture original-model centroids BEFORE compression destroys them  #
    # ------------------------------------------------------------------ #
    original_centroids       = None
    original_class_centroids = None
    class_dataloaders        = None

    if config.enable_imprinting and domain_dataloader is not None:
        # Global output-space centroids (dimension-stable, works for any domain)
        print("\n  Pre-computing domain centroids from original model (output space)...")
        original_centroids = DomainImprinter.compute_output_centroids(
            model=inspector.model,
            inspector=inspector,
            domain_dataloader=domain_dataloader,
            device=inspector.device,
            max_batches=16,
        )
        print(f"  Global centroids captured for {len(original_centroids)} layers.")

        # Class-conditional centroids (biomedical: yes/no/maybe/general)
        # These allow the imprinting step to restore per-class geometry, not
        # just the global domain mean — particularly important for protecting
        # the fragile "maybe" uncertainty circuit.
        if config.target_domain == "biomedical":
            try:
                from .mri.studies_domain_compression import load_class_biomedical_datasets
                print("\n  Pre-computing per-class domain centroids from original model...")
                class_datasets = load_class_biomedical_datasets(
                    inspector.tokenizer,
                    max_seq_len=config.max_length,
                    n_samples=32,
                )
                if class_datasets:
                    class_dataloaders = {
                        cls_name: get_dataloader(ds, batch_size=config.batch_size)
                        for cls_name, ds in class_datasets.items()
                    }
                    original_class_centroids = DomainImprinter.compute_class_output_centroids(
                        model=inspector.model,
                        inspector=inspector,
                        class_dataloaders=class_dataloaders,
                        device=inspector.device,
                        max_batches=16,
                    )
                    print(
                        f"  Per-class centroids captured: "
                        f"{list(original_class_centroids.keys())}"
                    )
                else:
                    print("  Warning: per-class dataset loading returned empty — "
                          "falling back to global imprinting.")
            except Exception as exc:
                print(
                    f"  Warning: per-class centroid computation failed ({exc}) — "
                    f"falling back to global imprinting."
                )

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
        inspector=inspector,
    )

    result = compressor.compress()
    return result, original_centroids, original_class_centroids, class_dataloaders


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

    # ---- Built-in custom domains (have their own loaders, no path required) ----
    if domain_name in _BUILTIN_DOMAINS:
        try:
            if domain_name == "biomedical":
                from .mri.studies_domain_compression import load_biomedical_dataset
                dataset = load_biomedical_dataset(
                    tokenizer, max_seq_len=max_seq_len, n_samples=128)
                return get_dataloader(dataset, batch_size=batch_size)
        except Exception as e:
            print(f"  Warning: Failed to load built-in domain '{domain_name}': {e}")
            return None

    # ---- Custom domain with explicit path ----
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
                        text = "\n".join(
                            str(row[field_name]) for row in ds
                            if len(str(row[field_name])) > 50
                        )
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

    # ---- Standard built-in domains (english / math / code / italian) ----
    try:
        from .mri.studies_domain import load_domain_datasets
        domain_datasets = load_domain_datasets(
            tokenizer, max_seq_len=max_seq_len, samples_per_domain=128,
        )
        if domain_name in domain_datasets:
            return get_dataloader(domain_datasets[domain_name], batch_size=batch_size)
        else:
            print(
                f"  Warning: Domain '{domain_name}' not found in standard domains "
                f"({list(domain_datasets.keys())}).  "
                f"Use --domain-dataset to provide custom text."
            )
            return None
    except Exception as e:
        print(f"  Warning: Failed to load domain datasets: {e}")
        return None


def run_evaluation_stage(
    inspector: ModelInspector,
    config: ExperimentConfig,
    baseline_ppl: float,
    domain_baseline_ppl=None,
    baseline_benchmark: dict = None,
) -> dict:
    """Stage 4: Evaluate compressed model (PPL + optional domain benchmark)."""
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

    # Count parameters
    total_params = sum(p.numel() for p in inspector.model.parameters())

    # ---- Domain PPL (primary metric for domain_compress mode) ----
    domain_compressed_ppl = None
    if config.target_domain:
        print(f"\n  Loading {config.target_domain} domain dataset for evaluation...")
        domain_loader = _create_domain_dataloader(
            config.target_domain, inspector.tokenizer,
            max_seq_len=config.max_length, batch_size=config.batch_size,
            custom_path=config.custom_domain_path,
            custom_name=config.custom_domain_name,
        )
        if domain_loader is not None:
            domain_compressed_ppl = evaluate_perplexity(
                inspector.model, domain_loader, inspector.device,
                max_batches=8,
            )

    # ---- Domain benchmark (accuracy on downstream task) ----
    compressed_benchmark = None
    if config.target_domain:
        from .benchmark import run_domain_benchmark, DOMAIN_BENCHMARKS
        if config.target_domain in DOMAIN_BENCHMARKS:
            compressed_benchmark = run_domain_benchmark(
                config.target_domain,
                inspector.model,
                inspector.tokenizer,
                inspector.device,
                n_samples=200,
            )

    # ---- Print ----
    if config.target_domain and domain_compressed_ppl is not None:
        # Domain mode: domain PPL is the headline; wikitext shown as tradeoff reference
        if domain_baseline_ppl is not None:
            dom_delta = domain_compressed_ppl - domain_baseline_ppl
            dom_delta_pct = (dom_delta / domain_baseline_ppl) * 100
            print(f"\n  Domain ({config.target_domain}) PPL:  "
                  f"{domain_baseline_ppl:.2f} → {domain_compressed_ppl:.2f}"
                  f"  ({dom_delta_pct:+.1f}%)")
        else:
            print(f"\n  Domain ({config.target_domain}) PPL (compressed):  {domain_compressed_ppl:.2f}")
        print(f"  Generic PPL (wikitext):  {baseline_ppl:.2f} → {compressed_ppl:.2f}"
              f"  ({ppl_increase_pct:+.1f}%)")
        print(f"  (Generic PPL increase is expected — neurons irrelevant to"
              f" {config.target_domain} were removed)")
    else:
        print(f"\n  Baseline PPL:    {baseline_ppl:.2f}")
        print(f"  Compressed PPL:  {compressed_ppl:.2f}")
        print(f"  PPL increase:    {ppl_increase:.2f} ({ppl_increase_pct:+.1f}%)")

    # Print benchmark comparison if we have both baseline and compressed scores
    if compressed_benchmark and baseline_benchmark:
        base_acc = baseline_benchmark.get("accuracy", 0.0)
        comp_acc = compressed_benchmark.get("accuracy", 0.0)
        delta    = comp_acc - base_acc
        print(f"\n  Benchmark ({compressed_benchmark.get('dataset', config.target_domain)}):")
        print(f"    Accuracy:  {base_acc:.1%} → {comp_acc:.1%}  ({delta:+.1%})")
    elif compressed_benchmark:
        comp_acc = compressed_benchmark.get("accuracy", 0.0)
        print(f"\n  Benchmark ({compressed_benchmark.get('dataset', config.target_domain)}):")
        print(f"    Compressed accuracy: {comp_acc:.1%}")

    print(f"  Total parameters: {total_params:,}")

    eval_report = {
        "baseline_ppl": baseline_ppl,
        "compressed_ppl": compressed_ppl,
        "ppl_increase": ppl_increase,
        "ppl_increase_pct": ppl_increase_pct,
        "total_parameters": total_params,
    }
    if config.target_domain and domain_compressed_ppl is not None:
        eval_report["domain"] = config.target_domain
        eval_report["domain_baseline_ppl"] = domain_baseline_ppl
        eval_report["domain_compressed_ppl"] = domain_compressed_ppl
    if compressed_benchmark:
        eval_report["benchmark"] = compressed_benchmark
    if baseline_benchmark:
        eval_report["baseline_benchmark"] = baseline_benchmark

    # Save evaluation report
    report_path = os.path.join(config.output_dir, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(eval_report, f, indent=2)
    print(f"\n  Evaluation report saved to {report_path}")

    return eval_report


def run_pipeline(args):
    """Run the full or partial pipeline."""
    t_start = time.time()

    # ---- Expand preset mode into concrete args FIRST ----
    apply_preset_mode(args)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Print what we're about to do
    if args.mode:
        mode_desc = PRESET_MODES[args.mode]["description"]
        print(f"\n  Mode: {args.mode}  —  {mode_desc}")
        if getattr(args, "domain", None):
            print(f"  Domain: {args.domain}")

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
        target_domain=getattr(args, "target_domain", None),
        custom_domain_path=getattr(args, "custom_domain_path", None),
        custom_domain_name=getattr(args, "custom_domain_name", None),
        domain_unnecessary_frac=getattr(args, "domain_unnecessary_frac", 0.05),
        domain_critical_frac=getattr(args, "domain_critical_frac", 0.10),
        enable_low_rank=not args.disable_low_rank,
        enable_static_fold=not args.disable_static_fold,
        enable_weight_sharing=args.enable_weight_sharing,
        enable_imprinting=getattr(args, "imprint", False),
        imprinting_scale=getattr(args, "imprint_scale", 0.05),
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
        _s24 = summary.get("aggregated", {}).get("study24", {})
        domain_baseline_ppl = float(_s24["baseline_ppl"]) if _s24.get("baseline_ppl") else None
    else:
        if not args.studies:
            raise SystemExit(
                "No studies specified.  Use --mode <mode> or --studies <numbers>."
            )
        # Parse study list
        studies = [int(s.strip()) for s in args.studies.split(",")]
        print(f"  Studies: {sorted(studies)}")

        # Stage 1: MRI
        summary = run_mri_stage(config, inspector, studies)
        summary["output_dir"] = config.output_dir
        baseline_ppl = summary.get("baseline_ppl", 0)
        _s24 = summary.get("aggregated", {}).get("study24", {})
        domain_baseline_ppl = float(_s24["baseline_ppl"]) if _s24.get("baseline_ppl") else None

    # Stage 2-4: Compression pipeline (if requested)
    if args.compress:
        if summary is None:
            print("ERROR: No MRI summary available. Run studies or provide --from-summary.")
            return

        # Stage 2: Diagnosis
        prescription = run_diagnosis_stage(summary, config)

        # Baseline benchmark — measured on the ORIGINAL model before compression
        # so we can report a clean before/after comparison in Stage 4.
        baseline_benchmark = None
        if config.target_domain:
            from .benchmark import run_domain_benchmark, DOMAIN_BENCHMARKS
            if config.target_domain in DOMAIN_BENCHMARKS:
                print(f"\n  Measuring baseline benchmark ({config.target_domain}) "
                      f"on original model...")
                baseline_benchmark = run_domain_benchmark(
                    config.target_domain,
                    inspector.model,
                    inspector.tokenizer,
                    inspector.device,
                    n_samples=200,
                )

        # Stage 3: Compression
        # Returns (result, original_centroids, original_class_centroids, class_dataloaders).
        # The last two are populated for biomedical domain + imprinting enabled;
        # they are None otherwise (graceful fallback to global imprinting).
        result, original_centroids, original_class_centroids, class_dataloaders = (
            run_compression_stage(inspector, prescription, config, summary)
        )

        # Stage 3.5: Domain Imprinting (optional, requires --imprint and --domain)
        imprint_report = {}
        if config.enable_imprinting and config.target_domain:
            imprint_report = run_imprinting_stage(
                inspector, config,
                pre_computed_centroids=original_centroids,
                pre_computed_class_centroids=original_class_centroids,
                class_dataloaders=class_dataloaders,
            )
            # Persist imprinting report alongside the evaluation report
            imprint_path = os.path.join(config.output_dir, "imprinting_report.json")
            with open(imprint_path, "w") as _f:
                import json as _json
                _json.dump(imprint_report, _f, indent=2, default=str)
            print(f"\n  Imprinting report saved to {imprint_path}")

        # Stage 4: Evaluation
        eval_report = run_evaluation_stage(
            inspector, config, baseline_ppl,
            domain_baseline_ppl=domain_baseline_ppl,
            baseline_benchmark=baseline_benchmark,
        )

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
    mode_list = "\n".join(
        f"    {name:20s}  {info['description']}"
        for name, info in PRESET_MODES.items()
    )

    parser = argparse.ArgumentParser(
        description="MRI-to-Compression Pipeline for LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
AVAILABLE MODES
{mode_list}

EXAMPLES
  # Biomedical specialisation (auto-downloads PubMed data)
  python -m mri_compressor.pipeline --model Qwen/Qwen3.5-0.8B-Base \\
      --mode domain_compress --domain biomedical --save-model

  # Custom domain from a local text file
  python -m mri_compressor.pipeline --model Qwen/Qwen3.5-0.8B-Base \\
      --mode domain_compress --domain legal --domain-dataset /data/legal.txt

  # General compression
  python -m mri_compressor.pipeline --model Qwen/Qwen3.5-0.8B-Base --mode compress

  # Manual study list (legacy / advanced)
  python -m mri_compressor.pipeline --model gpt2 --studies 1,5,10
        """,
    )

    # ---- Model ----
    parser.add_argument("--model", type=str, default="gpt2",
                        help="HuggingFace model name or local path")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto | cuda | cpu")

    # ---- Mode (recommended) ----
    parser.add_argument(
        "--mode", type=str, default=None,
        choices=list(PRESET_MODES),
        metavar="MODE",
        help=(
            "Preset pipeline mode.  Choices: "
            + ", ".join(PRESET_MODES)
            + "  (see epilog for descriptions)"
        ),
    )

    # ---- Domain (for domain_scan / domain_compress modes) ----
    parser.add_argument(
        "--domain", type=str, default=None,
        metavar="NAME",
        help=(
            "Target domain for domain-aware modes.  "
            "Built-in: biomedical.  "
            "Standard: english, math, code, italian.  "
            "Custom: any name combined with --domain-dataset."
        ),
    )
    parser.add_argument(
        "--domain-dataset", type=str, default=None,
        dest="domain_dataset",
        metavar="PATH",
        help=(
            "Path to a custom domain text file or HuggingFace dataset id.  "
            "Used when --domain is not one of the built-in names."
        ),
    )

    # ---- MRI (manual / advanced) ----
    parser.add_argument("--studies", type=str, default=None,
                        help="Comma-separated study numbers (overrides --mode preset)")
    parser.add_argument("--from-summary", type=str, default=None,
                        help="Skip MRI, load existing summary.json and go straight to compression")

    # ---- Data ----
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--max-length", type=int, default=512)

    # ---- Compression flags ----
    parser.add_argument("--compress", action="store_true",
                        help="Run compression after MRI (automatically set by some modes)")
    parser.add_argument("--enable-attn", action="store_true",
                        help="Enable attention head pruning")
    parser.add_argument("--enable-depth", action="store_true",
                        help="Enable depth pruning (remove entire layers)")
    parser.add_argument("--disable-merge", action="store_true",
                        help="Disable neuron merging")
    parser.add_argument("--reconstruction-steps", type=int, default=200,
                        help="Local reconstruction fine-tuning steps per layer (default: 200)")
    parser.add_argument("--reconstruction-lr", type=float, default=1e-4,
                        help="Learning rate for local reconstruction (default: 1e-4)")

    # ---- Domain compression fine-tuning (advanced) ----
    parser.add_argument("--target-domain", type=str, default=None,
                        help="Override target domain for compression (set automatically by --domain)")
    parser.add_argument("--custom-domain-path", type=str, default=None,
                        help="Override custom domain path (set automatically by --domain-dataset)")
    parser.add_argument("--custom-domain-name", type=str, default=None,
                        help="Override custom domain name (set automatically by --domain)")
    parser.add_argument("--domain-unnecessary-frac", type=float, default=0.05,
                        help="Fallback fraction of neurons to remove per layer when Study 24 is "
                             "unavailable (default: 0.05 = 5%%)")
    parser.add_argument("--domain-critical-frac", type=float, default=0.10,
                        help="Fraction of top domain-critical neurons to protect per layer "
                             "(default: 0.10 = 10%%)")

    # ---- Module toggles ----
    parser.add_argument("--disable-low-rank", action="store_true",
                        help="Disable low-rank MLP factorization")
    parser.add_argument("--disable-static-fold", action="store_true",
                        help="Disable static neuron folding")
    parser.add_argument("--enable-weight-sharing", action="store_true",
                        help="Enable weight sharing between similar layers (experimental)")

    # ---- Domain Imprinting ----
    parser.add_argument(
        "--imprint", action="store_true", dest="imprint",
        help=(
            "After compression, inject per-layer domain activation centroids into "
            "down_proj biases (domain imprinting).  Requires --domain.  "
            "No retraining — pure bias shift toward the domain manifold."
        ),
    )
    parser.add_argument(
        "--imprint-scale", type=float, default=0.05, dest="imprint_scale",
        metavar="S",
        help=(
            "Scale factor for domain centroid injection (default: 0.05).  "
            "Typical range 0.01–0.10.  Higher = stronger domain prior, "
            "may hurt general PPL."
        ),
    )

    # ---- Output ----
    parser.add_argument("--output", type=str, default="./results",
                        help="Output directory (default: ./results)")
    parser.add_argument("--save-model", action="store_true",
                        help="Save the compressed model to <output>/compressed_model/")

    args = parser.parse_args()

    # Validate: need either --mode or --studies or --from-summary
    if args.mode is None and args.studies is None and args.from_summary is None:
        parser.error(
            "Specify at least one of: --mode <mode>, --studies <list>, or --from-summary <path>"
        )

    run_pipeline(args)


if __name__ == "__main__":
    main()

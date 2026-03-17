"""
mri_compressor — Diagnostic-guided LLM compression without quantization.

Quick start:

    import mri_compressor

    # Run MRI studies only
    summary = mri_compressor.run_mri("gpt2", studies=[1, 3, 4, 5, 6, 8, 9, 10])

    # Compress from an existing MRI summary
    result = mri_compressor.compress("gpt2", summary)

    # Full pipeline in one call
    result = mri_compressor.run_full_pipeline("gpt2")

    # Pass a pre-loaded model instead of a name
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    summary = mri_compressor.run_mri((model, tokenizer), studies=[1, 3, 5])

Advanced usage — access the underlying classes directly:

    from mri_compressor import ExperimentConfig, MRIRunner, MRIDiagnostician, MRICompressor
"""

__version__ = "0.1.0"

# --- Public class exports (advanced / composable usage) ---
from .config import ExperimentConfig
from .model_utils import ModelInspector
from .mri.runner import MRIRunner
from .compression.diagnostician import MRIDiagnostician
from .compression.compressor import MRICompressor
from .compression.prescription import CompressionPrescription, CompressionStrategy

# --- Default study set used when studies=None ---
_DEFAULT_STUDIES = [1, 3, 4, 5, 6, 8, 9, 10]


def _make_inspector(model, device: str) -> "ModelInspector":
    """Accept either a model-name string or a (model, tokenizer) tuple."""
    if isinstance(model, str):
        return ModelInspector(model, device=device)
    # Pre-loaded: (model, tokenizer) tuple — bypass __init__ and call _detect_architecture
    import torch
    model_obj, tokenizer = model
    inspector = ModelInspector.__new__(ModelInspector)
    inspector.model_name = getattr(model_obj.config, "_name_or_path", "custom")
    inspector.device = device
    inspector.dtype = next(model_obj.parameters()).dtype
    inspector.model = model_obj.to(device)
    inspector.model.eval()
    inspector.tokenizer = tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    inspector._detect_architecture()
    return inspector


def run_mri(
    model,
    studies=None,
    output_dir: str = "./results",
    device: str = "cuda",
    batch_size: int = 4,
    max_batches: int = 16,
    max_samples: int = 256,
) -> dict:
    """Run MRI diagnostic studies on a model.

    Parameters
    ----------
    model:
        HuggingFace model ID string (e.g. ``"gpt2"``, ``"Qwen/Qwen2.5-0.5B"``)
        **or** a ``(model, tokenizer)`` tuple for pre-loaded models.
    studies:
        List of study IDs to run, e.g. ``[1, 3, 4, 5]``.
        Defaults to ``[1, 3, 4, 5, 6, 8, 9, 10]``.
        Available studies: 1-11, 12-22 (see README for descriptions).
    output_dir:
        Directory where plots and ``summary.json`` are saved.
    device:
        PyTorch device string. Defaults to ``"cuda"``.
    batch_size:
        Calibration dataloader batch size.
    max_batches:
        Maximum number of calibration batches per study.
    max_samples:
        Number of WikiText samples to use for calibration data.

    Returns
    -------
    dict
        Full MRI summary including ``baseline_ppl``, architecture info, and
        per-layer results for every study that was run.
    """
    from .pipeline import run_mri_stage

    if studies is None:
        studies = _DEFAULT_STUDIES

    model_name = model if isinstance(model, str) else getattr(
        model[0].config, "_name_or_path", "custom"
    )

    config = ExperimentConfig(
        model_name=model_name,
        output_dir=output_dir,
        device=device,
        batch_size=batch_size,
        max_batches=max_batches,
        max_samples=max_samples,
    )

    inspector = _make_inspector(model, device)
    return run_mri_stage(config, inspector, studies)


def compress(
    model,
    mri_summary: dict,
    output_dir: str = "./results",
    device: str = "cuda",
    batch_size: int = 4,
    enable_attn_pruning: bool = True,
    enable_depth_pruning: bool = False,
    enable_low_rank: bool = True,
    enable_static_fold: bool = True,
    enable_merge: bool = True,
    target_domain: str = None,
    save_path: str = None,
) -> dict:
    """Apply MRI-guided compression to a model.

    Parameters
    ----------
    model:
        HuggingFace model ID string **or** a ``(model, tokenizer)`` tuple.
    mri_summary:
        Summary dict returned by :func:`run_mri`.
    output_dir:
        Directory for saving evaluation results.
    device:
        PyTorch device string. Defaults to ``"cuda"``.
    batch_size:
        Batch size for calibration data during reconstruction.
    enable_attn_pruning:
        Whether to prune low-importance attention heads.
    enable_depth_pruning:
        Whether to remove entire redundant layers (disabled by default; high risk).
    enable_low_rank:
        Whether to apply SVD-based low-rank factorization to MLP weights.
    enable_static_fold:
        Whether to fold static neurons into bias terms.
    enable_merge:
        Whether to merge similar neurons.
    target_domain:
        Optional domain name for domain-conditional compression (e.g. ``"math"``).
    save_path:
        If set, save the compressed model to this directory via
        ``model.save_pretrained(save_path)``.

    Returns
    -------
    dict
        Compression result with keys ``original_ppl``, ``compressed_ppl``,
        ``param_reduction_pct``, and per-layer statistics.
    """
    from .pipeline import run_diagnosis_stage, run_compression_stage

    model_name = model if isinstance(model, str) else getattr(
        model[0].config, "_name_or_path", "custom"
    )

    config = ExperimentConfig(
        model_name=model_name,
        output_dir=output_dir,
        device=device,
        batch_size=batch_size,
        enable_compression=True,
        enable_attn_pruning=enable_attn_pruning,
        enable_depth_pruning=enable_depth_pruning,
        enable_low_rank=enable_low_rank,
        enable_static_fold=enable_static_fold,
        enable_merge=enable_merge,
        target_domain=target_domain,
        save_compressed_model=save_path is not None,
    )

    inspector = _make_inspector(model, device)
    prescription = run_diagnosis_stage(mri_summary, config)
    result = run_compression_stage(inspector, prescription, config, mri_summary)

    if save_path:
        inspector.model.save_pretrained(save_path)
        inspector.tokenizer.save_pretrained(save_path)
        print(f"Compressed model saved to: {save_path}")

    return {
        "original_ppl": result.original_ppl,
        "compressed_ppl": result.compressed_ppl,
        "param_reduction_pct": (
            1.0 - result.total_params_compressed / result.total_params_original
        ) * 100 if hasattr(result, "total_params_compressed") else None,
        "result": result,
    }


def run_full_pipeline(
    model,
    studies=None,
    output_dir: str = "./results",
    device: str = "cuda",
    batch_size: int = 4,
    max_batches: int = 16,
    max_samples: int = 256,
    enable_attn_pruning: bool = True,
    enable_depth_pruning: bool = False,
    enable_low_rank: bool = True,
    enable_static_fold: bool = True,
    enable_merge: bool = True,
    target_domain: str = None,
    save_path: str = None,
) -> dict:
    """Run the full MRI → Diagnosis → Compression pipeline.

    Convenience wrapper that calls :func:`run_mri` followed by :func:`compress`.

    Returns
    -------
    dict
        Combined result with keys ``mri_summary`` and ``compression`` (the dict
        returned by :func:`compress`).
    """
    summary = run_mri(
        model,
        studies=studies,
        output_dir=output_dir,
        device=device,
        batch_size=batch_size,
        max_batches=max_batches,
        max_samples=max_samples,
    )

    compression_result = compress(
        model,
        summary,
        output_dir=output_dir,
        device=device,
        batch_size=batch_size,
        enable_attn_pruning=enable_attn_pruning,
        enable_depth_pruning=enable_depth_pruning,
        enable_low_rank=enable_low_rank,
        enable_static_fold=enable_static_fold,
        enable_merge=enable_merge,
        target_domain=target_domain,
        save_path=save_path,
    )

    return {
        "mri_summary": summary,
        "compression": compression_result,
    }


__all__ = [
    # High-level functions
    "run_mri",
    "compress",
    "run_full_pipeline",
    # Classes for advanced usage
    "ExperimentConfig",
    "ModelInspector",
    "MRIRunner",
    "MRIDiagnostician",
    "MRICompressor",
    "CompressionPrescription",
    "CompressionStrategy",
]

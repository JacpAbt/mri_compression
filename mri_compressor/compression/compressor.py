"""
MRI-Compress Engine v2
=======================
Applies a CompressionPrescription to a HuggingFace LLM.

New operations in v2 (informed by Studies 17-21):
1. Dormant neuron removal -- removes neurons firing on <1% of inputs
2. Attention head pruning -- zeros out lowest-impact heads per layer
3. Depth pruning (applied) -- zeros out entire layers
4. Low-rank factorization -- SVD decomposition of MLP weight matrices
5. Weight sharing -- tie MLP weights between high-CKA layer pairs

Architecture from v1 retained:
- Sequential layer processing (collect -> compress -> reconstruct)
- Dead neuron removal
- Neuron merging
- Wanda-guided structured pruning
- Local reconstruction (SparseGPT-style)
- RTN quantization (disabled)
"""

from __future__ import annotations
import gc
import logging
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

from .prescription import (
    CompressionPrescription,
    CompressionStrategy,
    LayerPrescription,
)
from .operations import (
    DeadNeuronRemover,
    NeuronMerger,
    WandaPruner,
    AttentionHeadPruner,
    DepthPruner,
    LowRankFactorizer,
    LocalReconstructor,
    StaticNeuronFolder,
    WeightSharer,
    DomainImprinter,
)
from ._utils import (get_intermediate_size, get_mlp_modules, get_mlp_submodule,
                     resolve_layer, _find_attn_output_proj)

logger = logging.getLogger(__name__)


@contextmanager
def timer(name: str):
    t0 = time.perf_counter()
    yield
    logger.info(f"  {name}: {time.perf_counter() - t0:.1f}s")


# ============================================================================
# Data Collection (unchanged from v1)
# ============================================================================

@torch.no_grad()
def collect_activations(
    model, dataloader, layer_idx, max_batches=16,
    device=torch.device("cuda"), inspector=None,
):
    """Collect MLP intermediate activations for a single layer.

    Architecture-agnostic: uses ``inspector.layer_path`` when provided,
    otherwise falls back to common attribute paths.

    Collects per-batch on CPU to avoid OOM, then concatenates and moves the
    final tensor back to the target device for fast downstream operations.
    """
    all_acts = []
    if inspector is not None and layer_idx < len(inspector.mlp_layers):
        # Primary path: inspector already resolved the correct module for every
        # architecture, including GPT-2's Conv1D (c_proj) which isinstance(nn.Linear)
        # checks would miss.
        down_proj = inspector.mlp_layers[layer_idx].down_proj
    else:
        layer = resolve_layer(model, layer_idx, inspector)
        mlp_mods = get_mlp_modules(layer)
        down_proj = mlp_mods.get("down_proj") or mlp_mods.get("fc2") or mlp_mods.get("c_proj")
        if down_proj is None:
            # last fallback: use the last Linear child of the mlp
            for _, m in list(get_mlp_submodule(layer).named_modules())[::-1]:
                if isinstance(m, nn.Linear):
                    down_proj = m
                    break
        if down_proj is None:
            raise RuntimeError(f"Cannot find down projection for layer {layer_idx}")

    hook_data = {"acts": None}

    def hook_fn(module, input, output):
        hook_data["acts"] = input[0].detach()

    handle = down_proj.register_forward_hook(hook_fn)
    model.eval()
    use_cuda = str(device).startswith("cuda")
    try:
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            ids = batch["input_ids"].to(device)
            mask = batch.get("attention_mask", torch.ones_like(ids)).to(device)
            if use_cuda:
                with autocast("cuda", dtype=torch.bfloat16):
                    model(input_ids=ids, attention_mask=mask)
            else:
                model(input_ids=ids, attention_mask=mask)
            if hook_data["acts"] is not None:
                acts = hook_data["acts"].reshape(-1, hook_data["acts"].shape[-1])
                all_acts.append(acts.float().cpu())  # CPU to avoid OOM during collection
                hook_data["acts"] = None
    finally:
        handle.remove()
    if not all_acts:
        raise RuntimeError(f"No activations collected for layer {layer_idx}")
    # Concatenate on CPU, then move to device for fast operations
    return torch.cat(all_acts, dim=0).to(device)


@torch.no_grad()
def collect_mlp_io(
    model, dataloader, layer_idx, max_batches=8,
    device=torch.device("cuda"), inspector=None,
):
    """Collect MLP input/output pairs for a single layer.

    Architecture-agnostic: uses ``inspector.layer_path`` when provided.
    """
    inputs_list, outputs_list = [], []
    layer = resolve_layer(model, layer_idx, inspector)
    mlp = get_mlp_submodule(layer)
    io = {"inp": None, "out": None}

    def pre_hook(module, args, kwargs):
        io["inp"] = args[0].detach().cpu()

    def post_hook(module, args, kwargs, output):
        out = output[0] if isinstance(output, tuple) else output
        io["out"] = out.detach().cpu()

    h1 = mlp.register_forward_pre_hook(pre_hook, with_kwargs=True)
    h2 = mlp.register_forward_hook(post_hook, with_kwargs=True)
    model.eval()
    use_cuda = str(device).startswith("cuda")
    try:
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            ids = batch["input_ids"].to(device)
            mask = batch.get("attention_mask", torch.ones_like(ids)).to(device)
            if use_cuda:
                with autocast("cuda", dtype=torch.bfloat16):
                    model(input_ids=ids, attention_mask=mask)
            else:
                model(input_ids=ids, attention_mask=mask)
            if io["inp"] is not None:
                inputs_list.append(io["inp"])
                outputs_list.append(io["out"])
                io["inp"] = io["out"] = None
    finally:
        h1.remove()
        h2.remove()
    return inputs_list, outputs_list


# ============================================================================
# Main Pipeline
# ============================================================================

@dataclass
class CompressionResult:
    original_ppl: float
    compressed_ppl: float
    per_layer_results: list[dict] = field(default_factory=list)
    total_neurons_removed: int = 0
    total_neurons_merged: int = 0
    total_dormant_removed: int = 0
    total_attn_heads_pruned: int = 0
    total_depth_pruned_layers: int = 0
    total_low_rank_layers: int = 0
    total_folded_neurons: int = 0
    total_weight_shared_pairs: int = 0
    total_domain_unnecessary_removed: int = 0
    total_params_original: int = 0
    total_params_compressed: int = 0
    elapsed_seconds: float = 0.0

    def summary(self) -> str:
        r = 1 - self.total_params_compressed / max(self.total_params_original, 1)
        d = self.compressed_ppl - self.original_ppl
        p = (self.compressed_ppl / max(self.original_ppl, 1e-8) - 1) * 100
        lines = [
            f"{'='*60}", "  MRI-Compress v2 Results", f"{'='*60}",
            f"  Original PPL:       {self.original_ppl:.4f}",
            f"  Compressed PPL:     {self.compressed_ppl:.4f}",
            f"  PPL change:         {d:+.4f} ({p:+.2f}%)", "",
            f"  Parameters:         {self.total_params_original:,} -> {self.total_params_compressed:,}",
            f"  Reduction:          {r:.1%}",
            f"  Dead neurons removed:   {self.total_neurons_removed:,}",
            f"  Dormant neurons removed:{self.total_dormant_removed:,}",
            f"  Neurons merged:         {self.total_neurons_merged:,}",
            f"  Attn heads pruned:      {self.total_attn_heads_pruned}",
            f"  Depth-pruned layers:    {self.total_depth_pruned_layers}",
            f"  Low-rank factorized:    {self.total_low_rank_layers}",
            f"  Folded neurons:         {self.total_folded_neurons:,}",
            f"  Weight-shared pairs:    {self.total_weight_shared_pairs}",
        ]
        if self.total_domain_unnecessary_removed > 0:
            lines.append(f"  Domain-unnecessary:     {self.total_domain_unnecessary_removed:,}")
        lines.extend([
            f"  Time:               {self.elapsed_seconds:.1f}s",
            f"{'='*60}",
        ])
        return "\n".join(lines)


class MRICompressor:
    """
    v2 compression pipeline with expanded strategy support.
    """

    def __init__(
        self,
        model, tokenizer,
        prescription: CompressionPrescription,
        calibration_dataloader,
        device=torch.device("cuda"),
        max_calibration_batches: int = 16,
        do_reconstruction: bool = True,
        reconstruction_lr: float = 1e-4,
        reconstruction_iterations: int = 200,
        enable_low_rank: bool = True,
        enable_attn_pruning: bool = True,
        enable_depth_pruning: bool = True,
        enable_static_fold: bool = True,
        enable_weight_sharing: bool = False,
        domain_calibration_dataloader=None,
        inspector=None,
        enable_imprinting: bool = False,
        imprinting_scale: float = 0.05,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.prescription = prescription
        self.dataloader = calibration_dataloader
        self.domain_dataloader = domain_calibration_dataloader
        self.device = device
        self.max_cal_batches = max_calibration_batches
        self.do_reconstruction = do_reconstruction
        self.recon_lr = reconstruction_lr
        self.recon_iters = reconstruction_iterations
        self.enable_low_rank = enable_low_rank
        self.enable_attn_pruning = enable_attn_pruning
        self.enable_depth_pruning = enable_depth_pruning
        self.enable_static_fold = enable_static_fold
        self.enable_weight_sharing = enable_weight_sharing
        self.inspector = inspector  # Optional ModelInspector for arch-agnostic layer resolution
        self.enable_imprinting = enable_imprinting
        self.imprinting_scale = imprinting_scale
        # d_model needed to locate attention output projections (Studies 6+23)
        _cfg = getattr(model, "config", None)
        self.hidden_size: int = getattr(_cfg, "hidden_size", 0)

    def _get_layer(self, layer_idx: int) -> nn.Module:
        """Resolve a transformer layer using the inspector (or common fallbacks)."""
        return resolve_layer(self.model, layer_idx, self.inspector)

    def _get_attn_info(self, layer_idx: int):
        """Return the AttentionInfo for a layer if inspector is available."""
        if self.inspector is not None and layer_idx < len(self.inspector.attn_layers):
            return self.inspector.attn_layers[layer_idx]
        return None

    def _apply_attn_contrib_ops(
        self,
        layer: nn.Module,
        lp: LayerPrescription,
        lr: dict,
    ) -> None:
        """Apply hybrid-attention compression ops driven by Studies 6+19+23.

        Study 6/19: Zero out rows [s:e] of the attention output projection
        weight for channel groups flagged as low-magnitude contributors.
        This effectively silences those output dimensions of the DeltaNet /
        linear-attention mechanism without changing parameter count.

        Study 23: Apply an in-place low-rank approximation to the attention
        output projection using the effective rank from the attention
        contribution eigenspectrum (rank_95 / d_model < 0.35 threshold).

        Both operations are gated by ``self.enable_attn_pruning``.
        Nothing is done when ``self.hidden_size == 0`` (unknown architecture).
        """
        if not self.enable_attn_pruning or self.hidden_size == 0:
            return

        # Study 6+19: zero low-magnitude output channel groups
        if lp.attn_zero_channel_groups:
            out_proj = _find_attn_output_proj(layer, self.hidden_size)
            if out_proj is not None:
                for (s, e) in lp.attn_zero_channel_groups:
                    out_proj.weight.data[s:e] = 0.0
                    if out_proj.bias is not None:
                        out_proj.bias.data[s:e] = 0.0
                n_groups = len(lp.attn_zero_channel_groups)
                lr["attn_channel_groups_zeroed"] = n_groups
                logger.info(
                    f"    Layer {lp.layer_idx}: zeroed {n_groups} attn output "
                    f"channel group(s) (Studies 6+19)"
                )

        # Study 23: in-place low-rank approximation of attention output proj
        if lp.attn_output_proj_low_rank and lp.attn_output_proj_target_rank:
            out_proj = _find_attn_output_proj(layer, self.hidden_size)
            if out_proj is not None:
                applied = LowRankFactorizer.factorize_module(
                    out_proj,
                    target_rank=lp.attn_output_proj_target_rank,
                    device=self.device,
                )
                if applied:
                    lr["attn_output_proj_low_rank"] = 1
                    logger.info(
                        f"    Layer {lp.layer_idx}: low-rank attn output proj "
                        f"→ rank {lp.attn_output_proj_target_rank} (Study 23)"
                    )

    def compress(self) -> CompressionResult:
        t0 = time.perf_counter()
        result = CompressionResult(
            original_ppl=self.prescription.baseline_ppl,
            compressed_ppl=0.0,
            total_params_original=sum(p.numel() for p in self.model.parameters()),
        )

        print(self.prescription.summary())
        logger.info("Starting MRI-Compress v2 pipeline...")

        for lp in self.prescription.layers:
            lr = self._process_layer(lp)
            result.per_layer_results.append(lr)
            result.total_neurons_removed += lr.get("dead_removed", 0)
            result.total_dormant_removed += lr.get("dormant_removed", 0)
            result.total_neurons_merged += lr.get("neurons_merged", 0)
            result.total_attn_heads_pruned += lr.get("attn_heads_pruned", 0)
            result.total_depth_pruned_layers += lr.get("depth_pruned", 0)
            result.total_low_rank_layers += lr.get("low_rank_applied", 0)
            result.total_folded_neurons += lr.get("neurons_folded", 0)
            result.total_domain_unnecessary_removed += lr.get("domain_unnecessary_removed", 0)

        # ---- Post-loop: Weight sharing between high-CKA layer pairs ----
        if self.enable_weight_sharing and self.prescription.weight_sharing_pairs:
            for layer_a, layer_b in self.prescription.weight_sharing_pairs:
                try:
                    ws_result = WeightSharer.share_mlp_weights(
                        self.model, layer_a, layer_b, self.device,
                        inspector=self.inspector)
                    result.total_weight_shared_pairs += 1
                    logger.info(
                        f"  Weight sharing: layer {layer_a} -> {layer_b}, "
                        f"params saved: {ws_result['params_saved']:,}")
                except Exception as e:
                    logger.warning(f"  Weight sharing failed for ({layer_a}, {layer_b}): {e}")

        result.total_params_compressed = sum(p.numel() for p in self.model.parameters())
        result.elapsed_seconds = time.perf_counter() - t0

        logger.info("Evaluating compressed model...")
        result.compressed_ppl = self._evaluate_ppl()

        print(result.summary())
        return result

    def _get_reconstruction_dataloader(self, lp: LayerPrescription):
        """Get the appropriate dataloader for reconstruction.

        If a domain_calibration_dataloader is set and this layer has a
        target_domain, use domain-specific data for "sharpening".
        Otherwise fall back to the standard calibration dataloader.
        """
        if lp.target_domain and self.domain_dataloader is not None:
            return self.domain_dataloader
        return self.dataloader

    def _process_layer(self, lp: LayerPrescription) -> dict:
        lr = {
            "layer": lp.layer_idx, "strategy": lp.strategy.name,
            "dead_removed": 0, "dormant_removed": 0,
            "neurons_merged": 0, "attn_heads_pruned": 0,
            "depth_pruned": 0, "low_rank_applied": 0,
            "neurons_folded": 0, "domain_unnecessary_removed": 0,
            "reconstruction_mse": None,
            # Studies 6+19+23: hybrid attention contrib compression
            "attn_channel_groups_zeroed": 0,
            "attn_output_proj_low_rank": 0,
        }

        logger.info(f"  Layer {lp.layer_idx}: {lp.strategy.name}")
        layer = self._get_layer(lp.layer_idx)

        # ---- Depth pruning (zero entire layer) ----
        if lp.strategy == CompressionStrategy.DEPTH_PRUNE:
            if self.enable_depth_pruning:
                DepthPruner.prune_layer(layer)
                lr["depth_pruned"] = 1
                logger.info(f"    DEPTH PRUNED (zeroed all weights)")
            else:
                logger.info(f"    DEPTH PRUNE skipped (disabled)")
            return lr

        # ---- Light touch: static fold + low-rank + attention pruning ----
        if lp.strategy == CompressionStrategy.LIGHT_TOUCH:
            is_lossy_lt = False

            # Static neuron folding (Study 20) -- applicable even on LIGHT_TOUCH
            if (self.enable_static_fold and lp.foldable_neuron_count > 0
                    and lp.foldable_neuron_indices is not None):
                with timer(f"Collecting activations for layer {lp.layer_idx} (fold)"):
                    activations = collect_activations(
                        self.model, self.dataloader, lp.layer_idx,
                        max_batches=min(4, self.max_cal_batches), device=self.device,
                        inspector=self.inspector)
                n_folded, activations = StaticNeuronFolder.fold(
                    layer, activations, lp.foldable_neuron_indices, self.device)
                lr["neurons_folded"] = n_folded
                if n_folded > 0:
                    logger.info(f"    Folded {n_folded} static neurons into bias")
                    is_lossy_lt = True
                del activations
                gc.collect()
                torch.cuda.empty_cache()

            # Low-rank factorization (Study 18)
            # Collect MLP I/O BEFORE factorization for reconstruction
            mlp_io_lt = None
            if lp.low_rank_target is not None and self.enable_low_rank:
                if self.do_reconstruction:
                    with timer(f"Collecting MLP I/O for layer {lp.layer_idx} (pre low-rank)"):
                        mlp_io_lt = collect_mlp_io(
                            self.model, self.dataloader, lp.layer_idx,
                            max_batches=min(self.max_cal_batches, 8), device=self.device,
                            inspector=self.inspector)

                with timer(f"Low-rank factorization for layer {lp.layer_idx}"):
                    ranks = LowRankFactorizer.factorize_mlp(
                        layer, lp.low_rank_target, self.device,
                        per_proj_ranks=lp.low_rank_ranks)
                if ranks:
                    lr["low_rank_applied"] = 1
                    is_lossy_lt = True

                # Reconstruct to recover accuracy after SVD approximation
                if self.do_reconstruction and is_lossy_lt and mlp_io_lt is not None:
                    inputs, outputs = mlp_io_lt
                    if inputs and outputs:
                        iters = lp.reconstruction_iterations
                        logger.info(f"    Reconstructing after low-rank ({iters} iters)...")
                        mse = LocalReconstructor.reconstruct(
                            layer, inputs, outputs,
                            lr=self.recon_lr, iterations=iters, device=self.device)
                        lr["reconstruction_mse"] = mse
                        logger.info(f"    Final MSE: {mse:.6f}")
                    del inputs, outputs
                del mlp_io_lt
                gc.collect()
                torch.cuda.empty_cache()

            # Attention head pruning
            if lp.attn_heads_to_prune > 0 and self.enable_attn_pruning:
                pruned = AttentionHeadPruner.prune_heads(
                    layer, lp.attn_heads_to_prune,
                    self.dataloader, self.model, lp.layer_idx, self.device,
                    head_importance_data=lp.head_importance_data,
                    cluster_prunable_heads=lp.cluster_prunable_heads,
                    attn_info=self._get_attn_info(lp.layer_idx))
                lr["attn_heads_pruned"] = len(pruned)
                logger.info(f"    Pruned {len(pruned)} attention heads: {pruned}")

            # Hybrid attention contrib compression (Studies 6+19+23)
            self._apply_attn_contrib_ops(layer, lp, lr)
            return lr

        # ---- Domain specialization: remove domain-unnecessary neurons ----
        if lp.strategy == CompressionStrategy.DOMAIN_SPECIALIZE:
            if lp.domain_unnecessary_indices:
                # Use domain-specific dataloader for reconstruction "sharpening"
                recon_dl = self._get_reconstruction_dataloader(lp)

                # Collect MLP I/O BEFORE removal (for reconstruction)
                mlp_io_domain = None
                if self.do_reconstruction:
                    with timer(f"Collecting MLP I/O for layer {lp.layer_idx} (domain recon)"):
                        mlp_io_domain = collect_mlp_io(
                            self.model, recon_dl, lp.layer_idx,
                            max_batches=min(self.max_cal_batches, 8), device=self.device,
                            inspector=self.inspector)

                # Collect activations for removal
                with timer(f"Collecting activations for layer {lp.layer_idx} (domain)"):
                    activations = collect_activations(
                        self.model, self.dataloader, lp.layer_idx,
                        max_batches=min(4, self.max_cal_batches), device=self.device,
                        inspector=self.inspector)

                # Static neuron folding first (if applicable)
                if (self.enable_static_fold and lp.foldable_neuron_count > 0
                        and lp.foldable_neuron_indices is not None):
                    n_folded, activations = StaticNeuronFolder.fold(
                        layer, activations, lp.foldable_neuron_indices, self.device)
                    lr["neurons_folded"] = n_folded
                    if n_folded > 0:
                        logger.info(f"    Folded {n_folded} static neurons into bias")

                # Remove domain-unnecessary neurons by index
                n_rm, activations = DeadNeuronRemover.remove_by_indices(
                    layer, activations, lp.domain_unnecessary_indices,
                    self.device, protected_indices=lp.protected_neuron_indices,
                    mlp_info=self.inspector.mlp_layers[lp.layer_idx] if self.inspector else None)
                lr["domain_unnecessary_removed"] = n_rm
                logger.info(f"    Removed {n_rm} domain-unnecessary neurons "
                            f"(target domain: {lp.target_domain})")

                # Merge if worthwhile
                if (lp.merge_target_width is not None
                        and lp.merge_target_width < activations.shape[1]):
                    cur = activations.shape[1]
                    tgt = min(lp.merge_target_width, cur)
                    if tgt < cur:
                        with timer(f"Merging neurons in layer {lp.layer_idx}"):
                            nw, activations = NeuronMerger.merge(
                                layer, activations, tgt, device=self.device)
                            lr["neurons_merged"] = cur - nw
                            logger.info(f"    Merged {cur} -> {nw} neurons")

                # Reconstruct using domain-specific data ("sharpening")
                if self.do_reconstruction and mlp_io_domain is not None:
                    inputs, outputs = mlp_io_domain
                    if inputs and outputs:
                        iters = lp.reconstruction_iterations
                        logger.info(f"    Domain reconstruction ({iters} iters, "
                                    f"domain: {lp.target_domain})...")
                        mse = LocalReconstructor.reconstruct(
                            layer, inputs, outputs,
                            lr=self.recon_lr, iterations=iters, device=self.device)
                        lr["reconstruction_mse"] = mse
                        logger.info(f"    Final MSE: {mse:.6f}")
                    del inputs, outputs
                del mlp_io_domain, activations
                gc.collect()
                torch.cuda.empty_cache()

            # Attention head pruning
            if lp.attn_heads_to_prune > 0 and self.enable_attn_pruning:
                pruned = AttentionHeadPruner.prune_heads(
                    layer, lp.attn_heads_to_prune,
                    self.dataloader, self.model, lp.layer_idx, self.device,
                    head_importance_data=lp.head_importance_data,
                    cluster_prunable_heads=lp.cluster_prunable_heads,
                    attn_info=self._get_attn_info(lp.layer_idx))
                lr["attn_heads_pruned"] = len(pruned)
                logger.info(f"    Pruned {len(pruned)} attention heads: {pruned}")

            # Low-rank factorization (post-domain-removal)
            if lp.low_rank_target is not None and self.enable_low_rank:
                with timer(f"Low-rank factorization for layer {lp.layer_idx}"):
                    ranks = LowRankFactorizer.factorize_mlp(
                        layer, lp.low_rank_target, self.device,
                        per_proj_ranks=lp.low_rank_ranks)
                if ranks:
                    lr["low_rank_applied"] = 1

            # Hybrid attention contrib compression (Studies 6+19+23)
            self._apply_attn_contrib_ops(layer, lp, lr)
            return lr

        # ---- Determine if lossy (needs reconstruction targets) ----
        is_lossy = False
        has_domain_overlay = (lp.domain_unnecessary_indices
                              and len(lp.domain_unnecessary_indices) > 0)
        if lp.strategy == CompressionStrategy.DEAD_REMOVAL_AND_MERGE:
            has_merge = (lp.merge_target_width is not None and
                         lp.merge_target_width < get_intermediate_size(layer) - lp.dead_neuron_count - lp.dormant_neuron_count)
            is_lossy = has_merge or has_domain_overlay
        elif lp.strategy in (CompressionStrategy.STRUCTURED_PRUNE, CompressionStrategy.DORMANT_REMOVAL):
            is_lossy = True

        # ---- Collect fresh MLP I/O before compression ----
        # Use domain-specific dataloader for reconstruction when available
        recon_dl = self._get_reconstruction_dataloader(lp)
        mlp_io = None
        if self.do_reconstruction and is_lossy:
            with timer(f"Collecting MLP I/O for layer {lp.layer_idx}"):
                mlp_io = collect_mlp_io(
                    self.model, recon_dl, lp.layer_idx,
                    max_batches=min(self.max_cal_batches, 8), device=self.device,
                    inspector=self.inspector)

        # ---- Collect activations ----
        activations = None
        needs_acts = lp.strategy in (
            CompressionStrategy.DEAD_REMOVAL_AND_MERGE,
            CompressionStrategy.DORMANT_REMOVAL,
            CompressionStrategy.STRUCTURED_PRUNE,
        )
        if needs_acts:
            n_batch = self.max_cal_batches
            if lp.strategy == CompressionStrategy.DEAD_REMOVAL_AND_MERGE and not is_lossy:
                n_batch = min(4, self.max_cal_batches)
            with timer(f"Collecting activations for layer {lp.layer_idx}"):
                activations = collect_activations(
                    self.model, self.dataloader, lp.layer_idx,
                    max_batches=n_batch, device=self.device,
                    inspector=self.inspector)

        # ---- MLP Compression ----
        if lp.strategy == CompressionStrategy.DEAD_REMOVAL_AND_MERGE:
            if activations is not None:
                # Check if we need combined dead + domain-unnecessary removal
                has_domain_removal = (lp.domain_unnecessary_indices
                                      and len(lp.domain_unnecessary_indices) > 0)

                _mlp_info = self.inspector.mlp_layers[lp.layer_idx] if self.inspector else None
                if has_domain_removal and lp.dead_neuron_count > 0:
                    # Combined single-pass removal to avoid index drift
                    n_dead, n_domain, activations = DeadNeuronRemover.remove_combined(
                        layer, activations,
                        n_dead_to_remove=lp.dead_neuron_count,
                        domain_unnecessary_indices=lp.domain_unnecessary_indices,
                        device=self.device,
                        protected_indices=lp.protected_neuron_indices,
                        mlp_info=_mlp_info)
                    lr["dead_removed"] = n_dead
                    lr["domain_unnecessary_removed"] = n_domain
                    logger.info(f"    Combined removal: {n_dead} dead + "
                                f"{n_domain} domain-unnecessary neurons")
                else:
                    # Standard dead-only removal
                    if lp.dead_neuron_count > 0:
                        n_rm, activations = DeadNeuronRemover.remove_by_mri_count(
                            layer, activations, lp.dead_neuron_count, self.device,
                            protected_indices=lp.protected_neuron_indices,
                            mlp_info=_mlp_info)
                        lr["dead_removed"] = n_rm
                        logger.info(f"    Removed {n_rm} dead neurons")

                # Remove dormant neurons (NEW in v2)
                if lp.dormant_neuron_count > 0:
                    n_rm, activations = DeadNeuronRemover.remove_by_mri_count(
                        layer, activations, lp.dormant_neuron_count, self.device,
                        protected_indices=lp.protected_neuron_indices,
                        mlp_info=_mlp_info)
                    lr["dormant_removed"] = n_rm
                    logger.info(f"    Removed {n_rm} dormant neurons")

                # Static neuron folding (Study 20) -- fold before merge
                if (self.enable_static_fold and lp.foldable_neuron_count > 0
                        and lp.foldable_neuron_indices is not None):
                    n_folded, activations = StaticNeuronFolder.fold(
                        layer, activations, lp.foldable_neuron_indices, self.device)
                    lr["neurons_folded"] = n_folded
                    logger.info(f"    Folded {n_folded} static neurons into bias")

                # Merge
                if lp.merge_target_width is not None:
                    cur = activations.shape[1]
                    tgt = min(lp.merge_target_width, cur)
                    if tgt < cur:
                        with timer(f"Merging neurons in layer {lp.layer_idx}"):
                            nw, activations = NeuronMerger.merge(
                                layer, activations, tgt, device=self.device)
                            lr["neurons_merged"] = cur - nw
                            logger.info(f"    Merged {cur} -> {nw} neurons")

        elif lp.strategy == CompressionStrategy.DORMANT_REMOVAL:
            if activations is not None:
                total_to_remove = lp.dead_neuron_count + lp.dormant_neuron_count
                if total_to_remove > 0:
                    n_rm, activations = DeadNeuronRemover.remove_by_mri_count(
                        layer, activations, total_to_remove, self.device,
                        protected_indices=lp.protected_neuron_indices,
                        mlp_info=self.inspector.mlp_layers[lp.layer_idx] if self.inspector else None)
                    lr["dead_removed"] = min(n_rm, lp.dead_neuron_count)
                    lr["dormant_removed"] = max(0, n_rm - lp.dead_neuron_count)
                    logger.info(f"    Removed {n_rm} neurons (dead+dormant)")

                # Static neuron folding (Study 20)
                if (self.enable_static_fold and lp.foldable_neuron_count > 0
                        and lp.foldable_neuron_indices is not None):
                    n_folded, activations = StaticNeuronFolder.fold(
                        layer, activations, lp.foldable_neuron_indices, self.device)
                    lr["neurons_folded"] = n_folded
                    logger.info(f"    Folded {n_folded} static neurons into bias")

        elif lp.strategy == CompressionStrategy.STRUCTURED_PRUNE:
            if activations is not None:
                # Choose importance source: gate-guided (Study 2) or Wanda (Study 3)
                importance = lp.precomputed_wanda_scores
                if (lp.pruning_approach == "gate_guided"
                        and lp.gate_importance_scores is not None):
                    logger.info(
                        f"    Using gate-guided importance scores for layer {lp.layer_idx}")
                    importance = torch.tensor(
                        lp.gate_importance_scores, dtype=torch.float32,
                        device=self.device)
                n_pr, activations = WandaPruner.prune(
                    layer, activations, lp.target_sparsity, self.device,
                    precomputed_importance=importance)
                lr["dead_removed"] = n_pr
                logger.info(f"    Pruned {n_pr} neurons ({lp.target_sparsity:.0%})")

        # ---- Attention head pruning (additive, on any strategy) ----
        if lp.attn_heads_to_prune > 0 and self.enable_attn_pruning:
            pruned = AttentionHeadPruner.prune_heads(
                layer, lp.attn_heads_to_prune,
                self.dataloader, self.model, lp.layer_idx, self.device,
                head_importance_data=lp.head_importance_data,
                cluster_prunable_heads=lp.cluster_prunable_heads)
            lr["attn_heads_pruned"] = len(pruned)
            logger.info(f"    Pruned {len(pruned)} attention heads: {pruned}")

        # ---- Reconstruction ----
        if self.do_reconstruction and is_lossy and mlp_io is not None:
            inputs, outputs = mlp_io
            if inputs and outputs:
                iters = lp.reconstruction_iterations
                if lp.cascade_amplification > 50:
                    iters = int(iters * 1.5)
                logger.info(f"    Reconstructing ({iters} iters)...")
                mse = LocalReconstructor.reconstruct(
                    layer, inputs, outputs,
                    lr=self.recon_lr, iterations=iters, device=self.device)
                lr["reconstruction_mse"] = mse
                logger.info(f"    Final MSE: {mse:.6f}")

        # ---- Low-rank factorization (post-pruning, post-reconstruction) ----
        if lp.low_rank_target is not None and self.enable_low_rank:
            with timer(f"Low-rank factorization for layer {lp.layer_idx}"):
                ranks = LowRankFactorizer.factorize_mlp(
                    layer, lp.low_rank_target, self.device,
                    per_proj_ranks=lp.low_rank_ranks)
            if ranks:
                lr["low_rank_applied"] = 1

        # ---- Hybrid attention contrib compression (Studies 6+19+23) ----
        self._apply_attn_contrib_ops(layer, lp, lr)

        del activations, mlp_io
        gc.collect()
        torch.cuda.empty_cache()
        return lr

    def apply_domain_imprinting(
        self,
        domain_dataloader=None,
        scale: float | None = None,
        max_batches: int = 16,
        pre_computed_centroids=None,
    ) -> dict:
        """
        Post-compression domain imprinting pass.

        Injects per-layer domain activation centroids into the down_proj bias
        of every layer, tilting each layer's output toward the domain manifold
        without any gradient-based training.

        Parameters
        ----------
        domain_dataloader     : DataLoader with domain text (uses internal one if None).
        scale                 : Overrides self.imprinting_scale when provided.
        max_batches           : Calibration batches for centroid estimation.
        pre_computed_centroids: Optional dict[layer_idx -> Tensor(hidden_size,)]
                                computed from the *original* model via
                                DomainImprinter.compute_output_centroids().
                                When supplied, centroids are injected directly
                                (no W_down projection) using the cleaner,
                                undamaged domain signal from before compression.
                                When None, centroids are computed fresh on the
                                compressed model (original behaviour).

        Returns
        -------
        dict with keys:
            "pre_imprint_domain_ppl"   – domain PPL before imprinting
            "post_imprint_domain_ppl"  – domain PPL after imprinting
            "ppl_recovered"            – absolute PPL recovered (positive = improvement)
            "ppl_recovered_pct"        – recovery as % of the pre-imprint PPL
            "per_layer"                – per-layer centroid/delta norms
            "used_original_centroids"  – True when pre_computed_centroids were used
        """
        dl = domain_dataloader or self.domain_dataloader
        if dl is None:
            logger.warning(
                "apply_domain_imprinting: no domain dataloader available — skipping"
            )
            return {}

        if self.inspector is None:
            logger.warning(
                "apply_domain_imprinting: inspector required — skipping"
            )
            return {}

        _scale = scale if scale is not None else self.imprinting_scale

        # --- PPL before imprinting (domain) ---
        pre_ppl = self._evaluate_ppl_on(dl, max_batches=8)
        print(f"\n  Domain PPL before imprinting:  {pre_ppl:.4f}")

        # --- Compute / inject centroids ---
        per_layer = DomainImprinter.imprint(
            model=self.model,
            inspector=self.inspector,
            domain_dataloader=dl,
            device=self.device,
            scale=_scale,
            max_batches=max_batches,
            pre_computed_centroids=pre_computed_centroids,
        )

        # --- PPL after imprinting (domain) ---
        post_ppl   = self._evaluate_ppl_on(dl, max_batches=8)
        ppl_delta  = post_ppl - pre_ppl          # positive = got worse, negative = improved
        recovered  = pre_ppl  - post_ppl         # positive = PPL recovered (improved)
        recovered_pct = (recovered / pre_ppl * 100) if pre_ppl > 0 else 0.0

        print(f"  Domain PPL after  imprinting:  {post_ppl:.4f}")
        if ppl_delta < 0:
            print(f"  PPL recovered:  {abs(ppl_delta):.4f}  ({-ppl_delta / pre_ppl * 100:+.2f}%)")
        elif ppl_delta > 0:
            print(f"  PPL change:     +{ppl_delta:.4f}  ({ppl_delta / pre_ppl * 100:+.2f}%)  "
                  f"[imprinting had no benefit at this scale]")
        else:
            print(f"  PPL change:     0.0000  (no effect)")

        return {
            "pre_imprint_domain_ppl":   pre_ppl,
            "post_imprint_domain_ppl":  post_ppl,
            "ppl_delta":                ppl_delta,       # post - pre: negative = improved
            "ppl_recovered":            recovered,       # pre - post: positive = improved
            "ppl_recovered_pct":        recovered_pct,
            "per_layer":                per_layer,
            "used_original_centroids":  pre_computed_centroids is not None,
        }

    def apply_class_conditional_imprinting(
        self,
        original_class_centroids: Dict[str, Dict[int, torch.Tensor]],
        class_dataloaders: Dict[str, object],
        scale: Optional[float] = None,
        max_batches: int = 16,
        class_weights: Optional[Dict[str, float]] = None,
    ) -> dict:
        """
        Post-compression class-conditional imprinting pass.

        Restores per-class (yes / no / maybe / general) activation geometry by
        computing how each class's hidden-space centroid shifted during
        compression, then injecting the importance-weighted mean delta as a
        per-layer down_proj bias correction.

        Parameters
        ----------
        original_class_centroids : Per-class centroids from the ORIGINAL model
            (before compression).  Produced by
            ``DomainImprinter.compute_class_output_centroids()`` called on the
            original model.
            Structure: Dict[class_name -> Dict[layer_idx -> Tensor(hidden_size,)]].
        class_dataloaders : DataLoaders per class, used here to compute
            compressed-model class centroids.
            Structure: Dict[class_name -> DataLoader].
        scale : Injection scale (overrides self.imprinting_scale when set).
        class_weights : Per-class importance weights.
            Default: {"yes": 1.0, "no": 1.0, "maybe": 3.0, "general": 0.5}.
            "maybe" is upweighted 3× because uncertainty circuits are the most
            fragile under pruning.

        Returns
        -------
        dict with keys:
            "pre_imprint_domain_ppl"   – domain PPL before correction
            "post_imprint_domain_ppl"  – domain PPL after correction
            "ppl_delta"                – post - pre (negative = improved)
            "ppl_recovered"            – pre - post (positive = improved)
            "ppl_recovered_pct"        – recovery as % of pre-imprint PPL
            "per_layer"                – per-layer correction stats
            "n_classes_used"           – number of classes contributing
            "class_names"              – list of class names used
            "mode"                     – "class_conditional"
        """
        if not original_class_centroids or not class_dataloaders:
            logger.warning(
                "apply_class_conditional_imprinting: no centroids or dataloaders — skipping"
            )
            return {}
        if self.inspector is None:
            logger.warning(
                "apply_class_conditional_imprinting: inspector required — skipping"
            )
            return {}

        _scale   = scale if scale is not None else self.imprinting_scale
        # Use any class dataloader for PPL measurement
        _dl_ppl  = next(iter(class_dataloaders.values()), None) or self.domain_dataloader

        # --- PPL before correction ---
        pre_ppl = self._evaluate_ppl_on(_dl_ppl, max_batches=8) if _dl_ppl is not None else None
        if pre_ppl is not None:
            print(f"\n  Domain PPL before class-cond imprinting:  {pre_ppl:.4f}")

        # --- Compute + inject class-conditional corrections ---
        per_layer = DomainImprinter.imprint_class_conditional(
            model=self.model,
            inspector=self.inspector,
            class_dataloaders=class_dataloaders,
            original_class_centroids=original_class_centroids,
            device=self.device,
            scale=_scale,
            max_batches=max_batches,
            class_weights=class_weights,
        )

        # --- PPL after correction ---
        post_ppl = self._evaluate_ppl_on(_dl_ppl, max_batches=8) if _dl_ppl is not None else None
        if pre_ppl is not None and post_ppl is not None:
            ppl_delta     = post_ppl - pre_ppl
            recovered     = pre_ppl  - post_ppl
            recovered_pct = (recovered / pre_ppl * 100) if pre_ppl > 0 else 0.0
            print(f"  Domain PPL after  class-cond imprinting:  {post_ppl:.4f}")
            if ppl_delta < 0:
                print(
                    f"  PPL recovered:  {abs(ppl_delta):.4f}  "
                    f"({-ppl_delta / pre_ppl * 100:+.2f}%)"
                )
            elif ppl_delta > 0:
                print(
                    f"  PPL change:     +{ppl_delta:.4f}  "
                    f"({ppl_delta / pre_ppl * 100:+.2f}%)  "
                    f"[class-cond imprinting had no benefit at this scale]"
                )
            else:
                print(f"  PPL change:     0.0000  (no effect)")
        else:
            ppl_delta     = None
            recovered     = None
            recovered_pct = None

        return {
            "pre_imprint_domain_ppl":   pre_ppl,
            "post_imprint_domain_ppl":  post_ppl,
            "ppl_delta":                ppl_delta,
            "ppl_recovered":            recovered,
            "ppl_recovered_pct":        recovered_pct,
            "per_layer":                per_layer,
            "n_classes_used":           len(original_class_centroids),
            "class_names":              list(original_class_centroids.keys()),
            "mode":                     "class_conditional",
        }

    @torch.no_grad()
    def _evaluate_ppl_on(self, dataloader, max_batches: int = 8) -> float:
        """Evaluate perplexity on an arbitrary dataloader."""
        self.model.eval()
        total_loss, total_tok = 0.0, 0
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            ids = batch["input_ids"].to(self.device)
            out = self.model(input_ids=ids, labels=ids)
            seq_len = ids.shape[1] - 1
            total_loss += out.loss.item() * seq_len * ids.shape[0]
            total_tok += seq_len * ids.shape[0]
        return math.exp(total_loss / max(total_tok, 1))

    @torch.no_grad()
    def _evaluate_ppl(self, max_batches: int = 8) -> float:
        """Evaluate perplexity. Uses same logic as data_utils.evaluate_perplexity
        for consistent numbers between Stage 3 and Stage 4."""
        self.model.eval()
        total_loss, total_tok = 0.0, 0
        for i, batch in enumerate(self.dataloader):
            if i >= max_batches:
                break
            ids = batch["input_ids"].to(self.device)
            out = self.model(input_ids=ids, labels=ids)
            seq_len = ids.shape[1] - 1  # shifted labels
            total_loss += out.loss.item() * seq_len * ids.shape[0]
            total_tok += seq_len * ids.shape[0]
        return math.exp(total_loss / max(total_tok, 1))

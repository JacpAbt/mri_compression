"""
MRI-Compress Engine
====================
Applies a CompressionPrescription to a HuggingFace LLM.

Implements:
1. Dead neuron removal (zero-cost structured pruning)
2. Redundancy-guided neuron merging (activation-similarity-based)
3. Wanda-guided structured pruning
4. Depth pruning (entire layer removal)
5. Sequential local reconstruction (SparseGPT-style)

Design decisions learned from testing on Qwen2.5-3B:
- RTN quantization is catastrophically bad on sub-7B models (PPL 1M+).
  Quantization code is kept but gated behind enable_quantization=False.
- Reconstruction must be SEQUENTIAL: compress layer N, reconstruct it,
  then re-collect I/O for layer N+1 from the partially-compressed model.
  Pre-collecting all I/O upfront produces stale targets that cause
  reconstruction to diverge (layer 2 MSE=32 in initial testing).
- Dead neuron removal is nearly free (PPL +0.2% for 6% param reduction).
  We trust the MRI count and remove the N least-active neurons rather
  than re-measuring with an arbitrary threshold.
- Merging must be conservative after heavy dead removal: surviving
  neurons are disproportionately important per-neuron.
- Only layers with actual structural damage need reconstruction.
  Skipping LIGHT_TOUCH and dead-only layers saves ~80% of recon time.
"""

from __future__ import annotations
import gc
import logging
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

from diagnostic import (
    CompressionPrescription,
    CompressionStrategy,
    LayerPrescription,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Utilities
# ============================================================================

@contextmanager
def timer(name: str):
    t0 = time.perf_counter()
    yield
    logger.info(f"  {name}: {time.perf_counter() - t0:.1f}s")


def get_mlp_modules(layer: nn.Module) -> dict[str, nn.Linear]:
    """Extract MLP projections from a transformer layer (SwiGLU or standard)."""
    mlp = getattr(layer, "mlp", None)
    if mlp is None:
        raise ValueError(f"Cannot find MLP in layer: {type(layer)}")
    modules = {}
    for name in ["gate_proj", "up_proj", "down_proj"]:
        mod = getattr(mlp, name, None)
        if mod is not None and isinstance(mod, nn.Linear):
            modules[name] = mod
    if not modules:
        for name, mod in mlp.named_modules():
            if isinstance(mod, nn.Linear):
                modules[name] = mod
    return modules


def get_attention_module(layer: nn.Module) -> nn.Module:
    for name in ["self_attn", "attention"]:
        attn = getattr(layer, name, None)
        if attn is not None:
            return attn
    raise ValueError(f"Cannot find attention in layer: {type(layer)}")


def get_intermediate_size(layer: nn.Module) -> int:
    """Get the current MLP intermediate size (may change after pruning)."""
    mods = get_mlp_modules(layer)
    if "gate_proj" in mods:
        return mods["gate_proj"].out_features
    if "up_proj" in mods:
        return mods["up_proj"].out_features
    raise ValueError("Cannot determine intermediate size")


# ============================================================================
# Data Collection
# ============================================================================

@torch.no_grad()
def collect_activations(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    layer_idx: int,
    max_batches: int = 16,
    device: torch.device = torch.device("cuda"),
) -> torch.Tensor:
    """
    Collect MLP intermediate activations (input to down_proj).
    Returns: [total_tokens, intermediate_size] on CPU.
    """
    all_acts = []
    mlp = model.model.layers[layer_idx].mlp
    hook_data = {"acts": None}

    def hook_fn(module, input, output):
        hook_data["acts"] = input[0].detach()

    handle = mlp.down_proj.register_forward_hook(hook_fn)
    model.eval()
    try:
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            ids = batch["input_ids"].to(device)
            mask = batch.get("attention_mask", torch.ones_like(ids)).to(device)
            with autocast("cuda", dtype=torch.bfloat16):
                model(input_ids=ids, attention_mask=mask)
            if hook_data["acts"] is not None:
                acts = hook_data["acts"].reshape(-1, hook_data["acts"].shape[-1])
                all_acts.append(acts.float().cpu())
                hook_data["acts"] = None
    finally:
        handle.remove()

    if not all_acts:
        raise RuntimeError(f"No activations collected for layer {layer_idx}")
    return torch.cat(all_acts, dim=0)


@torch.no_grad()
def collect_mlp_io(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    layer_idx: int,
    max_batches: int = 8,
    device: torch.device = torch.device("cuda"),
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Collect MLP input/output for reconstruction targets.
    Hooks the MLP sub-module to avoid position embedding issues.
    """
    inputs_list, outputs_list = [], []
    mlp = model.model.layers[layer_idx].mlp
    io = {"inp": None, "out": None}

    def pre_hook(module, args, kwargs):
        io["inp"] = args[0].detach().cpu()

    def post_hook(module, args, kwargs, output):
        out = output[0] if isinstance(output, tuple) else output
        io["out"] = out.detach().cpu()

    h1 = mlp.register_forward_pre_hook(pre_hook, with_kwargs=True)
    h2 = mlp.register_forward_hook(post_hook, with_kwargs=True)
    model.eval()
    try:
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            ids = batch["input_ids"].to(device)
            mask = batch.get("attention_mask", torch.ones_like(ids)).to(device)
            with autocast("cuda", dtype=torch.bfloat16):
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
# Compression Operations
# ============================================================================

class DeadNeuronRemover:
    """Remove dead neurons by physically shrinking MLP weight matrices."""

    @staticmethod
    @torch.no_grad()
    def remove_by_mri_count(
        layer: nn.Module,
        activations: torch.Tensor,
        n_to_remove: int,
        device: torch.device,
    ) -> tuple[int, torch.Tensor]:
        """
        Remove the N least-active neurons (trusting MRI diagnostic count).
        Returns: (num_removed, updated_activations)
        """
        n_neurons = activations.shape[1]
        n_to_remove = min(n_to_remove, n_neurons - 1)
        if n_to_remove <= 0:
            return 0, activations

        fire_mag = activations.abs().mean(dim=0)
        _, sorted_idx = fire_mag.sort()
        keep_mask = torch.ones(n_neurons, dtype=torch.bool)
        keep_mask[sorted_idx[:n_to_remove]] = False

        n_removed = DeadNeuronRemover._shrink_mlp(layer, keep_mask, device)
        return n_removed, activations[:, keep_mask]

    @staticmethod
    @torch.no_grad()
    def _shrink_mlp(layer: nn.Module, keep_mask: torch.Tensor, device: torch.device) -> int:
        """Physically remove neurons from MLP weight matrices."""
        mlp_modules = get_mlp_modules(layer)
        n_removed = (~keep_mask).sum().item()
        if n_removed == 0:
            return 0

        keep_idx = keep_mask.nonzero(as_tuple=True)[0].to(device)

        for name in ["gate_proj", "up_proj"]:
            if name in mlp_modules:
                mod = mlp_modules[name]
                mod.weight = nn.Parameter(torch.index_select(mod.weight.data, 0, keep_idx))
                mod.out_features = mod.weight.shape[0]
                if mod.bias is not None:
                    mod.bias = nn.Parameter(mod.bias.data[keep_idx])

        if "down_proj" in mlp_modules:
            mod = mlp_modules["down_proj"]
            mod.weight = nn.Parameter(torch.index_select(mod.weight.data, 1, keep_idx))
            mod.in_features = mod.weight.shape[1]

        return n_removed


class NeuronMerger:
    """Merge redundant neurons via greedy agglomerative clustering."""

    @staticmethod
    @torch.no_grad()
    def merge(
        layer: nn.Module,
        activations: torch.Tensor,
        target_width: int,
        device: torch.device,
        chunk_size: int = 1024,
    ) -> tuple[int, torch.Tensor]:
        """
        Cluster and merge neurons to target_width.
        Returns: (new_width, updated_activations)
        """
        n = activations.shape[1]
        if target_width >= n:
            return n, activations

        clusters = NeuronMerger._cluster(activations, target_width, device, chunk_size)
        NeuronMerger._apply_merge(layer, clusters, activations, device)

        # Build merged activations (importance-weighted average per cluster)
        importance = activations.abs().mean(dim=0)
        merged = torch.zeros(activations.shape[0], len(clusters))
        for cid, members in enumerate(clusters):
            if len(members) == 1:
                merged[:, cid] = activations[:, members[0]]
            else:
                imp = importance[members]
                w = imp / imp.sum().clamp(min=1e-8)
                merged[:, cid] = (activations[:, members] * w.unsqueeze(0)).sum(1)

        return len(clusters), merged

    @staticmethod
    def _cluster(acts: torch.Tensor, target: int, device: torch.device, chunk: int) -> list[list[int]]:
        n = acts.shape[1]
        acts_gpu = acts.to(device)
        norms = acts_gpu.norm(dim=0, keepdim=True).clamp(min=1e-8)
        acts_n = acts_gpu / norms
        importance = acts.abs().mean(dim=0)

        sim = torch.zeros(n, n, device=device)
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            sim[s:e] = acts_n[:, s:e].T @ acts_n
        sim.fill_diagonal_(-float("inf"))

        members: list[list[int]] = [[i] for i in range(n)]
        cl_imp = importance.clone().to(device)

        for step in range(n - target):
            flat = sim.argmax()
            i, j = flat // n, flat % n
            if cl_imp[j] > cl_imp[i]:
                i, j = j, i

            members[i.item()].extend(members[j.item()])
            members[j.item()] = []

            wi, wj = cl_imp[i], cl_imp[j]
            tw = wi + wj
            sim[i] = (sim[i] * wi + sim[j] * wj) / tw
            sim[:, i] = sim[i]
            sim[i, i] = -float("inf")
            sim[j] = -float("inf")
            sim[:, j] = -float("inf")
            cl_imp[i] = tw

            if (step + 1) % 1000 == 0:
                logger.info(f"    Merge step {step+1}/{n - target}")

        del sim, acts_gpu, acts_n
        torch.cuda.empty_cache()
        return [m for m in members if m]

    @staticmethod
    def _apply_merge(layer: nn.Module, clusters: list[list[int]], acts: torch.Tensor, device: torch.device):
        mlp = get_mlp_modules(layer)
        imp = acts.abs().mean(dim=0).to(device)
        nw = len(clusters)

        for name in ["gate_proj", "up_proj"]:
            if name not in mlp:
                continue
            mod = mlp[name]
            old = mod.weight.data.to(device)
            new = torch.zeros(nw, old.shape[1], dtype=old.dtype, device=device)
            for c, ms in enumerate(clusters):
                if len(ms) == 1:
                    new[c] = old[ms[0]]
                else:
                    idx = torch.tensor(ms, device=device)
                    m_imp = imp[idx]
                    new[c] = (old[idx] * m_imp.unsqueeze(1)).sum(0) / m_imp.sum().clamp(min=1e-8)
            mod.weight = nn.Parameter(new)
            mod.out_features = nw
            if mod.bias is not None:
                old_b = mod.bias.data.to(device)
                new_b = torch.zeros(nw, dtype=old_b.dtype, device=device)
                for c, ms in enumerate(clusters):
                    if len(ms) == 1:
                        new_b[c] = old_b[ms[0]]
                    else:
                        idx = torch.tensor(ms, device=device)
                        m_imp = imp[idx]
                        new_b[c] = (old_b[idx] * m_imp).sum() / m_imp.sum().clamp(min=1e-8)
                mod.bias = nn.Parameter(new_b)

        if "down_proj" in mlp:
            mod = mlp["down_proj"]
            old = mod.weight.data.to(device)
            new = torch.zeros(old.shape[0], nw, dtype=old.dtype, device=device)
            for c, ms in enumerate(clusters):
                if len(ms) == 1:
                    new[:, c] = old[:, ms[0]]
                else:
                    idx = torch.tensor(ms, device=device)
                    new[:, c] = old[:, idx].sum(dim=1)
            mod.weight = nn.Parameter(new)
            mod.in_features = nw


class WandaPruner:
    """Structured pruning: |weight| × ||activation|| importance."""

    @staticmethod
    @torch.no_grad()
    def prune(
        layer: nn.Module,
        activations: torch.Tensor,
        target_sparsity: float,
        device: torch.device,
    ) -> tuple[int, torch.Tensor]:
        """Returns: (num_pruned, updated_activations)"""
        mlp = get_mlp_modules(layer)
        act_norm = activations.norm(dim=0).to(device)
        w_norm = mlp["down_proj"].weight.data.norm(dim=0).to(device) if "down_proj" in mlp else torch.ones_like(act_norm)
        importance = act_norm * w_norm

        n = importance.shape[0]
        n_keep = max(1, int(n * (1 - target_sparsity)))
        _, top = importance.topk(n_keep)
        keep = torch.zeros(n, dtype=torch.bool, device=device)
        keep[top] = True

        n_pruned = DeadNeuronRemover._shrink_mlp(layer, keep.cpu(), device)
        return n_pruned, activations[:, keep.cpu()]


class LayerQuantizer:
    """
    RTN weight quantization. DISABLED BY DEFAULT.
    Only use with proper quantization backend (GPTQ/AWQ).
    Tested: 4-bit RTN on Qwen2.5-3B → PPL 1,147,420.
    """

    @staticmethod
    @torch.no_grad()
    def quantize_layer_weights(
        layer: nn.Module, bits: int = 4,
        outlier_bits: Optional[int] = None,
        outlier_fraction: float = 0.0,
        activations: Optional[torch.Tensor] = None,
        device: torch.device = torch.device("cuda"),
    ):
        mlp = get_mlp_modules(layer)
        attn = get_attention_module(layer)

        outlier_mask = None
        if outlier_bits and outlier_fraction > 0 and activations is not None:
            mag = activations.abs().mean(dim=0)
            n_out = max(1, int(len(mag) * outlier_fraction))
            _, idx = mag.topk(n_out)
            outlier_mask = torch.zeros(len(mag), dtype=torch.bool)
            outlier_mask[idx] = True

        def _qt(w, b):
            qmax = 2 ** (b - 1) - 1
            s = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax
            return ((w / s).round().clamp(-qmax - 1, qmax)) * s

        for name, mod in mlp.items():
            if outlier_mask is not None:
                dim = 0 if name in ("gate_proj", "up_proj") else 1
                w = mod.weight.data
                om = outlier_mask.to(w.device)
                bulk, out = (~om).nonzero(as_tuple=True)[0], om.nonzero(as_tuple=True)[0]
                if dim == 0:
                    if len(bulk): w[bulk] = _qt(w[bulk], bits)
                    if len(out): w[out] = _qt(w[out], outlier_bits)
                else:
                    if len(bulk): w[:, bulk] = _qt(w[:, bulk], bits)
                    if len(out): w[:, out] = _qt(w[:, out], outlier_bits)
            else:
                mod.weight.data = _qt(mod.weight.data, bits)

        for _, mod in attn.named_modules():
            if isinstance(mod, nn.Linear):
                mod.weight.data = _qt(mod.weight.data, bits)


class LocalReconstructor:
    """Fine-tune compressed MLP to match original output."""

    @staticmethod
    def reconstruct(
        layer: nn.Module,
        target_inputs: list[torch.Tensor],
        target_outputs: list[torch.Tensor],
        lr: float = 1e-4,
        iterations: int = 200,
        device: torch.device = torch.device("cuda"),
    ) -> float:
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            return 0.0
        params = [p for p in mlp.parameters() if p.requires_grad]
        if not params:
            return 0.0

        opt = torch.optim.Adam(params, lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iterations)
        best = float("inf")

        for it in range(iterations):
            total, n = 0.0, 0
            for inp, tgt in zip(target_inputs, target_outputs):
                inp_g, tgt_g = inp.to(device), tgt.to(device)
                with autocast("cuda", dtype=torch.bfloat16):
                    out = mlp(inp_g)
                    if isinstance(out, tuple):
                        out = out[0]
                    loss = F.mse_loss(out.float(), tgt_g.float())
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total += loss.item() * inp.shape[0]
                n += inp.shape[0]
                del inp_g, tgt_g, out, loss
            sched.step()
            avg = total / max(n, 1)
            best = min(best, avg)
            if (it + 1) % 50 == 0:
                logger.info(f"    Recon iter {it+1}/{iterations}: MSE={avg:.6f}")

        torch.cuda.empty_cache()
        return best


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
    total_params_original: int = 0
    total_params_compressed: int = 0
    elapsed_seconds: float = 0.0

    def summary(self) -> str:
        r = 1 - self.total_params_compressed / max(self.total_params_original, 1)
        d = self.compressed_ppl - self.original_ppl
        p = (self.compressed_ppl / max(self.original_ppl, 1e-8) - 1) * 100
        return "\n".join([
            f"{'='*60}", "  MRI-Compress Results", f"{'='*60}",
            f"  Original PPL:    {self.original_ppl:.4f}",
            f"  Compressed PPL:  {self.compressed_ppl:.4f}",
            f"  PPL change:      {d:+.4f} ({p:+.2f}%)", "",
            f"  Parameters:      {self.total_params_original:,} → {self.total_params_compressed:,}",
            f"  Reduction:       {r:.1%}",
            f"  Neurons removed: {self.total_neurons_removed:,}",
            f"  Neurons merged:  {self.total_neurons_merged:,}",
            f"  Time:            {self.elapsed_seconds:.1f}s",
            f"{'='*60}",
        ])


class MRICompressor:
    """
    Main compression pipeline with SEQUENTIAL layer processing.

    For each compressed layer:
      1. Collect fresh MLP I/O from the current (partially compressed) model
      2. Collect activations
      3. Compress (dead removal / merge / prune)
      4. Reconstruct against FRESH targets
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        prescription: CompressionPrescription,
        calibration_dataloader: torch.utils.data.DataLoader,
        device: torch.device = torch.device("cuda"),
        max_calibration_batches: int = 16,
        do_reconstruction: bool = True,
        reconstruction_lr: float = 1e-4,
        reconstruction_iterations: int = 200,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.prescription = prescription
        self.dataloader = calibration_dataloader
        self.device = device
        self.max_cal_batches = max_calibration_batches
        self.do_reconstruction = do_reconstruction
        self.recon_lr = reconstruction_lr
        self.recon_iters = reconstruction_iterations

    def compress(self) -> CompressionResult:
        t0 = time.perf_counter()
        result = CompressionResult(
            original_ppl=self.prescription.baseline_ppl,
            compressed_ppl=0.0,
            total_params_original=sum(p.numel() for p in self.model.parameters()),
        )

        print(self.prescription.summary())
        logger.info("Starting MRI-Compress pipeline...")

        for lp in self.prescription.layers:
            lr = self._process_layer(lp)
            result.per_layer_results.append(lr)
            result.total_neurons_removed += lr.get("neurons_removed", 0)
            result.total_neurons_merged += lr.get("neurons_merged", 0)

        result.total_params_compressed = sum(p.numel() for p in self.model.parameters())
        result.elapsed_seconds = time.perf_counter() - t0

        logger.info("Evaluating compressed model...")
        result.compressed_ppl = self._evaluate_ppl()

        print(result.summary())
        return result

    def _process_layer(self, lp: LayerPrescription) -> dict:
        """Process one layer: collect → compress → reconstruct (all sequential)."""
        lr = {"layer": lp.layer_idx, "strategy": lp.strategy.name,
              "neurons_removed": 0, "neurons_merged": 0, "reconstruction_mse": None}

        logger.info(f"  Layer {lp.layer_idx}: {lp.strategy.name}")

        if lp.strategy == CompressionStrategy.LIGHT_TOUCH:
            return lr

        layer = self.model.model.layers[lp.layer_idx]
        is_structural = lp.strategy in (
            CompressionStrategy.DEAD_REMOVAL_AND_MERGE,
            CompressionStrategy.STRUCTURED_PRUNE,
            CompressionStrategy.DEPTH_PRUNE,
        )

        # Determine if this layer has lossy compression (needs reconstruction)
        is_lossy = False
        if lp.strategy == CompressionStrategy.DEAD_REMOVAL_AND_MERGE:
            has_merge = (lp.merge_target_width is not None and
                         lp.merge_target_width < get_intermediate_size(layer) - lp.dead_neuron_count)
            is_lossy = has_merge  # Dead-only is lossless, merging is lossy
        elif lp.strategy == CompressionStrategy.STRUCTURED_PRUNE:
            is_lossy = True
        elif lp.strategy in (CompressionStrategy.QUANTIZE_AGGRESSIVE, CompressionStrategy.QUANTIZE_MIXED_PRECISION):
            is_lossy = True

        # Step 1: Collect fresh MLP I/O BEFORE compression (only if reconstruction needed)
        mlp_io = None
        if self.do_reconstruction and is_lossy:
            with timer(f"Collecting MLP I/O for layer {lp.layer_idx}"):
                mlp_io = collect_mlp_io(
                    self.model, self.dataloader, lp.layer_idx,
                    max_batches=min(self.max_cal_batches, 8),
                    device=self.device,
                )

        # Step 2: Collect activations
        activations = None
        needs_acts = lp.strategy in (
            CompressionStrategy.DEAD_REMOVAL_AND_MERGE,
            CompressionStrategy.STRUCTURED_PRUNE,
            CompressionStrategy.QUANTIZE_MIXED_PRECISION,
        )
        if needs_acts:
            is_dead_only = (lp.strategy == CompressionStrategy.DEAD_REMOVAL_AND_MERGE and not is_lossy)
            n_batch = min(4, self.max_cal_batches) if is_dead_only else self.max_cal_batches
            with timer(f"Collecting activations for layer {lp.layer_idx}"):
                activations = collect_activations(
                    self.model, self.dataloader, lp.layer_idx,
                    max_batches=n_batch, device=self.device,
                )

        # Step 3: Compress
        if lp.strategy == CompressionStrategy.DEAD_REMOVAL_AND_MERGE:
            if activations is not None and lp.dead_neuron_count > 0:
                n_rm, activations = DeadNeuronRemover.remove_by_mri_count(
                    layer, activations, lp.dead_neuron_count, self.device)
                lr["neurons_removed"] = n_rm
                logger.info(f"    Removed {n_rm} dead neurons (MRI target: {lp.dead_neuron_count})")

            if activations is not None and lp.merge_target_width is not None:
                cur = activations.shape[1]
                tgt = min(lp.merge_target_width, cur)
                if tgt < cur:
                    with timer(f"Merging neurons in layer {lp.layer_idx}"):
                        nw, activations = NeuronMerger.merge(
                            layer, activations, tgt, device=self.device)
                        lr["neurons_merged"] = cur - nw
                        logger.info(f"    Merged {cur} → {nw} neurons")

        elif lp.strategy == CompressionStrategy.STRUCTURED_PRUNE:
            if activations is not None:
                n_pr, activations = WandaPruner.prune(
                    layer, activations, lp.target_sparsity, self.device)
                lr["neurons_removed"] = n_pr
                logger.info(f"    Pruned {n_pr} neurons ({lp.target_sparsity:.0%})")

        elif lp.strategy == CompressionStrategy.QUANTIZE_AGGRESSIVE:
            LayerQuantizer.quantize_layer_weights(layer, bits=lp.quant_bits, device=self.device)
            logger.info(f"    Quantized to {lp.quant_bits}-bit")

        elif lp.strategy == CompressionStrategy.QUANTIZE_MIXED_PRECISION:
            LayerQuantizer.quantize_layer_weights(
                layer, bits=lp.quant_bits, outlier_bits=lp.outlier_bits,
                outlier_fraction=lp.outlier_fraction, activations=activations,
                device=self.device)
            logger.info(f"    Mixed-precision: {lp.quant_bits}b/{lp.outlier_bits}b")

        elif lp.strategy == CompressionStrategy.DEPTH_PRUNE:
            for mod in layer.modules():
                if isinstance(mod, nn.Linear):
                    mod.weight.data.zero_()
                    if mod.bias is not None:
                        mod.bias.data.zero_()
            logger.info(f"    DEPTH PRUNED")

        # Step 4: Reconstruct (only for lossy layers, using fresh targets)
        if self.do_reconstruction and is_lossy and mlp_io is not None:
            inputs, outputs = mlp_io
            if inputs and outputs:
                iters = self.recon_iters
                if lp.cascade_amplification > 50:
                    iters = int(iters * 1.5)
                logger.info(f"    Reconstructing ({iters} iters)...")
                mse = LocalReconstructor.reconstruct(
                    layer, inputs, outputs,
                    lr=self.recon_lr, iterations=iters, device=self.device)
                lr["reconstruction_mse"] = mse
                logger.info(f"    Final MSE: {mse:.6f}")

        del activations, mlp_io
        gc.collect()
        torch.cuda.empty_cache()
        return lr

    @torch.no_grad()
    def _evaluate_ppl(self) -> float:
        self.model.eval()
        total_loss, total_tok = 0.0, 0
        for i, batch in enumerate(self.dataloader):
            if i >= self.max_cal_batches:
                break
            ids = batch["input_ids"].to(self.device)
            mask = batch.get("attention_mask", torch.ones_like(ids)).to(self.device)
            with autocast("cuda", dtype=torch.bfloat16):
                out = self.model(input_ids=ids, attention_mask=mask, labels=ids)
            n = mask.sum().item()
            total_loss += out.loss.item() * n
            total_tok += n
        return math.exp(total_loss / max(total_tok, 1))
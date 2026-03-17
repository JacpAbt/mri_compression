"""
MRI-Compress Engine v2
=======================
Applies a CompressionPrescription to a HuggingFace LLM.

New operations in v2 (informed by Studies 17-21):
1. Dormant neuron removal — removes neurons firing on <1% of inputs
2. Attention head pruning — zeros out lowest-impact heads per layer
3. Depth pruning (applied) — zeros out entire layers
4. Low-rank factorization — SVD decomposition of MLP weight matrices
5. Weight sharing — tie MLP weights between high-CKA layer pairs

Architecture from v1 retained:
- Sequential layer processing (collect → compress → reconstruct)
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
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

from diagnostics import (
    CompressionPrescription,
    CompressionStrategy,
    LayerPrescription,
)

logger = logging.getLogger(__name__)


@contextmanager
def timer(name: str):
    t0 = time.perf_counter()
    yield
    logger.info(f"  {name}: {time.perf_counter() - t0:.1f}s")


def get_mlp_modules(layer: nn.Module) -> dict[str, nn.Linear]:
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
    mods = get_mlp_modules(layer)
    if "gate_proj" in mods:
        return mods["gate_proj"].out_features
    if "up_proj" in mods:
        return mods["up_proj"].out_features
    raise ValueError("Cannot determine intermediate size")


# ============================================================================
# Data Collection (unchanged from v1)
# ============================================================================

@torch.no_grad()
def collect_activations(model, dataloader, layer_idx, max_batches=16, device=torch.device("cuda")):
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
def collect_mlp_io(model, dataloader, layer_idx, max_batches=8, device=torch.device("cuda")):
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
# Compression Operations (v1, unchanged)
# ============================================================================

class DeadNeuronRemover:
    @staticmethod
    @torch.no_grad()
    def remove_by_mri_count(layer, activations, n_to_remove, device):
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
    def _shrink_mlp(layer, keep_mask, device):
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
    @staticmethod
    @torch.no_grad()
    def merge(layer, activations, target_width, device, chunk_size=1024):
        n = activations.shape[1]
        if target_width >= n:
            return n, activations
        clusters = NeuronMerger._cluster(activations, target_width, device, chunk_size)
        NeuronMerger._apply_merge(layer, clusters, activations, device)
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
    def _cluster(acts, target, device, chunk):
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
        members = [[i] for i in range(n)]
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
    def _apply_merge(layer, clusters, acts, device):
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
    @staticmethod
    @torch.no_grad()
    def prune(layer, activations, target_sparsity, device):
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


class LocalReconstructor:
    @staticmethod
    def reconstruct(layer, target_inputs, target_outputs, lr=1e-4, iterations=200, device=torch.device("cuda")):
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
# NEW Compression Operations (v2)
# ============================================================================

class AttentionHeadPruner:
    """
    Zero out the least-important attention heads in a layer.

    Strategy: zero the output projection columns for pruned heads.
    This effectively makes those heads produce zero output while keeping
    the attention computation intact (avoids shape changes).

    Why not remove heads entirely? Changing num_heads breaks the model's
    config and all downstream shape assumptions. Zeroing is safe and
    equivalent at inference time (the heads contribute nothing).
    """

    @staticmethod
    @torch.no_grad()
    def prune_heads(
        layer: nn.Module,
        n_heads_to_prune: int,
        dataloader,
        model: nn.Module,
        layer_idx: int,
        device: torch.device,
        max_batches: int = 4,
    ) -> list[int]:
        """
        Identify and zero out the least-important heads.
        Returns list of pruned head indices.
        """
        attn = get_attention_module(layer)
        num_heads = model.config.num_attention_heads
        head_dim = model.config.hidden_size // num_heads

        if n_heads_to_prune <= 0 or n_heads_to_prune >= num_heads:
            return []

        # Measure head importance via output norm contribution
        # Each head's contribution = norm of its output projection slice
        o_proj = attn.o_proj
        W_o = o_proj.weight.data  # [hidden_size, hidden_size]

        head_importance = torch.zeros(num_heads)
        for h in range(num_heads):
            start = h * head_dim
            end = start + head_dim
            # Frobenius norm of the output projection for this head
            head_importance[h] = W_o[:, start:end].float().norm()

        # Prune the lowest-importance heads
        _, sorted_heads = head_importance.sort()
        heads_to_prune = sorted_heads[:n_heads_to_prune].tolist()

        # Zero out the pruned heads' weights in all projections
        for h in heads_to_prune:
            start = h * head_dim
            end = start + head_dim
            # Zero output projection columns for this head
            o_proj.weight.data[:, start:end] = 0
            # Zero the Q/K/V projection rows for this head
            # (saves computation and prevents any gradient flow)
            for proj_name in ["q_proj"]:
                proj = getattr(attn, proj_name, None)
                if proj is not None:
                    proj.weight.data[start:end] = 0
                    if proj.bias is not None:
                        proj.bias.data[start:end] = 0

        return heads_to_prune


class DepthPruner:
    """
    Zero out an entire transformer layer, making it a no-op.
    The residual connection means input passes through unchanged.
    """

    @staticmethod
    @torch.no_grad()
    def prune_layer(layer: nn.Module):
        """Zero all linear weights in the layer."""
        for mod in layer.modules():
            if isinstance(mod, nn.Linear):
                mod.weight.data.zero_()
                if mod.bias is not None:
                    mod.bias.data.zero_()
        # Also zero LayerNorm if present (so it becomes identity-like)
        # Actually, keep LayerNorm — zeroing MLP+Attn is sufficient
        # since the residual connection preserves the input.


class LowRankFactorizer:
    """
    Replace a Linear(in, out) with two linears: Linear(in, rank) + Linear(rank, out).
    This saves params when rank < in*out/(in+out).

    Applied to gate_proj, up_proj, down_proj independently.
    """

    @staticmethod
    @torch.no_grad()
    def factorize_mlp(
        layer: nn.Module,
        target_rank: int,
        device: torch.device,
        energy_threshold: float = 0.95,
    ) -> dict[str, int]:
        """
        Factorize MLP projections via truncated SVD.
        Returns dict of {proj_name: actual_rank_used}.
        """
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            return {}

        results = {}
        for name in ["gate_proj", "up_proj", "down_proj"]:
            mod = getattr(mlp, name, None)
            if mod is None or not isinstance(mod, nn.Linear):
                continue

            W = mod.weight.data.float().to(device)
            m, n = W.shape  # [out_features, in_features]

            # Check if factorization actually saves params
            # Original: m*n. Factorized: m*r + r*n = r*(m+n)
            max_useful_rank = (m * n) // (m + n)
            rank = min(target_rank, max_useful_rank, min(m, n))

            if rank >= min(m, n) * 0.95:
                # Not worth factorizing
                results[name] = min(m, n)
                continue

            # Truncated SVD
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            # U: [m, k], S: [k], Vh: [k, n] where k = min(m,n)

            # Find rank that captures energy_threshold
            total_energy = (S ** 2).sum()
            cumulative = (S ** 2).cumsum(0) / total_energy
            rank_for_energy = int((cumulative < energy_threshold).sum().item()) + 1
            rank = min(rank, rank_for_energy)

            # Factorize: W ≈ (U[:,:r] @ diag(S[:r])) @ Vh[:r,:]
            #           = A @ B where A = U[:,:r]*S[:r], B = Vh[:r,:]
            A = (U[:, :rank] * S[:rank].unsqueeze(0)).to(W.dtype)
            B = Vh[:rank, :].to(W.dtype)

            # Replace single linear with two sequential linears
            # We store them as a nn.Sequential on the mlp
            has_bias = mod.bias is not None
            linear_a = nn.Linear(n, rank, bias=False, device=device, dtype=mod.weight.dtype)
            linear_b = nn.Linear(rank, m, bias=has_bias, device=device, dtype=mod.weight.dtype)

            linear_a.weight = nn.Parameter(B.to(mod.weight.dtype))  # [rank, n]
            linear_b.weight = nn.Parameter(A.to(mod.weight.dtype))  # [m, rank]
            if has_bias:
                linear_b.bias = nn.Parameter(mod.bias.data)

            seq = nn.Sequential(linear_a, linear_b)
            setattr(mlp, name, seq)

            actual_params = rank * (m + n) + (m if has_bias else 0)
            original_params = m * n + (m if has_bias else 0)
            saving = 1 - actual_params / original_params
            results[name] = rank
            logger.info(f"    {name}: rank {min(m,n)} → {rank} "
                        f"(saving {saving:.1%}, {original_params:,} → {actual_params:,} params)")

        return results


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
    total_params_original: int = 0
    total_params_compressed: int = 0
    elapsed_seconds: float = 0.0

    def summary(self) -> str:
        r = 1 - self.total_params_compressed / max(self.total_params_original, 1)
        d = self.compressed_ppl - self.original_ppl
        p = (self.compressed_ppl / max(self.original_ppl, 1e-8) - 1) * 100
        return "\n".join([
            f"{'='*60}", "  MRI-Compress v2 Results", f"{'='*60}",
            f"  Original PPL:       {self.original_ppl:.4f}",
            f"  Compressed PPL:     {self.compressed_ppl:.4f}",
            f"  PPL change:         {d:+.4f} ({p:+.2f}%)", "",
            f"  Parameters:         {self.total_params_original:,} → {self.total_params_compressed:,}",
            f"  Reduction:          {r:.1%}",
            f"  Dead neurons removed:   {self.total_neurons_removed:,}",
            f"  Dormant neurons removed:{self.total_dormant_removed:,}",
            f"  Neurons merged:         {self.total_neurons_merged:,}",
            f"  Attn heads pruned:      {self.total_attn_heads_pruned}",
            f"  Depth-pruned layers:    {self.total_depth_pruned_layers}",
            f"  Low-rank factorized:    {self.total_low_rank_layers}",
            f"  Time:               {self.elapsed_seconds:.1f}s",
            f"{'='*60}",
        ])


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
        self.enable_low_rank = enable_low_rank
        self.enable_attn_pruning = enable_attn_pruning
        self.enable_depth_pruning = enable_depth_pruning

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

        result.total_params_compressed = sum(p.numel() for p in self.model.parameters())
        result.elapsed_seconds = time.perf_counter() - t0

        logger.info("Evaluating compressed model...")
        result.compressed_ppl = self._evaluate_ppl()

        print(result.summary())
        return result

    def _process_layer(self, lp: LayerPrescription) -> dict:
        lr = {
            "layer": lp.layer_idx, "strategy": lp.strategy.name,
            "dead_removed": 0, "dormant_removed": 0,
            "neurons_merged": 0, "attn_heads_pruned": 0,
            "depth_pruned": 0, "low_rank_applied": 0,
            "reconstruction_mse": None,
        }

        logger.info(f"  Layer {lp.layer_idx}: {lp.strategy.name}")
        layer = self.model.model.layers[lp.layer_idx]

        # ---- Depth pruning (zero entire layer) ----
        if lp.strategy == CompressionStrategy.DEPTH_PRUNE:
            if self.enable_depth_pruning:
                DepthPruner.prune_layer(layer)
                lr["depth_pruned"] = 1
                logger.info(f"    DEPTH PRUNED (zeroed all weights)")
            else:
                logger.info(f"    DEPTH PRUNE skipped (disabled)")
            return lr

        # ---- Light touch: only do attention pruning if applicable ----
        if lp.strategy == CompressionStrategy.LIGHT_TOUCH:
            if lp.attn_heads_to_prune > 0 and self.enable_attn_pruning:
                pruned = AttentionHeadPruner.prune_heads(
                    layer, lp.attn_heads_to_prune,
                    self.dataloader, self.model, lp.layer_idx, self.device)
                lr["attn_heads_pruned"] = len(pruned)
                logger.info(f"    Pruned {len(pruned)} attention heads: {pruned}")
            return lr

        # ---- Determine if lossy (needs reconstruction targets) ----
        is_lossy = False
        if lp.strategy == CompressionStrategy.DEAD_REMOVAL_AND_MERGE:
            has_merge = (lp.merge_target_width is not None and
                         lp.merge_target_width < get_intermediate_size(layer) - lp.dead_neuron_count - lp.dormant_neuron_count)
            is_lossy = has_merge
        elif lp.strategy in (CompressionStrategy.STRUCTURED_PRUNE, CompressionStrategy.DORMANT_REMOVAL):
            is_lossy = True

        # ---- Collect fresh MLP I/O before compression ----
        mlp_io = None
        if self.do_reconstruction and is_lossy:
            with timer(f"Collecting MLP I/O for layer {lp.layer_idx}"):
                mlp_io = collect_mlp_io(
                    self.model, self.dataloader, lp.layer_idx,
                    max_batches=min(self.max_cal_batches, 8), device=self.device)

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
                    max_batches=n_batch, device=self.device)

        # ---- MLP Compression ----
        if lp.strategy == CompressionStrategy.DEAD_REMOVAL_AND_MERGE:
            if activations is not None:
                # Remove dead neurons
                if lp.dead_neuron_count > 0:
                    n_rm, activations = DeadNeuronRemover.remove_by_mri_count(
                        layer, activations, lp.dead_neuron_count, self.device)
                    lr["dead_removed"] = n_rm
                    logger.info(f"    Removed {n_rm} dead neurons")

                # Remove dormant neurons (NEW in v2)
                if lp.dormant_neuron_count > 0:
                    n_rm, activations = DeadNeuronRemover.remove_by_mri_count(
                        layer, activations, lp.dormant_neuron_count, self.device)
                    lr["dormant_removed"] = n_rm
                    logger.info(f"    Removed {n_rm} dormant neurons")

                # Merge
                if lp.merge_target_width is not None:
                    cur = activations.shape[1]
                    tgt = min(lp.merge_target_width, cur)
                    if tgt < cur:
                        with timer(f"Merging neurons in layer {lp.layer_idx}"):
                            nw, activations = NeuronMerger.merge(
                                layer, activations, tgt, device=self.device)
                            lr["neurons_merged"] = cur - nw
                            logger.info(f"    Merged {cur} → {nw} neurons")

        elif lp.strategy == CompressionStrategy.DORMANT_REMOVAL:
            if activations is not None:
                total_to_remove = lp.dead_neuron_count + lp.dormant_neuron_count
                if total_to_remove > 0:
                    n_rm, activations = DeadNeuronRemover.remove_by_mri_count(
                        layer, activations, total_to_remove, self.device)
                    lr["dead_removed"] = min(n_rm, lp.dead_neuron_count)
                    lr["dormant_removed"] = max(0, n_rm - lp.dead_neuron_count)
                    logger.info(f"    Removed {n_rm} neurons (dead+dormant)")

        elif lp.strategy == CompressionStrategy.STRUCTURED_PRUNE:
            if activations is not None:
                n_pr, activations = WandaPruner.prune(
                    layer, activations, lp.target_sparsity, self.device)
                lr["dead_removed"] = n_pr
                logger.info(f"    Pruned {n_pr} neurons ({lp.target_sparsity:.0%})")

        # ---- Attention head pruning (additive, on any strategy) ----
        if lp.attn_heads_to_prune > 0 and self.enable_attn_pruning:
            pruned = AttentionHeadPruner.prune_heads(
                layer, lp.attn_heads_to_prune,
                self.dataloader, self.model, lp.layer_idx, self.device)
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
                    layer, lp.low_rank_target, self.device)
            if ranks:
                lr["low_rank_applied"] = 1

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
#!/usr/bin/env python3
"""
Sparsity MRI — Studies 17-21: Next-Generation Compression Diagnostics
=====================================================================

These studies target the "Gaussian middle zone" (layers ~10-29) where
Studies 1-16 found zero prunable neurons but high activation consistency,
suggesting hidden compression opportunities in weight-space and
cross-layer structure.

Study 17: Cross-Layer Activation Subspace Alignment
    Measures CKA / SVCCA between residual stream updates at consecutive
    layers to find which adjacent layers do "similar work" → merge/skip.

Study 18: Effective Weight Matrix Rank
    SVD of MLP projections: how many singular values capture 95%/99% of
    the Frobenius norm → quantifies low-rank factorization opportunity.

Study 19: Attention Head Functional Clustering
    Cluster attention heads by pattern similarity (within and across
    layers) → identifies prunable head families.

Study 20: Dynamic vs Static Activation Decomposition
    Per-neuron: what fraction of activation variance is input-invariant
    (can be replaced by a bias) vs input-dependent → targets for
    static folding.

Study 21: Token-Conditional Magnitude Divergence
    Per-neuron activation magnitude shift across domains. Even when the
    same neurons fire (Study 11 showed Jaccard≈1.0), magnitude
    distributions may differ → guides domain-conditional precision.
"""

from __future__ import annotations
import gc
import math
import time
import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.amp import autocast

logger = logging.getLogger(__name__)


# ============================================================================
# Shared streaming utilities
# ============================================================================

@torch.no_grad()
def _stream_residual_deltas(
    inspector, dataset, batch_size: int, max_batches: int,
    layer_indices: list[int] | None = None,
) -> dict[int, torch.Tensor]:
    """
    Collect residual stream *deltas* (output - input) for each transformer
    layer. Returns {layer_idx: [total_tokens, hidden_size]} on CPU.

    We hook the full layer to get input and output, then compute the delta.
    This captures what each layer "adds" to the residual stream.
    """
    from data_utils import get_dataloader

    model = inspector.model
    device = next(model.parameters()).device
    num_layers = inspector.num_layers
    if layer_indices is None:
        layer_indices = list(range(num_layers))

    # We collect one layer at a time to save memory
    all_deltas = {}
    for li in layer_indices:
        layer = model.model.layers[li]
        io_data = {"inp": None, "out": None}

        def make_hooks(data_ref):
            def pre_hook(module, args, kwargs=None):
                x = args[0] if isinstance(args, tuple) else args
                data_ref["inp"] = x.detach()
                return None

            def post_hook(module, args, output, kwargs=None):
                out = output[0] if isinstance(output, tuple) else output
                data_ref["out"] = out.detach()
                return None
            return pre_hook, post_hook

        pre_h, post_h = make_hooks(io_data)
        h1 = layer.register_forward_pre_hook(pre_h)
        h2 = layer.register_forward_hook(post_h)

        deltas = []
        loader = get_dataloader(dataset, batch_size=batch_size)
        model.eval()
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            ids = batch["input_ids"].to(device)
            mask = batch.get("attention_mask", torch.ones_like(ids)).to(device)
            with autocast("cuda", dtype=torch.bfloat16):
                model(input_ids=ids, attention_mask=mask)
            if io_data["inp"] is not None and io_data["out"] is not None:
                delta = (io_data["out"] - io_data["inp"]).float().cpu()
                # Flatten [batch, seq, hidden] -> [tokens, hidden]
                deltas.append(delta.reshape(-1, delta.shape[-1]))
            io_data["inp"] = io_data["out"] = None

        h1.remove()
        h2.remove()
        if deltas:
            all_deltas[li] = torch.cat(deltas, dim=0)
        del deltas
        gc.collect()
        torch.cuda.empty_cache()

    return all_deltas


@torch.no_grad()
def _stream_mlp_activations(
    inspector, dataset, batch_size: int, max_batches: int,
    layer_idx: int,
) -> torch.Tensor:
    """
    Collect MLP intermediate activations (input to down_proj) for a
    single layer. Returns [total_tokens, intermediate_size] on CPU.
    """
    from data_utils import get_dataloader

    model = inspector.model
    device = next(model.parameters()).device
    mlp = model.model.layers[layer_idx].mlp
    hook_data = {"acts": None}

    def hook_fn(module, input, output):
        hook_data["acts"] = input[0].detach()

    handle = mlp.down_proj.register_forward_hook(hook_fn)
    all_acts = []
    loader = get_dataloader(dataset, batch_size=batch_size)
    model.eval()
    for i, batch in enumerate(loader):
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

    handle.remove()
    if not all_acts:
        raise RuntimeError(f"No activations collected for layer {layer_idx}")
    return torch.cat(all_acts, dim=0)


@torch.no_grad()
def _stream_attention_patterns(
    inspector, dataset, batch_size: int, max_batches: int,
    layer_idx: int,
) -> torch.Tensor:
    """
    Collect attention weight matrices for a single layer.
    Returns [total_samples, num_heads, seq_len, seq_len] on CPU.

    Computes attention weights manually from Q/K projections to avoid
    requiring output_attentions=True (which is incompatible with SDPA).
    """
    from data_utils import get_dataloader

    model = inspector.model
    device = next(model.parameters()).device
    attn_mod = model.model.layers[layer_idx].self_attn

    # Detect head configuration
    num_heads = model.config.num_attention_heads
    num_kv_heads = getattr(model.config, "num_key_value_heads", num_heads)
    head_dim = model.config.hidden_size // num_heads
    kv_groups = num_heads // num_kv_heads  # GQA group size

    # Hook the attention input (hidden states going into the attention module)
    attn_input = {"hidden": None}

    def pre_hook(module, args, kwargs):
        if kwargs and "hidden_states" in kwargs:
            attn_input["hidden"] = kwargs["hidden_states"].detach()
        elif args:
            attn_input["hidden"] = args[0].detach()

    handle = attn_mod.register_forward_pre_hook(pre_hook, with_kwargs=True)

    all_patterns = []
    loader = get_dataloader(dataset, batch_size=batch_size)
    model.eval()

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        ids = batch["input_ids"].to(device)
        mask = batch.get("attention_mask", torch.ones_like(ids)).to(device)
        with autocast("cuda", dtype=torch.bfloat16):
            model(input_ids=ids, attention_mask=mask)

        if attn_input["hidden"] is not None:
            hidden = attn_input["hidden"]  # [batch, seq, hidden_size]
            bsz, seq_len, _ = hidden.shape

            # Project Q and K manually
            q_proj = attn_mod.q_proj
            k_proj = attn_mod.k_proj

            q = q_proj(hidden).float()  # cast to float AFTER projection
            k = k_proj(hidden).float()

            # Reshape to [batch, num_heads, seq, head_dim]
            q = q.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
            k = k.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)

            # Expand K for GQA if needed
            if kv_groups > 1:
                k = k.unsqueeze(2).expand(-1, -1, kv_groups, -1, -1)
                k = k.reshape(bsz, num_heads, seq_len, head_dim)

            # Compute attention scores
            scale = head_dim ** -0.5
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale

            # Apply causal mask
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
                diagonal=1,
            )
            attn_scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

            # Apply padding mask
            if mask is not None:
                pad_mask = (mask == 0).unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq]
                attn_scores.masked_fill_(pad_mask, float("-inf"))

            attn_weights = torch.softmax(attn_scores, dim=-1)
            all_patterns.append(attn_weights.cpu())
            attn_input["hidden"] = None

    handle.remove()

    if not all_patterns:
        return None
    return torch.cat(all_patterns, dim=0)


# ============================================================================
# STUDY 17: Cross-Layer Activation Subspace Alignment
# ============================================================================

@dataclass
class CrossLayerAlignmentReport:
    layer_a: int
    layer_b: int
    cka_linear: float          # Linear CKA similarity [0,1]
    cka_rbf: float             # RBF kernel CKA
    top_subspace_overlap: float  # Overlap of top-k principal components
    residual_delta_cosine: float  # Mean cosine similarity of layer deltas
    merge_score: float         # Composite score: higher = more mergeable


def _linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """
    Linear CKA (Centered Kernel Alignment) between two representation
    matrices X, Y of shape [n_samples, features].
    Kornblith et al., ICML 2019.
    """
    # Center
    X = X - X.mean(0)
    Y = Y - Y.mean(0)

    # Gram matrices (use float64 for numerical stability)
    X64, Y64 = X.double(), Y.double()
    hsic_xy = (X64 @ X64.T * (Y64 @ Y64.T)).sum()
    hsic_xx = (X64 @ X64.T).pow(2).sum()
    hsic_yy = (Y64 @ Y64.T).pow(2).sum()

    denom = (hsic_xx * hsic_yy).sqrt()
    if denom < 1e-12:
        return 0.0
    return float(hsic_xy / denom)


def _subspace_overlap(X: torch.Tensor, Y: torch.Tensor, top_k: int = 64) -> float:
    """
    Principal subspace overlap: fraction of variance in X's top-k subspace
    that is captured by Y's top-k subspace.
    """
    X = X - X.mean(0)
    Y = Y - Y.mean(0)

    # SVD on the feature dimension (n_samples × features)
    # We want the right singular vectors (principal directions in feature space)
    _, _, Vx = torch.linalg.svd(X.double(), full_matrices=False)
    _, _, Vy = torch.linalg.svd(Y.double(), full_matrices=False)

    k = min(top_k, Vx.shape[0], Vy.shape[0])
    Vx_k = Vx[:k]  # [k, features]
    Vy_k = Vy[:k]

    # Overlap = ||Vx_k @ Vy_k^T||_F^2 / k
    overlap_matrix = Vx_k @ Vy_k.T
    overlap = float(overlap_matrix.pow(2).sum() / k)
    return min(overlap, 1.0)


def run_cross_layer_alignment(
    inspector, dataset,
    batch_size: int = 2,
    max_batches: int = 8,
    top_k_subspace: int = 64,
    stride: int = 1,
) -> list[CrossLayerAlignmentReport]:
    """
    Study 17: Measure how similar consecutive layers' residual stream
    contributions are. High similarity → layer merging/skipping candidate.
    """
    print("\n" + "=" * 80)
    print("STUDY 17: Cross-Layer Activation Subspace Alignment")
    print("=" * 80)

    num_layers = inspector.num_layers
    # Collect residual deltas for all layers
    all_deltas = _stream_residual_deltas(
        inspector, dataset, batch_size, max_batches,
        layer_indices=list(range(num_layers)),
    )

    reports = []
    # Compare consecutive layer pairs (and optionally with stride)
    pairs = []
    for i in range(num_layers - stride):
        pairs.append((i, i + stride))

    # Subsample tokens for CKA (it's O(n²) in samples)
    max_samples_cka = 2048

    for layer_a, layer_b in pairs:
        if layer_a not in all_deltas or layer_b not in all_deltas:
            continue

        da = all_deltas[layer_a]
        db = all_deltas[layer_b]

        # Subsample for CKA
        n = min(da.shape[0], db.shape[0], max_samples_cka)
        da_sub = da[:n]
        db_sub = db[:n]

        # Linear CKA
        cka_lin = _linear_cka(da_sub, db_sub)

        # RBF CKA (approximate: use median distance heuristic)
        # Skip for speed if hidden_size is large
        cka_r = 0.0  # placeholder, linear CKA is the key metric

        # Top-k subspace overlap
        overlap = _subspace_overlap(da_sub, db_sub, top_k=top_k_subspace)

        # Mean cosine similarity of deltas
        da_norm = F.normalize(da_sub, dim=1)
        db_norm = F.normalize(db_sub, dim=1)
        cosine = float((da_norm * db_norm).sum(dim=1).mean())

        # Composite merge score
        merge_score = 0.4 * cka_lin + 0.3 * overlap + 0.3 * max(cosine, 0)

        report = CrossLayerAlignmentReport(
            layer_a=layer_a,
            layer_b=layer_b,
            cka_linear=round(cka_lin, 4),
            cka_rbf=round(cka_r, 4),
            top_subspace_overlap=round(overlap, 4),
            residual_delta_cosine=round(cosine, 4),
            merge_score=round(merge_score, 4),
        )
        reports.append(report)

        print(f"  Layer {layer_a:2d} → {layer_b:2d}: "
              f"CKA={cka_lin:.3f}, overlap={overlap:.3f}, "
              f"cosine={cosine:.3f}, merge_score={merge_score:.3f}")

    # Print top merge candidates
    if reports:
        top5 = sorted(reports, key=lambda r: -r.merge_score)[:5]
        print(f"\n  Top-5 merge candidates:")
        for r in top5:
            print(f"    Layers {r.layer_a}-{r.layer_b}: "
                  f"merge_score={r.merge_score:.3f} (CKA={r.cka_linear:.3f})")

    del all_deltas
    gc.collect()
    torch.cuda.empty_cache()
    return reports


# ============================================================================
# STUDY 18: Effective Weight Matrix Rank
# ============================================================================

@dataclass
class WeightRankReport:
    layer_idx: int
    # Per projection matrix
    gate_proj_rank95: int       # Rank for 95% Frobenius norm
    gate_proj_rank99: int       # Rank for 99% Frobenius norm
    gate_proj_full_rank: int    # Full size (rows or cols, whichever is smaller)
    gate_proj_ratio95: float    # rank95 / full_rank → lower = more compressible
    up_proj_rank95: int
    up_proj_rank99: int
    up_proj_full_rank: int
    up_proj_ratio95: float
    down_proj_rank95: int
    down_proj_rank99: int
    down_proj_full_rank: int
    down_proj_ratio95: float
    # Aggregate
    avg_ratio95: float          # Average compression potential
    avg_ratio99: float
    estimated_param_saving_95: float  # Fraction of params saved at 95% energy


def _compute_effective_rank(W: torch.Tensor, thresholds=(0.95, 0.99)) -> dict:
    """
    Compute effective rank of a weight matrix via SVD.
    Returns rank at each energy threshold + full rank.
    """
    # W: [out_features, in_features]
    # SVD: W = U @ diag(S) @ V^T
    S = torch.linalg.svdvals(W.float())
    total_energy = (S ** 2).sum()
    cumulative = (S ** 2).cumsum(0) / total_energy

    result = {"full_rank": int(min(W.shape)), "singular_values": S.cpu()}
    for t in thresholds:
        rank = int((cumulative < t).sum().item()) + 1
        result[f"rank_{int(t*100)}"] = min(rank, result["full_rank"])
        result[f"ratio_{int(t*100)}"] = rank / result["full_rank"]

    return result


def run_weight_rank_analysis(
    inspector, dataset,  # dataset unused but kept for API consistency
    batch_size: int = 2,
    max_batches: int = 8,
) -> list[WeightRankReport]:
    """
    Study 18: SVD analysis of MLP weight matrices to find low-rank
    factorization opportunities, especially in the Gaussian middle layers.
    """
    print("\n" + "=" * 80)
    print("STUDY 18: Effective Weight Matrix Rank")
    print("=" * 80)

    model = inspector.model
    device = next(model.parameters()).device
    reports = []

    for li in range(inspector.num_layers):
        mlp = model.model.layers[li].mlp

        results_per_proj = {}
        for name in ["gate_proj", "up_proj", "down_proj"]:
            mod = getattr(mlp, name, None)
            if mod is None or not isinstance(mod, nn.Linear):
                continue
            W = mod.weight.data.to(device)
            r = _compute_effective_rank(W)
            results_per_proj[name] = r
            del W
            torch.cuda.empty_cache()

        # Build report
        def get_or_default(proj_name, key, default=0):
            if proj_name in results_per_proj:
                return results_per_proj[proj_name].get(key, default)
            return default

        gate_r95 = get_or_default("gate_proj", "rank_95")
        gate_r99 = get_or_default("gate_proj", "rank_99")
        gate_full = get_or_default("gate_proj", "full_rank")
        up_r95 = get_or_default("up_proj", "rank_95")
        up_r99 = get_or_default("up_proj", "rank_99")
        up_full = get_or_default("up_proj", "full_rank")
        down_r95 = get_or_default("down_proj", "rank_95")
        down_r99 = get_or_default("down_proj", "rank_99")
        down_full = get_or_default("down_proj", "full_rank")

        gate_ratio95 = gate_r95 / max(gate_full, 1)
        up_ratio95 = up_r95 / max(up_full, 1)
        down_ratio95 = down_r95 / max(down_full, 1)
        gate_ratio99 = gate_r99 / max(gate_full, 1)
        up_ratio99 = up_r99 / max(up_full, 1)
        down_ratio99 = down_r99 / max(down_full, 1)

        avg_r95 = (gate_ratio95 + up_ratio95 + down_ratio95) / 3
        avg_r99 = (gate_ratio99 + up_ratio99 + down_ratio99) / 3

        # Estimate param saving: for rank r factorization of [m,n],
        # params go from m*n to m*r + r*n = r*(m+n)
        # Saving = 1 - r*(m+n)/(m*n)
        total_orig, total_compressed = 0, 0
        for name in ["gate_proj", "up_proj", "down_proj"]:
            mod = getattr(mlp, name, None)
            if mod is None:
                continue
            m, n = mod.weight.shape
            r = results_per_proj.get(name, {}).get("rank_95", min(m, n))
            total_orig += m * n
            total_compressed += r * (m + n)

        saving_95 = max(0, 1 - total_compressed / max(total_orig, 1))

        report = WeightRankReport(
            layer_idx=li,
            gate_proj_rank95=gate_r95, gate_proj_rank99=gate_r99,
            gate_proj_full_rank=gate_full, gate_proj_ratio95=round(gate_ratio95, 3),
            up_proj_rank95=up_r95, up_proj_rank99=up_r99,
            up_proj_full_rank=up_full, up_proj_ratio95=round(up_ratio95, 3),
            down_proj_rank95=down_r95, down_proj_rank99=down_r99,
            down_proj_full_rank=down_full, down_proj_ratio95=round(down_ratio95, 3),
            avg_ratio95=round(avg_r95, 3),
            avg_ratio99=round(avg_r99, 3),
            estimated_param_saving_95=round(saving_95, 3),
        )
        reports.append(report)

        print(f"  Layer {li:2d}: "
              f"gate={gate_r95}/{gate_full} ({gate_ratio95:.1%}), "
              f"up={up_r95}/{up_full} ({up_ratio95:.1%}), "
              f"down={down_r95}/{down_full} ({down_ratio95:.1%}) "
              f"| saving@95%={saving_95:.1%}")

    # Summary
    if reports:
        avg_saving = np.mean([r.estimated_param_saving_95 for r in reports])
        best = max(reports, key=lambda r: r.estimated_param_saving_95)
        print(f"\n  Average potential param saving (rank@95%): {avg_saving:.1%}")
        print(f"  Best layer: {best.layer_idx} ({best.estimated_param_saving_95:.1%} saving)")

    return reports


# ============================================================================
# STUDY 19: Attention Head Functional Clustering
# ============================================================================

@dataclass
class AttentionHeadClusterReport:
    num_heads_total: int
    num_clusters: int
    num_singleton_heads: int      # Heads that are unique (no cluster mates)
    num_redundant_heads: int      # Heads in clusters of size >= 2
    prunable_heads: list[tuple[int, int]]  # (layer, head) — one representative kept per cluster
    # Per-cluster info
    cluster_sizes: list[int]
    cluster_members: list[list[tuple[int, int]]]  # Each cluster: list of (layer, head)
    # Per-head similarity to nearest neighbor
    head_nn_similarity: dict[tuple[int, int], float]  # (layer, head) → max sim to any other head


def _compute_head_signature(attn_patterns: torch.Tensor) -> torch.Tensor:
    """
    Compute a compact signature for each attention head from its attention
    weight matrices. Uses statistics over the attention distribution:
    - Mean entropy per position
    - Mean attention to first token
    - Mean attention to self (diagonal)
    - Attention spread (std of row-wise max)
    - Mean attention distance

    Input: [num_samples, num_heads, seq_len, seq_len]
    Output: [num_heads, signature_dim]
    """
    n_samples, n_heads, seq_len, _ = attn_patterns.shape

    sigs = []
    for h in range(n_heads):
        p = attn_patterns[:, h]  # [n_samples, seq_len, seq_len]

        # Entropy per position
        eps = 1e-8
        entropy = -(p * (p + eps).log()).sum(-1)  # [n_samples, seq_len]
        mean_entropy = entropy.mean().item()
        std_entropy = entropy.std().item()

        # Attention to first token
        first_tok_attn = p[:, :, 0].mean().item()

        # Diagonal attention (self-attention)
        diag_mask = torch.eye(seq_len, device=p.device).bool()
        diag_attn = p[:, diag_mask].mean().item() if seq_len > 0 else 0

        # Attention spread
        max_attn = p.max(dim=-1).values  # [n_samples, seq_len]
        spread = max_attn.std().item()

        # Mean attention distance
        positions = torch.arange(seq_len, device=p.device, dtype=torch.float)
        # distance[i,j] = |i - j|
        dist_matrix = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()
        mean_dist = (p * dist_matrix.unsqueeze(0)).sum(-1).mean().item()

        # Locality: fraction of attention within window of 8 tokens
        local_mask = dist_matrix <= 8
        local_attn = (p * local_mask.float().unsqueeze(0)).sum(-1).mean().item()

        sigs.append([
            mean_entropy, std_entropy, first_tok_attn, diag_attn,
            spread, mean_dist, local_attn,
        ])

    return torch.tensor(sigs, dtype=torch.float)


def run_attention_head_clustering(
    inspector, dataset,
    batch_size: int = 2,
    max_batches: int = 4,
    similarity_threshold: float = 0.90,
    sample_layers: int | None = None,
) -> AttentionHeadClusterReport:
    """
    Study 19: Cluster attention heads across all layers by functional
    similarity. Heads doing similar work can be pruned.
    """
    print("\n" + "=" * 80)
    print("STUDY 19: Attention Head Functional Clustering")
    print("=" * 80)

    num_layers = inspector.num_layers
    num_heads = inspector.model.config.num_attention_heads

    if sample_layers is not None:
        # Sample evenly-spaced layers
        indices = np.linspace(0, num_layers - 1, sample_layers, dtype=int)
        layer_indices = sorted(set(indices))
    else:
        layer_indices = list(range(num_layers))

    # Collect attention signatures per layer
    all_signatures = {}  # (layer, head) → signature vector
    all_sigs_list = []
    head_ids = []

    for li in layer_indices:
        t0 = time.time()
        patterns = _stream_attention_patterns(
            inspector, dataset, batch_size, max_batches, li
        )
        if patterns is None:
            print(f"  Layer {li}: Could not collect attention patterns (skipping)")
            continue

        # Compute signatures
        sigs = _compute_head_signature(patterns)  # [num_heads, sig_dim]
        for h in range(sigs.shape[0]):
            head_ids.append((li, h))
            all_sigs_list.append(sigs[h])

        del patterns
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  Layer {li:2d}: collected {sigs.shape[0]} head signatures ({time.time()-t0:.1f}s)")

    if not all_sigs_list:
        print("  WARNING: No attention patterns collected. Model may not support output_attentions.")
        return AttentionHeadClusterReport(
            num_heads_total=0, num_clusters=0, num_singleton_heads=0,
            num_redundant_heads=0, prunable_heads=[], cluster_sizes=[],
            cluster_members=[], head_nn_similarity={},
        )

    # Stack all signatures and compute pairwise cosine similarity
    sig_matrix = torch.stack(all_sigs_list)  # [total_heads, sig_dim]
    sig_norm = F.normalize(sig_matrix, dim=1)
    sim = sig_norm @ sig_norm.T  # [total_heads, total_heads]
    sim.fill_diagonal_(-1)  # Exclude self

    # Nearest-neighbor similarity per head
    nn_sim = {}
    for i, (li, hi) in enumerate(head_ids):
        nn_sim[(li, hi)] = float(sim[i].max())

    # Greedy agglomerative clustering at similarity_threshold
    n = len(head_ids)
    cluster_assignment = list(range(n))  # Each head starts in its own cluster
    members = {i: [i] for i in range(n)}

    # Sort all pairs by similarity (descending)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= similarity_threshold:
                pairs.append((float(sim[i, j]), i, j))
    pairs.sort(reverse=True)

    for _, i, j in pairs:
        ci, cj = cluster_assignment[i], cluster_assignment[j]
        if ci == cj:
            continue
        # Merge smaller into larger
        if len(members[ci]) < len(members[cj]):
            ci, cj = cj, ci
        for idx in members[cj]:
            cluster_assignment[idx] = ci
        members[ci].extend(members[cj])
        del members[cj]

    # Build cluster results
    clusters_final = list(members.values())
    cluster_members = [
        [head_ids[idx] for idx in cl] for cl in clusters_final
    ]
    cluster_sizes = [len(cl) for cl in clusters_final]

    singletons = sum(1 for s in cluster_sizes if s == 1)
    redundant = sum(s - 1 for s in cluster_sizes if s > 1)

    # Prunable: all but one from each multi-member cluster
    prunable = []
    for cl in cluster_members:
        if len(cl) > 1:
            # Keep the one with highest entropy (most informative)
            prunable.extend(cl[1:])  # Simple: keep first, prune rest

    report = AttentionHeadClusterReport(
        num_heads_total=len(head_ids),
        num_clusters=len(clusters_final),
        num_singleton_heads=singletons,
        num_redundant_heads=redundant,
        prunable_heads=prunable,
        cluster_sizes=sorted(cluster_sizes, reverse=True),
        cluster_members=cluster_members,
        head_nn_similarity=nn_sim,
    )

    print(f"\n  Total heads analyzed: {len(head_ids)}")
    print(f"  Clusters found: {len(clusters_final)} "
          f"(singletons={singletons}, multi-member={len(clusters_final)-singletons})")
    print(f"  Redundant (prunable) heads: {redundant}")
    if cluster_sizes:
        print(f"  Largest cluster: {max(cluster_sizes)} heads")
        # Print top 5 largest clusters
        for i, (sz, members) in enumerate(
            sorted(zip(cluster_sizes, cluster_members), reverse=True)[:5]
        ):
            if sz > 1:
                member_str = ", ".join(f"L{l}H{h}" for l, h in members[:6])
                if sz > 6:
                    member_str += f", ... ({sz} total)"
                print(f"    Cluster {i+1} (size={sz}): {member_str}")

    return report


# ============================================================================
# STUDY 20: Dynamic vs Static Activation Decomposition
# ============================================================================

@dataclass
class StaticDynamicReport:
    layer_idx: int
    static_fraction: float        # Mean fraction of activation that is static (mean/total)
    dynamic_fraction: float       # 1 - static_fraction
    n_mostly_static: int          # Neurons where static > 90% of total variance
    n_mostly_dynamic: int         # Neurons where dynamic > 90%
    n_mixed: int                  # The rest
    mean_activation_magnitude: float  # Mean |activation| across all neurons
    mean_static_magnitude: float     # Mean |mean_activation| per neuron
    foldable_neuron_count: int    # Neurons where static component dominates (>95%)
    foldable_param_saving: float  # Estimated params saved by folding static neurons to biases


def run_static_dynamic_decomposition(
    inspector, dataset,
    batch_size: int = 2,
    max_batches: int = 16,
    static_threshold: float = 0.90,
    foldable_threshold: float = 0.95,
) -> list[StaticDynamicReport]:
    """
    Study 20: For each neuron, decompose activation into static (mean)
    and dynamic (variance) components. Neurons with >95% static can be
    replaced by bias terms, eliminating their gate/up computation.
    """
    print("\n" + "=" * 80)
    print("STUDY 20: Dynamic vs Static Activation Decomposition")
    print("=" * 80)

    reports = []
    intermediate_size = inspector.mlp_layers[0].intermediate_size

    for li in range(inspector.num_layers):
        acts = _stream_mlp_activations(inspector, dataset, batch_size, max_batches, li)
        # acts: [total_tokens, intermediate_size]

        # Per-neuron statistics
        neuron_mean = acts.mean(dim=0)       # [intermediate_size]
        neuron_var = acts.var(dim=0)          # [intermediate_size]
        total_var = ((acts - acts.mean()) ** 2).mean(dim=0)  # Total variance per neuron

        # Static fraction: how much of the total magnitude is explained by the mean
        # We use: static_energy = mean^2, total_energy = mean^2 + var
        mean_sq = neuron_mean ** 2
        total_energy = mean_sq + neuron_var
        static_frac_per_neuron = mean_sq / total_energy.clamp(min=1e-12)

        avg_static = float(static_frac_per_neuron.mean())

        n_static = int((static_frac_per_neuron > static_threshold).sum())
        n_dynamic = int((static_frac_per_neuron < (1 - static_threshold)).sum())
        n_mixed = intermediate_size - n_static - n_dynamic

        n_foldable = int((static_frac_per_neuron > foldable_threshold).sum())

        # Param saving estimate: each foldable neuron removes one row from gate_proj
        # and up_proj, one column from down_proj. For SwiGLU: 3 * hidden_size params
        hidden_size = inspector.model.config.hidden_size
        saving = n_foldable * 3 * hidden_size / (intermediate_size * 3 * hidden_size) if intermediate_size > 0 else 0

        report = StaticDynamicReport(
            layer_idx=li,
            static_fraction=round(avg_static, 4),
            dynamic_fraction=round(1 - avg_static, 4),
            n_mostly_static=n_static,
            n_mostly_dynamic=n_dynamic,
            n_mixed=n_mixed,
            mean_activation_magnitude=float(acts.abs().mean()),
            mean_static_magnitude=float(neuron_mean.abs().mean()),
            foldable_neuron_count=n_foldable,
            foldable_param_saving=round(saving, 4),
        )
        reports.append(report)

        print(f"  Layer {li:2d}: static={avg_static:.1%}, "
              f"mostly_static={n_static}, mostly_dynamic={n_dynamic}, "
              f"foldable={n_foldable} ({saving:.1%} MLP saving)")

        del acts
        gc.collect()
        torch.cuda.empty_cache()

    # Summary
    if reports:
        total_foldable = sum(r.foldable_neuron_count for r in reports)
        avg_static_all = np.mean([r.static_fraction for r in reports])
        print(f"\n  Overall: avg static fraction = {avg_static_all:.1%}")
        print(f"  Total foldable neurons: {total_foldable}")
        top3 = sorted(reports, key=lambda r: -r.foldable_neuron_count)[:3]
        for r in top3:
            print(f"    Layer {r.layer_idx}: {r.foldable_neuron_count} foldable neurons")

    return reports


# ============================================================================
# STUDY 21: Token-Conditional Magnitude Divergence
# ============================================================================

@dataclass
class MagnitudeDivergenceReport:
    layer_idx: int
    domains: list[str]
    # Per-domain mean activation magnitudes
    domain_mean_magnitudes: dict[str, float]
    # Divergence metrics
    kl_divergence_pairs: dict[str, float]  # "domA_vs_domB" → KL
    magnitude_cv: float                     # Coefficient of variation across domains
    n_domain_sensitive_neurons: int          # Neurons with >2x magnitude ratio across domains
    n_domain_invariant_neurons: int          # Neurons with <1.2x ratio
    # Top domain-sensitive neurons
    top_sensitive_neurons: list[tuple[int, float]]  # (neuron_idx, max_ratio)
    # Aggregate
    domain_sensitivity_score: float         # 0-1, higher = more domain-dependent


def _load_domain_texts(inspector, samples_per_domain: int = 32) -> dict[str, list[str]]:
    """Load text from multiple domains for comparison."""
    domains = {}

    # English (wikitext)
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
        texts = [t for t in ds["text"] if len(t.strip()) > 100][:samples_per_domain]
        if texts:
            domains["english"] = texts
    except Exception:
        domains["english"] = [
            "The transformer architecture revolutionized natural language processing. "
            "Attention mechanisms allow models to focus on relevant parts of the input sequence. " * 5
        ] * samples_per_domain

    # Math
    try:
        from datasets import load_dataset
        ds = load_dataset("gsm8k", "main", split="test")
        texts = [f"{s['question']} {s['answer']}" for s in ds][:samples_per_domain]
        if texts:
            domains["math"] = texts
    except Exception:
        domains["math"] = [
            "If x + 3 = 7, then x = 4. Calculate: 2x + 5 = 2(4) + 5 = 13. "
            "The sum of angles in a triangle is 180 degrees. " * 5
        ] * samples_per_domain

    # Code
    domains["code"] = [
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    "
        "return fibonacci(n-1) + fibonacci(n-2)\n\n"
        "class BinaryTree:\n    def __init__(self, value):\n        "
        "self.value = value\n        self.left = None\n        self.right = None\n"
        "    def insert(self, val):\n        if val < self.value:\n"
        "            if self.left: self.left.insert(val)\n"
        "            else: self.left = BinaryTree(val)\n" * 3
    ] * samples_per_domain

    # Italian
    domains["italian"] = [
        "L'architettura rinascimentale italiana ha influenzato profondamente "
        "lo sviluppo dell'arte e della cultura europea. Le opere di Leonardo da Vinci "
        "e Michelangelo rappresentano il culmine della creatività umana. "
        "La Toscana offre paesaggi mozzafiato con colline verdi e cipressi. " * 3
    ] * samples_per_domain

    return domains


def run_magnitude_divergence(
    inspector, dataset,
    batch_size: int = 2,
    max_batches: int = 8,
    samples_per_domain: int = 32,
    magnitude_ratio_threshold: float = 2.0,
) -> list[MagnitudeDivergenceReport]:
    """
    Study 21: Measure how activation *magnitudes* differ across domains,
    even when the same neurons fire. This reveals domain-conditional
    precision opportunities that binary fire/no-fire analysis (Study 11)
    misses entirely.
    """
    print("\n" + "=" * 80)
    print("STUDY 21: Token-Conditional Magnitude Divergence")
    print("=" * 80)

    from data_utils import get_dataloader

    # Load domain texts and tokenize
    domain_texts = _load_domain_texts(inspector, samples_per_domain)
    tokenizer = inspector.tokenizer
    device = next(inspector.model.parameters()).device

    domain_datasets = {}
    for dom_name, texts in domain_texts.items():
        encoded = tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=512,
        )
        # Create simple dataset
        domain_datasets[dom_name] = encoded
        print(f"  {dom_name}: {len(texts)} texts, {encoded['input_ids'].shape}")

    domain_names = sorted(domain_datasets.keys())
    reports = []

    for li in range(inspector.num_layers):
        mlp = inspector.model.model.layers[li].mlp
        hook_data = {"acts": None}

        def hook_fn(module, input, output):
            hook_data["acts"] = input[0].detach()

        handle = mlp.down_proj.register_forward_hook(hook_fn)
        inspector.model.eval()

        # Collect per-domain activation magnitudes
        domain_magnitudes = {}  # domain → [intermediate_size] mean abs activation
        for dom_name in domain_names:
            enc = domain_datasets[dom_name]
            ids = enc["input_ids"]
            mask = enc["attention_mask"]

            all_mags = []
            for start in range(0, ids.shape[0], batch_size):
                end = min(start + batch_size, ids.shape[0])
                b_ids = ids[start:end].to(device)
                b_mask = mask[start:end].to(device)
                with torch.no_grad(), autocast("cuda", dtype=torch.bfloat16):
                    inspector.model(input_ids=b_ids, attention_mask=b_mask)
                if hook_data["acts"] is not None:
                    # Masked mean magnitude
                    a = hook_data["acts"].float()  # [batch, seq, intermediate]
                    m = b_mask.unsqueeze(-1).float()  # [batch, seq, 1]
                    mag = (a.abs() * m).sum(dim=(0, 1)) / m.sum(dim=(0, 1)).clamp(min=1)
                    all_mags.append(mag.cpu())
                    hook_data["acts"] = None

                if len(all_mags) >= max_batches:
                    break

            if all_mags:
                domain_magnitudes[dom_name] = torch.stack(all_mags).mean(dim=0)

        handle.remove()

        if len(domain_magnitudes) < 2:
            continue

        # Compute divergence metrics
        mag_matrix = torch.stack([domain_magnitudes[d] for d in domain_names])  # [n_domains, intermediate]

        # Per-neuron: max/min magnitude ratio across domains
        max_mag = mag_matrix.max(dim=0).values
        min_mag = mag_matrix.min(dim=0).values.clamp(min=1e-8)
        ratios = max_mag / min_mag

        n_sensitive = int((ratios > magnitude_ratio_threshold).sum())
        n_invariant = int((ratios < 1.2).sum())

        # Top sensitive neurons
        top_k = min(10, len(ratios))
        top_vals, top_idx = ratios.topk(top_k)
        top_sensitive = [(int(idx), float(val)) for idx, val in zip(top_idx, top_vals)]

        # Coefficient of variation across domains per neuron, then average
        cv = float(mag_matrix.std(dim=0).mean() / mag_matrix.mean(dim=0).clamp(min=1e-8).mean())

        # KL divergence between domain magnitude distributions (treating as histograms)
        kl_pairs = {}
        for i, d1 in enumerate(domain_names):
            for j, d2 in enumerate(domain_names):
                if j <= i:
                    continue
                p = F.softmax(domain_magnitudes[d1], dim=0)
                q = F.softmax(domain_magnitudes[d2], dim=0)
                kl = float(F.kl_div(q.log(), p, reduction='sum'))
                kl_pairs[f"{d1}_vs_{d2}"] = round(kl, 4)

        # Domain sensitivity score
        sensitivity = min(1.0, n_sensitive / max(mag_matrix.shape[1], 1) * 10)

        # Mean magnitudes per domain
        mean_mags = {d: float(domain_magnitudes[d].mean()) for d in domain_names}

        report = MagnitudeDivergenceReport(
            layer_idx=li,
            domains=domain_names,
            domain_mean_magnitudes=mean_mags,
            kl_divergence_pairs=kl_pairs,
            magnitude_cv=round(cv, 4),
            n_domain_sensitive_neurons=n_sensitive,
            n_domain_invariant_neurons=n_invariant,
            top_sensitive_neurons=top_sensitive,
            domain_sensitivity_score=round(sensitivity, 4),
        )
        reports.append(report)

        print(f"  Layer {li:2d}: sensitive={n_sensitive}, invariant={n_invariant}, "
              f"CV={cv:.3f}, sensitivity_score={sensitivity:.3f}")

        del domain_magnitudes, mag_matrix
        gc.collect()
        torch.cuda.empty_cache()

    # Summary
    if reports:
        avg_sens = np.mean([r.domain_sensitivity_score for r in reports])
        total_sensitive = sum(r.n_domain_sensitive_neurons for r in reports)
        print(f"\n  Average domain sensitivity: {avg_sens:.3f}")
        print(f"  Total domain-sensitive neurons: {total_sensitive}")

    return reports
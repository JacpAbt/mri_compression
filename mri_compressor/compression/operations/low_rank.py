"""
Low-Rank Factorization Operation
===================================
Replace a Linear(in, out) with two linears: Linear(in, rank) + Linear(rank, out).
This saves params when rank < in*out/(in+out).

Applied to gate_proj, up_proj, down_proj independently.
"""

from __future__ import annotations
import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class LowRankFactorizer:

    @staticmethod
    @torch.no_grad()
    def factorize_mlp(
        layer: nn.Module,
        target_rank: int,
        device: torch.device,
        energy_threshold: float = 0.99,
        per_proj_ranks: Optional[dict] = None,
    ) -> dict[str, int]:
        """
        Factorize MLP projections via truncated SVD.
        Returns dict of {proj_name: actual_rank_used}.

        Args:
            per_proj_ranks: Optional dict mapping projection name → target rank.
                When provided, only projections present in this dict are factorized
                (projections absent are skipped — their low-rank structure is
                insufficient to justify compression). Overrides target_rank per
                projection. When None, all projections use target_rank.
        """
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            return {}

        results = {}
        for name in ["gate_proj", "up_proj", "down_proj"]:
            mod = getattr(mlp, name, None)
            if mod is None or not isinstance(mod, nn.Linear):
                continue

            # Determine per-projection rank target
            if per_proj_ranks is not None:
                if name not in per_proj_ranks:
                    continue   # not enough low-rank structure for this projection
                proj_target_rank = per_proj_ranks[name]
            else:
                proj_target_rank = target_rank

            W = mod.weight.data.float().to(device)
            m, n = W.shape  # [out_features, in_features]

            # Check if factorization actually saves params
            # Original: m*n. Factorized: m*r + r*n = r*(m+n)
            max_useful_rank = (m * n) // (m + n)
            rank = min(proj_target_rank, max_useful_rank, min(m, n))

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

            # Factorize: W ~ (U[:,:r] @ diag(S[:r])) @ Vh[:r,:]
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
            logger.info(f"    {name}: rank {min(m,n)} -> {rank} "
                        f"(saving {saving:.1%}, {original_params:,} -> {actual_params:,} params)")

        return results

    @staticmethod
    @torch.no_grad()
    def factorize_module(
        mod: nn.Linear,
        target_rank: int,
        device: torch.device,
        energy_threshold: float = 0.99,
    ) -> bool:
        """
        In-place low-rank approximation of a single nn.Linear weight matrix.

        Replaces mod.weight.data with its rank-r SVD approximation:
            W ≈ U[:,:r] @ diag(S[:r]) @ Vh[:r,:]

        Unlike factorize_mlp, this keeps the same module structure (no
        Sequential replacement) so it can be applied to modules found via
        _find_attn_output_proj without needing a reference to the parent.

        Returns True if the approximation was applied, False if skipped
        (e.g. target_rank is already close to full rank).
        """
        if not isinstance(mod, nn.Linear):
            return False

        W = mod.weight.data.float().to(device)
        m, n = W.shape  # [out_features, in_features]

        # Skip if rank is essentially full
        rank = min(target_rank, min(m, n))
        if rank >= min(m, n) * 0.95:
            return False

        # Truncated SVD
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        # U: [m, k], S: [k], Vh: [k, n] where k = min(m, n)

        # Optionally reduce rank further to capture energy_threshold
        total_energy = (S ** 2).sum()
        if total_energy > 0:
            cumulative = (S ** 2).cumsum(0) / total_energy
            rank_for_energy = int((cumulative < energy_threshold).sum().item()) + 1
            rank = min(rank, rank_for_energy)

        if rank >= min(m, n) * 0.95:
            return False

        # Reconstruct rank-r approximation in-place
        W_approx = (U[:, :rank] * S[:rank].unsqueeze(0)) @ Vh[:rank, :]
        mod.weight.data.copy_(W_approx.to(mod.weight.dtype))

        orig_rank = min(m, n)
        savings_pct = (1 - rank / orig_rank) * 100
        logger.info(
            f"    factorize_module: rank {orig_rank} -> {rank} "
            f"({savings_pct:.1f}% rank reduction, {m}x{n} matrix)"
        )
        return True

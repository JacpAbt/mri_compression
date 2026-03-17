"""
Low-Rank Factorization Operation
===================================
Replace a Linear(in, out) with two linears: Linear(in, rank) + Linear(rank, out).
This saves params when rank < in*out/(in+out).

Applied to gate_proj, up_proj, down_proj independently.
"""

from __future__ import annotations
import logging

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

"""
Local Reconstruction Operation
================================
SparseGPT-style local reconstruction to recover quality after compression.
"""

from __future__ import annotations
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

logger = logging.getLogger(__name__)


class LocalReconstructor:
    @staticmethod
    def reconstruct(layer, target_inputs, target_outputs, lr=1e-4, iterations=200, device=torch.device("cuda")):
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            return 0.0
        params = [p for p in mlp.parameters() if p.requires_grad]
        if not params:
            return 0.0

        # Pre-move all targets to GPU once (avoids repeated CPU→GPU per iteration)
        gpu_pairs = [(inp.to(device), tgt.to(device))
                     for inp, tgt in zip(target_inputs, target_outputs)]

        opt = torch.optim.Adam(params, lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iterations)
        best = float("inf")
        for it in range(iterations):
            total, n = 0.0, 0
            for inp_g, tgt_g in gpu_pairs:
                with autocast("cuda", dtype=torch.bfloat16):
                    out = mlp(inp_g)
                    if isinstance(out, tuple):
                        out = out[0]
                    loss = F.mse_loss(out.float(), tgt_g.float())
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total += loss.item() * inp_g.shape[0]
                n += inp_g.shape[0]
                del out, loss
            sched.step()
            avg = total / max(n, 1)
            best = min(best, avg)
            if (it + 1) % 50 == 0:
                logger.info(f"    Recon iter {it+1}/{iterations}: MSE={avg:.6f}")
        del gpu_pairs
        torch.cuda.empty_cache()
        return best

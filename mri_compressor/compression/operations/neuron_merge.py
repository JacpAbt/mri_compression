"""
Neuron Merge Operation
========================
Merges redundant neurons via hierarchical clustering.
"""

from __future__ import annotations
import logging

import torch
import torch.nn as nn

from .._utils import get_mlp_modules

logger = logging.getLogger(__name__)


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
        merged = torch.zeros(activations.shape[0], len(clusters),
                             dtype=activations.dtype, device=activations.device)
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
        sim = torch.zeros(n, n, device=device)
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            sim[s:e] = acts_n[:, s:e].T @ acts_n
        sim.fill_diagonal_(-float("inf"))
        members = [[i] for i in range(n)]
        cl_imp = acts_gpu.abs().mean(dim=0)  # stay on GPU
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
    @torch.no_grad()
    def _apply_merge(layer, clusters, acts, device):
        mlp = get_mlp_modules(layer)
        imp = acts.to(device).abs().mean(dim=0)
        nw = len(clusters)
        # Pre-build cluster index tensors on GPU once
        cluster_indices = [torch.tensor(ms, device=device) for ms in clusters]

        for name in ["gate_proj", "up_proj"]:
            if name not in mlp:
                continue
            mod = mlp[name]
            old = mod.weight.data  # already on device
            new = torch.zeros(nw, old.shape[1], dtype=old.dtype, device=device)
            for c, (ms, idx) in enumerate(zip(clusters, cluster_indices)):
                if len(ms) == 1:
                    new[c] = old[ms[0]]
                else:
                    m_imp = imp[idx]
                    new[c] = (old[idx] * m_imp.unsqueeze(1)).sum(0) / m_imp.sum().clamp(min=1e-8)
            mod.weight = nn.Parameter(new)
            mod.out_features = nw
            if mod.bias is not None:
                old_b = mod.bias.data  # already on device
                new_b = torch.zeros(nw, dtype=old_b.dtype, device=device)
                for c, (ms, idx) in enumerate(zip(clusters, cluster_indices)):
                    if len(ms) == 1:
                        new_b[c] = old_b[ms[0]]
                    else:
                        m_imp = imp[idx]
                        new_b[c] = (old_b[idx] * m_imp).sum() / m_imp.sum().clamp(min=1e-8)
                mod.bias = nn.Parameter(new_b)
        if "down_proj" in mlp:
            mod = mlp["down_proj"]
            old = mod.weight.data  # already on device
            new = torch.zeros(old.shape[0], nw, dtype=old.dtype, device=device)
            for c, (ms, idx) in enumerate(zip(clusters, cluster_indices)):
                if len(ms) == 1:
                    new[:, c] = old[:, ms[0]]
                else:
                    new[:, c] = old[:, idx].sum(dim=1)
            mod.weight = nn.Parameter(new)
            mod.in_features = nw

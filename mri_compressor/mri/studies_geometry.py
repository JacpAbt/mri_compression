"""
Study 25: MLP Write-Vector Geometry
=====================================
Measures *what* the MLP writes to the residual stream during inference.

Each MLP layer adds a write vector

    Δ_l(x) = W_down @ (gate(x) * up(x))

to the residual.  This module hooks down_proj INPUT (intermediate activations
a_j) and OUTPUT (write vectors Δ_l) using a *single* forward pass with all
layer hooks registered simultaneously (not one layer at a time), keeping the
computational cost to O(N_batches) rather than O(N_batches × N_layers).

Metrics computed
────────────────
Static (weight-only, free):
  neuron_out_norms          ‖W_down[:,j]‖₂  per neuron

Dynamic (from a forward pass over a calibration set):
  mean_write_length         E[‖Δ_l‖₂]
  write_length_cv           std / mean of per-sample write-vector norms
  direction_coherence       ‖E[Δ̂_l]‖₂  ∈ [0, 1]
                            (0 = random direction each token,
                             1 = always writes the same direction)
  intrinsic_dim_95          Randomised PCA rank at 95 % explained variance
  dominant_direction        (hidden,) top-1 PC tensor (saved to .pt)

Per-neuron:
  geometric_importance      E[|a_j|] · ‖W_down[:,j]‖₂
                            — actual expected residual-stream contribution,
                            more informative than Wanda (which uses input norm)
  geom_importance_concentration
                            Fraction of neurons that carry 80 % of total
                            geometric importance

Cross-domain (optional, second pass):
  domain_direction_divergences
                            Per-layer angle (degrees) between generic and
                            domain mean write directions
"""

from __future__ import annotations

import gc
import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class LayerGeometryReport:
    """Geometry report for a single MLP layer under a single domain pass."""

    layer_idx: int
    hidden_size: int
    intermediate_size: int
    domain_name: str          # "generic" or a custom domain name
    n_vectors_used: int       # number of write vectors retained for PCA

    # ── Static (computed once from weights, no forward pass) ──────────────
    neuron_out_norms: torch.Tensor   # (intermediate,) ‖W_down[:,j]‖₂

    # ── Dynamic write-vector geometry ─────────────────────────────────────
    mean_write_length: float         # E[‖Δ_l‖₂]
    write_length_cv: float           # std / mean of write-vector norms
    direction_coherence: float       # ‖E[Δ̂_l]‖₂  ∈ [0, 1]
    intrinsic_dim_95: int            # PCA effective rank at 95 % variance
    dominant_direction: torch.Tensor  # (hidden,) first principal component

    # ── Per-neuron geometric importance ───────────────────────────────────
    geometric_importance: torch.Tensor   # (intermediate,) E[|a_j|]·‖W_down[:,j]‖₂
    geom_importance_concentration: float  # fraction of neurons holding 80 % of total


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _neuron_norms_and_hidden(mlp_info) -> tuple[torch.Tensor, int]:
    """
    Return (neuron_out_norms, hidden_size) from the down_proj weight.

    Handles both nn.Linear layout (hidden, intermediate) and Conv1D layout
    (intermediate, hidden) used by GPT-2.
    """
    W = mlp_info.down_proj.weight.detach().float()  # weight matrix
    inter = mlp_info.intermediate_size

    # Conv1D: weight shape is (in_features, out_features) = (intermediate, hidden)
    # nn.Linear: weight shape is (out_features, in_features) = (hidden, intermediate)
    if W.shape[0] == inter:          # Conv1D layout
        neuron_out_norms = W.norm(dim=1).cpu()   # (intermediate,)
        hidden_size = W.shape[1]
    else:                            # nn.Linear layout
        neuron_out_norms = W.norm(dim=0).cpu()   # (intermediate,)
        hidden_size = W.shape[0]

    return neuron_out_norms, hidden_size


def _collect_write_vectors(
    inspector,
    dataset,
    batch_size: int,
    max_batches: int,
    max_vectors: int,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """
    Register simultaneous INPUT and OUTPUT hooks on every down_proj, run
    max_batches forward passes, then return:
      act_tensors[layer_idx]   (N, intermediate)  — input  to down_proj
      write_tensors[layer_idx] (N, hidden)         — output of down_proj

    Both tensors are on CPU, float32, with at most max_vectors rows
    (randomly sub-sampled if necessary).
    """
    from ..data_utils import get_dataloader

    n_layers = len(inspector.mlp_layers)
    act_bufs:   dict[int, list[torch.Tensor]] = {i: [] for i in range(n_layers)}
    write_bufs: dict[int, list[torch.Tensor]] = {i: [] for i in range(n_layers)}
    hooks: list = []

    def _make_input_hook(idx: int):
        def _hook(module, inp, out):
            x = inp[0] if isinstance(inp, tuple) else inp
            # x: (batch, seq, intermediate)  — flatten batch*seq later
            act_bufs[idx].append(x.detach().float().cpu())
        return _hook

    def _make_output_hook(idx: int):
        def _hook(module, inp, out):
            # out: (batch, seq, hidden)
            write_bufs[idx].append(out.detach().float().cpu())
        return _hook

    # Register all hooks before any forward pass
    for idx, mlp_info in enumerate(inspector.mlp_layers):
        h1 = mlp_info.down_proj.register_forward_hook(_make_input_hook(idx))
        h2 = mlp_info.down_proj.register_forward_hook(_make_output_hook(idx))
        hooks.extend([h1, h2])

    # Single forward pass for all layers
    dataloader = get_dataloader(dataset, batch_size=batch_size)
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            input_ids = batch["input_ids"].to(inspector.device)
            inspector.model(input_ids=input_ids)

    for h in hooks:
        h.remove()

    def _concat_and_subsample(buf: list[torch.Tensor]) -> torch.Tensor:
        if not buf:
            return torch.empty(0)
        combined = torch.cat(buf, dim=0)          # (total_batch, seq, D) or (total_batch*seq, D)
        if combined.dim() == 3:
            combined = combined.view(-1, combined.shape[-1])   # (N, D)
        if combined.shape[0] > max_vectors:
            idx = torch.randperm(combined.shape[0])[:max_vectors]
            combined = combined[idx]
        return combined

    act_tensors:   dict[int, torch.Tensor] = {}
    write_tensors: dict[int, torch.Tensor] = {}
    for i in range(n_layers):
        act_tensors[i]   = _concat_and_subsample(act_bufs[i])
        write_tensors[i] = _concat_and_subsample(write_bufs[i])
        act_bufs[i].clear()
        write_bufs[i].clear()

    return act_tensors, write_tensors


def _compute_layer_geometry(
    layer_idx: int,
    mlp_info,
    act_tensor: torch.Tensor,    # (N, intermediate) — may be empty
    write_tensor: torch.Tensor,  # (N, hidden)        — may be empty
    domain_name: str,
) -> LayerGeometryReport:
    """Compute all geometry metrics for one layer from pre-collected tensors."""

    neuron_out_norms, hidden_size = _neuron_norms_and_hidden(mlp_info)
    intermediate_size = mlp_info.intermediate_size

    # ── Defaults (in case tensors are empty) ──────────────────────────────
    n_vecs = write_tensor.shape[0] if write_tensor.numel() > 0 else 0
    mean_write_length  = 0.0
    write_length_cv    = 0.0
    direction_coherence = 0.0
    intrinsic_dim_95   = 1
    dominant_direction = torch.zeros(hidden_size)

    if n_vecs > 1:
        wv = write_tensor.float()              # (N, hidden)
        norms = wv.norm(dim=1)                 # (N,)

        mean_write_length = float(norms.mean().item())
        if mean_write_length > 1e-8:
            write_length_cv = float((norms.std() / norms.mean()).item())

        # Direction coherence: ‖E[Δ̂_l]‖₂
        unit_vecs = wv / norms.unsqueeze(1).clamp(min=1e-8)
        mean_dir  = unit_vecs.mean(0)          # (hidden,)
        direction_coherence = float(mean_dir.norm().item())

        # Intrinsic dimensionality via randomised PCA
        centered = wv - wv.mean(0, keepdim=True)
        q = min(64, n_vecs - 1, hidden_size - 1)
        if q >= 1:
            try:
                # torch.pca_lowrank(A, q) with center=False (we already centered)
                # returns (U, S, V) where V is (n_features, q), columns = PCs
                _, S, V = torch.pca_lowrank(centered, q=q, center=False)
                var_explained = S ** 2
                total_var = var_explained.sum()
                if total_var > 0:
                    cumvar = var_explained.cumsum(0) / total_var
                    intrinsic_dim_95 = int((cumvar < 0.95).sum().item()) + 1
                    dominant_direction = V[:, 0].cpu()    # (hidden,) first PC
            except Exception as exc:
                logger.debug(f"  Layer {layer_idx}: PCA failed — {exc}")

    # ── Per-neuron geometric importance ───────────────────────────────────
    geom_importance = torch.zeros(intermediate_size)
    concentration   = 0.0

    if act_tensor.numel() > 0 and n_vecs > 0:
        mean_abs_act  = act_tensor.float().abs().mean(0).cpu()   # (intermediate,)
        geom_importance = mean_abs_act * neuron_out_norms         # (intermediate,)

        total = float(geom_importance.sum().item())
        if total > 0.0:
            sorted_imp = geom_importance.sort(descending=True).values
            cumulative = sorted_imp.cumsum(0) / total
            n_for_80   = int((cumulative < 0.80).sum().item()) + 1
            concentration = n_for_80 / intermediate_size

    return LayerGeometryReport(
        layer_idx=layer_idx,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        domain_name=domain_name,
        n_vectors_used=n_vecs,
        neuron_out_norms=neuron_out_norms,
        mean_write_length=mean_write_length,
        write_length_cv=write_length_cv,
        direction_coherence=direction_coherence,
        intrinsic_dim_95=intrinsic_dim_95,
        dominant_direction=dominant_direction,
        geometric_importance=geom_importance,
        geom_importance_concentration=concentration,
    )


# ─── Public API ───────────────────────────────────────────────────────────────

def run_geometry_analysis(
    inspector,
    dataset,
    domain_datasets: Optional[Dict[str, object]] = None,
    batch_size: int = 4,
    max_vectors: int = 2000,
    max_batches: int = 16,
) -> dict:
    """
    Study 25: MLP Write-Vector Geometry.

    Runs one forward pass (all layers hooked simultaneously) over `dataset`,
    then optionally a second pass per domain in `domain_datasets`.

    Parameters
    ----------
    inspector        : ModelInspector with .mlp_layers, .model, .device
    dataset          : Calibration TextDataset (generic / wikitext)
    domain_datasets  : Optional {domain_name: TextDataset}
    batch_size       : Forward-pass batch size
    max_vectors      : Maximum write vectors retained per layer for PCA
                       (random sub-sample if more are collected)
    max_batches      : Maximum batches per forward pass

    Returns
    -------
    dict with:
      "layer_reports"      list[LayerGeometryReport]  — generic pass
      "domain_reports"     dict[str, list[LayerGeometryReport]]
      "domain_divergences" dict[str, list[float]]  angles in degrees per layer
    """
    print(
        f"  Study 25: collecting write vectors "
        f"(max {max_vectors}/layer, {max_batches} batches)..."
    )

    # ── Generic calibration pass ──────────────────────────────────────────
    act_t, write_t = _collect_write_vectors(
        inspector, dataset, batch_size, max_batches, max_vectors
    )

    layer_reports: list[LayerGeometryReport] = []
    for idx, mlp_info in enumerate(inspector.mlp_layers):
        report = _compute_layer_geometry(
            idx, mlp_info,
            act_t[idx], write_t[idx],
            domain_name="generic",
        )
        layer_reports.append(report)

    del act_t, write_t
    gc.collect()

    print(f"    Generic pass: {len(layer_reports)} layers analysed.")

    # ── Print per-layer table (generic pass) ──────────────────────────────
    if layer_reports:
        avg_coherence = sum(r.direction_coherence for r in layer_reports) / len(layer_reports)
        avg_dim       = sum(r.intrinsic_dim_95    for r in layer_reports) / len(layer_reports)
        avg_conc      = sum(r.geom_importance_concentration for r in layer_reports) / len(layer_reports)
        hdr = (
            f"\n  Study 25 — Write-Vector Geometry (per-layer breakdown):\n"
            f"  {'Layer':>5} | {'CoherDir':>8} | {'IntrDim':>7} | {'GeomConc':>8}"
        )
        sep = "  " + "-" * (6 + 11 + 10 + 11)
        rows = []
        for r in layer_reports:
            rows.append(
                f"  {r.layer_idx:>5} | {r.direction_coherence:>8.3f} | "
                f"{r.intrinsic_dim_95:>7.1f} | {r.geom_importance_concentration:>8.3f}"
            )
        avg_row = (
            f"  {'Avg':>5} | {avg_coherence:>8.3f} | "
            f"{avg_dim:>7.1f} | {avg_conc:>8.3f}"
        )
        print(hdr)
        print(sep)
        print("\n".join(rows))
        print(sep)
        print(avg_row)
        print()

    # ── Domain passes ─────────────────────────────────────────────────────
    domain_reports:    dict[str, list[LayerGeometryReport]] = {}
    domain_divergences: dict[str, list[float]] = {}

    if domain_datasets:
        for domain_name, domain_ds in domain_datasets.items():
            print(f"    Domain pass: {domain_name}...")
            d_act, d_write = _collect_write_vectors(
                inspector, domain_ds, batch_size, max_batches, max_vectors
            )
            d_reports: list[LayerGeometryReport] = []
            for idx, mlp_info in enumerate(inspector.mlp_layers):
                r = _compute_layer_geometry(
                    idx, mlp_info,
                    d_act[idx], d_write[idx],
                    domain_name=domain_name,
                )
                d_reports.append(r)
            del d_act, d_write
            gc.collect()

            domain_reports[domain_name] = d_reports

            # Cross-domain divergence: angle between mean write directions
            # Use dominant_direction (first PC of write vectors) as the proxy
            divergences: list[float] = []
            for gen_r, dom_r in zip(layer_reports, d_reports):
                dir_gen = gen_r.dominant_direction.float()
                dir_dom = dom_r.dominant_direction.float()
                ng = dir_gen.norm()
                nd = dir_dom.norm()
                if ng > 1e-8 and nd > 1e-8:
                    cos_sim = (dir_gen / ng * dir_dom / nd).sum().clamp(-1.0, 1.0)
                    angle = math.degrees(math.acos(float(cos_sim.item())))
                else:
                    angle = 0.0
                divergences.append(angle)

            domain_divergences[domain_name] = divergences
            avg_div = sum(divergences) / max(1, len(divergences))
            # Print per-layer divergence alongside aggregate
            div_str = "  ".join(f"L{i}:{d:.1f}°" for i, d in enumerate(divergences))
            print(f"    {domain_name}: avg domain divergence = {avg_div:.1f}°")
            print(f"      Per-layer: {div_str}")

    return {
        "layer_reports":      layer_reports,
        "domain_reports":     domain_reports,
        "domain_divergences": domain_divergences,
    }

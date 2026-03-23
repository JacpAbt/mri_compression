"""
Domain Imprinting
=================
Post-compression step that injects per-layer domain activation centroids
into each layer's down_proj bias.

Intuition
---------
After domain compression the model has had neurons irrelevant to the target
domain removed.  The remaining neurons still operate in their original (general)
activation distribution.  "Imprinting" shifts each layer's output by the
expected domain contribution: we compute the mean activation on domain data
(the centroid of the domain manifold) and fold it permanently into the
down_proj bias — pure linear algebra, no gradient descent.

Two centroid modes
------------------
1. Compressed-model centroids  (original behaviour, intermediate space)
   Centroids are computed on the *already compressed* model.  The centroid
   lives in intermediate activation space (D_intermediate, post-compression
   size), so it is projected through W_down before injection:

       down_proj.bias += scale * W_down @ centroid          # [hidden_size]

2. Original-model centroids  (preferred, hidden/output space)
   Centroids are computed on the *original, pre-compression* model by hooking
   the down_proj OUTPUT instead of its input.  This gives a [hidden_size]
   vector that is dimension-stable: hidden_size is never altered by neuron
   removal, so the centroid computed on the original model can be injected
   into the compressed model's bias without any projection or size mismatch:

       down_proj.bias += scale * centroid                   # [hidden_size]

   This is the recommended path because the centroid captures the full,
   undamaged domain signal from the original model rather than the already
   degraded signal from the compressed one.
"""

from __future__ import annotations

import gc
import logging
from typing import Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _is_cuda(device) -> bool:
    """Return True if *device* refers to a CUDA device (str or torch.device)."""
    if isinstance(device, torch.device):
        return device.type == "cuda"
    return str(device).startswith("cuda")


class DomainImprinter:
    """Compute and inject per-layer domain centroids into down_proj bias."""

    # ------------------------------------------------------------------
    # Mode 1: intermediate-space centroids (compressed model)
    # ------------------------------------------------------------------

    @staticmethod
    @torch.no_grad()
    def compute_centroids(
        model: nn.Module,
        domain_dataloader,
        inspector,
        device,
        max_batches: int = 16,
    ) -> Dict[int, torch.Tensor]:
        """
        Compute per-layer domain activation centroids on the COMPRESSED model.

        Hooks the *input* to each down_proj (intermediate activation space,
        size = compressed intermediate_size) and averages across domain tokens.
        Must be called AFTER compression so tensor shapes match the reduced
        down_proj input dimension.

        Returns
        -------
        dict[layer_idx -> Tensor(intermediate_size_compressed,)]  CPU / float32.
        """
        from .._utils import resolve_layer, get_mlp_modules

        centroids: Dict[int, torch.Tensor] = {}
        model.eval()

        for layer_idx in range(inspector.num_layers):
            all_acts: list[torch.Tensor] = []

            try:
                layer     = resolve_layer(model, layer_idx, inspector)
                mlp_mods  = get_mlp_modules(layer)
                down_proj = mlp_mods.get("down_proj")
                if down_proj is None:
                    logger.warning("  Layer %d: down_proj not found — skipping", layer_idx)
                    continue
            except Exception as exc:
                logger.warning("  Layer %d: resolution error (%s) — skipping", layer_idx, exc)
                continue

            def _make_hook(store: list):
                def _hook(module, inp, out):
                    x = inp[0].detach().cpu().float()
                    store.append(x.reshape(-1, x.shape[-1]))
                return _hook

            handle = down_proj.register_forward_hook(_make_hook(all_acts))
            try:
                for i, batch in enumerate(domain_dataloader):
                    if i >= max_batches:
                        break
                    model(input_ids=batch["input_ids"].to(device))
            finally:
                handle.remove()

            if all_acts:
                stacked = torch.cat(all_acts, dim=0)
                centroids[layer_idx] = stacked.mean(dim=0)
                del stacked
            else:
                logger.warning("  Layer %d: no activations captured", layer_idx)

            del all_acts
            gc.collect()
            if _is_cuda(device):
                torch.cuda.empty_cache()

        return centroids

    # ------------------------------------------------------------------
    # Mode 2: output-space centroids (original model, preferred)
    # ------------------------------------------------------------------

    @staticmethod
    @torch.no_grad()
    def compute_output_centroids(
        model: nn.Module,
        inspector,
        domain_dataloader,
        device,
        max_batches: int = 16,
    ) -> Dict[int, torch.Tensor]:
        """
        Compute per-layer domain centroids in HIDDEN (output) space.

        Unlike compute_centroids() — which hooks down_proj *inputs* and
        produces tensors of shape [intermediate_size] — this method hooks
        down_proj *outputs* and produces tensors of shape [hidden_size].

        hidden_size is never changed by neuron removal (only intermediate_size
        shrinks), so these centroids are valid injection targets on both the
        original and the compressed model.

        Intended use
        ------------
        Call on the ORIGINAL (pre-compression) model to capture the clean,
        undamaged domain manifold centroid.  Then pass the result as
        ``pre_computed_centroids`` to ``imprint()``, which will apply them to
        the compressed model via ``apply_direct()`` — no W_down projection
        needed, no dimension mismatch possible.

        Returns
        -------
        dict[layer_idx -> Tensor(hidden_size,)]  CPU / float32.
        """
        from .._utils import resolve_layer, get_mlp_modules

        centroids: Dict[int, torch.Tensor] = {}
        model.eval()

        for layer_idx in range(inspector.num_layers):
            all_acts: list[torch.Tensor] = []

            try:
                layer     = resolve_layer(model, layer_idx, inspector)
                mlp_mods  = get_mlp_modules(layer)
                down_proj = mlp_mods.get("down_proj")
                if down_proj is None:
                    logger.warning(
                        "  Layer %d: down_proj not found — skipping original centroid",
                        layer_idx,
                    )
                    continue
            except Exception as exc:
                logger.warning("  Layer %d: resolution error (%s) — skipping", layer_idx, exc)
                continue

            # Hook the OUTPUT (not input) of down_proj → [hidden_size]
            def _make_hook(store: list):
                def _hook(module, inp, out):
                    x = out.detach().cpu().float()
                    store.append(x.reshape(-1, x.shape[-1]))
                return _hook

            handle = down_proj.register_forward_hook(_make_hook(all_acts))
            try:
                for i, batch in enumerate(domain_dataloader):
                    if i >= max_batches:
                        break
                    model(input_ids=batch["input_ids"].to(device))
            finally:
                handle.remove()

            if all_acts:
                stacked = torch.cat(all_acts, dim=0)        # (N, hidden_size)
                centroids[layer_idx] = stacked.mean(dim=0)  # (hidden_size,)
                del stacked
            else:
                logger.warning(
                    "  Layer %d: no activations captured for original centroid", layer_idx
                )

            del all_acts
            gc.collect()
            if _is_cuda(device):
                torch.cuda.empty_cache()

        return centroids

    # ------------------------------------------------------------------
    # Bias injection — Mode 1: intermediate space (W_down projection)
    # ------------------------------------------------------------------

    @staticmethod
    @torch.no_grad()
    def apply(
        model: nn.Module,
        inspector,
        centroids: Dict[int, torch.Tensor],
        scale: float,
        device,
    ) -> Dict[int, dict]:
        """
        Inject intermediate-space centroids via W_down projection.

        For each layer l:
            Δbias = scale * W_down @ centroid_l      # [hidden_size]
            down_proj.bias += Δbias

        centroid_l must match the current (post-compression) down_proj
        in_features.  Creates the bias parameter if absent.

        Returns dict[layer_idx -> {"centroid_norm", "delta_norm"}].
        """
        from .._utils import resolve_layer, get_mlp_modules

        applied: Dict[int, dict] = {}

        for layer_idx, centroid in centroids.items():
            try:
                layer     = resolve_layer(model, layer_idx, inspector)
                mlp_mods  = get_mlp_modules(layer)
                down_proj = mlp_mods.get("down_proj")
                if down_proj is None:
                    continue
            except Exception:
                continue

            weight       = down_proj.weight.data                        # [out, in]
            centroid_dev = centroid.to(device=weight.device, dtype=weight.dtype)

            if centroid_dev.shape[0] != weight.shape[1]:
                logger.warning(
                    "  Layer %d: centroid size %d ≠ down_proj in_features %d — skipping",
                    layer_idx, centroid_dev.shape[0], weight.shape[1],
                )
                continue

            if down_proj.bias is None:
                down_proj.bias = nn.Parameter(
                    torch.zeros(weight.shape[0], device=weight.device, dtype=weight.dtype)
                )

            bias_delta = scale * (weight @ centroid_dev)    # [hidden_size]
            down_proj.bias.data.add_(bias_delta)

            applied[layer_idx] = {
                "centroid_norm": float(centroid.norm().item()),
                "delta_norm":    float(bias_delta.norm().item()),
            }
            logger.debug(
                "  Layer %d imprinted: centroid_norm=%.4f  delta_norm=%.4f",
                layer_idx,
                applied[layer_idx]["centroid_norm"],
                applied[layer_idx]["delta_norm"],
            )

        return applied

    # ------------------------------------------------------------------
    # Bias injection — Mode 2: hidden space (direct addition)
    # ------------------------------------------------------------------

    @staticmethod
    @torch.no_grad()
    def apply_direct(
        model: nn.Module,
        inspector,
        centroids_hidden: Dict[int, torch.Tensor],
        scale: float,
        device,
    ) -> Dict[int, dict]:
        """
        Inject hidden-space centroids directly into down_proj bias.

        For each layer l:
            down_proj.bias += scale * centroid_l      # centroid already [hidden_size]

        No W_down projection needed.  Because centroid_l is in output space
        and hidden_size never changes through compression, this method works
        identically on the original and the compressed model.

        centroids_hidden should come from compute_output_centroids().

        Returns dict[layer_idx -> {"centroid_norm", "delta_norm"}].
        """
        from .._utils import resolve_layer, get_mlp_modules

        applied: Dict[int, dict] = {}

        for layer_idx, centroid in centroids_hidden.items():
            try:
                layer     = resolve_layer(model, layer_idx, inspector)
                mlp_mods  = get_mlp_modules(layer)
                down_proj = mlp_mods.get("down_proj")
                if down_proj is None:
                    continue
            except Exception:
                continue

            weight       = down_proj.weight.data                        # [out, in]
            centroid_dev = centroid.to(device=weight.device, dtype=weight.dtype)

            # Centroid must match out_features (hidden_size)
            if centroid_dev.shape[0] != weight.shape[0]:
                logger.warning(
                    "  Layer %d: centroid hidden size %d ≠ down_proj out_features %d — skipping",
                    layer_idx, centroid_dev.shape[0], weight.shape[0],
                )
                continue

            if down_proj.bias is None:
                down_proj.bias = nn.Parameter(
                    torch.zeros(weight.shape[0], device=weight.device, dtype=weight.dtype)
                )

            bias_delta = scale * centroid_dev
            down_proj.bias.data.add_(bias_delta)

            applied[layer_idx] = {
                "centroid_norm": float(centroid.norm().item()),
                "delta_norm":    float(bias_delta.norm().item()),
            }
            logger.debug(
                "  Layer %d (direct) imprinted: centroid_norm=%.4f  delta_norm=%.4f",
                layer_idx,
                applied[layer_idx]["centroid_norm"],
                applied[layer_idx]["delta_norm"],
            )

        return applied

    # ------------------------------------------------------------------
    # Convenience: compute + apply in one call
    # ------------------------------------------------------------------

    @classmethod
    @torch.no_grad()
    def imprint(
        cls,
        model: nn.Module,
        inspector,
        domain_dataloader,
        device,
        scale: float = 0.05,
        max_batches: int = 16,
        pre_computed_centroids: Optional[Dict[int, torch.Tensor]] = None,
    ) -> Dict[int, dict]:
        """
        Full imprinting pass.

        If *pre_computed_centroids* is provided (output-space centroids from
        the original model via ``compute_output_centroids()``), they are
        injected directly using ``apply_direct()`` — no W_down projection,
        dimension-stable across compression.

        Otherwise, centroids are computed from *domain_dataloader* on the
        current (compressed) model and injected via ``apply()`` (intermediate
        space, requires matching down_proj in_features).

        Returns per-layer application stats (centroid_norm, delta_norm).
        """
        if pre_computed_centroids is not None:
            n = len(pre_computed_centroids)
            print(
                f"\n  Using pre-computed original-model centroids "
                f"({n} layers, output space, scale={scale:.3f})..."
            )
            applied = cls.apply_direct(
                model, inspector, pre_computed_centroids, scale, device
            )
        else:
            print(f"\n  Computing domain activation centroids (scale={scale:.3f})...")
            centroids = cls.compute_centroids(
                model, domain_dataloader, inspector, device,
                max_batches=max_batches,
            )
            print(f"  Centroids computed for {len(centroids)}/{inspector.num_layers} layers.")
            applied = cls.apply(model, inspector, centroids, scale, device)

        print(f"  Domain imprinting complete: {len(applied)} layers updated.")
        return applied

"""
Weight Sharing Operation
=========================
Makes two consecutive layers share MLP weights.

For high-CKA layer pairs identified by Study 17, the layers produce nearly
identical representations, so layer B's MLP projections can be replaced with
references to layer A's parameters.  This saves memory at inference time
without any additional approximation error.
"""

from __future__ import annotations
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_SHARED_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


class WeightSharer:
    @staticmethod
    @torch.no_grad()
    def share_mlp_weights(model, layer_a_idx, layer_b_idx, device):
        """
        Make layer B's MLP share weights with layer A.

        For high-CKA pairs, layer B's MLP projections become references
        to layer A's parameters.  This saves memory at inference time.

        Args:
            model: The full model.
            layer_a_idx: Source layer index (keeps its weights).
            layer_b_idx: Target layer index (gets shared weights).
            device: Torch device.

        Returns:
            dict with sharing stats::

                {
                    "params_saved": int,   # total number of parameters saved
                    "projections_shared": list,  # names of shared projections
                }
        """
        mlp_a = model.model.layers[layer_a_idx].mlp
        mlp_b = model.model.layers[layer_b_idx].mlp

        params_saved = 0
        projections_shared: list[str] = []

        for proj_name in _SHARED_PROJECTIONS:
            mod_a = getattr(mlp_a, proj_name, None)
            mod_b = getattr(mlp_b, proj_name, None)

            if mod_a is None or mod_b is None:
                logger.warning(
                    "Projection %s not found in one of the layers (%d, %d); skipping",
                    proj_name, layer_a_idx, layer_b_idx,
                )
                continue

            # Count parameters that will be freed from layer B.
            n_weight_params = mod_b.weight.numel()
            n_bias_params = mod_b.bias.numel() if mod_b.bias is not None else 0

            # Point layer B's weight to layer A's Parameter object.
            mod_b.weight = mod_a.weight

            # Share bias as well if layer A has one.
            if mod_a.bias is not None:
                mod_b.bias = mod_a.bias
            elif mod_b.bias is not None:
                # Layer A has no bias but layer B does -- drop it to match.
                mod_b.bias = None

            params_saved += n_weight_params + n_bias_params
            projections_shared.append(proj_name)

        logger.info(
            "Shared MLP weights: layer %d -> layer %d  |  projections=%s  |  params_saved=%d",
            layer_a_idx, layer_b_idx, projections_shared, params_saved,
        )

        return {
            "params_saved": params_saved,
            "projections_shared": projections_shared,
        }

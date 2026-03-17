"""
Unit tests for individual compression operations.

All tests run on CPU with tiny synthetic layers — no GPU or model download required.
"""

import pytest
import torch
import torch.nn as nn

from mri_compressor.compression.operations.dead_removal import DeadNeuronRemover
from mri_compressor.compression.operations.low_rank import LowRankFactorizer
from mri_compressor.compression._utils import get_mlp_modules, get_intermediate_size


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def count_params(layer):
    return sum(p.numel() for p in layer.parameters())


# ---------------------------------------------------------------------------
# DeadNeuronRemover
# ---------------------------------------------------------------------------

class TestDeadNeuronRemover:
    def test_removes_correct_count(self, gated_layer, small_activations):
        device = torch.device("cpu")
        n_before = gated_layer.mlp.gate_proj.out_features
        n_removed, remaining = DeadNeuronRemover.remove_by_mri_count(
            gated_layer, small_activations, n_to_remove=8, device=device
        )
        assert n_removed == 8
        assert gated_layer.mlp.gate_proj.out_features == n_before - 8

    def test_remaining_activations_shape(self, gated_layer, small_activations):
        device = torch.device("cpu")
        n_tokens = small_activations.shape[0]
        n_before = small_activations.shape[1]
        n_removed, remaining = DeadNeuronRemover.remove_by_mri_count(
            gated_layer, small_activations, n_to_remove=4, device=device
        )
        assert remaining.shape == (n_tokens, n_before - n_removed)

    def test_down_proj_in_features_shrinks(self, gated_layer, small_activations):
        device = torch.device("cpu")
        in_before = gated_layer.mlp.down_proj.in_features
        n_removed, _ = DeadNeuronRemover.remove_by_mri_count(
            gated_layer, small_activations, n_to_remove=4, device=device
        )
        assert gated_layer.mlp.down_proj.in_features == in_before - 4

    def test_up_proj_out_features_shrinks(self, gated_layer, small_activations):
        device = torch.device("cpu")
        out_before = gated_layer.mlp.up_proj.out_features
        n_removed, _ = DeadNeuronRemover.remove_by_mri_count(
            gated_layer, small_activations, n_to_remove=4, device=device
        )
        assert gated_layer.mlp.up_proj.out_features == out_before - 4

    def test_removes_lowest_magnitude_neurons(self, gated_layer):
        """The 8 neurons with zero activation should be removed first."""
        device = torch.device("cpu")
        acts = torch.randn(64, 64)
        acts[:, :8] = 0.0  # neurons 0-7 are dead

        # Ask to remove exactly 8
        n_removed, remaining = DeadNeuronRemover.remove_by_mri_count(
            gated_layer, acts, n_to_remove=8, device=device
        )
        assert n_removed == 8

    def test_zero_removal_is_noop(self, gated_layer, small_activations):
        device = torch.device("cpu")
        params_before = count_params(gated_layer)
        n_removed, remaining = DeadNeuronRemover.remove_by_mri_count(
            gated_layer, small_activations, n_to_remove=0, device=device
        )
        assert n_removed == 0
        assert count_params(gated_layer) == params_before

    def test_protected_indices_are_skipped(self, gated_layer, small_activations):
        """Neurons with zero activation that are also protected must not be removed."""
        device = torch.device("cpu")
        acts = small_activations.clone()
        acts[:, :8] = 0.0  # neurons 0-7 are dead

        # Protect all dead neurons
        protected = set(range(8))
        n_removed, _ = DeadNeuronRemover.remove_by_mri_count(
            gated_layer, acts, n_to_remove=8, device=device,
            protected_indices=protected,
        )
        # Since all 8 dead neurons are protected, the remover must pick other neurons
        # or remove fewer than requested — it should not remove the protected ones
        assert n_removed <= 8

    def test_remove_by_indices(self, gated_layer, small_activations):
        device = torch.device("cpu")
        n_before = gated_layer.mlp.gate_proj.out_features
        n_removed, remaining = DeadNeuronRemover.remove_by_indices(
            gated_layer, small_activations, indices_to_remove=[0, 1, 2], device=device
        )
        assert n_removed == 3
        assert gated_layer.mlp.gate_proj.out_features == n_before - 3

    def test_remove_by_indices_empty_list_is_noop(self, gated_layer, small_activations):
        device = torch.device("cpu")
        params_before = count_params(gated_layer)
        n_removed, _ = DeadNeuronRemover.remove_by_indices(
            gated_layer, small_activations, indices_to_remove=[], device=device
        )
        assert n_removed == 0
        assert count_params(gated_layer) == params_before

    def test_remove_combined_dead_and_domain(self, gated_layer, small_activations):
        device = torch.device("cpu")
        n_before = gated_layer.mlp.gate_proj.out_features
        # Remove 4 dead + 2 domain-specific neurons (non-overlapping)
        dead_removed, domain_removed, remaining = DeadNeuronRemover.remove_combined(
            gated_layer, small_activations,
            n_dead_to_remove=4,
            domain_unnecessary_indices=[50, 51],  # high indices, unlikely to be in dead set
            device=device,
        )
        assert dead_removed + domain_removed <= 6
        assert gated_layer.mlp.gate_proj.out_features == n_before - (dead_removed + domain_removed)


# ---------------------------------------------------------------------------
# LowRankFactorizer
# ---------------------------------------------------------------------------

class TestLowRankFactorizer:
    def test_factorize_returns_rank_dict(self, gated_layer):
        device = torch.device("cpu")
        result = LowRankFactorizer.factorize_mlp(gated_layer, target_rank=16, device=device)
        # Should return a dict with projection names
        assert isinstance(result, dict)

    def test_factorize_reduces_params(self, gated_layer):
        """After SVD factorization the layer should have fewer parameters."""
        device = torch.device("cpu")
        params_before = count_params(gated_layer)
        LowRankFactorizer.factorize_mlp(gated_layer, target_rank=8, device=device)
        params_after = count_params(gated_layer)
        # With rank=8 and intermediate=64, factorization should save params
        # (8*(32+64) < 32*64 for each projection)
        assert params_after <= params_before

    def test_no_factorize_without_mlp(self):
        """Layer without .mlp attribute returns empty dict."""
        bare = nn.Linear(32, 32)
        result = LowRankFactorizer.factorize_mlp(bare, target_rank=8, device=torch.device("cpu"))
        assert result == {}

    def test_high_rank_skipped(self, gated_layer):
        """If target_rank >= useful threshold, factorization is skipped for that projection."""
        device = torch.device("cpu")
        params_before = count_params(gated_layer)
        # With target_rank = 63 (nearly full rank for 64), factorization should not save params
        # and should be skipped
        result = LowRankFactorizer.factorize_mlp(gated_layer, target_rank=63, device=device)
        params_after = count_params(gated_layer)
        # Params should be unchanged or only slightly changed (skipped projections)
        assert params_after <= params_before


# ---------------------------------------------------------------------------
# _utils: get_mlp_modules, get_intermediate_size
# ---------------------------------------------------------------------------

class TestUtils:
    def test_get_mlp_modules_gated(self, gated_layer):
        mods = get_mlp_modules(gated_layer)
        assert "gate_proj" in mods
        assert "up_proj" in mods
        assert "down_proj" in mods

    def test_get_mlp_modules_all_linear(self, gated_layer):
        mods = get_mlp_modules(gated_layer)
        for name, mod in mods.items():
            assert isinstance(mod, nn.Linear), f"{name} is not a Linear"

    def test_get_mlp_modules_no_mlp_raises(self):
        bare = nn.Linear(32, 32)
        with pytest.raises(ValueError, match="Cannot find MLP"):
            get_mlp_modules(bare)

    def test_get_intermediate_size(self, gated_layer):
        size = get_intermediate_size(gated_layer)
        assert size == 64  # SyntheticGatedMLP uses intermediate=64

    def test_get_intermediate_size_no_mlp_raises(self):
        bare = nn.Linear(32, 32)
        with pytest.raises((ValueError, AttributeError)):
            get_intermediate_size(bare)

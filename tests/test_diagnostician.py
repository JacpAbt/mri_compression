"""Tests for MRIDiagnostician.diagnose_from_summary()."""

import pytest
from mri_compressor import MRIDiagnostician, CompressionPrescription, CompressionStrategy
from mri_compressor.compression.prescription import LayerPrescription


class TestMRIDiagnosticianConstruction:
    def test_default_construction(self):
        d = MRIDiagnostician()
        assert d.ppl_budget == 2.0
        assert d.enable_attn_pruning is True
        assert d.enable_depth_pruning is False
        assert d.enable_merging is True

    def test_custom_ppl_budget(self):
        d = MRIDiagnostician(ppl_budget=5.0)
        assert d.ppl_budget == 5.0

    def test_disable_attn_pruning(self):
        d = MRIDiagnostician(enable_attn_pruning=False)
        assert d.enable_attn_pruning is False

    def test_enable_depth_pruning(self):
        d = MRIDiagnostician(enable_depth_pruning=True)
        assert d.enable_depth_pruning is True

    def test_domain_target(self):
        d = MRIDiagnostician(target_domain="math")
        assert d.target_domain == "math"


class TestDiagnoseFromSummary:
    def test_returns_prescription(self, minimal_mri_summary):
        d = MRIDiagnostician()
        p = d.diagnose_from_summary(minimal_mri_summary)
        assert isinstance(p, CompressionPrescription)

    def test_prescription_has_correct_layer_count(self, minimal_mri_summary):
        d = MRIDiagnostician()
        p = d.diagnose_from_summary(minimal_mri_summary)
        expected_layers = minimal_mri_summary["architecture"]["num_layers"]
        assert len(p.layers) == expected_layers

    def test_prescription_model_name(self, minimal_mri_summary):
        d = MRIDiagnostician()
        p = d.diagnose_from_summary(minimal_mri_summary)
        assert p.model_name == "synthetic-test-model"

    def test_prescription_baseline_ppl(self, minimal_mri_summary):
        d = MRIDiagnostician()
        p = d.diagnose_from_summary(minimal_mri_summary)
        assert p.baseline_ppl == pytest.approx(25.0)

    def test_each_layer_has_prescription(self, minimal_mri_summary):
        d = MRIDiagnostician()
        p = d.diagnose_from_summary(minimal_mri_summary)
        for lp in p.layers:
            assert isinstance(lp, LayerPrescription)
            assert isinstance(lp.strategy, CompressionStrategy)

    def test_layer_indices_are_sequential(self, minimal_mri_summary):
        d = MRIDiagnostician()
        p = d.diagnose_from_summary(minimal_mri_summary)
        indices = [lp.layer_idx for lp in p.layers]
        assert indices == list(range(len(p.layers)))

    def test_dead_neurons_propagated(self, minimal_mri_summary):
        d = MRIDiagnostician()
        p = d.diagnose_from_summary(minimal_mri_summary)
        # Summary has dead_count=5 per layer; at least some layers should reflect that
        dead_counts = [lp.dead_neuron_count for lp in p.layers]
        assert any(c > 0 for c in dead_counts)

    def test_depth_pruning_disabled_by_default(self, minimal_mri_summary):
        d = MRIDiagnostician(enable_depth_pruning=False)
        p = d.diagnose_from_summary(minimal_mri_summary)
        assert p.depth_prune_applied == []
        for lp in p.layers:
            assert lp.depth_prune is False

    def test_attn_pruning_disabled(self, minimal_mri_summary):
        d = MRIDiagnostician(enable_attn_pruning=False)
        p = d.diagnose_from_summary(minimal_mri_summary)
        for lp in p.layers:
            assert lp.attn_heads_to_prune == 0

    def test_minimal_summary_no_per_layer(self):
        """Diagnostician should handle a summary with no per_layer data gracefully."""
        summary = {
            "model": "fallback-model",
            "baseline_ppl": 30.0,
            "architecture": {
                "num_layers": 2,
                "intermediate_size": 64,
                "num_attention_heads": 4,
            },
            "per_layer": {},
        }
        d = MRIDiagnostician()
        p = d.diagnose_from_summary(summary)
        assert isinstance(p, CompressionPrescription)
        assert len(p.layers) == 2

    def test_prescription_summary_is_valid_string(self, minimal_mri_summary):
        d = MRIDiagnostician()
        p = d.diagnose_from_summary(minimal_mri_summary)
        s = p.summary()
        assert isinstance(s, str)
        assert "synthetic-test-model" in s

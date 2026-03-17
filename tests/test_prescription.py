"""Tests for CompressionPrescription, LayerPrescription, and CompressionStrategy."""

import pytest
from mri_compressor import CompressionPrescription, CompressionStrategy
from mri_compressor.compression.prescription import LayerPrescription


class TestCompressionStrategy:
    def test_all_strategies_exist(self):
        expected = [
            "DEAD_REMOVAL_AND_MERGE",
            "DORMANT_REMOVAL",
            "STRUCTURED_PRUNE",
            "LIGHT_TOUCH",
            "DEPTH_PRUNE",
            "ATTENTION_PRUNE",
            "LOW_RANK_FACTORIZE",
            "DOMAIN_SPECIALIZE",
        ]
        names = {s.name for s in CompressionStrategy}
        for name in expected:
            assert name in names, f"Missing strategy: {name}"

    def test_strategies_are_unique(self):
        values = [s.value for s in CompressionStrategy]
        assert len(values) == len(set(values))


class TestLayerPrescription:
    def test_minimal_construction(self):
        lp = LayerPrescription(
            layer_idx=0,
            strategy=CompressionStrategy.LIGHT_TOUCH,
        )
        assert lp.layer_idx == 0
        assert lp.strategy == CompressionStrategy.LIGHT_TOUCH
        assert lp.dead_neuron_count == 0
        assert lp.dormant_neuron_count == 0
        assert lp.attn_heads_to_prune == 0

    def test_with_dead_neurons(self):
        lp = LayerPrescription(
            layer_idx=3,
            strategy=CompressionStrategy.DEAD_REMOVAL_AND_MERGE,
            dead_neuron_count=12,
            dormant_neuron_count=5,
        )
        assert lp.dead_neuron_count == 12
        assert lp.dormant_neuron_count == 5

    def test_protected_indices_default_none(self):
        lp = LayerPrescription(layer_idx=0, strategy=CompressionStrategy.LIGHT_TOUCH)
        assert lp.protected_neuron_indices is None

    def test_with_protected_indices(self):
        protected = {0, 5, 42}
        lp = LayerPrescription(
            layer_idx=1,
            strategy=CompressionStrategy.STRUCTURED_PRUNE,
            protected_neuron_indices=protected,
        )
        assert lp.protected_neuron_indices == {0, 5, 42}

    def test_low_rank_fields(self):
        lp = LayerPrescription(
            layer_idx=2,
            strategy=CompressionStrategy.LOW_RANK_FACTORIZE,
            low_rank_target=32,
            low_rank_ranks={"gate_proj": 32, "up_proj": 32, "down_proj": 32},
        )
        assert lp.low_rank_target == 32
        assert lp.low_rank_ranks["gate_proj"] == 32

    def test_domain_fields(self):
        lp = LayerPrescription(
            layer_idx=5,
            strategy=CompressionStrategy.DOMAIN_SPECIALIZE,
            target_domain="math",
            domain_unnecessary_count=8,
            domain_unnecessary_indices=[10, 20, 30],
            domain_critical_indices={1, 2, 3},
        )
        assert lp.target_domain == "math"
        assert lp.domain_unnecessary_count == 8
        assert 10 in lp.domain_unnecessary_indices
        assert 1 in lp.domain_critical_indices


class TestCompressionPrescription:
    def _make_prescription(self, num_layers=4):
        layers = [
            LayerPrescription(
                layer_idx=i,
                strategy=CompressionStrategy.DEAD_REMOVAL_AND_MERGE,
                dead_neuron_count=i * 2,
                dormant_neuron_count=i,
            )
            for i in range(num_layers)
        ]
        return CompressionPrescription(
            model_name="test-model",
            baseline_ppl=25.0,
            num_layers=num_layers,
            intermediate_size=128,
            num_attention_heads=4,
            layers=layers,
            total_dead_neurons=sum(i * 2 for i in range(num_layers)),
            total_dormant_neurons=sum(range(num_layers)),
        )

    def test_construction(self):
        p = self._make_prescription(4)
        assert p.model_name == "test-model"
        assert p.baseline_ppl == 25.0
        assert p.num_layers == 4
        assert len(p.layers) == 4

    def test_layer_count_matches(self):
        p = self._make_prescription(6)
        assert len(p.layers) == 6

    def test_summary_returns_string(self):
        p = self._make_prescription(4)
        s = p.summary()
        assert isinstance(s, str)
        assert len(s) > 0

    def test_summary_contains_model_name(self):
        p = self._make_prescription(4)
        assert "test-model" in p.summary()

    def test_summary_contains_baseline_ppl(self):
        p = self._make_prescription(4)
        assert "25.00" in p.summary()

    def test_summary_contains_strategy_names(self):
        p = self._make_prescription(4)
        s = p.summary()
        assert "DEAD_REMOVAL_AND_MERGE" in s

    def test_summary_contains_per_layer_rows(self):
        p = self._make_prescription(4)
        s = p.summary()
        # Each layer index should appear in the table
        for i in range(4):
            assert str(i) in s

    def test_empty_layers(self):
        p = CompressionPrescription(
            model_name="empty",
            baseline_ppl=10.0,
            num_layers=0,
            intermediate_size=0,
        )
        s = p.summary()
        assert isinstance(s, str)

    def test_depth_prune_in_summary(self):
        p = CompressionPrescription(
            model_name="m",
            baseline_ppl=20.0,
            num_layers=8,
            intermediate_size=64,
            depth_prune_applied=[3],
        )
        assert "3" in p.summary()

    def test_weight_sharing_in_summary(self):
        p = CompressionPrescription(
            model_name="m",
            baseline_ppl=20.0,
            num_layers=8,
            intermediate_size=64,
            weight_sharing_pairs=[(2, 4)],
        )
        assert "2" in p.summary() or "4" in p.summary()

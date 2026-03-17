"""Tests for ExperimentConfig."""

import os
import tempfile
import pytest
from mri_compressor import ExperimentConfig


class TestExperimentConfigDefaults:
    def test_default_model(self):
        cfg = ExperimentConfig()
        assert cfg.model_name == "gpt2"

    def test_default_device(self):
        cfg = ExperimentConfig()
        assert cfg.device == "cuda"

    def test_default_batch_size(self):
        cfg = ExperimentConfig()
        assert cfg.batch_size == 4

    def test_default_max_samples(self):
        cfg = ExperimentConfig()
        assert cfg.max_samples == 256

    def test_default_compression_disabled(self):
        cfg = ExperimentConfig()
        assert cfg.enable_compression is False

    def test_default_depth_pruning_disabled(self):
        cfg = ExperimentConfig()
        assert cfg.enable_depth_pruning is False

    def test_default_low_rank_enabled(self):
        cfg = ExperimentConfig()
        assert cfg.enable_low_rank is True

    def test_default_weight_sharing_disabled(self):
        cfg = ExperimentConfig()
        assert cfg.enable_weight_sharing is False

    def test_default_target_domain_none(self):
        cfg = ExperimentConfig()
        assert cfg.target_domain is None


class TestExperimentConfigCustom:
    def test_model_override(self):
        cfg = ExperimentConfig(model_name="Qwen/Qwen2.5-0.5B")
        assert cfg.model_name == "Qwen/Qwen2.5-0.5B"

    def test_device_override(self):
        cfg = ExperimentConfig(device="cpu")
        assert cfg.device == "cpu"

    def test_compression_flags(self):
        cfg = ExperimentConfig(
            enable_compression=True,
            enable_attn_pruning=True,
            enable_depth_pruning=True,
        )
        assert cfg.enable_compression is True
        assert cfg.enable_attn_pruning is True
        assert cfg.enable_depth_pruning is True

    def test_target_sparsities_default(self):
        cfg = ExperimentConfig()
        assert 0.0 in cfg.target_sparsities
        assert 0.5 in cfg.target_sparsities

    def test_output_dir_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "nested", "output")
            cfg = ExperimentConfig(output_dir=out)
            assert os.path.isdir(out)

    def test_low_rank_energy_threshold(self):
        cfg = ExperimentConfig(low_rank_energy_threshold=0.90)
        assert cfg.low_rank_energy_threshold == 0.90

    def test_domain_fractions(self):
        cfg = ExperimentConfig(
            domain_unnecessary_frac=0.10,
            domain_critical_frac=0.20,
        )
        assert cfg.domain_unnecessary_frac == 0.10
        assert cfg.domain_critical_frac == 0.20

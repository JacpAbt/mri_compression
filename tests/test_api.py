"""
Tests for the public API surface in mri_compressor.__init__.

These tests verify the module structure, callable signatures, and basic wiring
without downloading any model or running GPU operations.
"""

import inspect
import pytest
import mri_compressor


class TestPublicAPIExists:
    """Verify that all documented public names exist and are callable."""

    def test_version_string(self):
        assert hasattr(mri_compressor, "__version__")
        assert isinstance(mri_compressor.__version__, str)
        assert len(mri_compressor.__version__) > 0

    def test_run_mri_callable(self):
        assert callable(mri_compressor.run_mri)

    def test_compress_callable(self):
        assert callable(mri_compressor.compress)

    def test_run_full_pipeline_callable(self):
        assert callable(mri_compressor.run_full_pipeline)

    def test_experiment_config_importable(self):
        from mri_compressor import ExperimentConfig
        assert ExperimentConfig is not None

    def test_model_inspector_importable(self):
        from mri_compressor import ModelInspector
        assert ModelInspector is not None

    def test_mri_runner_importable(self):
        from mri_compressor import MRIRunner
        assert MRIRunner is not None

    def test_mri_diagnostician_importable(self):
        from mri_compressor import MRIDiagnostician
        assert MRIDiagnostician is not None

    def test_mri_compressor_class_importable(self):
        from mri_compressor import MRICompressor
        assert MRICompressor is not None

    def test_compression_prescription_importable(self):
        from mri_compressor import CompressionPrescription
        assert CompressionPrescription is not None

    def test_compression_strategy_importable(self):
        from mri_compressor import CompressionStrategy
        assert CompressionStrategy is not None

    def test_all_exports_present(self):
        for name in mri_compressor.__all__:
            assert hasattr(mri_compressor, name), f"__all__ member missing: {name}"


class TestRunMriSignature:
    def test_has_model_param(self):
        sig = inspect.signature(mri_compressor.run_mri)
        assert "model" in sig.parameters

    def test_has_studies_param_with_default(self):
        sig = inspect.signature(mri_compressor.run_mri)
        assert "studies" in sig.parameters
        assert sig.parameters["studies"].default is None

    def test_has_output_dir(self):
        sig = inspect.signature(mri_compressor.run_mri)
        assert "output_dir" in sig.parameters
        assert sig.parameters["output_dir"].default == "./results"

    def test_has_device_param(self):
        sig = inspect.signature(mri_compressor.run_mri)
        assert "device" in sig.parameters
        assert sig.parameters["device"].default == "cuda"

    def test_has_batch_size(self):
        sig = inspect.signature(mri_compressor.run_mri)
        assert "batch_size" in sig.parameters

    def test_has_max_batches(self):
        sig = inspect.signature(mri_compressor.run_mri)
        assert "max_batches" in sig.parameters

    def test_has_max_samples(self):
        sig = inspect.signature(mri_compressor.run_mri)
        assert "max_samples" in sig.parameters


class TestCompressSignature:
    def test_has_model_param(self):
        sig = inspect.signature(mri_compressor.compress)
        assert "model" in sig.parameters

    def test_has_mri_summary_param(self):
        sig = inspect.signature(mri_compressor.compress)
        assert "mri_summary" in sig.parameters

    def test_has_device_param(self):
        sig = inspect.signature(mri_compressor.compress)
        assert "device" in sig.parameters

    def test_has_enable_attn_pruning(self):
        sig = inspect.signature(mri_compressor.compress)
        assert "enable_attn_pruning" in sig.parameters
        assert sig.parameters["enable_attn_pruning"].default is True

    def test_has_enable_depth_pruning(self):
        sig = inspect.signature(mri_compressor.compress)
        assert "enable_depth_pruning" in sig.parameters
        assert sig.parameters["enable_depth_pruning"].default is False

    def test_has_enable_low_rank(self):
        sig = inspect.signature(mri_compressor.compress)
        assert "enable_low_rank" in sig.parameters

    def test_has_target_domain(self):
        sig = inspect.signature(mri_compressor.compress)
        assert "target_domain" in sig.parameters
        assert sig.parameters["target_domain"].default is None

    def test_has_save_path(self):
        sig = inspect.signature(mri_compressor.compress)
        assert "save_path" in sig.parameters
        assert sig.parameters["save_path"].default is None


class TestRunFullPipelineSignature:
    def test_has_model_param(self):
        sig = inspect.signature(mri_compressor.run_full_pipeline)
        assert "model" in sig.parameters

    def test_has_studies_param(self):
        sig = inspect.signature(mri_compressor.run_full_pipeline)
        assert "studies" in sig.parameters

    def test_has_output_dir(self):
        sig = inspect.signature(mri_compressor.run_full_pipeline)
        assert "output_dir" in sig.parameters

    def test_has_compression_flags(self):
        sig = inspect.signature(mri_compressor.run_full_pipeline)
        for flag in ["enable_attn_pruning", "enable_depth_pruning", "enable_low_rank"]:
            assert flag in sig.parameters, f"Missing: {flag}"


class TestDefaultStudySet:
    def test_default_studies_exist(self):
        assert hasattr(mri_compressor, "_DEFAULT_STUDIES")

    def test_default_studies_is_list(self):
        assert isinstance(mri_compressor._DEFAULT_STUDIES, list)

    def test_default_studies_nonempty(self):
        assert len(mri_compressor._DEFAULT_STUDIES) > 0

    def test_default_studies_are_integers(self):
        for s in mri_compressor._DEFAULT_STUDIES:
            assert isinstance(s, int)

    def test_default_studies_in_valid_range(self):
        for s in mri_compressor._DEFAULT_STUDIES:
            assert 1 <= s <= 22, f"Study {s} out of valid range 1-22"

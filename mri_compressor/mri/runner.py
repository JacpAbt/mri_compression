"""
MRI Runner: Orchestrates execution of all MRI studies on a model.

Replaces the original main.py with a cleaner class-based architecture.
"""

import argparse
import gc
import os
import time
import torch
from pathlib import Path
from typing import Optional

from ..config import ExperimentConfig
from ..model_utils import ModelInspector
from ..data_utils import load_wikitext_data, TextDataset, get_dataloader, evaluate_perplexity
from .summary import build_summary
from .visualize import generate_all_plots


class MRIRunner:
    """Orchestrates MRI study execution on a model."""

    def __init__(self, config: ExperimentConfig, inspector: Optional[ModelInspector] = None):
        self.config = config
        self.inspector = inspector or ModelInspector(config.model_name, config.device)
        self.dataset = None
        self.results = {}
        self.baseline_ppl = None

    def load_data(self):
        """Load calibration dataset."""
        print(f"Loading dataset: {self.config.dataset_name}")
        self.dataset = load_wikitext_data(
            self.inspector.tokenizer,
            max_seq_len=self.config.max_length,
            num_samples=self.config.max_samples,
        )
        print(f"  Loaded {len(self.dataset)} samples")

    def compute_baseline(self):
        """Compute baseline perplexity."""
        if self.dataset is None:
            self.load_data()
        loader = get_dataloader(self.dataset, batch_size=self.config.batch_size)
        self.baseline_ppl = evaluate_perplexity(
            self.inspector.model, loader, self.inspector.device,
            max_batches=self.config.max_eval_batches,
        )
        print(f"Baseline perplexity: {self.baseline_ppl:.2f}")
        return self.baseline_ppl

    def get_architecture_info(self) -> dict:
        """Extract architecture metadata from the model."""
        model = self.inspector.model
        config = model.config
        return {
            "num_layers": self.inspector.num_layers,
            "intermediate_size": getattr(config, "intermediate_size", 0),
            "hidden_size": getattr(config, "hidden_size", 0),
            "num_attention_heads": getattr(config, "num_attention_heads", 0),
            "num_kv_heads": getattr(config, "num_key_value_heads",
                                    getattr(config, "num_attention_heads", 0)),
            "is_gated": self.inspector.is_gated,
            "activation_fn": "silu" if self.inspector.is_gated else "gelu",
        }

    def run_study(self, study_num: int):
        """Run a single study by number."""
        if self.dataset is None:
            self.load_data()

        t0 = time.time()
        # Study 2 needs gradients (gate training); all others are inference-only
        grad_ctx = torch.no_grad() if study_num != 2 else torch.enable_grad()
        grad_ctx.__enter__()

        if study_num == 1:
            from .studies_activation import run_activation_profiling
            self.results["activation_profiles"] = run_activation_profiling(
                self.inspector, self.dataset,
                batch_size=self.config.batch_size,
                max_batches=self.config.max_batches,
            )

        elif study_num == 2:
            from .studies_gates import run_gate_training
            self.results["gate_training"] = run_gate_training(
                self.inspector, self.dataset,
                target_sparsities=self.config.target_sparsities,
                batch_size=self.config.batch_size,
            )

        elif study_num == 3:
            from .studies_importance import compute_wanda_scores
            self.results["wanda_scores"] = compute_wanda_scores(
                self.inspector, self.dataset,
                batch_size=self.config.batch_size,
                max_batches=self.config.max_batches,
            )

        elif study_num == 4:
            from .studies_activation import run_massive_activation_scan
            self.results["massive_activations"] = run_massive_activation_scan(
                self.inspector, self.dataset,
                batch_size=self.config.batch_size,
                max_batches=self.config.max_batches,
            )

        elif study_num == 5:
            from .studies_neuron_health import run_dead_neuron_analysis
            self.results["dead_neurons"] = run_dead_neuron_analysis(
                self.inspector, self.dataset,
                batch_size=self.config.batch_size,
                max_batches=self.config.max_batches,
            )

        elif study_num == 6:
            from .studies_attention import run_attention_head_importance
            self.results["attention_heads"] = run_attention_head_importance(
                self.inspector, self.dataset,
                batch_size=self.config.batch_size,
                max_batches=self.config.max_batches,
            )

        elif study_num == 7:
            from .studies_gates import run_gate_wanda_correlation
            if "gate_training" in self.results and "wanda_scores" in self.results:
                # Pick the middle sparsity target
                sparsities = sorted(self.results["gate_training"].keys())
                mid_sp = sparsities[len(sparsities) // 2]
                gate_patterns = self.results["gate_training"][mid_sp][0]
                self.results["gate_wanda_correlation"] = run_gate_wanda_correlation(
                    gate_patterns, self.results["wanda_scores"],
                    self.inspector.num_layers,
                )
            else:
                print("  Study 7 requires Studies 2 and 3 to run first. Skipping.")

        elif study_num == 8:
            from .studies_structure import run_sparsity_structure_analysis
            self.results["sparsity_structure"] = run_sparsity_structure_analysis(
                self.inspector, self.dataset,
                batch_size=self.config.batch_size,
                max_batches=self.config.max_batches,
            )

        elif study_num == 9:
            from .studies_importance import run_critical_neuron_search
            self.results["critical_neurons"] = run_critical_neuron_search(
                self.inspector, self.dataset,
                batch_size=self.config.batch_size,
                max_eval_batches=self.config.max_eval_batches,
                prior_results=self.results,
            )

        elif study_num == 10:
            from .studies_layer import run_layer_redundancy
            self.results["layer_redundancy"] = run_layer_redundancy(
                self.inspector, self.dataset,
                batch_size=self.config.batch_size,
                max_eval_batches=self.config.max_eval_batches,
            )

        elif study_num == 11:
            from .studies_domain import run_domain_divergence_study
            self.results["domain_divergence"] = run_domain_divergence_study(
                self.inspector,
                batch_size=self.config.batch_size,
                max_batches=self.config.max_batches,
            )

        elif study_num == 12:
            from .studies_cross import run_cross_layer_motif_analysis
            self.results["cross_layer_motifs"] = run_cross_layer_motif_analysis(
                self.inspector, self.dataset,
                batch_size=self.config.batch_size,
                max_batches=self.config.max_batches,
                prior_results=self.results,
            )

        elif study_num == 13:
            from .studies_cross import run_information_bottleneck_profile
            self.results["information_bottleneck"] = run_information_bottleneck_profile(
                self.inspector, self.dataset,
                batch_size=self.config.batch_size,
                max_eval_batches=self.config.max_eval_batches,
            )

        elif study_num == 14:
            from .studies_advanced import run_functional_redundancy_census
            self.results["functional_redundancy"] = run_functional_redundancy_census(
                self.inspector, self.dataset,
                batch_size=self.config.batch_size,
                max_batches=self.config.max_batches,
                prior_results=self.results,
            )

        elif study_num == 15:
            from .studies_advanced import run_perturbation_cascade_analysis
            self.results["perturbation_cascade"] = run_perturbation_cascade_analysis(
                self.inspector, self.dataset,
                batch_size=self.config.batch_size,
                max_eval_batches=self.config.max_eval_batches,
                prior_results=self.results,
            )

        elif study_num == 16:
            from .studies_advanced import run_phase_transition_analysis
            self.results["phase_transition"] = run_phase_transition_analysis(
                self.inspector, self.dataset,
                batch_size=self.config.batch_size,
                max_batches=self.config.max_batches,
                prior_results=self.results,
            )

        elif study_num == 17:
            from .studies_nextgen import run_cross_layer_alignment
            self.results["cross_layer_alignment"] = run_cross_layer_alignment(
                self.inspector, self.dataset,
                batch_size=self.config.batch_size,
                max_batches=self.config.max_batches,
            )

        elif study_num == 18:
            from .studies_nextgen import run_weight_rank_analysis
            self.results["weight_rank"] = run_weight_rank_analysis(
                self.inspector, self.dataset,
                batch_size=self.config.batch_size,
                max_batches=self.config.max_batches,
            )

        elif study_num == 19:
            from .studies_nextgen import run_attention_head_clustering
            self.results["head_clustering"] = run_attention_head_clustering(
                self.inspector, self.dataset,
                batch_size=self.config.batch_size,
                max_batches=self.config.max_batches,
            )

        elif study_num == 20:
            from .studies_nextgen import run_static_dynamic_decomposition
            self.results["static_dynamic"] = run_static_dynamic_decomposition(
                self.inspector, self.dataset,
                batch_size=self.config.batch_size,
                max_batches=self.config.max_batches,
            )

        elif study_num == 21:
            from .studies_nextgen import run_magnitude_divergence
            self.results["magnitude_divergence"] = run_magnitude_divergence(
                self.inspector, self.dataset,
                batch_size=self.config.batch_size,
                max_batches=self.config.max_batches,
            )

        elif study_num == 22:
            from .studies_domain_importance import run_domain_conditional_importance
            custom = self._load_custom_domain() if self._has_custom_domain() else None
            self.results["domain_conditional_importance"] = run_domain_conditional_importance(
                self.inspector,
                batch_size=self.config.batch_size,
                max_batches=self.config.max_batches,
                custom_domain_datasets=custom,
                prior_results=self.results,
            )

        elif study_num == 23:
            from .studies_hybrid_attention import run_linear_attention_rank_analysis
            self.results["linear_attention_rank"] = run_linear_attention_rank_analysis(
                self.inspector, self.dataset,
                batch_size=self.config.batch_size,
                max_batches=self.config.max_batches,
            )

        elif study_num == 24:
            from .studies_domain_compression import (
                run_domain_compression_curve,
                load_biomedical_dataset,
            )

            # Resolve target domain dataset
            domain_name = getattr(self.config, "custom_domain_name", None) or "biomedical"
            domain_dataset = None

            # Use custom domain dataset if provided and matches the target domain
            if self._has_custom_domain():
                custom = self._load_custom_domain()
                if custom and domain_name in custom:
                    domain_dataset = custom[domain_name]

            # Fall back to the built-in biomedical loader
            if domain_dataset is None:
                if domain_name != "biomedical":
                    print(f"  WARNING: custom domain '{domain_name}' not loaded; "
                          f"using built-in biomedical dataset instead")
                domain_name = "biomedical"
                domain_dataset = load_biomedical_dataset(
                    self.inspector.tokenizer,
                    max_seq_len=self.config.max_length,
                    n_samples=min(64, self.config.max_samples),
                )

            # Attach the registered benchmark function for the domain (if any).
            # For biomedical this gives the PubMedQA accuracy curve alongside PPL.
            from ..benchmark import DOMAIN_BENCHMARKS
            benchmark_fn = DOMAIN_BENCHMARKS.get(domain_name)

            self.results["domain_compression_curve"] = run_domain_compression_curve(
                self.inspector,
                domain_dataset=domain_dataset,
                domain_name=domain_name,
                batch_size=self.config.batch_size,
                max_batches_wanda=self.config.max_batches,
                max_batches_eval=min(8, self.config.max_batches),
                prior_results=self.results,
                benchmark_fn=benchmark_fn,
                acc_threshold_absolute=0.05,
                benchmark_n_samples=100,
            )

        elif study_num == 25:
            from .studies_geometry import run_geometry_analysis
            domain_ds = None
            if self._has_custom_domain():
                custom = self._load_custom_domain()
                if custom:
                    domain_ds = custom
            self.results["write_vector_geometry"] = run_geometry_analysis(
                self.inspector,
                self.dataset,
                domain_datasets=domain_ds,
                batch_size=self.config.batch_size,
                max_vectors=2000,
                max_batches=self.config.max_batches,
            )

        else:
            print(f"  Unknown study number: {study_num}")
            return

        grad_ctx.__exit__(None, None, None)
        elapsed = time.time() - t0
        print(f"  Study {study_num} completed in {elapsed:.1f}s")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Domain names that ship their own data loaders — no file path required.
    _BUILTIN_CUSTOM_DOMAINS = {"biomedical"}

    def _has_custom_domain(self) -> bool:
        """Check if config has custom domain settings.

        Returns True when:
          - custom_domain_name is set AND custom_domain_path is set  (file or
            HuggingFace dataset path), OR
          - custom_domain_name is one of the built-in domains with their own
            data loaders (e.g. "biomedical" → PubMed).  No path is required.
        """
        name = getattr(self.config, "custom_domain_name", None)
        if not name:
            return False
        # Built-in domain — path is optional
        if name in self._BUILTIN_CUSTOM_DOMAINS:
            return True
        # Path-based custom domain
        return getattr(self.config, "custom_domain_path", None) is not None

    def _load_custom_domain(self) -> dict:
        """Load a custom domain dataset.

        Handles three cases in order:
          1. Built-in domain (e.g. "biomedical") — uses its own loader.
          2. Local text file at custom_domain_path.
          3. HuggingFace dataset at custom_domain_path.
        """
        name = self.config.custom_domain_name
        path = getattr(self.config, "custom_domain_path", None)
        max_seq_len = self.config.max_length

        # ---- Built-in custom domain ----
        if name == "biomedical":
            from .studies_domain_compression import load_biomedical_dataset
            print(f"  Loading built-in domain 'biomedical' (PubMed abstracts)")
            dataset = load_biomedical_dataset(
                self.inspector.tokenizer,
                max_seq_len=max_seq_len,
                n_samples=min(64, self.config.max_samples),
            )
            return {name: dataset}

        # ---- Path-based custom domain ----
        print(f"  Loading custom domain '{name}' from {path}")

        if os.path.isfile(path):
            # Load from text file
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            texts = [chunk for chunk in text.split("\n\n") if len(chunk.strip()) > 50]
            if not texts:
                texts = [text]
        else:
            # Try loading as HuggingFace dataset
            try:
                from datasets import load_dataset
                ds = load_dataset(path, split="train", trust_remote_code=True)
                for field_name in ["text", "content", "question", "prompt"]:
                    if field_name in ds.column_names:
                        texts = [str(row[field_name]) for row in ds
                                 if len(str(row[field_name]).strip()) > 50]
                        if texts:
                            break
                else:
                    print(f"    WARNING: Could not find text field in dataset {path}")
                    return {}
            except Exception as e:
                print(f"    ERROR loading custom domain: {e}")
                return {}

        # Tokenize and chunk
        all_text = "\n".join(texts[:1000])
        tokens = self.inspector.tokenizer.encode(all_text, return_tensors="pt")[0]
        n_chunks = min(64, len(tokens) // max_seq_len)
        if n_chunks < 4:
            repeat_factor = (4 * max_seq_len // len(tokens)) + 1
            tokens = tokens.repeat(repeat_factor)
            n_chunks = min(64, len(tokens) // max_seq_len)

        chunks = tokens[:n_chunks * max_seq_len].reshape(n_chunks, max_seq_len)
        dataset = TextDataset(chunks)
        print(f"    Custom domain '{name}': {n_chunks} chunks of {max_seq_len} tokens")
        return {name: dataset}

    def run_studies(self, studies: list[int]):
        """Run multiple studies in order."""
        print(f"\n{'='*80}")
        print(f"MRI SCAN: {self.config.model_name}")
        print(f"Studies to run: {studies}")
        print(f"{'='*80}\n")

        for study_num in studies:
            try:
                self.run_study(study_num)
            except Exception as e:
                print(f"  ERROR in Study {study_num}: {e}")
                import traceback
                traceback.print_exc()
                # If run_study entered a no_grad context but crashed before
                # __exit__, torch.no_grad() stays active permanently and breaks
                # any subsequent operation that needs gradients (e.g. LocalReconstructor).
                if not torch.is_grad_enabled():
                    torch.set_grad_enabled(True)

    def save(self, output_dir: str) -> dict:
        """Save enriched summary and generate plots."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        if self.baseline_ppl is None:
            self.compute_baseline()

        summary = build_summary(
            model_name=self.config.model_name,
            baseline_ppl=self.baseline_ppl,
            architecture=self.get_architecture_info(),
            results=self.results,
            output_dir=output_dir,
        )

        # Generate plots
        try:
            generate_all_plots(self.results, self.config.model_name, output_dir)
        except Exception as e:
            print(f"  Warning: Plot generation failed: {e}")

        return summary


def main():
    """CLI entry point for standalone MRI runs."""
    parser = argparse.ArgumentParser(description="MRI LLM Analysis Suite")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name or path")
    parser.add_argument("--studies", type=str, default="1,3,4,5,6,8,9,10",
                        help="Comma-separated study numbers to run")
    parser.add_argument("--output", type=str, default="./mri_results", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    config = ExperimentConfig(
        model_name=args.model,
        device=device,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        max_samples=args.max_samples,
        max_length=args.max_length,
        max_eval_batches=min(args.max_batches, 8),
    )

    studies = [int(s.strip()) for s in args.studies.split(",")]

    runner = MRIRunner(config)
    runner.run_studies(studies)
    summary = runner.save(args.output)

    print(f"\nMRI scan complete. Results saved to {args.output}")
    return runner, summary


if __name__ == "__main__":
    main()

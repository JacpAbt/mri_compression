"""
Configuration for the Sparsity MRI experiment suite.

This experiment performs a comprehensive analysis of internal model structure
during activation sparsification, inspired by and extending several lines of research:

Core Studies:
  - CATS (2024): Learned token-adaptive sparsity via sigmoid gates on activations
  - MaskLLM (2024): Learned Gumbel-sigmoid masks for semi-structured sparsity
  - ProxSparse (2024): Proximal gradient for activation sparsity
  - Wanda (2023): Weight * activation magnitude pruning metric

Extended Analyses (exploratory):
  - Massive Activations (Sun et al., 2024 COLM): Input-agnostic outlier activations
  - Super Weights (Yu et al., 2024): Single weights that destroy model if pruned
  - Dead/Dormant Neurons (Voita et al., 2023; ACL 2024 Findings): Never-firing MLP neurons
  - Attention Head Specialization (Voita et al., 2019): Which heads do the heavy lifting
  - Outlier Feature Emergence (He et al., NeurIPS 2024): Why outliers form during training
  - Critical Neurons (2025): Ultra-sparse neuron subsets governing model behavior
  - Attention Sinks (Xiao et al., 2023): Disproportionate attention to initial tokens

Model Selection Rationale:
  GPT-2 (124M): GELU activation, no gated MLP. Good as a baseline and for fast iteration.
    - Cons: GELU doesn't produce natural sparsity; no gate_proj/up_proj split
    - Pros: Well-studied, fits easily on any GPU, huge literature for comparison
  
  Qwen2.5-0.5B or TinyLlama-1.1B: SwiGLU activation, gated MLP (gate * up -> down)
    - This is where CATS-style sparsity actually makes architectural sense
    - Natural ~50% sparsity from SiLU gating
    - Still fits on a 3090 with room for gradient computation
  
  Recommendation: Run BOTH. Design code to be model-agnostic, compare findings.
  The key question is: which phenomena are architecture-dependent vs universal?
"""

from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class ExperimentConfig:
    """Master configuration for all analyses."""

    # ---- Model ----
    model_name: str = "gpt2"  # or "Qwen/Qwen2.5-0.5B", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    # ---- Data ----
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-103-raw-v1"
    max_length: int = 512
    max_samples: int = 256  # calibration samples

    # ---- Gate Training (CATS-style, Study 2) ----
    gate_lr: float = 1e-2
    gate_training_steps: int = 500
    gate_warmup_steps: int = 50
    target_sparsities: list = field(default_factory=lambda: [0.0, 0.25, 0.50, 0.75, 0.90])
    sparsity_loss_weight: float = 2.0

    # ---- Analysis ----
    batch_size: int = 4
    max_batches: int = 16
    max_eval_batches: int = 8
    device: str = "cuda"
    seed: int = 42
    output_dir: str = "./results"

    # ---- Compression ----
    enable_compression: bool = False
    enable_attn_pruning: bool = False
    enable_depth_pruning: bool = False
    enable_merge: bool = True
    reconstruction_steps: int = 200
    reconstruction_lr: float = 1e-4
    save_compressed_model: bool = False
    target_domain: Optional[str] = None  # for domain-conditional compression
    custom_domain_path: Optional[str] = None  # path to custom domain text file or HF dataset
    custom_domain_name: Optional[str] = None  # name for the custom domain
    domain_unnecessary_frac: float = 0.05  # max fraction of neurons to remove per layer
    domain_critical_frac: float = 0.10     # fraction of top neurons to protect per layer
    enable_low_rank: bool = True
    enable_static_fold: bool = True
    enable_weight_sharing: bool = False  # off by default, experimental
    low_rank_energy_threshold: float = 0.95

    # ---- Domain Imprinting (post-compression bias injection) ----
    enable_imprinting: bool = False   # inject domain centroid into down_proj bias
    imprinting_scale: float = 0.05   # scale factor for centroid projection (tune 0.01–0.10)

    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

# mri-compressor

Diagnostic-guided LLM compression without quantization or hardware-specific tricks.

The core idea: before compressing a model, run an "MRI scan" — 22 diagnostic studies that reveal exactly which neurons are dead, dormant, load-bearing, domain-specific, or redundant. Use those findings to build a per-layer prescription, then apply compression operations guided by the data.

**Supported techniques:** dead/dormant neuron removal, neuron merging, Wanda-guided structured pruning, SVD low-rank factorization, static neuron folding, attention head pruning, depth pruning, weight sharing, and SparseGPT-style local reconstruction.

**No quantization.** All operations are exact and architecture-preserving.

---

## Install

```bash
pip install -e .
# or, from PyPI once published:
pip install mri-compressor
```

Dependencies: `torch>=2.1.0`, `transformers>=4.35.0`, `datasets>=2.14.0`, `scipy>=1.10.0`, `matplotlib>=3.8.0`, `numpy>=1.24.0`

---

## Quick Start (Library)

```python
import mri_compressor

# --- MRI only: understand the model before touching it ---
summary = mri_compressor.run_mri(
    "Qwen/Qwen2.5-0.5B",
    studies=[1, 3, 4, 5, 6, 8, 9, 10],   # see study list below
    output_dir="./results",
    device="cuda",
)
print("Baseline PPL:", summary["baseline_ppl"])

# --- Compress using MRI findings ---
result = mri_compressor.compress(
    "Qwen/Qwen2.5-0.5B",
    summary,
    enable_attn_pruning=True,
    enable_low_rank=True,
    save_path="./compressed_model",
)
print(f"PPL: {result['original_ppl']:.2f} -> {result['compressed_ppl']:.2f}")

# --- Or run everything in one call ---
result = mri_compressor.run_full_pipeline("gpt2", output_dir="./results")

# --- Pass a pre-loaded model (e.g. already fine-tuned) ---
from transformers import AutoModelForCausalLM, AutoTokenizer
model     = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
summary   = mri_compressor.run_mri((model, tokenizer), studies=[1, 3, 5])
```

### Advanced (class-level access)

```python
from mri_compressor import (
    ExperimentConfig,
    MRIRunner,
    MRIDiagnostician,
    MRICompressor,
    CompressionPrescription,
)

config = ExperimentConfig(model_name="gpt2", device="cuda", output_dir="./out")

runner = MRIRunner(config)
runner.run_studies([1, 3, 4, 5, 10])
summary = runner.save("./out")

diagnostician = MRIDiagnostician(enable_attn_pruning=True, ppl_budget=2.0)
prescription = diagnostician.diagnose_from_summary(summary)
print(prescription.summary())
```

## Quick Start (CLI)

```bash
# MRI scan only (~10-60 min depending on model and studies)
python -m mri_compressor.pipeline \
    --model Qwen/Qwen2.5-0.5B \
    --studies 1,3,4,5,6,8,9,10 \
    --output ./results

# Full pipeline: MRI + compression
python -m mri_compressor.pipeline \
    --model Qwen/Qwen2.5-0.5B \
    --studies 1,3,4,5,6,8,9,10,11 \
    --compress --enable-attn --save-model \
    --output ./results

# Compression from a saved MRI summary (skip re-scanning)
python -m mri_compressor.pipeline \
    --model Qwen/Qwen2.5-0.5B \
    --from-summary ./results/summary.json \
    --compress --save-model

# Domain-specific compression (protect math-critical neurons)
python -m mri_compressor.pipeline \
    --model Qwen/Qwen2.5-0.5B \
    --compress --target-domain math \
    --output ./results
```

---

## The 4-Stage Pipeline

```
Model + Calibration Data
        |
        v
+----------------------+
|   STAGE 1: MRI SCAN  |  Run 1-22 diagnostic studies -> summary.json + plots
+----------+-----------+
           |
           v
+----------------------+
| STAGE 2: DIAGNOSIS   |  Analyze findings -> per-layer CompressionPrescription
+----------+-----------+
           |
           v
+----------------------+
| STAGE 3: COMPRESSION |  Apply prescription layer-by-layer -> compressed model
+----------+-----------+
           |
           v
+----------------------+
| STAGE 4: EVALUATION  |  Measure perplexity delta and parameter reduction
+----------------------+
```

The diagnosis stage is purely data-driven: it reads the MRI findings and uses PPL deltas, dead neuron counts, activation statistics, and CKA similarity scores to decide what to do to each layer. A configurable PPL budget prevents over-compression.

---

## Compression Operations

| Operation | Informed by | What it does |
|-----------|-------------|--------------|
| Dead neuron removal | Study 5 dead counts | Removes neurons that never fire across calibration data |
| Dormant neuron removal | Study 5 dormant counts | Removes neurons active on <1% of tokens |
| Neuron merging | Study 14 redundancy | Merges neurons with high cosine similarity |
| Wanda pruning | Study 3 scores | Structured pruning via `|weight| x ||activation||` |
| Low-rank factorization | Study 18 rank analysis | SVD: replaces Linear(in, out) with two smaller matrices |
| Static folding | Study 20 static analysis | Folds near-constant neurons into bias terms |
| Attention head pruning | Study 6 entropy | Zeros low-importance attention heads |
| Depth pruning | Study 10 redundancy | Removes entire redundant layers (disabled by default) |
| Local reconstruction | Post-compression | SparseGPT-style weight update to recover accuracy |

Protected neurons (massive activations from Study 4, critical neurons from Study 9) are never touched regardless of which operations are enabled.

---

## Model Support

| Model | Architecture | Activation | Notes |
|-------|-------------|-----------|-------|
| **GPT-2 (124M)** | Standard MLP (fc1->GELU->fc2) | GELU | Baseline. No natural sparsity. |
| **Qwen2.5-0.5B** | Gated MLP (gate*up->down) | SwiGLU | ~50% natural sparsity from SiLU gate. Recommended. |
| **TinyLlama-1.1B** | Gated MLP | SwiGLU | Same architecture as Qwen, slightly larger. |

The key architectural difference: gated MLPs (Qwen, Llama, Mistral) have a multiplicative gate that naturally produces ~50% near-zero activations. Standard GELU MLPs (GPT-2) don't, so aggressive MLP compression is harder. The diagnostics work on both; compression yields vary.

---

## The 22 Studies

### Core (1-11)

#### Study 1: Activation Profiling
What do activation distributions look like layer-by-layer? Computes mean, std, kurtosis, Gini coefficient, natural sparsity, and outlier ratios. This is the baseline — CATS assumes activations cluster near zero (true for SiLU, not for GELU).

**Related work:** CATS (2024), Massive Activations (Sun et al., 2024), Understanding Outlier Features (He et al., NeurIPS 2024)

#### Study 2: CATS-style Gate Training
Learn per-neuron sigmoid gates via LM loss + sparsity regularization. The resulting gate values reveal the model's own "opinion" about which neurons matter. Expensive but informative.

**Related work:** CATS (2024), MaskLLM (2024), Voita et al. (2019)

#### Study 3: Wanda Importance Scores
Compute `|weight| x ||activation||` for each neuron. Zero-cost, no-training. Study 7 checks whether this agrees with learned gates.

**Related work:** Wanda (Sun et al., ICLR 2024), SparseGPT (Frantar & Alistarh, 2023)

#### Study 4: Massive Activation Scan
Flag neurons with disproportionately large, input-agnostic activations (mean >> layer median, low variance). These act as implicit bias terms and must never be pruned. Outputs to the "never prune" protection list.

**Related work:** Massive Activations (Sun et al., 2024 COLM), Super Weights (Yu et al., 2024), DuQuant (NeurIPS 2024)

#### Study 5: Dead / Dormant Neuron Census
Classify each neuron as dead (0% firing rate), dormant (<1%), rare (<10%), or hyperactive (>99%). Dead and dormant neurons are free to remove.

**Related work:** "Neurons in LLMs: Dead, N-gram, Positional" (ACL 2024 Findings), "The Achilles' Heel of LLMs" (2025)

#### Study 6: Attention Head Importance
Compute per-head entropy, first-token attention fraction, and max concentration. Low-entropy heads are specialized; high-entropy heads may be redundant.

**Related work:** Voita et al. (ACL 2019), Attention Sinks (Xiao et al., 2023), From Attention to Activation (Kaul et al., 2024)

#### Study 7: Gate-Wanda Correlation
Pearson and Spearman correlation between learned gates (Study 2) and Wanda scores (Study 3), plus top-K overlap at multiple sparsity levels. High correlation = Wanda is sufficient. Low correlation = gradients capture structure that heuristics miss.

#### Study 8: Sparsity Structure Analysis
Is sparsity input-dependent (different neurons fire for different inputs) or mostly static? If mostly static, dynamic methods like CATS are over-engineering the problem. Informs whether static or gate-guided pruning is better.

**Related work:** CATS key claim on "contextually-aware" sparsity. DejaVu (Liu et al., 2023).

#### Study 9: Critical Neuron Search
Zero out individual neurons one-at-a-time and measure perplexity impact. Finds "super weights" — single neurons whose removal is catastrophic. Outputs to the "never prune" protection list.

**Related work:** Super Weights (Yu et al., 2024), The Achilles' Heel of LLMs (2025)

#### Study 10: Layer Redundancy Analysis
Zero out each layer's MLP or attention output and measure perplexity impact. Reveals the model's depth structure: early and late layers tend to be critical; middle layers may be partially redundant.

**Related work:** ShortGPT (Men et al., 2024), Layer-Selective Rank Reduction (Sharma et al., 2024)

#### Study 11: Domain-Specific Activation Divergence
Run activation collection on four domain datasets (English prose, math, code, Italian). Compute per-layer Jaccard similarity between domain neuron sets, identify universal neurons and domain-specific ones. Determines whether domain-specific pruning is viable and where in the network it is most effective.

**Related work:** "Sharing Matters: Analysing Neurons Across Languages and Tasks" (ACL 2024), "Language-Specific Neurons" (Tang et al., 2024)

### Extended (12-22)

| Study | Name | What it measures |
|-------|------|-----------------|
| 12 | Cross-Layer Motif Analysis | Recurring activation patterns across layers |
| 13 | Information Bottleneck Profile | How much information each layer preserves |
| 14 | Functional Redundancy Census | Neurons with high cosine similarity (merge candidates) |
| 15 | Perturbation Cascade Analysis | How strongly removing neurons propagates downstream |
| 16 | Phase Transition Analysis | Power-law tail exponent (heavy tail = compression-resistant) |
| 17 | Cross-Layer Alignment (CKA) | Representational similarity between adjacent layers |
| 18 | Weight Rank Analysis | Effective rank of weight matrices (low rank = factorize) |
| 19 | Attention Head Clustering | Which heads are functionally similar (prune redundant clusters) |
| 20 | Static-Dynamic Decomposition | Which neuron activations are near-constant (fold to bias) |
| 21 | Magnitude Divergence | Domain-sensitive activation magnitude difference |
| 22 | Domain-Conditional Importance | Which neurons are critical for a specific target domain |

---

## Output Structure

```
results/
+-- summary.json                          # Full MRI findings (input to diagnosis stage)
+-- study1_activation_profiles.png        # 6-panel activation distribution analysis
+-- study2_gate_training.png              # Gate training dynamics + learned patterns
+-- study3_wanda_scores.png               # Wanda importance distributions
+-- study4_massive_activations.png        # Massive activation locations
+-- study5_dead_neurons.png               # Dead/dormant neuron census
+-- study6_attention_heads.png            # Attention head entropy/specialization
+-- study7_gate_wanda_correlation.png     # Gate vs Wanda agreement
+-- study10_layer_redundancy.png          # Layer importance breakdown
+-- study11_domain_divergence.png         # 7-panel domain specialization analysis
```

`summary.json` is the key artifact: it contains per-layer results for every study that was run and can be reused to run compression multiple times without re-scanning the model.

---

## Package Structure

```
mri_compressor/
+-- __init__.py                # Public API: run_mri(), compress(), run_full_pipeline()
+-- pipeline.py                # Stage functions chained by __init__.py wrappers
+-- config.py                  # ExperimentConfig dataclass
+-- model_utils.py             # ModelInspector (GPT-2, Qwen, Llama, etc.)
+-- data_utils.py              # WikiText loading and perplexity evaluation
+-- mri/
|   +-- runner.py              # MRIRunner: orchestrates studies 1-22
|   +-- studies_activation.py  # Studies 1, 4
|   +-- studies_gates.py       # Studies 2, 7
|   +-- studies_importance.py  # Studies 3, 9
|   +-- studies_neuron_health.py  # Study 5
|   +-- studies_attention.py   # Study 6
|   +-- studies_structure.py   # Study 8
|   +-- studies_layer.py       # Study 10
|   +-- studies_domain.py      # Study 11
|   +-- studies_cross.py       # Studies 12, 13
|   +-- studies_advanced.py    # Studies 14, 15, 16
|   +-- studies_nextgen.py     # Studies 17-21
|   +-- studies_domain_importance.py  # Study 22
|   +-- summary.py             # Builds summary.json from study results
|   +-- visualize.py           # Plot generation
+-- compression/
    +-- compressor.py          # MRICompressor: applies prescription to model
    +-- diagnostician.py       # MRIDiagnostician: MRI results -> prescription
    +-- prescription.py        # CompressionPrescription, LayerPrescription dataclasses
    +-- evaluate.py            # Perplexity evaluation utilities
    +-- neuron_recycling.py    # Neuron merge/recombination logic
    +-- _utils.py              # MLP/attention module accessors
    +-- operations/
        +-- dead_removal.py    # DeadNeuronRemover
        +-- neuron_merge.py    # NeuronMerger
        +-- wanda_pruner.py    # WandaPruner
        +-- attention_pruner.py  # AttentionHeadPruner
        +-- depth_pruner.py    # DepthPruner
        +-- low_rank.py        # LowRankFactorizer
        +-- static_fold.py     # StaticNeuronFolder
        +-- weight_sharing.py  # WeightSharer
        +-- reconstructor.py   # LocalReconstructor (SparseGPT-style)
```

---

## Key Hypotheses

1. **GELU vs SiLU natural sparsity**: SiLU models should show ~50% natural near-zero activations; GELU ~0%. (Study 1)

2. **Massive activations are gate-protected**: Learned gates keep massive activation neurons fully open. (Studies 2 + 4)

3. **Dead neurons are gate-pruned**: Learned gates assign near-zero values to dead/dormant neurons. (Studies 2 + 5)

4. **Wanda partially explains gates**: Moderate but not perfect correlation, suggesting gradients find contextual importance that heuristics miss. (Study 7)

5. **Critical neurons are outlier neurons**: The neurons most damaging to remove overlap with massive activations and high Wanda scores. (Studies 3, 4, 9)

6. **Middle layers are most compressible**: Layer redundancy shows a U-shape — early and late layers critical, middle layers partially redundant. (Study 10)

7. **Domain divergence peaks in middle layers**: Early layers have high domain overlap (shared syntax), middle layers diverge (domain-specific reasoning), late layers partially reconverge. (Study 11)

8. **Code and math share more structure than code and prose**: Both involve formal/logical reasoning patterns. (Study 11)

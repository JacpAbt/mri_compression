# mri-compressor

Diagnostic-guided LLM compression without quantization or hardware-specific tricks.

The core idea: before compressing a model, run an "MRI scan" — up to 25 diagnostic studies that reveal exactly which neurons are dead, dormant, load-bearing, domain-specific, or redundant. Use those findings to build a per-layer prescription, then apply compression operations guided by the data.

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

## The 4-Stage Pipeline

```
Model + Calibration Data
        |
        v
+----------------------+
|   STAGE 1: MRI SCAN  |  Run diagnostic studies -> summary.json + plots
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

The diagnosis stage is purely data-driven: it reads the MRI findings and uses PPL deltas, dead neuron counts, activation statistics, and geometric similarity scores to decide what to do to each layer. A configurable PPL budget prevents over-compression.

---

## Quick Start (CLI)

### Modes

The `--mode` flag selects a preset study bundle and compression target. You can override any mode with `--studies` for custom combinations.

| Mode | Studies run | Compresses | Domain required | Purpose |
|------|-------------|-----------|-----------------|---------|
| `general` | 1,3,4,5,6,8,9,10 | No | No | First look at any model — all diagnostics, no changes |
| `compress` | 1,3,4,5,6,8,9,10 | Yes | No | General-purpose compression |
| `domain_scan` | 1,5,11,22,24,25 | No | Yes | Domain analysis only |
| `domain_compress` | 1,3,4,5,6,8,9,10,11,22,24,25 | Yes | Yes | Domain-specialized compression |
| `full_scan` | 1–25 | No | No | Full research scan, all studies |

### Examples

#### 1 — General-purpose compression

Scan and compress a model without a domain target. Removes dead/dormant neurons, prunes low-importance attention heads, applies low-rank factorization and static folding where safe.

```bash
python -m mri_compressor.pipeline \
    --model Qwen/Qwen2.5-0.5B \
    --mode compress \
    --output ./results_general \
    --save-model
```

Expected output: `~3–8%` parameter reduction with `<5%` PPL increase on WikiText-103.

---

#### 2 — Domain-specialized compression (built-in: biomedical)

Run the full domain compression pipeline. Studies 22 and 24 identify which neurons are unnecessary for the target domain; Study 25 uses write-vector geometry to re-rank removal candidates. The model is compressed by removing neurons that are irrelevant to biomedical text while protecting those critical for it.

```bash
python -m mri_compressor.pipeline \
    --model Qwen/Qwen3.5-0.8B \
    --mode domain_compress \
    --domain biomedical \
    --output ./results_biomedical \
    --max-samples 256 \
    --max-batches 32 \
    --batch-size 4 \
    --save-model
```

Expected output: `~8–12%` parameter reduction. Domain PPL typically increases `<10%`; generic PPL increases significantly (expected — domain-irrelevant neurons were removed).

**Built-in domains:** `biomedical` (PubMed abstracts, auto-downloaded)

**Standard domains** (loaded from HuggingFace automatically): `english`, `math`, `code`, `italian`

---

#### 3 — Custom domain

Point to your own text file or HuggingFace dataset path. The domain name is arbitrary.

```bash
python -m mri_compressor.pipeline \
    --model Qwen/Qwen2.5-7B \
    --mode domain_compress \
    --domain legal \
    --domain-dataset /data/legal_contracts.txt \
    --output ./results_legal \
    --save-model
```

---

#### 4 — Scan only, then compress from saved summary

The MRI scan is the expensive part (~1–3 hours for a 7B model). Save the summary and reuse it to try different compression settings without re-scanning.

```bash
# Step 1: Run the MRI scan only
python -m mri_compressor.pipeline \
    --model Qwen/Qwen2.5-0.5B \
    --mode general \
    --output ./results

# Step 2: Run compression from the saved summary (skips the scan)
python -m mri_compressor.pipeline \
    --model Qwen/Qwen2.5-0.5B \
    --from-summary ./results/summary.json \
    --compress \
    --enable-attn \
    --enable-depth \
    --save-model \
    --output ./results_compressed
```

---

#### 5 — Full research scan (all studies)

Runs all 25 studies including write-vector geometry and domain analysis. No compression. Useful for understanding a new model before deciding a compression strategy.

```bash
python -m mri_compressor.pipeline \
    --model Qwen/Qwen2.5-0.5B \
    --mode full_scan \
    --domain biomedical \
    --output ./results_full
```

---

#### 6 — Custom study selection

Run any subset of studies. Use `--compress` to apply compression with findings from whichever studies were run.

```bash
python -m mri_compressor.pipeline \
    --model gpt2 \
    --studies 1,3,5,9,10 \
    --compress \
    --output ./results_custom
```

---

#### 7 — Larger models with tuned data parameters

For larger models or better calibration quality, increase batch counts and samples.

```bash
python -m mri_compressor.pipeline \
    --model Qwen/Qwen2.5-7B \
    --mode domain_compress \
    --domain biomedical \
    --max-samples 512 \
    --max-batches 64 \
    --batch-size 2 \
    --output ./results_7b \
    --save-model
```

---

### Full CLI Reference

```
python -m mri_compressor.pipeline [OPTIONS]
```

#### Model & device
| Argument | Default | Description |
|----------|---------|-------------|
| `--model TEXT` | `gpt2` | HuggingFace model name or local path |
| `--device {auto,cuda,cpu}` | `auto` | Inference device |

#### Mode & studies
| Argument | Default | Description |
|----------|---------|-------------|
| `--mode {general,compress,domain_scan,domain_compress,full_scan}` | — | Preset study bundle (see table above) |
| `--studies TEXT` | — | Comma-separated study numbers, overrides `--mode` (e.g. `1,5,10`) |
| `--from-summary PATH` | — | Skip the MRI scan; load an existing `summary.json` |

#### Domain
| Argument | Default | Description |
|----------|---------|-------------|
| `--domain NAME` | — | Target domain for domain-aware modes |
| `--domain-dataset PATH` | — | Path to custom domain text file or HuggingFace dataset |

#### Data
| Argument | Default | Description |
|----------|---------|-------------|
| `--batch-size INT` | `4` | Batch size for forward passes |
| `--max-batches INT` | `16` | Number of batches per study |
| `--max-samples INT` | `1000` | Maximum calibration samples |
| `--max-length INT` | `512` | Token sequence length |

#### Compression toggles
| Argument | Default | Description |
|----------|---------|-------------|
| `--compress` | off | Run compression after MRI (auto-set by compress modes) |
| `--enable-attn` | off | Enable attention head pruning |
| `--enable-depth` | off | Enable depth pruning (remove entire layers) |
| `--disable-merge` | off | Disable neuron merging |
| `--disable-low-rank` | off | Disable SVD low-rank factorization |
| `--disable-static-fold` | off | Disable static neuron folding |
| `--enable-weight-sharing` | off | Enable experimental weight sharing |

#### Reconstruction
| Argument | Default | Description |
|----------|---------|-------------|
| `--reconstruction-steps INT` | `200` | SparseGPT-style local fine-tuning steps per layer |
| `--reconstruction-lr FLOAT` | `1e-4` | Learning rate for reconstruction |

#### Output
| Argument | Default | Description |
|----------|---------|-------------|
| `--output PATH` | `./results` | Directory for summary, plots, and evaluation report |
| `--save-model` | off | Save compressed model to `<output>/compressed_model/` |

---

## Quick Start (Library)

```python
import mri_compressor

# MRI only: understand the model before touching it
summary = mri_compressor.run_mri(
    "Qwen/Qwen2.5-0.5B",
    studies=[1, 3, 4, 5, 6, 8, 9, 10],
    output_dir="./results",
    device="cuda",
)
print("Baseline PPL:", summary["baseline_ppl"])

# Compress using MRI findings
result = mri_compressor.compress(
    "Qwen/Qwen2.5-0.5B",
    summary,
    enable_attn_pruning=True,
    enable_low_rank=True,
    save_path="./compressed_model",
)
print(f"PPL: {result['original_ppl']:.2f} -> {result['compressed_ppl']:.2f}")

# Or run everything in one call
result = mri_compressor.run_full_pipeline("gpt2", output_dir="./results")

# Pass a pre-loaded model (e.g. already fine-tuned)
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

---

## The Studies

### Core Studies (1–11)

These studies run on generic calibration data (WikiText-103 by default) and characterize the model's internal structure.

---

#### Study 1 — Activation Profiling

**What it measures:** Per-layer MLP activation distribution statistics: mean, std, kurtosis, Gini coefficient, natural sparsity (`pct_near_zero`), top-1 ratio (max/mean), and outlier fractions.

**Why it matters:** The entire MLP compression strategy depends on whether activations cluster near zero. SwiGLU/SiLU models (Qwen, Llama, Mistral) produce ~70–90% near-zero activations naturally because the gate pushes most neurons to zero. Standard GELU models (GPT-2, BERT) have no such gate — near-zero sparsity is near 0%, making MLP compression much harder. Study 1 confirms which regime the model is in before any compression is attempted.

High kurtosis (heavy tails) signals outlier neurons — a small fraction of neurons dominate the total activation magnitude. These are usually aligned with the massive activations detected in Study 4 and the critical neurons from Study 9.

**Feeds into:** Everything. Baseline characterization for all other studies.

**Related work:** CATS (2024), Massive Activations (Sun et al., 2024 COLM), Understanding Outlier Features (He et al., NeurIPS 2024)

---

#### Study 3 — Wanda Importance Scores

**What it measures:** Per-neuron importance via `|weight_j| × ||activation_j||₂` — the product of a neuron's output weight norm and its mean input activation norm across calibration data. Zero-cost, no training required.

**Why it matters:** Wanda is the primary signal for deciding which neurons are safe to remove. A neuron with a small weight (little contribution to the residual stream) AND low activation (rarely triggered strongly) is doubly unimportant. The key insight over pure magnitude pruning is that activation scale matters: a neuron with a large weight but tiny activations contributes negligibly.

The scores are stored as tensors per layer and reused by the diagnostician during compression candidate selection. Study 9 uses Wanda to shortlist candidates for its more expensive PPL-based criticality test.

**Feeds into:** Study 9 (candidate shortlisting), diagnostician (pruning candidate ranking), Study 22 (domain-conditional version of the same metric).

**Related work:** Wanda (Sun et al., ICLR 2024), SparseGPT (Frantar & Alistarh, NeurIPS 2023)

---

#### Study 4 — Massive Activation Scan

**What it measures:** Neurons with disproportionately large, input-agnostic activations — high mean activation AND very low variance across inputs (the activation is nearly constant regardless of what token is being processed).

**Why it matters:** These neurons function as implicit bias terms baked into the MLP weights. They write a near-constant vector into the residual stream on every token, effectively shifting the model's output distribution. Removing them destroys generation coherence immediately and catastrophically. They must be added to the permanent protection list before any compression.

Typically affects 1–5 neurons per layer in models with massive activation phenomena. Layer 14 in Qwen3.5-0.8B shows 99.9% near-zero activations with 68,000+ kurtosis — characteristic of one or two massive neurons dominating the layer.

**Feeds into:** Global protection list — these neurons are never touched by any operation.

**Related work:** Massive Activations (Sun et al., 2024 COLM), Super Weights (Yu et al., 2024), DuQuant (NeurIPS 2024)

---

#### Study 5 — Dead / Dormant Neuron Census

**What it measures:** Per-neuron firing rate across calibration data. Classifies each neuron as:
- **Dead**: never fires (0% of tokens)
- **Dormant**: fires on <1% of tokens
- **Rare**: fires on <10% of tokens
- **Normal**: 10–99% firing rate
- **Hyperactive**: fires on >99% of tokens

**Why it matters:** Dead and dormant neurons are essentially free to remove — they contribute nothing (or negligible amounts) to any forward pass in the calibration distribution. This is the zero-cost, zero-risk first pass of compression. In gated MLP models, many neurons in deeper layers are effectively suppressed by the gate for nearly all inputs.

Hyperactive neurons (>99% firing rate) are flagged and tracked: they may be candidates for static folding (Study 20) since their activation is nearly constant.

**Feeds into:** Dead neuron removal and dormant neuron removal operations. Hyperactive neurons feed into Study 20.

**Related work:** "Neurons in LLMs: Dead, N-gram, Positional" (ACL 2024 Findings)

---

#### Study 6 — Attention Head Importance

**What it measures:** Per-head diagnostics across all attention layers:
- **Entropy**: High entropy = head attends broadly (potentially redundant); low entropy = head attends sharply to specific positions (specialized)
- **First-token attention fraction**: Measures "sink" behavior — heads that route most attention to the first token regardless of content
- **Max concentration**: Highest single-token attention weight

For hybrid models (e.g., Qwen3 with interleaved linear and standard attention), linear attention layers receive a separate channel-group magnitude analysis.

**Why it matters:** Heads with high entropy and low first-token attention fraction are the safest pruning targets — they aren't doing specialized pattern matching. However, attention head pruning is inherently riskier than neuron removal because heads interact: removing one head changes how other heads must distribute the load. The diagnostician applies a strict PPL-delta gate (< 0.30 PPL increase when ablated) before recommending any head for pruning, and caps at 2 heads per layer.

**Feeds into:** Attention head pruning decisions. Attention sink heads (high first-token fraction) are protected.

**Related work:** Voita et al. (ACL 2019), Attention Sinks (Xiao et al., 2023), From Attention to Activation (Kaul et al., 2024)

---

#### Study 8 — Sparsity Structure Analysis

**What it measures:** Whether the natural MLP sparsity (identified in Study 1) is *static* or *dynamic*:
- **Token-position sparsity variance**: Does sparsity change with position in the sequence?
- **Neuron consistency**: Do the same neurons fire consistently across different inputs?
- **Specialization entropy**: How evenly are neurons distributed across firing rate bins?
- **Co-activation degree**: Do neurons tend to fire together in correlated clusters?

**Why it matters:** Dynamic, context-sensitive sparsity (where different neurons fire for different inputs) is what methods like CATS and DejaVu target — they build runtime predictors to identify which neurons to skip. If sparsity is mostly static (the same neurons are near-zero across all inputs), then much simpler static structured pruning suffices and dynamic methods add runtime overhead with no benefit. Study 8 determines which regime the model is in.

In practice, SwiGLU models show high consistency in which neurons are near-zero — most of the measured sparsity is structural (some neurons are nearly always suppressed by the gate), not contextual. This justifies the static pruning approach used here.

**Feeds into:** Strategy selection in the diagnostician (static vs. dynamic mask approach).

**Related work:** CATS (2024) key contextuality claim; DejaVu (Liu et al., 2023)

---

#### Study 9 — Critical Neuron Search

**What it measures:** The "super weight" effect — single neurons whose individual removal causes catastrophic perplexity spikes. For each layer, Study 3 Wanda scores are used to shortlist the top-5 candidates, and each is zeroed individually while measuring PPL impact.

**Why it matters:** The existence of super weights (neurons where removal alone causes PPL to spike by +3 or more) shows that importance is not smoothly distributed — a tiny fraction of neurons are disproportionately load-bearing. These must be added to the protection list even if Wanda scores would suggest they could be removed.

Example from Qwen3.5-0.8B: Layer 6, Neuron 241 causes +3.57 PPL increase when removed alone; Layer 2 Neuron 1 causes +0.72. These are added to the permanent protection list and never touched.

**Feeds into:** Global protection list (merged with Study 4 massive activations).

**Related work:** Super Weights (Yu et al., 2024), "The Achilles' Heel of LLMs" (2025)

---

#### Study 10 — Layer Redundancy Analysis

**What it measures:** Each layer's contribution to model performance, measured by zeroing its MLP output, its attention output, or both and measuring the perplexity impact (`mlp_ppl_delta`, `attn_ppl_delta`).

**Why it matters:** Reveals the depth structure of the model. Typically shows a U-shape: early layers (tokenization, syntax) and late layers (output projection) are critical with high PPL deltas; middle layers are often partially redundant with lower deltas. This informs depth pruning (which layers could be removed entirely) and helps prioritize which layers to be most conservative with during neuron-level compression.

In Qwen3.5-0.8B: Layer 23 has MLP delta +17.93 (critical, never compress aggressively), Layer 6 has +5.46 (important), while Layers 10–12 have deltas of +0.39–0.66 (most compressible).

**Feeds into:** Depth pruning decisions (only layers with combined delta < 0.15 are candidates). Also used to weight PPL budget allocation across layers.

**Related work:** ShortGPT (Men et al., 2024), Layer-Selective Rank Reduction (Sharma et al., 2024)

---

#### Study 11 — Domain-Specific Activation Divergence

**What it measures:** Across four domains (English prose, math, code, Italian), computes the per-layer Jaccard similarity between the sets of active neurons. Low Jaccard = layers where different domains use completely different neurons (high domain specificity). High Jaccard = layers where most neurons fire for all domains (domain-agnostic).

**Outputs per layer:**
- `jaccard`: Fraction of neurons shared across all domain neuron sets
- `specificity`: Fraction of neurons that are domain-specific (1 - jaccard)
- `universal_neurons`: Count of neurons active across all domains
- Per-domain unique neuron counts

**Why it matters:** Determines whether domain-specialized compression is viable and where in the network it will be most effective. Early layers tend to be domain-agnostic (processing syntax shared across all text). Some middle layers show the highest specificity — where the model routes domain-specific patterns. Late layers often partially converge again.

Layers with low Jaccard (high specificity) are the best targets for domain compression: you can remove neurons that are only used for non-target domains without affecting target-domain performance.

**Feeds into:** Study 22 and 24 (which build on the domain divergence insight). Also informs which layers to be most conservative with in domain compression (highly specific layers = more risky).

**Related work:** "Sharing Matters: Analysing Neurons Across Languages and Tasks" (ACL 2024), "Language-Specific Neurons" (Tang et al., 2024)

---

### Domain Studies (22–25)

These studies require a `--domain` target and analyze the model's behavior on domain-specific data. They are the core of the `domain_compress` pipeline.

---

#### Study 22 — Domain-Conditional Wanda Importance

**What it measures:** The same Wanda metric from Study 3 (`|weight_j| × ||activation_j||₂`), but computed separately for each domain's data. This produces a per-domain importance score for every neuron in every layer.

**Key concept — domain necessity:** A neuron that is unimportant in the target domain AND more important in the generic distribution is a strong removal candidate: it is being "used" by other domains but not the target domain. A neuron that has *lower* importance on the target domain than on average generic text is "domain-unnecessary."

**Outputs:**
- Per-layer domain-conditional Wanda score tensors (saved to disk, reused in compression)
- `domain_critical` count: top 10% most important neurons per domain
- `domain_unnecessary` count: bottom 20% per domain AND below global mean Wanda
- Cross-domain cosine similarity between importance vectors (measures how differently the model weights neurons for each domain pair)

**Feeds into:** Study 24 (uses these scores to test per-layer sparsity budgets), Study 25 (used as primary removal ranking, re-ranked by geometry), diagnostician (protection/removal lists).

---

#### Study 24 — Domain Compression Curve

**What it measures:** For each layer independently, tests how much sparsity can be applied using the domain-Wanda ranking (from Study 22) before domain-specific perplexity crosses a threshold (+15% relative by default).

**Method:**
1. Rank neurons by domain-Wanda score (ascending = least important first)
2. For each sparsity level in `[10%, 20%, 30%, 40%, 50%, 55%, 60%, 65%, 70%]`, zero the bottom-K neurons
3. Measure domain PPL with those neurons zeroed (all other layers intact)
4. Record the highest sparsity level where PPL increase stays below the threshold
5. This is `safe_domain_sparsity` for that layer

**Why it matters:** Replaces the hardcoded 5% cap with per-layer evidence-based budgets. In practice for Qwen3.5-0.8B on biomedical, 23/24 layers safely tolerate 70% sparsity and one tolerates 60% — far beyond the fixed 5% cap. This is possible because the domain-specific ranking concentrates the "actually used" neurons into the top fraction: zeroing the bottom 70% by domain-Wanda removes neurons that contribute almost nothing to domain text.

**Important caveat:** Single-layer budgets are tested independently. Compound error when all layers are pruned simultaneously is significant (the global sweep shows that pruning 10% globally causes +6.7% domain PPL — much larger than single-layer estimates). The per-layer budget is used as the candidate pool limit in the diagnostician, not as a guarantee of compound safety.

**Outputs:**
- `safe_domain_sparsity` per layer (float, used by diagnostician as removal budget)
- Full PPL curve at each tested sparsity level
- Global cumulative sweep results (all layers simultaneously)
- `study24_domain_compression_curve.png`

**Feeds into:** Diagnostician — `max_removable = int(n_neurons * safe_domain_sparsity)` per layer.

---

#### Study 25 — MLP Write-Vector Geometry

**What it measures:** The geometric properties of what each MLP neuron actually *writes* to the residual stream. While Wanda (Study 3, 22) measures the *input* side (activation × weight), Study 25 measures the *output* side: `W_down[:, j]` — the column of the down-projection that determines which direction in residual space neuron `j` writes to.

**Three key metrics per layer:**

**1. Direction coherence** (`direction_coherence`): The alignment of write vectors across all neurons, measured as the mean pairwise cosine similarity. High coherence (→1.0) means all neurons write in a similar direction — the MLP output is dominated by a single subspace. Low coherence (→0.0) means neurons write in diverse directions. High coherence makes geometry more reliable as a pruning signal because the dominant direction is interpretable.

**2. Intrinsic dimensionality** (`intrinsic_dim_95`): The number of PCA components needed to explain 95% of variance in the write-vector matrix. A layer with intrinsic_dim=34 (out of 3584 neurons) uses only a 34-dimensional effective subspace in the residual stream — the remaining 3550 dimensions are a basis transformation of the same 34 directions. This is an upper bound on how many truly distinct things the MLP can write.

**3. Geometric importance concentration** (`geom_importance_concentration`): The fraction of neurons that collectively hold 80% of total geometric importance, where per-neuron geometric importance is `E[|a_j(x)|] × ‖W_down[:, j]‖₂` — expected activation magnitude times write-vector norm. A concentration of 0.70 means 70% of neurons contribute 80% of the total residual-stream write; the remaining 30% contribute 20%. Lower concentration = importance is more concentrated in fewer neurons = the layer has a more sparse-but-impactful structure.

**Domain divergence angle:** A second pass is run on the target domain's data to compute the angle between each layer's dominant write direction during domain inference vs. generic inference. Low angle (near 0°) = the MLP writes in the same direction regardless of domain (domain-agnostic layer). High angle (near 180°) = the domain causes the MLP to write in a nearly opposite direction (strongly domain-specific routing).

**How it affects compression:**

Study 25 does not set the removal *budget* (that is Study 24's job). It influences *which* neurons within the budget are removed first:

1. **Candidate pool sizing**: The pool of neurons considered for removal is `max(safe_domain_sparsity × 1.5, 0.25)`, model-relatively expanded: layers where `geom_importance_concentration` is below the model's own average get a slightly larger pool (more concentrated = fewer neurons dominate = the tail is safer to explore).

2. **Combined ranking**: Within the pool, neurons are ranked by `wanda_rank[j] + geom_weight × geom_rank[j]` where lower = remove first. This promotes neurons that are doubly unimportant (low domain-Wanda AND low geometric importance) and de-prioritizes neurons with low Wanda but non-trivial geometric contribution.

3. **Geometry trust weight**: `geom_weight = (0.5 + coherence) × (1.0 + (1.0 - div_factor) × 0.5)` where `div_factor = min(div_angle / 90°, 1.0)`. When coherence is high (geometry is a reliable signal) and divergence angle is low (domain-Wanda is less reliable because the layer barely distinguishes domains), geometry gets a higher vote in the combined ranking.

**Outputs:**
- Per-layer: `direction_coherence`, `intrinsic_dim_95`, `geom_importance_concentration`
- Per-layer per-domain: divergence angle in degrees
- Per-neuron geometric importance tensors (saved to disk, loaded during compression)
- `study25_write_vector_geometry.png`

---

### Extended Studies (12–21)

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

---

## Compression Operations

| Operation | Informed by | What it does |
|-----------|-------------|--------------|
| Dead neuron removal | Study 5 | Removes neurons that never fire — zero PPL cost |
| Dormant neuron removal | Study 5 | Removes neurons firing on <1% of tokens |
| Wanda pruning | Study 3 | Structured pruning via `\|weight\| × ‖activation‖` |
| Neuron merging | Study 14 | Merges neurons with high cosine similarity into one |
| Low-rank factorization | Study 18 | SVD: replaces `Linear(in, out)` with two smaller matrices |
| Static folding | Study 20 | Folds near-constant neurons into bias terms |
| Attention head pruning | Study 6 | Zeros low-importance attention heads (max 2/layer) |
| Depth pruning | Study 10 | Removes entire redundant layers (disabled by default) |
| Domain specialization | Studies 22+24+25 | Removes neurons unnecessary for the target domain |
| Local reconstruction | Post-compression | SparseGPT-style weight update to recover accuracy |

**Safety guards applied to all operations:**
- Neurons flagged by Study 4 (massive activations) or Study 9 (critical neurons) are **never removed**
- Configurable PPL budget (default: 2.0 PPL increase total)
- Depth pruning: off by default; max 1 layer; only if combined MLP+attn delta < 0.15
- Attention pruning: max 2 heads/layer; only if ablated PPL delta < 0.30
- No attention pruning within ±2 layers of a depth-pruned layer

---

## Model Support

| Model | Architecture | Activation | Notes |
|-------|-------------|-----------|-------|
| **GPT-2 (124M)** | Standard MLP (fc1→GELU→fc2) | GELU | Baseline. No natural sparsity. |
| **Qwen2.5-0.5B / 1.5B / 7B** | Gated MLP (gate×up→down) | SwiGLU | ~50–90% natural sparsity. Recommended. |
| **Qwen3.5-0.8B / 2B** | Hybrid: interleaved linear + standard attention | SwiGLU | Linear attention layers handled by Study 6 extension |
| **TinyLlama-1.1B** | Gated MLP | SwiGLU | Same architecture as Qwen2.5 |
| **Llama / Mistral family** | Gated MLP | SwiGLU | Should work; not extensively tested |

The key architectural split: gated MLPs (Qwen, Llama, Mistral) have a multiplicative gate that naturally produces ~50–90% near-zero activations via SiLU. Standard GELU MLPs (GPT-2) have no gate — near-zero sparsity is near 0%, making MLP compression much harder. The diagnostics work on both; compression yields vary significantly.

---

## Output Structure

```
results/
├── summary.json                          # Full MRI findings — reusable without re-scanning
├── summary_tensors/                      # Per-layer Wanda score tensors and domain tensors
│   ├── layer_00_wanda.pt
│   ├── layer_00_domain_biomedical.pt
│   └── ...
├── evaluation_report.json                # Stage 4 results (PPL before/after, param counts)
├── compressed_model/                     # Saved compressed model (if --save-model)
├── study1_activation_profiles.png        # 6-panel activation distribution analysis
├── study3_wanda_scores.png               # Wanda importance distributions per layer
├── study4_massive_activations.png        # Massive activation locations
├── study5_dead_neurons.png               # Dead/dormant neuron census
├── study6_attention_heads.png            # Attention head entropy and specialization
├── study10_layer_redundancy.png          # Layer-by-layer PPL contribution
├── study11_domain_divergence.png         # 7-panel domain specialization analysis
├── study24_domain_compression_curve.png  # Per-layer safe sparsity curves
└── study25_write_vector_geometry.png     # Write-vector geometry heatmap
```

`summary.json` is the key artifact: it contains per-layer results for every study that was run and feeds directly into the diagnosis and compression stages. Pass it to `--from-summary` to skip re-scanning.

---

## Package Structure

```
mri_compressor/
├── __init__.py                # Public API: run_mri(), compress(), run_full_pipeline()
├── pipeline.py                # CLI entry point and stage orchestration
├── config.py                  # ExperimentConfig dataclass
├── model_utils.py             # ModelInspector (layer detection, architecture handling)
├── data_utils.py              # WikiText loading, perplexity evaluation, domain datasets
├── mri/
│   ├── runner.py              # MRIRunner: orchestrates studies
│   ├── studies_activation.py  # Studies 1, 4
│   ├── studies_gates.py       # Studies 2, 7
│   ├── studies_importance.py  # Studies 3, 9
│   ├── studies_neuron_health.py  # Study 5
│   ├── studies_attention.py   # Study 6
│   ├── studies_structure.py   # Study 8
│   ├── studies_layer.py       # Study 10
│   ├── studies_domain.py      # Study 11
│   ├── studies_cross.py       # Studies 12, 13
│   ├── studies_advanced.py    # Studies 14, 15, 16
│   ├── studies_nextgen.py     # Studies 17–21
│   ├── studies_domain_importance.py  # Study 22
│   ├── studies_domain_compression.py # Study 24
│   ├── studies_geometry.py    # Study 25
│   ├── summary.py             # Builds summary.json from study results
│   └── visualize.py           # Plot generation
└── compression/
    ├── compressor.py          # MRICompressor: applies prescription to model
    ├── diagnostician.py       # MRIDiagnostician: MRI findings → prescription
    ├── prescription.py        # CompressionPrescription, LayerPrescription dataclasses
    ├── evaluate.py            # Perplexity evaluation utilities
    ├── neuron_recycling.py    # Neuron merge/recombination logic
    ├── _utils.py              # MLP/attention module accessors
    └── operations/
        ├── dead_removal.py    # DeadNeuronRemover
        ├── neuron_merge.py    # NeuronMerger
        ├── wanda_pruner.py    # WandaPruner
        ├── attention_pruner.py  # AttentionHeadPruner
        ├── depth_pruner.py    # DepthPruner
        ├── low_rank.py        # LowRankFactorizer
        ├── static_fold.py     # StaticNeuronFolder
        ├── weight_sharing.py  # WeightSharer
        └── reconstructor.py   # LocalReconstructor (SparseGPT-style)
```

---

## Key Hypotheses

1. **GELU vs SiLU natural sparsity**: SiLU models show ~50–90% near-zero activations; GELU ~0%. (Study 1)

2. **Massive activations are gate-protected**: Learned gates keep massive activation neurons fully open. (Studies 2 + 4)

3. **Dead neurons are gate-pruned**: Learned gates assign near-zero values to dead/dormant neurons. (Studies 2 + 5)

4. **Wanda partially explains gates**: Moderate but not perfect correlation; gradients find contextual importance that heuristics miss. (Study 7)

5. **Critical neurons are outlier neurons**: The neurons most damaging to remove overlap with massive activations and high Wanda scores. (Studies 3, 4, 9)

6. **Middle layers are most compressible**: Layer redundancy shows a U-shape — early and late layers critical, middle layers partially redundant. (Study 10)

7. **Domain divergence enables domain specialization**: Neurons with low domain-conditional importance but normal generic importance are true domain-unnecessary neurons — removing them shrinks the model without harming target-domain performance. (Studies 11, 22, 24)

8. **Geometry refines Wanda ranking**: A neuron can have low Wanda importance (input side) but still write a non-trivial vector into the residual stream (output side). Combining both signals selects better removal candidates than either alone. (Study 25)

9. **Write-vector geometry adapts to domain routing**: When the dominant MLP write direction shifts significantly between generic and domain inference (high divergence angle), domain-Wanda is a reliable separation signal. When the angle is low (the MLP writes the same direction regardless of domain), geometry is a better guide than domain-Wanda alone. (Study 25)

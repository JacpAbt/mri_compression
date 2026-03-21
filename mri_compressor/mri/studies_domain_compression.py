"""
Study 24: Domain Compression Curve
====================================
For a specific target domain (e.g. biomedical/PubMed), iteratively ablate
neurons per layer to find per-layer maximum safe pruning limits.

Key insight: The fixed 5% cap used by the Study 22 diagnostician overlay is
too conservative for many layers and potentially too aggressive for critical
ones. This study measures actual domain-PPL impact per layer at multiple
sparsity levels, providing evidence-based per-layer budgets.

Method:
  For each layer:
    1. Compute (or retrieve from Study 22) domain Wanda scores per neuron
    2. For sparsity levels [10%, 20%, 30%, 40%, 50%]:
       a. Zero the bottom-K neurons in down_proj (by domain Wanda score)
       b. Measure domain-specific PPL on a small eval set (8 batches)
       c. Restore the original weights
    3. Find "safe" level: highest consecutive sparsity where domain PPL
       increase stays below ppl_threshold_relative (default 15%)

Output:
  - Per-layer DomainCompressionCurveReport with safe_domain_sparsity
  - Wired into the diagnostician's Study 22 overlay to replace the
    hardcoded 5% domain_unnecessary_removal_frac cap with per-layer values

PubMed data loading:
  Attempts (in order):
    1. qiaojin/PubMedQA  (pqa_labeled, 1 000 QA pairs, field: "context")
    2. allenai/pubmed_abstracts  (streaming, millions of abstracts)
    3. Synthetic biomedical fallback (high-quality hand-written text)
"""

from __future__ import annotations

import gc
import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


# ===========================================================================
# Data class
# ===========================================================================

@dataclass
class DomainCompressionCurveReport:
    """Per-layer result of the domain compression curve scan."""
    layer_idx: int
    n_neurons: int                      # MLP intermediate size for this layer
    domain_baseline_ppl: float          # Model PPL on domain data before any pruning
    sparsity_levels: List[float]        # e.g. [0.10, 0.20, 0.30, 0.40, 0.50]
    domain_ppl_curve: List[float]       # PPL at each sparsity level (parallel to sparsity_levels)
    safe_domain_sparsity: float         # Highest consecutive level where ppl_delta < threshold
    elbow_ppl_delta: float              # PPL delta (absolute) at the safe level


@dataclass
class GlobalCompressionCurveReport:
    """Result of the global (all-layer simultaneous) compression curve sweep."""
    domain_baseline_ppl: float          # Same baseline as per-layer sweep
    sparsity_levels: List[float]        # e.g. [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
    global_ppl_curve: List[float]       # PPL with ALL layers pruned at each level
    safe_global_sparsity: float         # Highest level that stays within PPL budget
    elbow_ppl_delta: float              # PPL delta (absolute) at the safe global level


# ===========================================================================
# Utility: temporary neuron zeroing
# ===========================================================================

@torch.no_grad()
def _zero_neurons_in_down_proj(mlp_layer, neuron_indices: List[int]) -> torch.Tensor:
    """
    Zero the output contribution of specified MLP neurons in down_proj.
    Returns the original weight data (backup) for later restoration.

    Handles both weight orientations:
      - Standard layout: down_proj.weight shape = (hidden, intermediate)
        → columns are per-neuron output vectors → zero columns
      - Conv1D layout:   down_proj.weight shape = (intermediate, hidden)
        → rows are per-neuron output vectors → zero rows
    """
    down_proj = mlp_layer.down_proj
    W = down_proj.weight.data

    # Use intermediate_size (not a shape comparison) to detect layout unambiguously.
    # Conv1D (GPT-2 c_proj): weight is (intermediate, hidden) — W.shape[0] == intermediate
    # nn.Linear (standard):  weight is (hidden, intermediate) — W.shape[0] == hidden
    if W.shape[0] == mlp_layer.intermediate_size:
        # Conv1D layout: weight is (intermediate, hidden) — neurons are rows
        backup = W[neuron_indices, :].clone()
        W[neuron_indices, :] = 0.0
    else:
        # Standard nn.Linear layout: weight is (hidden, intermediate) — neurons are columns
        backup = W[:, neuron_indices].clone()
        W[:, neuron_indices] = 0.0

    return backup


@torch.no_grad()
def _restore_neurons_in_down_proj(
    mlp_layer,
    neuron_indices: List[int],
    backup: torch.Tensor,
) -> None:
    """Restore previously zeroed neurons using a saved backup."""
    down_proj = mlp_layer.down_proj
    W = down_proj.weight.data

    if W.shape[0] == mlp_layer.intermediate_size:
        W[neuron_indices, :] = backup
    else:
        W[:, neuron_indices] = backup


# ===========================================================================
# Utility: fast domain PPL evaluation
# ===========================================================================

@torch.no_grad()
def _quick_eval_ppl(
    model,
    dataset,
    device: torch.device,
    batch_size: int = 4,
    max_batches: int = 8,
) -> float:
    """
    Quickly evaluate perplexity on a TextDataset.
    Uses at most max_batches batches for speed.
    """
    from ..data_utils import get_dataloader

    dl = get_dataloader(dataset, batch_size=batch_size)
    model.eval()
    total_loss, total_tok = 0.0, 0

    for i, batch in enumerate(dl):
        if i >= max_batches:
            break
        ids = batch["input_ids"].to(device)
        out = model(input_ids=ids, labels=ids)
        seq_len = ids.shape[1] - 1
        total_loss += out.loss.item() * seq_len * ids.shape[0]
        total_tok += seq_len * ids.shape[0]

    return math.exp(total_loss / max(total_tok, 1))


# ===========================================================================
# Utility: get domain Wanda scores
# ===========================================================================

def _get_domain_wanda_scores(
    inspector,
    domain_dataset,
    domain_name: str,
    batch_size: int,
    max_batches: int,
    prior_results: Optional[Dict],
) -> Dict[int, torch.Tensor]:
    """
    Get per-layer domain Wanda scores.

    Prefers reusing Study 22 results from prior_results for speed.
    Falls back to computing fresh scores using the same streaming logic
    as Study 22, restricted to the target domain only.

    Returns:
        dict mapping layer_idx (int) -> tensor of shape (intermediate_size,)
    """
    # Try to reuse Study 22 results
    if prior_results is not None:
        dci = prior_results.get("domain_conditional_importance", {})
        dw = dci.get("domain_wanda_scores", {})
        if domain_name in dw and dw[domain_name]:
            available = len(dw[domain_name])
            if available >= inspector.num_layers * 0.8:
                print(f"  Reusing Study 22 Wanda scores for '{domain_name}' "
                      f"({available} layers cached)")
                return dw[domain_name]

    # Compute fresh
    print(f"  Computing domain Wanda scores for '{domain_name}' "
          f"({inspector.num_layers} layers)...")
    from .studies_domain_importance import compute_domain_wanda_scores_streaming
    domain_wanda = compute_domain_wanda_scores_streaming(
        inspector,
        {domain_name: domain_dataset},
        batch_size=batch_size,
        max_batches=max_batches,
    )
    return domain_wanda.get(domain_name, {})


# ===========================================================================
# Main study entry point
# ===========================================================================

def run_domain_compression_curve(
    inspector,
    domain_dataset,
    domain_name: str = "biomedical",
    batch_size: int = 4,
    max_batches_wanda: int = 16,
    max_batches_eval: int = 8,
    sparsity_levels: Optional[List[float]] = None,
    ppl_threshold_relative: float = 0.15,
    prior_results: Optional[Dict] = None,
) -> Dict:
    """
    Study 24: Domain Compression Curve.

    For each transformer layer, scans multiple sparsity levels to find the
    maximum safe fraction of domain-unimportant neurons that can be removed
    while keeping the target-domain PPL increase below ppl_threshold_relative.

    Args:
        inspector:              ModelInspector with loaded model.
        domain_dataset:         Pre-tokenized TextDataset for the target domain.
        domain_name:            Name of the target domain (for logging/matching).
        batch_size:             Batch size for Wanda score computation and eval.
        max_batches_wanda:      Max batches for computing Wanda scores (if not reused).
        max_batches_eval:       Max batches for each quick PPL evaluation.
        sparsity_levels:        Sparsity fractions to test (default: 10%–50%).
        ppl_threshold_relative: Max allowed fractional PPL increase (default: 15%).
        prior_results:          Optional runner results dict; if Study 22 was run
                                with this domain, its Wanda scores are reused.

    Returns:
        dict with keys:
            "reports":          List[DomainCompressionCurveReport], one per layer.
            "baseline_ppl":     Domain PPL before any pruning.
            "domain_name":      The domain name.
            "sparsity_levels":  The levels tested.
            "avg_safe_sparsity": Mean safe sparsity across layers.
    """
    if sparsity_levels is None:
        sparsity_levels = [0.10, 0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70]

    print("\n" + "=" * 80)
    print("STUDY 24: Domain Compression Curve")
    print(f"  Target domain:   {domain_name}")
    print(f"  Sparsity levels: {[f'{s:.0%}' for s in sparsity_levels]}")
    print(f"  PPL threshold:   +{ppl_threshold_relative * 100:.0f}% relative")
    print("=" * 80)

    # ---- Step 1: Domain Wanda scores ----
    wanda_by_layer = _get_domain_wanda_scores(
        inspector, domain_dataset, domain_name,
        batch_size, max_batches_wanda, prior_results,
    )

    # ---- Step 2: Baseline domain PPL ----
    print("\n  Measuring baseline domain PPL...")
    baseline_ppl = _quick_eval_ppl(
        inspector.model, domain_dataset, inspector.device,
        batch_size=batch_size, max_batches=max_batches_eval,
    )
    print(f"  Baseline domain PPL: {baseline_ppl:.3f}")

    # ---- Step 3: Per-layer compression curve ----
    reports: List[DomainCompressionCurveReport] = []

    # Header row
    print(f"\n  {'Layer':>5} | {'Base':>8} | ", end="")
    for sp in sparsity_levels:
        print(f"{'@'+f'{sp:.0%}':>8} | ", end="")
    print(f"{'Safe':>7} | {'Δ PPL':>7}")
    sep = "  " + "-" * (6 + 11 + 11 * len(sparsity_levels) + 9 + 9)
    print(sep)

    for layer_idx in range(inspector.num_layers):
        mlp_layer = inspector.mlp_layers[layer_idx]

        # Skip layers without a standard down_proj (e.g. pure-attention layers)
        if mlp_layer is None or not hasattr(mlp_layer, "down_proj"):
            continue

        wanda_scores = wanda_by_layer.get(layer_idx)
        if wanda_scores is None:
            continue

        n_neurons = wanda_scores.shape[0]
        # Sort indices by ascending importance (least important first)
        sorted_indices = wanda_scores.argsort()

        ppls: List[float] = []
        for sparsity in sparsity_levels:
            k = int(n_neurons * sparsity)
            if k == 0:
                ppls.append(baseline_ppl)
                continue

            # Indices of the k least domain-important neurons
            neuron_indices = sorted_indices[:k].tolist()

            # Temporarily zero their down_proj contribution
            backup = _zero_neurons_in_down_proj(mlp_layer, neuron_indices)

            # Measure domain PPL with these neurons zeroed
            ppl_at_level = _quick_eval_ppl(
                inspector.model, domain_dataset, inspector.device,
                batch_size=batch_size, max_batches=max_batches_eval,
            )

            # Restore
            _restore_neurons_in_down_proj(mlp_layer, neuron_indices, backup)

            ppls.append(ppl_at_level)

        # Find safe sparsity: highest consecutive level below threshold
        safe_sparsity = 0.0
        elbow_ppl_delta = 0.0
        for sp, ppl_val in zip(sparsity_levels, ppls):
            rel_delta = (ppl_val / baseline_ppl - 1.0) if baseline_ppl > 0 else 0.0
            if rel_delta < ppl_threshold_relative:
                safe_sparsity = sp
                elbow_ppl_delta = ppl_val - baseline_ppl
            else:
                # Once threshold is exceeded, higher sparsities will be worse
                break

        report = DomainCompressionCurveReport(
            layer_idx=layer_idx,
            n_neurons=n_neurons,
            domain_baseline_ppl=baseline_ppl,
            sparsity_levels=list(sparsity_levels),
            domain_ppl_curve=ppls,
            safe_domain_sparsity=safe_sparsity,
            elbow_ppl_delta=elbow_ppl_delta,
        )
        reports.append(report)

        # Print row
        ppl_strs = " | ".join(f"{p:>8.3f}" for p in ppls)
        print(
            f"  {layer_idx:>5} | {baseline_ppl:>8.3f} | {ppl_strs} | "
            f"{safe_sparsity:>7.1%} | {elbow_ppl_delta:>+7.3f}"
        )

        del sorted_indices
        gc.collect()

    # ---- Summary ----
    if reports:
        avg_safe = sum(r.safe_domain_sparsity for r in reports) / len(reports)
        max_safe = max(r.safe_domain_sparsity for r in reports)
        min_safe = min(r.safe_domain_sparsity for r in reports)
        n_zero = sum(1 for r in reports if r.safe_domain_sparsity == 0.0)
    else:
        avg_safe = max_safe = min_safe = 0.0
        n_zero = 0

    print(f"\n  Safe sparsity summary (domain: {domain_name}):")
    print(f"    avg = {avg_safe:.1%}, min = {min_safe:.1%}, max = {max_safe:.1%}")
    print(f"    Layers with no safe pruning: {n_zero}/{len(reports)}")
    print(f"    Previous fixed cap was 5% — "
          f"avg improvement: {max(0.0, avg_safe - 0.05):.1%} additional headroom")

    # ---- Global cumulative sweep ----
    global_sparsity_levels = [0.10, 0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70]
    global_curve = _run_global_compression_curve(
        inspector=inspector,
        domain_dataset=domain_dataset,
        domain_name=domain_name,
        batch_size=batch_size,
        max_batches_eval=max_batches_eval,
        wanda_by_layer=wanda_by_layer,
        baseline_ppl=baseline_ppl,
        per_layer_avg_safe=avg_safe,
        sparsity_levels=global_sparsity_levels,
        ppl_threshold_relative=ppl_threshold_relative,
    )

    return {
        "reports": reports,
        "baseline_ppl": baseline_ppl,
        "domain_name": domain_name,
        "sparsity_levels": list(sparsity_levels),
        "avg_safe_sparsity": avg_safe,
        "global_curve": global_curve,
    }


# ===========================================================================
# Study 24b: Global cumulative compression sweep
# ===========================================================================

def _run_global_compression_curve(
    inspector,
    domain_dataset,
    domain_name: str,
    batch_size: int,
    max_batches_eval: int,
    wanda_by_layer: Dict,
    baseline_ppl: float,
    per_layer_avg_safe: float,
    sparsity_levels: Optional[List[float]] = None,
    ppl_threshold_relative: float = 0.15,
) -> "GlobalCompressionCurveReport":
    """
    Global cumulative compression sweep for Study 24.

    Unlike the per-layer scan (which ablates one layer at a time), this zeros
    ALL MLP layers simultaneously at the same sparsity level, then measures
    a single PPL. This reveals the compound/cascading error that accumulates
    as the residual stream degrades through successive pruned layers.

    Reuses the Wanda scores and baseline PPL already computed by the per-layer
    sweep, so no additional forward passes for calibration are needed.

    Args:
        inspector:              ModelInspector with loaded model.
        domain_dataset:         Pre-tokenized TextDataset for the target domain.
        domain_name:            Domain name (for logging).
        batch_size:             Batch size for PPL evaluation.
        max_batches_eval:       Max batches per PPL evaluation.
        wanda_by_layer:         Dict[layer_idx → Tensor] of domain Wanda scores.
        baseline_ppl:           Domain PPL with no pruning applied.
        per_layer_avg_safe:     Average safe sparsity from per-layer scan (for report).
        sparsity_levels:        Sparsity fractions to test (default: 10%–60%).
        ppl_threshold_relative: Max fractional PPL increase allowed (default: 15%).

    Returns:
        GlobalCompressionCurveReport
    """
    if sparsity_levels is None:
        sparsity_levels = [0.10, 0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70]

    # Pre-compute sorted indices per layer (least-important first)
    sorted_by_layer: Dict[int, torch.Tensor] = {}
    for layer_idx, scores in wanda_by_layer.items():
        sorted_by_layer[layer_idx] = scores.argsort()

    global_ppls: List[float] = []

    for sparsity in sparsity_levels:
        # --- Zero ALL layers at this sparsity ---
        backups: Dict[int, tuple] = {}  # layer_idx → (neuron_indices, backup_tensor)
        for layer_idx in sorted(sorted_by_layer.keys()):
            mlp_layer = inspector.mlp_layers[layer_idx]
            if mlp_layer is None:
                continue
            scores = wanda_by_layer[layer_idx]
            n_neurons = scores.shape[0]
            k = int(n_neurons * sparsity)
            if k == 0:
                continue
            neuron_indices = sorted_by_layer[layer_idx][:k].tolist()
            backup = _zero_neurons_in_down_proj(mlp_layer, neuron_indices)
            backups[layer_idx] = (neuron_indices, backup)

        # --- Single PPL measurement with all layers pruned ---
        ppl_all = _quick_eval_ppl(
            inspector.model, domain_dataset, inspector.device,
            batch_size=batch_size, max_batches=max_batches_eval,
        )

        # --- Restore all layers ---
        for layer_idx, (neuron_indices, backup) in backups.items():
            mlp_layer = inspector.mlp_layers[layer_idx]
            _restore_neurons_in_down_proj(mlp_layer, neuron_indices, backup)

        global_ppls.append(ppl_all)
        del backups
        gc.collect()

    # Find safe global sparsity
    safe_global = 0.0
    elbow_delta = 0.0
    for sp, ppl_val in zip(sparsity_levels, global_ppls):
        rel_delta = (ppl_val / baseline_ppl - 1.0) if baseline_ppl > 0 else 0.0
        if rel_delta < ppl_threshold_relative:
            safe_global = sp
            elbow_delta = ppl_val - baseline_ppl
        else:
            break

    # Print global sweep table
    print(f"\n  Global cumulative sweep (all {len(sorted_by_layer)} layers zeroed simultaneously):")
    header = "  Sparsity |" + "".join(f" {sp:>6.0%} |" for sp in sparsity_levels)
    print(header)
    ppl_row = "  PPL      |" + "".join(f" {p:>6.2f} |" for p in global_ppls)
    print(ppl_row)
    pct_row = "  PPL Δ%   |" + "".join(
        f" {(p / baseline_ppl - 1) * 100:>+5.1f}% |" for p in global_ppls
    )
    print(pct_row)
    print(f"  Safe global sparsity: {safe_global:.0%}  "
          f"(per-layer avg was {per_layer_avg_safe:.1%})")
    if safe_global < per_layer_avg_safe:
        gap = per_layer_avg_safe - safe_global
        print(f"  ⚠  Compound error reduces safe limit by {gap:.1%} vs. single-layer estimate")

    return GlobalCompressionCurveReport(
        domain_baseline_ppl=baseline_ppl,
        sparsity_levels=list(sparsity_levels),
        global_ppl_curve=global_ppls,
        safe_global_sparsity=safe_global,
        elbow_ppl_delta=elbow_delta,
    )


# ===========================================================================
# PubMed / Biomedical data loading
# ===========================================================================

def _try_load_pubmed_qa(n: int = 640) -> Optional[List[str]]:
    """Load from qiaojin/PubMedQA (pqa_labeled, ~1000 examples)."""
    try:
        from datasets import load_dataset
        ds = load_dataset(
            "qiaojin/PubMedQA", "pqa_labeled",
            split="train", trust_remote_code=True,
        )
        texts = []
        for row in ds:
            context = row.get("context", {})
            if isinstance(context, dict):
                sentences = context.get("contexts", [])
                text = " ".join(sentences) if sentences else ""
            else:
                text = str(context)
            if len(text.strip()) > 100:
                texts.append(text)
            if len(texts) >= n:
                break
        # Also append long answers for richer text
        for row in ds:
            la = row.get("long_answer", "")
            if len(la.strip()) > 100:
                texts.append(la)
            if len(texts) >= n:
                break
        if texts:
            print(f"    PubMedQA: loaded {len(texts)} passages")
            return texts
        return None
    except Exception as e:
        print(f"    PubMedQA load failed: {e}")
        return None


def _try_load_pubmed_abstracts_streaming(n: int = 640) -> Optional[List[str]]:
    """Load from allenai/pubmed_abstracts via streaming (no full download)."""
    try:
        from datasets import load_dataset
        ds = load_dataset(
            "allenai/pubmed_abstracts",
            streaming=True,
            trust_remote_code=True,
        )
        texts = []
        for row in ds["train"]:
            abstract = row.get("abstract", "").strip()
            if len(abstract) > 100:
                texts.append(abstract)
            if len(texts) >= n:
                break
        if texts:
            print(f"    allenai/pubmed_abstracts: loaded {len(texts)} abstracts")
            return texts
        return None
    except Exception as e:
        print(f"    pubmed_abstracts streaming load failed: {e}")
        return None


def _synthesize_biomedical_texts(n: int = 200) -> List[str]:
    """
    Synthetic biomedical text for use when HuggingFace datasets are unavailable.
    Covers a range of biomedical sub-domains to provide realistic vocabulary.
    """
    texts = [
        # Pharmacology
        "The pharmacokinetics of metformin involves rapid oral absorption with peak plasma"
        " concentrations achieved within 2-3 hours. The drug is not protein-bound and is"
        " excreted unchanged by the kidneys with a half-life of approximately 5 hours."
        " Dosing must be adjusted in renal impairment to prevent lactic acidosis accumulation.",

        # Oncology
        "Chimeric antigen receptor T-cell (CAR-T) therapy has transformed treatment of"
        " relapsed/refractory B-cell lymphomas. Axicabtagene ciloleucel targets CD19 and"
        " achieves complete response rates of 54% in large B-cell lymphoma. Cytokine release"
        " syndrome and immune effector cell-associated neurotoxicity remain key toxicities.",

        # Neuroscience
        "Amyloid precursor protein undergoes sequential cleavage by beta-secretase (BACE1)"
        " and gamma-secretase to produce amyloid-beta peptides of varying lengths. Abeta42"
        " is the most aggregation-prone species and forms the core of neuritic plaques in"
        " Alzheimer's disease. The amyloid cascade hypothesis suggests this is the primary"
        " pathological driver, preceding tau tangles and neurodegeneration by years.",

        # Immunology
        "Programmed death-ligand 1 (PD-L1) expression on tumor cells enables immune evasion"
        " by binding PD-1 on cytotoxic T lymphocytes, suppressing their activation. Anti-PD-1"
        " checkpoint inhibitors such as pembrolizumab and nivolumab disrupt this interaction,"
        " restoring anti-tumor immunity. Response rates vary significantly based on tumor"
        " mutational burden and PD-L1 expression levels as measured by immunohistochemistry.",

        # Cardiology
        "Heart failure with reduced ejection fraction (HFrEF) requires neurohormonal blockade"
        " with ACE inhibitors or ARBs combined with beta-blockers and mineralocorticoid"
        " receptor antagonists. Recent trials demonstrated SGLT2 inhibitors reduce"
        " hospitalization and cardiovascular mortality independent of glycemic control."
        " Device therapy including ICD and CRT is indicated in selected patients.",

        # Molecular biology
        "CRISPR-Cas9 genome editing employs a guide RNA complementary to the target sequence"
        " to direct Cas9 nuclease to introduce a double-strand break. Non-homologous end"
        " joining creates frameshift mutations disrupting gene function, while homology-directed"
        " repair can achieve precise sequence correction using a donor template. Base editing"
        " and prime editing variants enable single-nucleotide changes without double-strand breaks.",

        # Infectious disease
        "SARS-CoV-2 enters host cells via its spike protein receptor-binding domain binding to"
        " angiotensin-converting enzyme 2 (ACE2). Following membrane fusion, the RNA genome is"
        " released and translated by host ribosomes into a polyprotein cleaved by viral proteases."
        " Remdesivir inhibits the viral RNA-dependent RNA polymerase, reducing viral replication"
        " in hospitalized patients requiring supplemental oxygen.",

        # Genetics/genomics
        "Whole-exome sequencing identifies pathogenic variants in approximately 25-30% of patients"
        " with rare undiagnosed diseases. Variants are classified using ACMG/AMP criteria as"
        " pathogenic, likely pathogenic, variant of uncertain significance, likely benign, or benign."
        " Copy number variants detected by chromosomal microarray remain the highest-yield test"
        " in neurodevelopmental disorders when standard exome is negative.",

        # Endocrinology
        "Type 1 diabetes mellitus results from autoimmune destruction of pancreatic beta cells,"
        " mediated by autoreactive CD4+ and CD8+ T cells targeting islet antigens such as GAD65"
        " and insulin. Continuous subcutaneous insulin infusion via closed-loop systems achieves"
        " superior glycemic control compared to multiple daily injections, reducing HbA1c and"
        " time in hypoglycemia. Islet cell transplantation can achieve insulin independence.",

        # Respiratory medicine
        "Idiopathic pulmonary fibrosis is characterized by progressive subpleural fibrosis"
        " with honeycombing on high-resolution CT. Nintedanib and pirfenidone slow FVC decline"
        " but do not reverse established fibrosis. The pathogenesis involves aberrant"
        " epithelial-mesenchymal transition driven by TGF-beta signaling and fibroblast"
        " activation. Lung transplantation remains the only curative option.",

        # Hematology
        "Sickle cell disease results from a point mutation in the HBB gene encoding beta-globin"
        " (E6V substitution), causing polymerization of deoxygenated hemoglobin S. Hydroxyurea"
        " induces fetal hemoglobin, preventing sickling. Gene therapy approaches using lentiviral"
        " vectors or CRISPR-mediated fetal hemoglobin reactivation have shown curative potential"
        " in clinical trials, achieving transfusion independence in treated patients.",

        # Pathology
        "The tumor microenvironment comprises cancer-associated fibroblasts, myeloid-derived"
        " suppressor cells, regulatory T cells, and tumor-associated macrophages that collectively"
        " promote immune evasion and therapeutic resistance. Single-cell RNA sequencing has"
        " revealed remarkable heterogeneity within these populations, identifying distinct"
        " functional states associated with prognosis and response to immunotherapy.",

        # Pharmacodynamics
        "Beta-blockers competitively antagonize catecholamines at beta-adrenergic receptors,"
        " reducing heart rate, myocardial contractility, and blood pressure. Cardioselective"
        " agents with beta-1 selectivity (metoprolol, bisoprolol) are preferred in obstructive"
        " airway disease. Intrinsic sympathomimetic activity in some agents (pindolol) partially"
        " offsets resting bradycardia while maintaining exercise response blunting.",

        # Clinical trials
        "Phase III randomized controlled trials represent the gold standard for evaluating"
        " therapeutic efficacy and safety. Adaptive trial designs allow pre-specified"
        " modifications based on interim analyses without inflating type I error. Bayesian"
        " adaptive randomization can enrich allocation to superior treatment arms. The primary"
        " endpoint selection must account for clinically meaningful differences and realistic"
        " event rates to ensure adequate statistical power.",

        # Biochemistry
        "The citric acid cycle (TCA cycle) oxidizes acetyl-CoA to CO2 while reducing NAD+"
        " and FAD to NADH and FADH2. Succinate dehydrogenase (Complex II) participates in"
        " both the TCA cycle and the electron transport chain. Mutations in succinate"
        " dehydrogenase subunits (SDHA/B/C/D) cause hereditary paraganglioma-pheochromocytoma"
        " syndromes through pseudohypoxic drive and dysregulated HIF-1alpha signaling.",

        # Drug resistance
        "Acquired resistance to EGFR tyrosine kinase inhibitors in non-small cell lung cancer"
        " most commonly involves the T790M gating mutation in 50-60% of cases, which"
        " sterically hinders osimertinib binding. Osimertinib itself can be overcome by"
        " C797S mutation in the covalent binding site, amplification of bypass pathways"
        " including MET and HER2, or histological transformation to small-cell carcinoma.",
    ]
    import random
    random.seed(42)
    return [random.choice(texts) for _ in range(n)]


def load_biomedical_dataset(
    tokenizer,
    max_seq_len: int = 512,
    n_samples: int = 64,
) -> "TextDataset":
    """
    Load a biomedical (PubMed) text dataset for domain analysis.

    Attempts to load from HuggingFace in order:
      1. qiaojin/PubMedQA (pqa_labeled) — small, well-maintained, 1 000 examples
      2. allenai/pubmed_abstracts — large but streamed, no full download required
      3. Synthetic fallback — high-quality hand-written biomedical text

    Args:
        tokenizer:   Model tokenizer for encoding.
        max_seq_len: Token chunk length.
        n_samples:   Number of chunks to produce.

    Returns:
        TextDataset of shape (n_chunks, max_seq_len).
    """
    from ..data_utils import TextDataset

    print("  Loading biomedical domain dataset...")
    texts = _try_load_pubmed_qa(n_samples * 10)
    if texts is None:
        texts = _try_load_pubmed_abstracts_streaming(n_samples * 10)
    if texts is None:
        print("    Biomedical: using synthetic fallback")
        texts = _synthesize_biomedical_texts(n_samples * 4)

    all_text = "\n\n".join(texts[:n_samples * 10])
    tokens = tokenizer.encode(all_text, return_tensors="pt")[0]
    n_chunks = min(n_samples, len(tokens) // max_seq_len)

    if n_chunks < 4:
        repeat_factor = (4 * max_seq_len // max(len(tokens), 1)) + 1
        tokens = tokens.repeat(repeat_factor)
        n_chunks = min(n_samples, len(tokens) // max_seq_len)

    chunks = tokens[:n_chunks * max_seq_len].reshape(n_chunks, max_seq_len)
    print(f"    Biomedical: {n_chunks} chunks of {max_seq_len} tokens")
    return TextDataset(chunks)

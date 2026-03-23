"""
Domain benchmark evaluation for MRI-Compress.

Provides lightweight zero-shot task-accuracy measurements for supported
domains, complementing perplexity as a downstream-capability metric.

Supported domains
-----------------
biomedical : PubMedQA yes/no/maybe classification (pqa_labeled split).
             Zero-shot: compare next-token log-probs of the three answer
             tokens given a short (context, question) prompt.

Adding new domains
------------------
Implement a function  run_<domain>_benchmark(model, tokenizer, device, n_samples)
and register it in DOMAIN_BENCHMARKS at the bottom of this file.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PubMedQA  (biomedical domain)
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_pubmedqa_accuracy(
    model,
    tokenizer,
    device,
    n_samples: int = 200,
) -> dict:
    """
    Zero-shot PubMedQA yes / no / maybe accuracy.

    For every (abstract, question, label) triple we:
      1. Build a short prompt ending with "Answer:"
      2. Run a single forward pass to obtain next-token logits
      3. Compare the log-probabilities of the three answer tokens
      4. Predict the highest-scoring candidate

    Returns
    -------
    dict
        accuracy    – float in [0, 1]
        n_correct   – int
        n_samples   – int (evaluated, may be < requested if labels are missing)
        per_class   – {"yes": {"n_correct": int, "n_total": int}, ...}
        dataset     – "PubMedQA (pqa_labeled)"
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.warning("datasets library not available — skipping PubMedQA benchmark")
        return {}

    print(f"\n  Running PubMedQA benchmark (n={n_samples})...")

    try:
        ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    except Exception as exc:
        logger.warning("Failed to load PubMedQA: %s — skipping benchmark", exc)
        return {}

    ds = ds.select(range(min(n_samples, len(ds))))

    # Pre-tokenise each candidate answer once.
    # We prefer the token produced with a leading space (standard BPE convention).
    candidates = ["yes", "no", "maybe"]
    cand_ids: Dict[str, int] = {}
    for ans in candidates:
        ids_space = tokenizer.encode(f" {ans}", add_special_tokens=False)
        ids_plain = tokenizer.encode(ans, add_special_tokens=False)
        tok = ids_space[0] if ids_space else (ids_plain[0] if ids_plain else None)
        if tok is None:
            logger.warning("Cannot tokenize '%s' — skipping benchmark", ans)
            return {}
        cand_ids[ans] = tok

    per_class: Dict[str, Dict] = {a: {"n_correct": 0, "n_total": 0} for a in candidates}
    correct = 0
    total   = 0

    model.eval()
    for sample in ds:
        label = sample.get("final_decision", "").lower().strip()
        if label not in candidates:
            continue

        # Build prompt — keep context short to stay within max_length
        contexts = sample.get("context", {}).get("contexts", [])
        ctx_text = " ".join(contexts)[:600]
        question = sample.get("question", "")
        prompt   = f"Context: {ctx_text}\nQuestion: {question}\nAnswer:"

        enc = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=480,
            add_special_tokens=True,
        )
        input_ids = enc["input_ids"].to(device)

        try:
            out    = model(input_ids=input_ids)
            logits = out.logits[0, -1, :]           # next-token logits
            lp     = F.log_softmax(logits, dim=-1)  # log-probabilities
        except Exception:
            continue

        scores    = {ans: lp[cand_ids[ans]].item() for ans in candidates}
        predicted = max(scores, key=scores.get)

        per_class[label]["n_total"] += 1
        if predicted == label:
            correct += 1
            per_class[label]["n_correct"] += 1
        total += 1

    accuracy = correct / total if total > 0 else 0.0
    print(f"  PubMedQA accuracy: {accuracy:.1%}  ({correct}/{total})")
    for ans in candidates:
        pc = per_class[ans]
        if pc["n_total"] > 0:
            acc_c = pc["n_correct"] / pc["n_total"]
            print(f"    {ans:6s}: {acc_c:.1%}  ({pc['n_correct']}/{pc['n_total']})")

    return {
        "accuracy":  accuracy,
        "n_correct": correct,
        "n_samples": total,
        "per_class": per_class,
        "dataset":   "PubMedQA (pqa_labeled)",
    }


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

DOMAIN_BENCHMARKS = {
    "biomedical": run_pubmedqa_accuracy,
}


def run_domain_benchmark(
    domain: str,
    model,
    tokenizer,
    device,
    n_samples: int = 200,
) -> Optional[dict]:
    """
    Run the registered benchmark for *domain* and return a metrics dict.
    Returns None if the domain has no registered benchmark.
    """
    fn = DOMAIN_BENCHMARKS.get(domain)
    if fn is None:
        logger.info("No benchmark registered for domain '%s'", domain)
        return None
    return fn(model, tokenizer, device, n_samples=n_samples)

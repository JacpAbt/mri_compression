#!/usr/bin/env python3
"""
MRI-Compress Evaluation Suite
==============================

Properly handles compressed models with variable per-layer sizes.

The problem: compressed models have layers with different intermediate_size
(e.g., layer 1 has 4947 neurons). HuggingFace from_pretrained() reads
config.json (still says 11008), creates wrong shapes, and reinitializes
mismatched layers with RANDOM WEIGHTS -- destroying the model.

The solution: smart loader that reads actual safetensor shapes, resizes
layers before loading weights, then wraps for lm-eval.

Usage:
  python evaluate.py quick --model ./compressed_v2_safe/model
  python evaluate.py bench --model ./compressed_v2_safe/model
  python evaluate.py bench --model ./compressed_v2_safe/model --tasks arc_challenge,hellaswag
  python evaluate.py compare --original Qwen/Qwen2.5-3B --compressed ./compressed_v2_safe/model
  python evaluate.py compare-quick --original Qwen/Qwen2.5-3B --compressed ./compressed_v2_safe/model --generate
"""

from __future__ import annotations
import argparse
import json
import logging
import torch
import math
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


# ============================================================================
# Smart Model Loader
# ============================================================================

def load_model_smart(model_path: str, device: str = "cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    from safetensors import safe_open
    import torch.nn as nn

    model_path = str(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Check if compressed (variable layer sizes)
    safetensor_files = sorted(Path(model_path).glob("*.safetensors"))
    is_compressed = False
    layer_sizes = {}  # layer_idx -> actual intermediate_size

    if safetensor_files:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        expected = getattr(config, 'intermediate_size', None)
        if expected:
            for sf_path in safetensor_files:
                with safe_open(str(sf_path), framework="pt") as f:
                    for key in f.keys():
                        if "mlp.gate_proj.weight" in key:
                            shape = f.get_tensor(key).shape
                            # Extract layer index
                            parts = key.split(".")
                            for i, p in enumerate(parts):
                                if p == "layers" and i + 1 < len(parts):
                                    li = int(parts[i + 1])
                                    layer_sizes[li] = shape[0]
                                    if shape[0] != expected:
                                        is_compressed = True
                                    break

    if is_compressed:
        logger.info(f"Compressed model detected -- resizing {sum(1 for s in layer_sizes.values() if s != expected)} layers")
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

        # Create model shell
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16, trust_remote_code=True)

        # Resize layers
        hidden_size = config.hidden_size
        for li, actual_size in layer_sizes.items():
            if actual_size != expected:
                layer = model.model.layers[li]
                layer.mlp.gate_proj = nn.Linear(hidden_size, actual_size, bias=False, dtype=torch.bfloat16)
                layer.mlp.up_proj = nn.Linear(hidden_size, actual_size, bias=False, dtype=torch.bfloat16)
                layer.mlp.down_proj = nn.Linear(actual_size, hidden_size, bias=False, dtype=torch.bfloat16)

        # Load all weights from safetensors
        all_tensors = {}
        for sf_path in safetensor_files:
            with safe_open(str(sf_path), framework="pt") as f:
                for key in f.keys():
                    all_tensors[key] = f.get_tensor(key)

        state_dict = model.state_dict()
        loaded, skipped = 0, 0
        for key, tensor in all_tensors.items():
            if key in state_dict and state_dict[key].shape == tensor.shape:
                state_dict[key] = tensor
                loaded += 1
            else:
                skipped += 1

        model.load_state_dict(state_dict, strict=False)
        model = model.to(device)
        logger.info(f"  Loaded {loaded} tensors, skipped {skipped}")
    else:
        logger.info(f"Loading standard model: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)

    model.eval()
    return model, tokenizer


# ============================================================================
# Perplexity
# ============================================================================

def evaluate_perplexity(model_path, datasets=None, max_samples=200, seq_len=2048,
                        batch_size=4, device="cuda"):
    if datasets is None:
        datasets = ["wikitext"]

    model, tokenizer = load_model_smart(model_path, device)
    results = {"model": str(model_path), "num_params": sum(p.numel() for p in model.parameters())}

    for ds_name in datasets:
        logger.info(f"Evaluating PPL on {ds_name}...")
        try:
            ppl = _compute_ppl(model, tokenizer, ds_name, max_samples, seq_len, batch_size, device)
            results[f"ppl_{ds_name}"] = round(ppl, 4)
            logger.info(f"  {ds_name}: PPL = {ppl:.4f}")
        except Exception as e:
            logger.warning(f"  {ds_name}: FAILED ({e})")
            results[f"ppl_{ds_name}"] = None

    return results


@torch.no_grad()
def _compute_ppl(model, tokenizer, dataset_name, max_samples, seq_len, batch_size, device):
    from torch.amp import autocast
    from datasets import load_dataset

    if dataset_name == "wikitext":
        for name in ["Salesforce/wikitext", "wikitext"]:
            try:
                ds = load_dataset(name, "wikitext-103-raw-v1", split="test")
                break
            except:
                continue
        text_key = "text"
    elif dataset_name == "c4":
        ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        texts = []
        for i, s in enumerate(ds):
            if i >= max_samples * 2: break
            if len(s["text"].strip()) > 50: texts.append(s["text"])
        class FakeDS:
            def __init__(s, t): s.data = t
            def __iter__(s): return iter([{"text": x} for x in s.data])
        ds = FakeDS(texts)
        text_key = "text"
    elif dataset_name == "wmdp_cyber_corpus":
        # cais/wmdp-corpora cyber split: domain-specific cybersecurity text corpus
        # Used to measure PPL on cybersecurity content (lower = better domain retention)
        ds = load_dataset("cais/wmdp-corpora", "cyber-forget-corpus", split="train", streaming=True)
        texts = []
        for i, s in enumerate(ds):
            if i >= max_samples * 2: break
            t = s.get("text", "")
            if len(t.strip()) > 50: texts.append(t)
        class FakeDS:
            def __init__(s, t): s.data = t
            def __iter__(s): return iter([{"text": x} for x in s.data])
        ds = FakeDS(texts)
        text_key = "text"
    else:
        ds = load_dataset(dataset_name, split="test")
        text_key = "text"

    buffer = []
    for sample in ds:
        text = sample[text_key] if isinstance(sample, dict) else sample
        if len(str(text).strip()) < 10: continue
        toks = tokenizer(str(text), return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        buffer.append(toks)
        if sum(len(t) for t in buffer) >= max_samples * seq_len + seq_len: break

    full = torch.cat(buffer, dim=0)
    n_chunks = min(len(full) // seq_len, max_samples)
    chunks = [full[i * seq_len:(i + 1) * seq_len] for i in range(n_chunks)]

    total_loss, total_tok = 0.0, 0
    for i in range(0, len(chunks), batch_size):
        batch_ids = torch.stack(chunks[i:i + batch_size]).to(device)
        mask = torch.ones_like(batch_ids)
        with autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=batch_ids, attention_mask=mask, labels=batch_ids)
        n = mask.sum().item()
        total_loss += out.loss.item() * n
        total_tok += n

    return math.exp(total_loss / max(total_tok, 1))


# ============================================================================
# Benchmarks (lm-eval)
# ============================================================================

STANDARD_BENCHMARKS = {
    "arc_it": {"task": "arc_it", "num_fewshot": 25, "metric": "acc_norm"},
    # "hellaswag":     {"task": "hellaswag",     "num_fewshot": 10, "metric": "acc_norm"},
    # "mmlu":          {"task": "mmlu",          "num_fewshot": 5,  "metric": "acc"},
    # "winogrande":    {"task": "winogrande",    "num_fewshot": 5,  "metric": "acc"},
    # "truthfulqa":    {"task": "truthfulqa_mc2", "num_fewshot": 0, "metric": "acc"},
    "wmdp_cyber":    {"task": "wmdp_cyber",    "num_fewshot": 0,  "metric": "acc"},
}


def evaluate_benchmarks(model_path, benchmarks=None, batch_size=8, device="cuda"):
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        logger.error("lm-eval not installed. pip install lm-eval")
        return {"error": "lm-eval not installed"}

    if benchmarks is None:
        benchmarks = list(STANDARD_BENCHMARKS.keys())

    tasks = [STANDARD_BENCHMARKS[b]["task"] for b in benchmarks if b in STANDARD_BENCHMARKS]
    logger.info(f"Running benchmarks: {tasks}")

    # Pre-load model correctly
    model, tokenizer = load_model_smart(model_path, device)

    # Wrap for lm-eval
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size, dtype="bfloat16")

    results_raw = lm_eval.simple_evaluate(
        model=lm, tasks=tasks, num_fewshot=None, batch_size=batch_size, device=device)

    results = {"model": str(model_path)}
    for b in benchmarks:
        info = STANDARD_BENCHMARKS[b]
        task_key = info["task"]
        metric = info["metric"]
        if task_key in results_raw["results"]:
            task_results = results_raw["results"][task_key]
            score = None
            for k, v in task_results.items():
                if metric in k:
                    score = v
                    break
            results[b] = round(score, 4) if score is not None else None
            logger.info(f"  {b}: {score:.4f}" if score else f"  {b}: N/A")
        else:
            results[b] = None

    return results


# ============================================================================
# Generation
# ============================================================================

def evaluate_generation(model_path, prompts=None, max_new_tokens=128):
    if prompts is None:
        prompts = [
            "The capital of France is",
            "def fibonacci(n):\n    '''Return the nth Fibonacci number.'''\n",
            "Il significato della vita è",
            "The derivative of x^3 + 2x^2 - 5x + 1 is",
        ]

    model, tokenizer = load_model_smart(model_path)
    results = []
    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        results.append({"prompt": prompt, "completion": text[len(prompt):]})
        logger.info(f"  Prompt: {prompt[:50]}...")
        logger.info(f"  Output: {text[len(prompt):len(prompt) + 100]}...")
    return results


# ============================================================================
# Commands
# ============================================================================

def cmd_quick(args):
    r = evaluate_perplexity(args.model, datasets=["wikitext"], max_samples=args.max_samples, batch_size=args.batch_size)
    print(json.dumps(r, indent=2))
    if args.output:
        with open(args.output, "w") as f: json.dump(r, f, indent=2)

def cmd_bench(args):
    benchmarks = args.tasks.split(",") if args.tasks else None
    r = evaluate_benchmarks(args.model, benchmarks=benchmarks, batch_size=args.batch_size)
    print(json.dumps(r, indent=2))
    if args.output:
        with open(args.output, "w") as f: json.dump(r, f, indent=2)

def cmd_compare_quick(args):
    print("=" * 60 + "\n  ORIGINAL MODEL\n" + "=" * 60)
    orig = evaluate_perplexity(args.original, datasets=["wikitext"],
                                max_samples=args.max_samples, batch_size=args.batch_size)
    print("\n" + "=" * 60 + "\n  COMPRESSED MODEL\n" + "=" * 60)
    comp = evaluate_perplexity(args.compressed, datasets=["wikitext"],
                                max_samples=args.max_samples, batch_size=args.batch_size)
    print("\n" + "=" * 60 + "\n  COMPARISON\n" + "=" * 60)
    for ds in ["wikitext"]:
        key = f"ppl_{ds}"
        if orig.get(key) and comp.get(key):
            d = comp[key] - orig[key]
            print(f"  {ds}: {orig[key]:.4f} -> {comp[key]:.4f} ({d:+.4f}, {d/orig[key]*100:+.2f}%)")
    po, pc = orig.get("num_params", 0), comp.get("num_params", 0)
    if po and pc:
        print(f"  Params: {po:,} -> {pc:,} ({(1-pc/po)*100:.1f}% reduction)")
    if args.generate:
        prompts = ["The capital of France is", "def fibonacci(n):\n", "Il significato della vita è"]
        print("\n--- ORIGINAL ---")
        evaluate_generation(args.original, prompts, 64)
        print("\n--- COMPRESSED ---")
        evaluate_generation(args.compressed, prompts, 64)
    if args.output:
        with open(args.output, "w") as f: json.dump({"original": orig, "compressed": comp}, f, indent=2)

def cmd_compare(args):
    benchmarks = args.tasks.split(",") if args.tasks else None
    print("=" * 60 + "\n  ORIGINAL\n" + "=" * 60)
    ob = evaluate_benchmarks(args.original, benchmarks=benchmarks, batch_size=args.batch_size)
    op = evaluate_perplexity(args.original, datasets=["wikitext"], max_samples=100, batch_size=args.batch_size)
    print("\n" + "=" * 60 + "\n  COMPRESSED\n" + "=" * 60)
    cb = evaluate_benchmarks(args.compressed, benchmarks=benchmarks, batch_size=args.batch_size)
    cp = evaluate_perplexity(args.compressed, datasets=["wikitext"], max_samples=100, batch_size=args.batch_size)
    print("\n" + "=" * 60 + "\n  COMPARISON\n" + "=" * 60)
    print(f"  {'Metric':<20} {'Original':>12} {'Compressed':>12} {'Delta':>12}")
    print(f"  {'-'*56}")
    for ds in ["wikitext"]:
        k = f"ppl_{ds}"
        o, c = op.get(k), cp.get(k)
        if o and c: print(f"  {'PPL '+ds:<20} {o:>12.4f} {c:>12.4f} {c-o:>+12.4f}")
    for b in (benchmarks or list(STANDARD_BENCHMARKS)):
        o, c = ob.get(b), cb.get(b)
        if o is not None and c is not None:
            print(f"  {b:<20} {o:>12.4f} {c:>12.4f} {c-o:>+12.4f}")
    po, pc = op.get("num_params", 0), cp.get("num_params", 0)
    if po and pc: print(f"  {'Params':<20} {po:>12,} {pc:>12,} {(1-pc/po)*100:>+11.1f}%")
    if args.output:
        with open(args.output, "w") as f:
            json.dump({"original": {**op, **ob}, "compressed": {**cp, **cb}}, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="MRI-Compress Evaluation Suite")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("quick")
    p.add_argument("--model", required=True)
    p.add_argument("--max-samples", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--output", default=None)

    p = sub.add_parser("bench")
    p.add_argument("--model", required=True)
    p.add_argument("--tasks", default=None, help="Comma-separated: arc_challenge,hellaswag,mmlu,winogrande,truthfulqa")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--output", default=None)

    p = sub.add_parser("compare-quick")
    p.add_argument("--original", required=True)
    p.add_argument("--compressed", required=True)
    p.add_argument("--max-samples", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--generate", action="store_true")
    p.add_argument("--output", default=None)

    p = sub.add_parser("compare")
    p.add_argument("--original", required=True)
    p.add_argument("--compressed", required=True)
    p.add_argument("--tasks", default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--output", default=None)

    args = parser.parse_args()
    {"quick": cmd_quick, "bench": cmd_bench, "compare-quick": cmd_compare_quick, "compare": cmd_compare}[args.command](args)


if __name__ == "__main__":
    main()

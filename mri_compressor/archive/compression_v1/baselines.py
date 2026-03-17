#!/usr/bin/env python3
"""
Baseline Comparisons for MRI-Compress
======================================

Runs standard compression baselines for comparison:
1. Uniform Wanda pruning (structured) at 25%, 50%, 75% sparsity
2. Uniform RTN quantization at 8-bit, 4-bit, 3-bit
3. Random structured pruning (sanity check)

Usage:
  python baselines.py \
    --model Qwen/Qwen2.5-3B \
    --output ./baseline_results
"""

from __future__ import annotations
import argparse
import gc
import json
import logging
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from compressor import (
    WandaPruner,
    LayerQuantizer,
    DeadNeuronRemover,
    collect_activations,
    get_mlp_modules,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_ppl(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int = 32,
) -> float:
    """Evaluate perplexity on calibration data."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with autocast("cuda", dtype=torch.bfloat16):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
        n = attention_mask.sum().item()
        total_loss += outputs.loss.item() * n
        total_tokens += n

    return math.exp(total_loss / max(total_tokens, 1))


def load_model_fresh(model_name: str, dtype, device):
    """Load a fresh copy of the model."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, device_map="auto", trust_remote_code=True,
    )
    model.eval()
    return model


def build_calibration(tokenizer, seq_len=2048, num_samples=128, batch_size=4):
    """Build calibration dataloader by concatenating text and chunking."""
    dataset = None
    for ds_name in ["Salesforce/wikitext", "wikitext"]:
        try:
            dataset = load_dataset(ds_name, "wikitext-103-raw-v1", split="validation")
            break
        except Exception:
            continue
    if dataset is None:
        raise RuntimeError("Could not load wikitext dataset")

    all_tokens = []
    target_total = num_samples * seq_len + seq_len
    for sample in dataset:
        text = sample["text"]
        if len(text.strip()) < 10:
            continue
        tokens = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        all_tokens.append(tokens)
        if sum(len(t) for t in all_tokens) >= target_total:
            break

    full_stream = torch.cat(all_tokens, dim=0)
    n_chunks = min(len(full_stream) // seq_len, num_samples)
    all_ids = [full_stream[i * seq_len : (i + 1) * seq_len] for i in range(n_chunks)]
    logger.info(f"Calibration: {len(all_ids)} samples × {seq_len} tokens")

    class TD(Dataset):
        def __init__(self, t): self.t = t
        def __len__(self): return len(self.t)
        def __getitem__(self, i):
            return {"input_ids": self.t[i], "attention_mask": torch.ones_like(self.t[i])}

    return DataLoader(TD(all_ids), batch_size=batch_size, shuffle=False)


def baseline_wanda_uniform(model_name, tokenizer, dataloader, sparsity, dtype, device, max_cal=16):
    """Uniform Wanda structured pruning across all layers."""
    logger.info(f"=== Baseline: Wanda uniform {sparsity:.0%} sparsity ===")
    model = load_model_fresh(model_name, dtype, device)
    num_layers = len(model.model.layers)
    total_removed = 0

    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]

        # Collect activations
        activations = collect_activations(model, dataloader, layer_idx, max_batches=max_cal, device=device)
        importance = WandaPruner.compute_importance(layer, activations, device=device)
        _, n_removed = WandaPruner.prune_neurons(layer, importance, sparsity, device=device)
        total_removed += n_removed

        del activations
        gc.collect()
        torch.cuda.empty_cache()

        if (layer_idx + 1) % 6 == 0:
            logger.info(f"  Processed {layer_idx+1}/{num_layers} layers, removed {total_removed} neurons so far")

    ppl = evaluate_ppl(model, dataloader, device)
    params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Wanda-{sparsity:.0%}: PPL={ppl:.4f}, params={params:,}")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {"method": f"wanda_uniform_{sparsity:.0%}", "ppl": ppl, "params": params, "neurons_removed": total_removed}


def baseline_quant_uniform(model_name, tokenizer, dataloader, bits, dtype, device):
    """Uniform weight quantization across all layers."""
    logger.info(f"=== Baseline: Uniform {bits}-bit quantization ===")
    model = load_model_fresh(model_name, dtype, device)
    num_layers = len(model.model.layers)

    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        LayerQuantizer.quantize_layer_weights(layer, bits=bits, device=device)

    ppl = evaluate_ppl(model, dataloader, device)
    params = sum(p.numel() for p in model.parameters())
    logger.info(f"  RTN-{bits}b: PPL={ppl:.4f}, params={params:,}")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {"method": f"rtn_uniform_{bits}b", "ppl": ppl, "params": params}


def baseline_random_prune(model_name, tokenizer, dataloader, sparsity, dtype, device):
    """Random structured pruning (sanity check — should be worst)."""
    logger.info(f"=== Baseline: Random structured {sparsity:.0%} ===")
    model = load_model_fresh(model_name, dtype, device)
    num_layers = len(model.model.layers)
    total_removed = 0

    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        mlp_modules = get_mlp_modules(layer)

        if "gate_proj" in mlp_modules:
            n = mlp_modules["gate_proj"].out_features
        elif "up_proj" in mlp_modules:
            n = mlp_modules["up_proj"].out_features
        else:
            continue

        n_keep = max(1, int(n * (1 - sparsity)))
        keep_mask = torch.zeros(n, dtype=torch.bool)
        keep_indices = torch.randperm(n)[:n_keep]
        keep_mask[keep_indices] = True

        n_removed = DeadNeuronRemover.remove_neurons(layer, keep_mask, device)
        total_removed += n_removed

    ppl = evaluate_ppl(model, dataloader, device)
    params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Random-{sparsity:.0%}: PPL={ppl:.4f}, params={params:,}")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {"method": f"random_prune_{sparsity:.0%}", "ppl": ppl, "params": params, "neurons_removed": total_removed}


def main():
    parser = argparse.ArgumentParser(description="Run compression baselines")
    parser.add_argument("--model", required=True, help="HF model name")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-cal-batches", type=int, default=16)
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=["wanda25", "wanda50", "quant8", "quant4", "random25"],
        help="Which baselines to run",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataloader = build_calibration(tokenizer, args.seq_len, args.num_samples, args.batch_size)

    # Get baseline PPL first
    logger.info("=== Baseline: Dense model (no compression) ===")
    model = load_model_fresh(args.model, dtype, device)
    dense_ppl = evaluate_ppl(model, dataloader, device)
    dense_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Dense: PPL={dense_ppl:.4f}, params={dense_params:,}")
    del model
    gc.collect()
    torch.cuda.empty_cache()

    results = [{"method": "dense", "ppl": dense_ppl, "params": dense_params}]

    baseline_map = {
        "wanda25": lambda: baseline_wanda_uniform(args.model, tokenizer, dataloader, 0.25, dtype, device, args.max_cal_batches),
        "wanda50": lambda: baseline_wanda_uniform(args.model, tokenizer, dataloader, 0.50, dtype, device, args.max_cal_batches),
        "wanda75": lambda: baseline_wanda_uniform(args.model, tokenizer, dataloader, 0.75, dtype, device, args.max_cal_batches),
        "quant8": lambda: baseline_quant_uniform(args.model, tokenizer, dataloader, 8, dtype, device),
        "quant4": lambda: baseline_quant_uniform(args.model, tokenizer, dataloader, 4, dtype, device),
        "quant3": lambda: baseline_quant_uniform(args.model, tokenizer, dataloader, 3, dtype, device),
        "random25": lambda: baseline_random_prune(args.model, tokenizer, dataloader, 0.25, dtype, device),
        "random50": lambda: baseline_random_prune(args.model, tokenizer, dataloader, 0.50, dtype, device),
    }

    for name in args.baselines:
        if name in baseline_map:
            t0 = time.perf_counter()
            res = baseline_map[name]()
            res["elapsed_seconds"] = time.perf_counter() - t0
            results.append(res)
        else:
            logger.warning(f"Unknown baseline: {name}")

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print comparison table
    print("\n" + "=" * 70)
    print(f"  {'Method':<25} {'PPL':>10} {'Params':>14} {'Reduction':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*14} {'-'*10}")
    for r in results:
        reduction = (1 - r["params"] / dense_params) * 100 if dense_params > 0 else 0
        print(f"  {r['method']:<25} {r['ppl']:>10.4f} {r['params']:>14,} {reduction:>9.1f}%")
    print("=" * 70)

    logger.info(f"Results saved to {output_dir / 'baseline_results.json'}")


if __name__ == "__main__":
    main()
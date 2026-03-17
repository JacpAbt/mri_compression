"""
Studies 12-16: Cross-disciplinary analyses for LLM compression.

  12. Feed-Forward Loop Motif Analysis       (from Gene Regulatory Networks)
  13. Information Bottleneck Layer Profile    (from Information Theory)
  14. Functional Redundancy Census            (from Ecology)
  15. Perturbation Cascade Depth              (from Epidemiology / Network Science)
  16. Activation Magnitude Phase Transition   (from Statistical Physics)

All studies use streaming layer-by-layer collection for memory efficiency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gc
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

from model_utils import ModelInspector, collect_single_layer
from data_utils import TextDataset, get_dataloader, evaluate_perplexity


# =============================================================================
# Study 12: Feed-Forward Loop Motif Analysis
# =============================================================================
# BIOLOGICAL INSPIRATION (Gene Regulatory Networks):
# FFLs are the only enriched motif in real GRNs (Alon 2007, PNAS 2011).
# A->B, A->C, B->C acts as noise filter / delay element.
# We measure cross-layer neuron "regulatory" structure: does neuron i in layer L
# reliably cause neuron j in layer L+1 to fire? Bypass paths = safe to prune.

# =============================================================================
# Study 12: Feed-Forward Loop Motif Analysis (FIXED)
# =============================================================================
# FIX: The original used P(j|i) > 0.8 as the "strong connection" threshold.
# When neurons in late layers fire on >80% of tokens (high base rate), P(j|i)
# trivially exceeds 0.8 for ALL pairs, giving density ≈ 1.0.
#
# Solution: Use LIFT = P(j|i) / P(j). Lift > 1 means i genuinely increases
# j's firing probability beyond chance. We use lift > 1.5 as threshold.
# Also add pointwise mutual information (PMI) as secondary metric.

@dataclass
class CrossLayerMotifReport:
    source_layer: int
    target_layer: int
    n_strong_connections: int       # pairs with lift > threshold
    n_inhibitory: int               # pairs with lift < 1/threshold
    n_bypass_neurons: int           # source neurons with 2+ strong targets
    n_bottleneck_neurons: int       # target neurons with exactly 1 strong source
    mean_lift: float                # average lift across strong connections
    relay_density: float            # fraction of pairs that are strong
    # NEW metrics
    mean_pmi: float                 # average PMI across strong connections
    max_lift: float                 # strongest single connection


def run_cross_layer_motif_analysis(
    inspector: ModelInspector,
    dataset: TextDataset,
    batch_size: int = 4,
    max_batches: int = 16,
    top_k_neurons: int = 200,
    fire_threshold: float = 0.01,
    lift_threshold: float = 1.5,
) -> List[CrossLayerMotifReport]:
    """
    Study 12: Analyze feed-forward relay structure between adjacent layers.
    Uses LIFT metric instead of raw conditional probability.
    """
    print("\n" + "="*80)
    print("STUDY 12: Feed-Forward Loop Motif Analysis")
    print(f"  (using lift threshold={lift_threshold}, "
          f"fire_threshold={fire_threshold})")
    print("="*80)

    reports = []
    prev_firing = None
    prev_layer_idx = None

    for layer_idx in range(inspector.num_layers):
        act = collect_single_layer(inspector, dataset, layer_idx,
                                   batch_size=batch_size, max_batches=max_batches)
        firing = (act.abs() > fire_threshold).float()
        D = firing.shape[1]

        if prev_firing is not None:
            # Sample most selectively-firing neurons (rate closest to 0.5)
            prev_rates = prev_firing.mean(dim=0)
            curr_rates = firing.mean(dim=0)
            prev_variability = (prev_rates - 0.5).abs()
            curr_variability = (curr_rates - 0.5).abs()

            k = min(top_k_neurons, D)
            prev_idx = prev_variability.topk(k, largest=False).indices
            curr_idx = curr_variability.topk(k, largest=False).indices

            prev_sampled = prev_firing[:, prev_idx]  # (N, k)
            curr_sampled = firing[:, curr_idx]        # (N, k)
            N = prev_sampled.shape[0]

            # Joint firing rate: P(i AND j)
            joint = prev_sampled.T @ curr_sampled / N      # (k, k)
            prev_marginal = prev_sampled.mean(dim=0)        # (k,)  P(i)
            curr_marginal = curr_sampled.mean(dim=0)        # (k,)  P(j)

            # LIFT = P(i,j) / (P(i) * P(j))
            expected = prev_marginal.unsqueeze(1) * curr_marginal.unsqueeze(0)  # (k, k)
            lift = joint / (expected + 1e-10)

            # PMI = log(P(i,j) / (P(i)*P(j)))
            pmi = torch.log(lift.clamp(min=1e-10))

            # Strong: lift > threshold (genuine positive association)
            strong = (lift > lift_threshold)
            n_strong = strong.sum().item()

            # Inhibitory: lift < 1/threshold (genuine suppression)
            inhibitory = (lift < (1.0 / lift_threshold)) & (joint > 0)
            n_inhibitory = inhibitory.sum().item()

            # Bypass: source neurons with 2+ strong targets
            sources_with_multiple = (strong.sum(dim=1) >= 2).sum().item()

            # Bottleneck: target neurons with exactly 1 strong source
            targets_with_single = (strong.sum(dim=0) == 1).sum().item()

            mean_lift_val = lift[strong].mean().item() if n_strong > 0 else 1.0
            max_lift_val = lift[strong].max().item() if n_strong > 0 else 1.0
            mean_pmi_val = pmi[strong].mean().item() if n_strong > 0 else 0.0
            relay_density = n_strong / (k * k)

            report = CrossLayerMotifReport(
                source_layer=prev_layer_idx, target_layer=layer_idx,
                n_strong_connections=int(n_strong),
                n_inhibitory=int(n_inhibitory),
                n_bypass_neurons=int(sources_with_multiple),
                n_bottleneck_neurons=int(targets_with_single),
                mean_lift=mean_lift_val,
                relay_density=relay_density,
                mean_pmi=mean_pmi_val,
                max_lift=max_lift_val,
            )
            reports.append(report)

            print(f"  Layer {prev_layer_idx:2d} -> {layer_idx:2d}: "
                  f"strong(lift>{lift_threshold})={n_strong:5d}, "
                  f"inhibit={n_inhibitory:5d}, "
                  f"bypass={sources_with_multiple:4d}, "
                  f"bottleneck={targets_with_single:4d}, "
                  f"density={relay_density:.4f}, "
                  f"mean_lift={mean_lift_val:.2f}")

            del prev_sampled, curr_sampled, joint, lift, pmi

        del prev_firing
        prev_firing = firing
        prev_layer_idx = layer_idx
        del act
        gc.collect()

    del prev_firing
    gc.collect()
    return reports


# =============================================================================
# Study 13: Information Bottleneck Layer Profile (FIXED)
# =============================================================================
# FIX: The original zeroed the OUTPUT of subsequent layers, which kills the
# entire residual stream (all prior layers' contributions flow through the
# output). This gives PPL = vocab_size for every layer except the last.
#
# Solution: Use IDENTITY SKIP hooks that return the INPUT unchanged, so
# subsequent layers contribute nothing new but the residual stream from
# layers 0..L is preserved through to the LM head.

@dataclass
class InformationBottleneckReport:
    layer_idx: int
    reconstruction_mse: float
    partial_lm_loss: float
    compression_ratio: float
    information_retained: float


def run_information_bottleneck_profile(
    inspector: ModelInspector,
    dataset: TextDataset,
    batch_size: int = 4,
    max_eval_batches: int = 8,
) -> List[InformationBottleneckReport]:
    """
    Study 13: Measure information compression and preservation per layer.

    1. Collect residual stream at each layer in one forward pass
    2. Linear probe: reconstruct input embeddings from h_L (MSE = I(X;Z) proxy)
    3. SKIP (identity) all layers after L, measure PPL (= I(Z;Y) proxy)
       This preserves the residual stream from layers 0..L while disabling
       computation in layers L+1..N.
    """
    print("\n" + "="*80)
    print("STUDY 13: Information Bottleneck Layer Profile")
    print("="*80)

    eval_loader = get_dataloader(dataset, batch_size=batch_size)
    layers = inspector.get_layers()

    # ---- Phase 1: Collect residual stream at every layer in one pass ----
    print("  Collecting residual stream (all layers, single pass)...")
    input_embeds_list = []
    residual_captured = defaultdict(list)
    residual_hooks = []

    def make_residual_hook(idx):
        def hook_fn(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            residual_captured[idx].append(h.detach().float().cpu())
        return hook_fn

    for idx in range(inspector.num_layers):
        hook = layers[idx].register_forward_hook(make_residual_hook(idx))
        residual_hooks.append(hook)

    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            if i >= max_eval_batches:
                break
            input_ids = batch["input_ids"].to(inspector.device)

            # Get input embeddings
            if hasattr(inspector.model, 'transformer'):
                embed = inspector.model.transformer.wte(input_ids)
            elif hasattr(inspector.model, 'model'):
                embed = inspector.model.model.embed_tokens(input_ids)
            else:
                embed = inspector.model.get_input_embeddings()(input_ids)

            input_embeds_list.append(embed.detach().float().cpu())
            inspector.model(input_ids=input_ids)

    for h in residual_hooks:
        h.remove()

    input_embeds = torch.cat(input_embeds_list, dim=0)
    input_flat = input_embeds.reshape(-1, input_embeds.shape[-1])
    baseline_mse = input_flat.var().item()

    # ---- Phase 2: Get full-model baseline PPL ----
    print("  Computing baseline PPL (full model)...")
    baseline_ppl = evaluate_perplexity(
        inspector.model, eval_loader, inspector.device,
        max_batches=max_eval_batches,
    )
    print(f"  Baseline PPL: {baseline_ppl:.2f}")

    # ---- Phase 3: Per-layer analysis ----
    reports = []
    for layer_idx in range(inspector.num_layers):
        hidden = torch.cat(residual_captured[layer_idx], dim=0)
        hidden_flat = hidden.reshape(-1, hidden.shape[-1])

        # --- Proxy I(X;Z): linear reconstruction MSE ---
        n_samples = min(5000, hidden_flat.shape[0])
        sample_idx = torch.randperm(hidden_flat.shape[0])[:n_samples]
        H = hidden_flat[sample_idx]
        X = input_flat[sample_idx]

        try:
            W = torch.linalg.lstsq(H, X).solution
            recon_mse = F.mse_loss(H @ W, X).item()
        except Exception:
            recon_mse = baseline_mse

        compression_ratio = recon_mse / (baseline_mse + 1e-10)

        # --- Proxy I(Z;Y): SKIP all layers after L, measure PPL ---
        # Identity hook: return the layer's input (= residual from prior layers)
        # as the layer's output, effectively making the layer a no-op.
        skip_hooks = []
        for later_idx in range(layer_idx + 1, inspector.num_layers):
            def make_skip_hook(captured_idx):
                def hook_fn(module, input, output):
                    # input[0] is the residual stream entering this layer
                    h_in = input[0] if isinstance(input, tuple) else input
                    if isinstance(output, tuple):
                        return (h_in,) + output[1:]
                    return h_in
                return hook_fn
            hook = layers[later_idx].register_forward_hook(
                make_skip_hook(later_idx))
            skip_hooks.append(hook)

        try:
            partial_ppl = evaluate_perplexity(
                inspector.model, eval_loader, inspector.device,
                max_batches=max_eval_batches,
            )
        except Exception:
            partial_ppl = float('inf')

        for h in skip_hooks:
            h.remove()

        # Information retained: how much of the output info is available at layer L
        if baseline_ppl > 0 and partial_ppl < float('inf'):
            info_retained = max(0.0, 1.0 - (partial_ppl - baseline_ppl) / (baseline_ppl + 1e-10))
        else:
            info_retained = 0.0

        reports.append(InformationBottleneckReport(
            layer_idx=layer_idx,
            reconstruction_mse=recon_mse,
            partial_lm_loss=partial_ppl,
            compression_ratio=compression_ratio,
            information_retained=info_retained,
        ))

        print(f"  Layer {layer_idx:2d}: recon_MSE={recon_mse:.4f} "
              f"(compression={compression_ratio:.3f}), "
              f"partial_PPL={partial_ppl:.2f}, "
              f"info_retained={info_retained:.3f}")

        del hidden, hidden_flat, H, X
        gc.collect()

    del input_embeds, input_flat, residual_captured
    gc.collect()
    return reports


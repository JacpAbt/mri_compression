"""
Studies 14-16: Continued cross-disciplinary analyses.

  14. Functional Redundancy Census            (from Ecology)
  15. Perturbation Cascade Depth              (from Epidemiology / Network Science)
  16. Activation Magnitude Phase Transition   (from Statistical Physics)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gc
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

from ..model_utils import ModelInspector, collect_single_layer
from ..data_utils import TextDataset, get_dataloader, evaluate_perplexity


# =============================================================================
# Study 14: Functional Redundancy Census
# =============================================================================
# ECOLOGY INSPIRATION (Gause's Competitive Exclusion):
# Two species in the same niche cannot coexist — one is eliminated.
# In a trained network, neurons should differentiate into distinct "niches."
# Neurons with identical activation patterns = functionally redundant.
# High max-cosine-similarity to any neighbor = safe to prune (backup exists).
# Low max-similarity = "keystone species" = critical.

@dataclass
class FunctionalRedundancyReport:
    layer_idx: int
    n_neurons: int
    mean_max_similarity: float
    median_max_similarity: float
    n_highly_redundant: int       # max_sim > 0.9
    n_keystone: int               # max_sim < 0.3
    n_redundancy_groups: int      # clusters of 3+ with mutual sim > 0.8
    safe_to_prune_count: int      # max_sim > 0.8
    max_similarity_distribution: torch.Tensor


def run_functional_redundancy_census(
    inspector: ModelInspector,
    dataset: TextDataset,
    batch_size: int = 4,
    max_batches: int = 16,
    sample_neurons: int = 1024,
) -> List[FunctionalRedundancyReport]:
    """
    Study 14: Measure neuron-level functional redundancy per layer.

    For each layer, each neuron's "activation profile" is its vector of activations
    across all tokens. The max cosine similarity to any other neuron in the same
    layer measures how replaceable that neuron is.
    """
    print("\n" + "="*80)
    print("STUDY 14: Functional Redundancy Census")
    print("="*80)

    reports = []
    for layer_idx in range(inspector.num_layers):
        act = collect_single_layer(inspector, dataset, layer_idx,
                                   batch_size=batch_size, max_batches=max_batches)
        N, D = act.shape

        # Subsample tokens for speed
        if N > 5000:
            act = act[torch.randperm(N)[:5000]]
            N = 5000

        # Sample neurons if D is large
        if D > sample_neurons:
            neuron_idx = torch.randperm(D)[:sample_neurons]
            act_sampled = act[:, neuron_idx]
        else:
            act_sampled = act
            neuron_idx = torch.arange(D)

        k = act_sampled.shape[1]

        # Cosine similarity matrix (k x k)
        norms = act_sampled.norm(dim=0, keepdim=True).clamp(min=1e-10)
        act_normed = act_sampled / norms
        sim_matrix = act_normed.T @ act_normed
        sim_matrix.fill_diagonal_(0.0)

        max_sim, _ = sim_matrix.max(dim=1)

        mean_max_sim = max_sim.mean().item()
        median_max_sim = max_sim.median().item()
        n_highly_redundant = (max_sim > 0.9).sum().item()
        n_keystone = (max_sim < 0.3).sum().item()

        # Redundancy groups: neurons with degree >= 2 in the sim>0.8 graph
        high_sim = (sim_matrix > 0.8).float()
        degree = high_sim.sum(dim=1)
        n_in_groups = (degree >= 2).sum().item()

        safe_count = (max_sim > 0.8).sum().item()

        # Scale back to full neuron count
        scale = D / k if D > sample_neurons else 1.0

        report = FunctionalRedundancyReport(
            layer_idx=layer_idx,
            n_neurons=D,
            mean_max_similarity=mean_max_sim,
            median_max_similarity=median_max_sim,
            n_highly_redundant=int(n_highly_redundant * scale),
            n_keystone=int(n_keystone * scale),
            n_redundancy_groups=int(n_in_groups * scale),
            safe_to_prune_count=int(safe_count * scale),
            max_similarity_distribution=max_sim,
        )
        reports.append(report)

        print(f"  Layer {layer_idx:2d}: mean_max_sim={mean_max_sim:.3f}, "
              f"redundant(>0.9)={int(n_highly_redundant * scale):4d}, "
              f"keystone(<0.3)={int(n_keystone * scale):4d}, "
              f"safe_to_prune={int(safe_count * scale):4d}/{D}")

        del act, act_sampled, sim_matrix, act_normed, max_sim
        gc.collect()

    return reports


# =============================================================================
# Study 15: Perturbation Cascade Depth
# =============================================================================
# EPIDEMIOLOGY / NETWORK SCIENCE INSPIRATION:
# R0 measures how far a perturbation propagates through a contact network.
# We zero one neuron in layer L and measure the L2 norm of the residual-stream
# perturbation at each subsequent layer L+1, L+2, ..., L_final.
# Fast decay = local error correction. Growth = fragile amplification.

@dataclass
class PerturbationCascadeReport:
    layer_idx: int
    neuron_idx: int
    cascade_amplitudes: List[float]        # absolute perturbation norm per downstream layer
    cascade_normalized: List[float]        # perturbation / baseline_norm per layer
    local_damping_ratios: List[float]      # amplitude[k+1] / amplitude[k]
    max_amplification: float               # max(normalized) / normalized[0]
    mean_damping_ratio: float              # geometric mean of local ratios
    peak_layer_offset: int                 # how many layers downstream is the peak
    initial_perturbation: float            # raw perturbation at first downstream layer
    relative_perturbation: float           # initial / baseline_norm


def run_perturbation_cascade_analysis(
    inspector: ModelInspector,
    dataset: TextDataset,
    batch_size: int = 2,
    max_eval_batches: int = 4,
    neurons_per_layer: int = 3,
    sample_layers: int = 8,
) -> List[PerturbationCascadeReport]:
    """
    Study 15: Track how single-neuron perturbations propagate downstream.
    Uses normalized perturbation and local damping ratios.
    """
    print("\n" + "="*80)
    print("STUDY 15: Perturbation Cascade Depth")
    print("="*80)

    eval_loader = get_dataloader(dataset, batch_size=batch_size)
    layers_module = inspector.get_layers()
    n_layers = inspector.num_layers

    # Select layers to test
    if n_layers <= sample_layers:
        test_layers = list(range(n_layers - 1))
    else:
        step = max(1, n_layers // sample_layers)
        test_layers = list(range(0, n_layers - 1, step))

    # Get one batch for all tests (same input = comparable)
    test_batch = next(iter(eval_loader))
    input_ids = test_batch["input_ids"].to(inspector.device)

    reports = []

    for source_layer in test_layers:
        # Pick top neurons by weight norm
        mlp = inspector.mlp_layers[source_layer]
        down_w = mlp.down_proj.weight.data.cpu().float()
        D = mlp.intermediate_size

        if down_w.shape[0] == D:
            neuron_importance = down_w.norm(dim=1)
        else:
            neuron_importance = down_w.norm(dim=0)

        top_neurons = neuron_importance.topk(min(neurons_per_layer, D)).indices.tolist()

        for neuron_idx in top_neurons:
            # --- Clean forward pass: capture full residual tensors ---
            clean_hiddens = {}

            def make_capture_hook(idx, storage):
                def hook_fn(module, input, output):
                    h = output[0] if isinstance(output, tuple) else output
                    storage[idx] = h.detach().float().cpu()
                return hook_fn

            clean_hooks = []
            for idx in range(source_layer, n_layers):
                hook = layers_module[idx].register_forward_hook(
                    make_capture_hook(idx, clean_hiddens))
                clean_hooks.append(hook)

            with torch.no_grad():
                inspector.model(input_ids=input_ids)

            for h in clean_hooks:
                h.remove()

            # --- Perturbed forward pass: zero neuron ---
            perturbed_hiddens = {}

            def make_zero_neuron_hook(n_idx):
                def hook_fn(module, input):
                    x = input[0] if isinstance(input, tuple) else input
                    x_mod = x.clone()
                    x_mod[..., n_idx] = 0.0
                    if isinstance(input, tuple):
                        return (x_mod,) + input[1:]
                    return (x_mod,)
                return hook_fn

            zero_hook = mlp.down_proj.register_forward_pre_hook(
                make_zero_neuron_hook(neuron_idx))

            perturb_hooks = []
            for idx in range(source_layer, n_layers):
                hook = layers_module[idx].register_forward_hook(
                    make_capture_hook(idx, perturbed_hiddens))
                perturb_hooks.append(hook)

            with torch.no_grad():
                inspector.model(input_ids=input_ids)

            zero_hook.remove()
            for h in perturb_hooks:
                h.remove()

            # --- Compute cascade with normalized metrics ---
            cascade_abs = []
            cascade_norm = []
            for idx in range(source_layer + 1, n_layers):
                if idx in clean_hiddens and idx in perturbed_hiddens:
                    delta = (clean_hiddens[idx] - perturbed_hiddens[idx]).norm().item()
                    baseline_norm = clean_hiddens[idx].norm().item()
                    cascade_abs.append(delta)
                    cascade_norm.append(delta / (baseline_norm + 1e-10))
                else:
                    cascade_abs.append(0.0)
                    cascade_norm.append(0.0)

            # Local damping ratios: ratio of consecutive amplitudes
            local_ratios = []
            for ci in range(1, len(cascade_abs)):
                if cascade_abs[ci - 1] > 1e-20:
                    local_ratios.append(cascade_abs[ci] / cascade_abs[ci - 1])
                else:
                    local_ratios.append(1.0)

            if len(cascade_norm) > 0 and cascade_norm[0] > 1e-10:
                initial_norm = cascade_norm[0]
                max_amp = max(cascade_norm) / initial_norm
                peak_offset = int(np.argmax(cascade_norm))

                # Geometric mean of local ratios
                if local_ratios:
                    log_ratios = [np.log(max(r, 1e-20)) for r in local_ratios]
                    mean_damping = float(np.exp(np.mean(log_ratios)))
                else:
                    mean_damping = 1.0

                initial_pert = cascade_abs[0]
                baseline_at_first = clean_hiddens.get(
                    source_layer + 1, torch.tensor(1.0))
                if isinstance(baseline_at_first, torch.Tensor):
                    baseline_at_first = baseline_at_first.norm().item()
                relative_pert = initial_pert / (baseline_at_first + 1e-10)
            else:
                max_amp = 0.0
                peak_offset = 0
                mean_damping = 1.0
                initial_pert = 0.0
                relative_pert = 0.0

            reports.append(PerturbationCascadeReport(
                layer_idx=source_layer,
                neuron_idx=neuron_idx,
                cascade_amplitudes=cascade_abs,
                cascade_normalized=cascade_norm,
                local_damping_ratios=local_ratios,
                max_amplification=max_amp,
                mean_damping_ratio=mean_damping,
                peak_layer_offset=peak_offset,
                initial_perturbation=initial_pert,
                relative_perturbation=relative_pert,
            ))

            del clean_hiddens, perturbed_hiddens

        # Print summary for this layer
        layer_reports = [r for r in reports if r.layer_idx == source_layer]
        avg_amp = np.mean([r.max_amplification for r in layer_reports])
        avg_damping = np.mean([r.mean_damping_ratio for r in layer_reports])
        avg_peak = np.mean([r.peak_layer_offset for r in layer_reports])
        avg_rel = np.mean([r.relative_perturbation for r in layer_reports])
        print(f"  Layer {source_layer:2d}: "
              f"max_amp={avg_amp:.2f}x, "
              f"mean_damping={avg_damping:.4f}, "
              f"peak_offset=+{avg_peak:.1f}, "
              f"rel_perturbation={avg_rel:.2e}")

        gc.collect()
        torch.cuda.empty_cache()

    return reports


# =============================================================================
# Study 16: Activation Magnitude Phase Transition
# =============================================================================
# STATISTICAL PHYSICS INSPIRATION:
# Phase transitions = qualitative behavior change at critical parameter.
# We fit P(|a| > x) ~ x^(-alpha) per layer and track alpha across depth.
# alpha < 2 = infinite variance (heavy tails, dangerous to prune)
# alpha > 3 = light tails (safe zone)
# Sharp alpha transitions = "critical layers" where representation changes.

@dataclass
class PhaseTransitionReport:
    layer_idx: int
    power_law_alpha: float
    tail_fraction: float            # fraction > 10x median
    is_heavy_tailed: bool           # alpha < 2.5
    gaussian_fit_ks: float          # KS statistic vs Gaussian (higher = less Gaussian)
    log_log_slope: float            # slope of log-log survival function
    activation_entropy: float       # Shannon entropy of discretized distribution


def fit_power_law_tail(magnitudes: torch.Tensor, x_min_percentile: float = 90.0) -> Tuple[float, float]:
    """
    Hill estimator for power-law exponent on the upper tail.
    alpha = 1 + n / sum(log(x_i / x_min))
    """
    sorted_mags, _ = magnitudes.sort(descending=True)
    x_min_idx = max(1, int(len(sorted_mags) * (1 - x_min_percentile / 100)))
    x_min = sorted_mags[x_min_idx].item()

    if x_min <= 0:
        return 0.0, 0.0

    tail = sorted_mags[:x_min_idx].float()
    tail = tail[tail > x_min]

    if len(tail) < 10:
        return 0.0, x_min

    log_ratio = torch.log(tail / x_min)
    alpha = 1.0 + len(tail) / (log_ratio.sum().item() + 1e-10)
    return alpha, x_min


def run_phase_transition_analysis(
    inspector: ModelInspector,
    dataset: TextDataset,
    batch_size: int = 4,
    max_batches: int = 16,
) -> List[PhaseTransitionReport]:
    """Study 16: Fit power-law exponents to activation magnitude distributions."""
    print("\n" + "="*80)
    print("STUDY 16: Activation Magnitude Phase Transition")
    print("="*80)

    reports = []
    for layer_idx in range(inspector.num_layers):
        act = collect_single_layer(inspector, dataset, layer_idx,
                                   batch_size=batch_size, max_batches=max_batches)

        magnitudes = act.abs().flatten()

        # --- Power-law fit ---
        alpha, x_min = fit_power_law_tail(magnitudes, x_min_percentile=90.0)

        # --- Tail fraction ---
        median_val = magnitudes.median().item()
        tail_fraction = (magnitudes > 10 * max(median_val, 1e-10)).float().mean().item()

        # --- Gaussian fit quality (KS statistic) ---
        mu = magnitudes.mean().item()
        sigma = magnitudes.std().item()

        n_test = min(10000, len(magnitudes))
        sample = magnitudes[torch.randperm(len(magnitudes))[:n_test]].sort().values
        theoretical_cdf = 0.5 * (1 + torch.erf(
            (sample - mu) / (sigma * np.sqrt(2) + 1e-10)))
        empirical_cdf = torch.linspace(0, 1, n_test)
        ks_stat = (theoretical_cdf - empirical_cdf).abs().max().item()

        # --- Log-log slope of survival function ---
        sorted_mags, _ = magnitudes.sort(descending=True)
        mag_min = max(sorted_mags[-1].item(), 1e-10)
        mag_max = max(sorted_mags[0].item(), 1e-10)

        if mag_max > mag_min:
            points = torch.logspace(np.log10(mag_min), np.log10(mag_max), 100)
            survival = [(magnitudes > p).float().mean().item() for p in points]
            valid = [(np.log10(p.item()), np.log10(s + 1e-10))
                     for p, s in zip(points, survival) if s > 0]
            if len(valid) > 2:
                x_arr, y_arr = zip(*valid)
                slope = np.polyfit(np.array(x_arr), np.array(y_arr), 1)[0]
            else:
                slope = 0.0
        else:
            slope = 0.0

        # --- Activation entropy ---
        hist = torch.histc(magnitudes, bins=100, min=0,
                           max=max(magnitudes.max().item(), 1e-10))
        prob = hist / hist.sum()
        prob = prob[prob > 0]
        entropy = -(prob * prob.log()).sum().item()

        is_heavy = alpha < 2.5 and alpha > 0

        report = PhaseTransitionReport(
            layer_idx=layer_idx,
            power_law_alpha=alpha,
            tail_fraction=tail_fraction,
            is_heavy_tailed=is_heavy,
            gaussian_fit_ks=ks_stat,
            log_log_slope=slope,
            activation_entropy=entropy,
        )
        reports.append(report)

        tail_marker = " ← HEAVY TAIL" if is_heavy else ""
        print(f"  Layer {layer_idx:2d}: alpha={alpha:.2f}, tail_frac={tail_fraction:.4f}, "
              f"KS={ks_stat:.3f}, entropy={entropy:.2f}{tail_marker}")

        del act, magnitudes
        gc.collect()

    # Summary
    alphas = [r.power_law_alpha for r in reports if r.power_law_alpha > 0]
    if alphas:
        print(f"\n  Alpha range: {min(alphas):.2f} — {max(alphas):.2f}")
        # Find steepest transition
        for i in range(1, len(reports)):
            delta = abs(reports[i].power_law_alpha - reports[i-1].power_law_alpha)
            if delta > 0.5:
                print(f"  Phase transition at layer {reports[i].layer_idx}: "
                      f"alpha jump {reports[i-1].power_law_alpha:.2f} -> "
                      f"{reports[i].power_law_alpha:.2f} (delta={delta:.2f})")

    return reports

"""
Shared fixtures and helpers for the test suite.

All tests are designed to run without a GPU and without downloading any model.
They use synthetic mini-transformers built entirely from torch.nn primitives.
"""

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Synthetic MLP layer that mimics the gated (SwiGLU) structure used by
# Qwen/Llama: mlp.gate_proj, mlp.up_proj, mlp.down_proj
# ---------------------------------------------------------------------------

class SyntheticGatedMLP(nn.Module):
    def __init__(self, hidden=32, intermediate=64):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj   = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class SyntheticAttention(nn.Module):
    def __init__(self, hidden=32, num_heads=4):
        super().__init__()
        head_dim = hidden // num_heads
        self.num_heads = num_heads
        self.num_kv_heads = num_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)


class SyntheticLayer(nn.Module):
    """A transformer-like layer with .mlp and .self_attn attributes."""
    def __init__(self, hidden=32, intermediate=64, num_heads=4):
        super().__init__()
        self.mlp       = SyntheticGatedMLP(hidden, intermediate)
        self.self_attn = SyntheticAttention(hidden, num_heads)

    def forward(self, x):
        return self.mlp(x)


@pytest.fixture
def gated_layer():
    """A single synthetic gated-MLP transformer layer (CPU, float32)."""
    torch.manual_seed(0)
    return SyntheticLayer(hidden=32, intermediate=64)


@pytest.fixture
def small_activations():
    """Synthetic activation tensor [n_tokens=128, n_neurons=64]."""
    torch.manual_seed(1)
    acts = torch.randn(128, 64)
    # Zero out the last 8 neurons to simulate dead neurons
    acts[:, -8:] = 0.0
    return acts


@pytest.fixture
def minimal_mri_summary():
    """
    Minimal summary dict that satisfies MRIDiagnostician.diagnose_from_summary().
    Only fields read by the diagnostician are populated; everything else is absent
    so it falls through to safe defaults.
    """
    num_layers = 4
    per_layer = {}
    for i in range(num_layers):
        per_layer[str(i)] = {
            "study1_activation_profile": {"kurtosis": 3.0, "gini_coefficient": 0.3},
            "study5_neuron_health": {"dead_count": 5, "dormant_count": 3},
            "study6_attention_heads": [
                {"head_idx": j, "mean_entropy": 1.5, "first_token_frac": 0.1}
                for j in range(4)
            ],
            "study7_gate_wanda_correlation": {"pearson_r": 0.55},
            "study8_sparsity_structure": {
                "n_coactivation_clusters": 2,
                "mean_consistency": 0.6,
            },
            "study10_layer_redundancy": {"mlp_ppl_delta": 0.05, "attn_ppl_delta": 0.03},
            "study14_functional_redundancy": {"safe_to_prune_count": 2, "safe_to_prune_frac": 0.04},
            "study15_perturbation_cascade": {"max_amplification": 1.2},
            "study16_phase_transition": {"power_law_alpha": 3.0, "is_heavy_tailed": False},
        }

    return {
        "model": "synthetic-test-model",
        "baseline_ppl": 25.0,
        "architecture": {
            "num_layers": num_layers,
            "intermediate_size": 128,
            "num_attention_heads": 4,
        },
        "per_layer": per_layer,
        "protection_lists": {
            "never_prune_neurons": [],
            "never_prune_heads": [],
        },
        "compression_hints": {},
    }

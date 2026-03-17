"""
Study 11: Domain-Specific Activation Divergence — Memory-Optimized

KEY CHANGE: Instead of collecting all layers for all domains at once,
we iterate: for each layer, collect across all domains, compute metrics, free.

Memory: O(num_domains × intermediate_size) per layer instead of 
        O(num_domains × num_layers × num_tokens × intermediate_size).
"""

import torch
import torch.nn as nn
import numpy as np
import gc
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

from model_utils import ModelInspector, collect_single_layer
from data_utils import TextDataset, get_dataloader


# =============================================================================
# Domain data loading (unchanged from original)
# =============================================================================

def _try_load_dataset(name: str, config: str, split: str, field_name: str = "text") -> Optional[List[str]]:
    try:
        from datasets import load_dataset
        ds = load_dataset(name, config, split=split, trust_remote_code=True)
        for f in [field_name, "question", "content", "text", "prompt"]:
            if f in ds.column_names:
                texts = [str(row[f]) for row in ds if len(str(row[f]).strip()) > 50]
                if texts:
                    return texts
        return None
    except Exception as e:
        print(f"    Could not load {name}/{config}: {e}")
        return None


def _synthesize_math_prompts(tokenizer, n: int = 200) -> List[str]:
    templates = [
        "Calculate the derivative of f(x) = {a}x^{b} + {c}x^2 - {d}x + {e}. First, apply the power rule to each term.",
        "If a train travels {a} kilometers in {b} hours, and another train travels {c} kilometers in {d} hours, what is the ratio of their speeds?",
        "Solve the equation {a}x + {b} = {c}x - {d}. To find x, first move all x terms to one side.",
        "A rectangle has a perimeter of {a} meters. If its length is {b} meters more than its width, find the dimensions.",
        "The sum of {a} consecutive integers starting from {b} can be calculated using the formula for arithmetic series.",
        "Find the probability of drawing {a} red balls from a bag containing {b} red and {c} blue balls without replacement.",
        "The matrix A = [[{a}, {b}], [{c}, {d}]] has determinant ad - bc = {a}*{d} - {b}*{c}.",
        "In a geometric sequence where a1 = {a} and r = {b}/10, find the sum of the first {c} terms.",
        "If log base {a} of x = {b}, then x = {a}^{b}. Computing this power:",
    ]
    import random
    random.seed(42)
    texts = []
    for i in range(n):
        tmpl = random.choice(templates)
        vals = {k: random.randint(2, 50) for k in 'abcde'}
        texts.append(tmpl.format(**vals))
    return texts


def _synthesize_code_prompts(tokenizer, n: int = 200) -> List[str]:
    snippets = [
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n\nfor i in range(10):\n    print(f'F({i}) = {fibonacci(i)}')",
        "import numpy as np\n\ndef matrix_multiply(A, B):\n    assert A.shape[1] == B.shape[0]\n    result = np.zeros((A.shape[0], B.shape[1]))\n    for i in range(A.shape[0]):\n        for j in range(B.shape[1]):\n            for k in range(A.shape[1]):\n                result[i][j] += A[i][k] * B[k][j]\n    return result",
        "class BinarySearchTree:\n    def __init__(self, value):\n        self.value = value\n        self.left = None\n        self.right = None\n    \n    def insert(self, val):\n        if val < self.value:\n            if self.left is None:\n                self.left = BinarySearchTree(val)\n            else:\n                self.left.insert(val)\n        else:\n            if self.right is None:\n                self.right = BinarySearchTree(val)\n            else:\n                self.right.insert(val)",
        "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)",
        "from dataclasses import dataclass\nfrom typing import List, Optional\n\n@dataclass\nclass Node:\n    value: int\n    children: List['Node'] = field(default_factory=list)\n    parent: Optional['Node'] = None\n\n    def add_child(self, child: 'Node'):\n        child.parent = self\n        self.children.append(child)",
        "import torch\nimport torch.nn as nn\n\nclass TransformerBlock(nn.Module):\n    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):\n        super().__init__()\n        self.attn = nn.MultiheadAttention(d_model, n_heads)\n        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))\n        self.ln1 = nn.LayerNorm(d_model)\n        self.ln2 = nn.LayerNorm(d_model)",
    ]
    import random
    random.seed(42)
    return [random.choice(snippets) for _ in range(n)]


def _synthesize_italian_prompts(tokenizer, n: int = 200) -> List[str]:
    texts = [
        "La storia dell'Italia è ricca di eventi che hanno plasmato la cultura europea. Dall'Impero Romano alle città-stato del Rinascimento, ogni epoca ha lasciato un'impronta indelebile sulla società moderna.",
        "Nel campo della fisica teorica, le equazioni di Maxwell descrivono il comportamento dei campi elettromagnetici. La relazione tra campo elettrico e campo magnetico è fondamentale per comprendere la propagazione della luce.",
        "L'architettura gotica italiana presenta caratteristiche uniche rispetto a quella del nord Europa. Le cattedrali di Milano, Siena e Orvieto mostrano un'interpretazione locale dello stile.",
        "La cucina italiana regionale riflette secoli di tradizione e innovazione. Ogni regione ha sviluppato piatti tipici basati sugli ingredienti locali: il pesto alla genovese in Liguria, la carbonara nel Lazio.",
        "Il sistema bancario moderno ha le sue origini nelle città italiane del Medioevo. I Medici di Firenze, i banchieri genovesi e i mercanti veneziani hanno sviluppato strumenti finanziari alla base dell'economia contemporanea.",
        "La meccanica quantistica introduce concetti che sfidano l'intuizione classica. Il principio di indeterminazione di Heisenberg stabilisce un limite fondamentale alla precisione con cui possiamo conoscere posizione e quantità di moto.",
        "Dante Alighieri nella Divina Commedia descrive un viaggio attraverso i tre regni dell'aldilà. L'opera non è solo un capolavoro letterario, ma anche un trattato di filosofia, teologia e politica.",
        "L'analisi dei dati nella ricerca scientifica richiede una comprensione approfondita della statistica. Il test di ipotesi, la regressione lineare e l'analisi della varianza sono strumenti essenziali.",
    ]
    import random
    random.seed(42)
    return [random.choice(texts) for _ in range(n)]


def load_domain_datasets(
    tokenizer, max_seq_len: int = 512, samples_per_domain: int = 64,
) -> Dict[str, TextDataset]:
    """Load or synthesize datasets for multiple domains."""
    domains = {}
    print("  Loading domain datasets...")
    
    english_texts = _try_load_dataset("wikitext", "wikitext-103-raw-v1", "validation")
    if english_texts is None:
        english_texts = _try_load_dataset("wikitext", "wikitext-2-raw-v1", "validation")
    if english_texts is None:
        print("    English: using synthetic fallback")
        english_texts = ["The history of natural language processing began with early computational linguistics. " * 10] * samples_per_domain
    else:
        print(f"    English: loaded {len(english_texts)} texts")
    
    math_texts = _try_load_dataset("gsm8k", "main", "test", field_name="question")
    if math_texts is None:
        math_texts = _try_load_dataset("openai/gsm8k", "main", "test", field_name="question")
    if math_texts is None:
        print("    Math: using synthetic fallback")
        math_texts = _synthesize_math_prompts(tokenizer, samples_per_domain * 4)
    else:
        print(f"    Math: loaded {len(math_texts)} problems")
    
    code_texts = _try_load_dataset("codeparrot/github-code", "all", "train", field_name="code")
    if code_texts is None:
        print("    Code: using synthetic fallback")
        code_texts = _synthesize_code_prompts(tokenizer, samples_per_domain * 4)
    else:
        print(f"    Code: loaded {len(code_texts)} snippets")
    
    italian_texts = _try_load_dataset("uonlp/CulturaX", "it", "train", field_name="text")
    if italian_texts is None:
        print("    Italian: using synthetic fallback")
        italian_texts = _synthesize_italian_prompts(tokenizer, samples_per_domain * 4)
    else:
        print(f"    Italian: loaded {len(italian_texts)} texts")
    
    for name, texts in [("english", english_texts), ("math", math_texts), 
                          ("code", code_texts), ("italian", italian_texts)]:
        all_text = "\n".join(texts[:samples_per_domain * 10])
        tokens = tokenizer.encode(all_text, return_tensors="pt")[0]
        n_chunks = min(samples_per_domain, len(tokens) // max_seq_len)
        
        if n_chunks < 4:
            print(f"    WARNING: {name} only produced {n_chunks} chunks, padding with repeats.")
            repeat_factor = (4 * max_seq_len // len(tokens)) + 1
            tokens = tokens.repeat(repeat_factor)
            n_chunks = min(samples_per_domain, len(tokens) // max_seq_len)
        
        chunks = tokens[:n_chunks * max_seq_len].reshape(n_chunks, max_seq_len)
        domains[name] = TextDataset(chunks)
        print(f"    {name}: {n_chunks} chunks of {max_seq_len} tokens")
    
    return domains


# =============================================================================
# Streaming domain activation collection
# =============================================================================

def collect_domain_firing_rates_streaming(
    inspector: ModelInspector,
    domain_datasets: Dict[str, TextDataset],
    batch_size: int = 4,
    max_batches: int = 16,
    fire_threshold: float = 0.01,
) -> Dict[str, Dict[int, torch.Tensor]]:
    """
    MEMORY-OPTIMIZED: Collect firing rates layer-by-layer.
    
    Instead of: for each domain -> hook all layers -> run forward
    We do:      for each layer -> for each domain -> hook one layer -> run forward
    
    This means more forward passes (num_layers × num_domains instead of num_domains)
    but massively less memory.
    """
    domains = sorted(domain_datasets.keys())
    domain_firing_rates = {d: {} for d in domains}
    
    for layer_idx in range(inspector.num_layers):
        if layer_idx % 6 == 0:
            print(f"    Layer {layer_idx}/{inspector.num_layers}...")
        
        for domain_name in domains:
            dataset = domain_datasets[domain_name]
            act = collect_single_layer(
                inspector, dataset, layer_idx,
                batch_size=batch_size, max_batches=max_batches,
            )
            # act: (N, D) in float32 on CPU
            firing = (act.abs() > fire_threshold).float()
            firing_rate = firing.mean(dim=0)  # (D,)
            domain_firing_rates[domain_name][layer_idx] = firing_rate
            
            del act, firing
        
        gc.collect()
        torch.cuda.empty_cache()
    
    return domain_firing_rates


# =============================================================================
# Divergence metrics (unchanged logic)
# =============================================================================

@dataclass
class DomainDivergenceReport:
    layer_idx: int
    domain_a: str
    domain_b: str
    jaccard_similarity: float
    cosine_similarity: float
    n_shared_active: int
    n_domain_a_specific: int
    n_domain_b_specific: int
    n_both_inactive: int
    magnitude_correlation: float


@dataclass 
class DomainOverviewReport:
    layer_idx: int
    n_universal_neurons: int
    n_dead_across_all: int
    mean_pairwise_jaccard: float
    domain_specificity_score: float
    per_domain_active_count: Dict[str, int]


def compute_domain_divergence(
    domain_firing_rates: Dict[str, Dict[int, torch.Tensor]],
    num_layers: int,
    active_threshold: float = 0.05,
    inactive_threshold: float = 0.01,
) -> Tuple[List[DomainDivergenceReport], List[DomainOverviewReport]]:
    domains = sorted(domain_firing_rates.keys())
    pairwise_reports = []
    overview_reports = []
    
    for layer_idx in range(num_layers):
        rates = {d: domain_firing_rates[d][layer_idx] for d in domains}
        D = rates[domains[0]].shape[0]
        
        active = {d: (rates[d] > active_threshold) for d in domains}
        inactive = {d: (rates[d] < inactive_threshold) for d in domains}
        
        jaccards = []
        for i, da in enumerate(domains):
            for db in domains[i+1:]:
                set_a, set_b = active[da], active[db]
                
                intersection = (set_a & set_b).sum().item()
                union = (set_a | set_b).sum().item()
                jaccard = intersection / (union + 1e-10)
                
                ra, rb = rates[da].float(), rates[db].float()
                cosine = (ra @ rb) / (ra.norm() * rb.norm() + 1e-10)
                
                ra_c = ra - ra.mean()
                rb_c = rb - rb.mean()
                pearson = (ra_c @ rb_c) / (ra_c.norm() * rb_c.norm() + 1e-10)
                
                n_shared = intersection
                n_a_only = (set_a & ~set_b).sum().item()
                n_b_only = (~set_a & set_b).sum().item()
                n_neither = (~set_a & ~set_b).sum().item()
                
                pairwise_reports.append(DomainDivergenceReport(
                    layer_idx=layer_idx, domain_a=da, domain_b=db,
                    jaccard_similarity=jaccard, cosine_similarity=cosine.item(),
                    n_shared_active=int(n_shared),
                    n_domain_a_specific=int(n_a_only),
                    n_domain_b_specific=int(n_b_only),
                    n_both_inactive=int(n_neither),
                    magnitude_correlation=pearson.item(),
                ))
                jaccards.append(jaccard)
        
        all_active = torch.stack([active[d] for d in domains]).all(dim=0)
        all_inactive = torch.stack([inactive[d] for d in domains]).all(dim=0)
        
        overview_reports.append(DomainOverviewReport(
            layer_idx=layer_idx,
            n_universal_neurons=int(all_active.sum().item()),
            n_dead_across_all=int(all_inactive.sum().item()),
            mean_pairwise_jaccard=float(np.mean(jaccards)),
            domain_specificity_score=float(1.0 - np.mean(jaccards)),
            per_domain_active_count={d: int(active[d].sum().item()) for d in domains},
        ))
    
    return pairwise_reports, overview_reports


# =============================================================================
# Main runner
# =============================================================================

def run_domain_divergence_study(
    inspector: ModelInspector,
    batch_size: int = 4,
    max_batches: int = 16,
    samples_per_domain: int = 64,
) -> Dict:
    """Study 11: Domain-Specific Activation Divergence — memory-optimized."""
    print("\n" + "=" * 80)
    print("STUDY 11: Domain-Specific Activation Divergence")
    print("=" * 80)
    
    domain_datasets = load_domain_datasets(
        inspector.tokenizer, max_seq_len=512, samples_per_domain=samples_per_domain,
    )
    
    print("\n  Collecting domain-specific activation patterns (streaming)...")
    domain_firing_rates = collect_domain_firing_rates_streaming(
        inspector, domain_datasets, batch_size=batch_size, max_batches=max_batches,
    )
    
    print("\n  Computing divergence metrics...")
    pairwise_reports, overview_reports = compute_domain_divergence(
        domain_firing_rates, inspector.num_layers,
    )
    
    # Print summary
    print("\n  Layer-by-layer specialization profile:")
    print(f"  {'Layer':>5} | {'Jaccard':>8} | {'Specificity':>11} | {'Universal':>9} | {'Dead(all)':>9} | " +
          " | ".join(f"{d:>8}" for d in sorted(domain_firing_rates.keys())))
    print("  " + "-" * 100)
    
    for ov in overview_reports:
        counts = " | ".join(f"{ov.per_domain_active_count[d]:>8.0f}" 
                           for d in sorted(ov.per_domain_active_count.keys()))
        print(f"  {ov.layer_idx:>5} | {ov.mean_pairwise_jaccard:>8.3f} | "
              f"{ov.domain_specificity_score:>11.3f} | {ov.n_universal_neurons:>9} | "
              f"{ov.n_dead_across_all:>9} | {counts}")
    
    most_specialized = max(overview_reports, key=lambda r: r.domain_specificity_score)
    least_specialized = min(overview_reports, key=lambda r: r.domain_specificity_score)
    print(f"\n  Most specialized layer:  {most_specialized.layer_idx} "
          f"(specificity={most_specialized.domain_specificity_score:.3f})")
    print(f"  Least specialized layer: {least_specialized.layer_idx} "
          f"(specificity={least_specialized.domain_specificity_score:.3f})")
    
    print("\n  Notable pairwise divergences:")
    for layer_idx in [0, inspector.num_layers // 4, inspector.num_layers // 2, 
                       3 * inspector.num_layers // 4, inspector.num_layers - 1]:
        layer_pairs = [r for r in pairwise_reports if r.layer_idx == layer_idx]
        for r in layer_pairs:
            if r.jaccard_similarity < 0.5 or layer_idx in [0, inspector.num_layers - 1]:
                print(f"    Layer {r.layer_idx:2d}: {r.domain_a:>8} vs {r.domain_b:>8} — "
                      f"Jaccard={r.jaccard_similarity:.3f}, "
                      f"specific_A={r.n_domain_a_specific}, "
                      f"specific_B={r.n_domain_b_specific}")
    
    return {
        "pairwise_reports": pairwise_reports,
        "overview_reports": overview_reports,
        "domain_firing_rates": domain_firing_rates,
        "domains": sorted(domain_firing_rates.keys()),
    }
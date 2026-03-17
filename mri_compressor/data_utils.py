"""
Data loading utilities for the sparsity MRI experiments.
"""

import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import PreTrainedTokenizer
from typing import List, Optional


class TextDataset(Dataset):
    """Simple dataset that returns pre-tokenized chunks."""
    
    def __init__(self, input_ids: torch.Tensor):
        self.input_ids = input_ids
    
    def __len__(self):
        return self.input_ids.shape[0]
    
    def __getitem__(self, idx):
        return {"input_ids": self.input_ids[idx]}


def load_wikitext_data(
    tokenizer: PreTrainedTokenizer,
    split: str = "validation",  
    max_seq_len: int = 512,
    num_samples: int = 256,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-103-raw-v1",
) -> TextDataset:
    """
    Load and tokenize WikiText data into fixed-length chunks.
    Returns a TextDataset of shape (num_samples, max_seq_len).
    """
    print(f"Loading {dataset_name}/{dataset_config} ({split})...")
    
    # Try loading; fall back to wikitext-2 if 103 fails
    try:
        dataset = load_dataset(dataset_name, dataset_config, split=split)
    except Exception:
        print("  Falling back to wikitext-2-raw-v1")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    
    # Concatenate all text
    all_text = "\n".join([t for t in dataset["text"] if len(t.strip()) > 0])
    
    # Tokenize in one shot
    tokens = tokenizer.encode(all_text, return_tensors="pt")[0]
    print(f"  Total tokens: {len(tokens):,}")
    
    # Chunk into sequences of max_seq_len
    n_chunks = min(num_samples, len(tokens) // max_seq_len)
    chunks = tokens[:n_chunks * max_seq_len].reshape(n_chunks, max_seq_len)
    
    print(f"  Created {n_chunks} chunks of length {max_seq_len}")
    return TextDataset(chunks)


def get_dataloader(dataset: TextDataset, batch_size: int = 4, shuffle: bool = False) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def evaluate_perplexity(model, dataloader, device: str = "cuda", max_batches: int = None) -> float:
    """Compute perplexity on a dataloader."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    for i, batch in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break
        
        input_ids = batch["input_ids"].to(device)
        outputs = model(input_ids=input_ids, labels=input_ids)
        
        # outputs.loss is mean over all tokens
        seq_len = input_ids.shape[1] - 1  # shifted labels
        total_loss += outputs.loss.item() * seq_len * input_ids.shape[0]
        total_tokens += seq_len * input_ids.shape[0]
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return perplexity

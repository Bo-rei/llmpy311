import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import List, Dict


class FastDataset(Dataset):
    """Tokenize upfront and keep tensors in RAM for fast access."""

    def __init__(self, texts: List[str], tokenizer: AutoTokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Tokenize all texts immediately
        enc = tokenizer(texts, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        self.input_ids = enc['input_ids']
        self.attention_mask = enc['attention_mask']

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx]
        }

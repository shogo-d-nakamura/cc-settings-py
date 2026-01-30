---
name: data-pipeline
description: Patterns for molecular ML data pipelines (datasets, tokenization, collate, splitting, and loaders).
---

# Data Pipeline Patterns

Patterns for building robust data pipelines for molecular ML.

## Dataset Structure

### SMILES Dataset

```python
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from rdkit import Chem

@dataclass
class MoleculeRecord:
    smiles: str
    label: float | int | None = None

class SmilesDataset:
    """Dataset for SMILES + optional labels."""

    def __init__(self, csv_path: Path, smiles_col: str = "smiles", label_col: str | None = None):
        df = pd.read_csv(csv_path)
        self.smiles = df[smiles_col].astype(str).tolist()
        self.labels = df[label_col].tolist() if label_col else None

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int) -> MoleculeRecord:
        smi = self.smiles[idx]
        # Validate once at load time (or in collate)
        if Chem.MolFromSmiles(smi) is None:
            raise ValueError(f"Invalid SMILES at index {idx}: {smi}")
        label = self.labels[idx] if self.labels is not None else None
        return MoleculeRecord(smiles=smi, label=label)
```

## Tokenization and Collation

### Character-Level Tokenizer

```python
class SmilesTokenizer:
    """Simple character-level tokenizer."""

    def __init__(self, vocab: list[str], pad: str = "<pad>", bos: str = "<bos>", eos: str = "<eos>"):
        self.vocab = [pad, bos, eos] + vocab
        self.stoi = {t: i for i, t in enumerate(self.vocab)}
        self.itos = {i: t for t, i in self.stoi.items()}
        self.pad_id = self.stoi[pad]
        self.bos_id = self.stoi[bos]
        self.eos_id = self.stoi[eos]

    def encode(self, smiles: str) -> list[int]:
        ids = [self.bos_id] + [self.stoi[c] for c in smiles] + [self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        tokens = [self.itos[i] for i in ids]
        tokens = [t for t in tokens if t not in {"<pad>", "<bos>", "<eos>"}]
        return "".join(tokens)
```

### Collate Function with Padding

```python
import torch
from torch.nn.utils.rnn import pad_sequence

def collate_smiles(batch: list[MoleculeRecord], tokenizer: SmilesTokenizer) -> dict:
    ids = [torch.tensor(tokenizer.encode(b.smiles), dtype=torch.long) for b in batch]
    padded = pad_sequence(ids, batch_first=True, padding_value=tokenizer.pad_id)
    lengths = torch.tensor([len(x) for x in ids], dtype=torch.long)

    labels = None
    if batch[0].label is not None:
        labels = torch.tensor([b.label for b in batch])

    return {
        "input_ids": padded,
        "lengths": lengths,
        "labels": labels,
    }
```

## Splits

### Scaffold Split (Bemis-Murcko)

```python
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
import numpy as np

def scaffold_split(smiles_list: list[str], frac_train: float = 0.8, frac_val: float = 0.1, seed: int = 42):
    scaffolds = defaultdict(list)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        scaffolds[Chem.MolToSmiles(scaffold, canonical=True)].append(i)

    rng = np.random.default_rng(seed)
    scaffold_sets = list(scaffolds.values())
    rng.shuffle(scaffold_sets)

    n_total = len(smiles_list)
    n_train = int(frac_train * n_total)
    n_val = int(frac_val * n_total)

    train_idx, val_idx, test_idx = [], [], []
    for group in scaffold_sets:
        if len(train_idx) + len(group) <= n_train:
            train_idx.extend(group)
        elif len(val_idx) + len(group) <= n_val:
            val_idx.extend(group)
        else:
            test_idx.extend(group)

    return train_idx, val_idx, test_idx
```

## DataLoader Setup

```python
from torch.utils.data import DataLoader

def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int, tokenizer: SmilesTokenizer):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda batch: collate_smiles(batch, tokenizer),
    )
```

## Graph Pipelines (PyG)

```python
# If using PyTorch Geometric, create Data objects with atom/bond features
# and rely on torch_geometric.data.DataLoader for batching.
# Keep featurization deterministic and versioned.
```

## Data Quality Checks

- De-duplicate SMILES after canonicalization.
- Track invalid SMILES rate and log dropped indices.
- Avoid leakage: split before normalization or augmentation.
- Persist split indices for reproducibility.

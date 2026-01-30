---
name: python-standards
description: Python coding standards for ML/cheminformatics projects. Includes type hints, docstrings, file organization, and modern Python patterns.
---

# Python Coding Standards

Standards and best practices for Python code in ML and cheminformatics projects.

## Type Hints (REQUIRED)

### Function Signatures

```python
# GOOD: Full type hints
def encode_smiles(
    smiles: str,
    max_length: int = 100,
    padding: bool = True,
) -> torch.Tensor:
    ...

def process_molecules(
    smiles_list: list[str],
    n_jobs: int = -1,
) -> tuple[list[Chem.Mol], list[int]]:
    """Returns (valid_mols, failed_indices)."""
    ...

# BAD: No type hints
def encode_smiles(smiles, max_length=100, padding=True):
    ...
```

### Complex Types

```python
from typing import Callable, TypeVar, Generic
from collections.abc import Iterator, Sequence

# Type aliases
SMILES = str
Fingerprint = np.ndarray
MoleculeCallback = Callable[[Chem.Mol], float]

# Generic types
T = TypeVar("T")

class DataLoader(Generic[T]):
    def __iter__(self) -> Iterator[T]:
        ...

# Optional and Union
def load_model(path: str | Path | None = None) -> nn.Module:
    ...

# Callable with specific signature
def apply_transform(
    data: list[str],
    transform: Callable[[str], str],
) -> list[str]:
    ...
```

### Class Attributes

```python
class MoleculeEncoder(nn.Module):
    """Encoder for molecular SMILES."""

    vocab_size: int
    hidden_dim: int
    latent_dim: int

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        latent_dim: int = 64,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
```

## Docstrings (Google Style)

### Function Docstrings

```python
def compute_fingerprint(
    smiles: str,
    radius: int = 2,
    n_bits: int = 2048,
) -> np.ndarray:
    """Compute Morgan fingerprint for a molecule.

    Converts SMILES to RDKit mol object and computes circular
    fingerprint (Morgan/ECFP style).

    Args:
        smiles: SMILES string representing the molecule.
        radius: Radius for Morgan fingerprint. radius=2 gives ECFP4.
            Defaults to 2.
        n_bits: Number of bits in the fingerprint vector.
            Defaults to 2048.

    Returns:
        Binary fingerprint as numpy array of shape (n_bits,).

    Raises:
        ValueError: If SMILES is invalid or cannot be parsed.

    Example:
        >>> fp = compute_fingerprint("CCO", radius=2, n_bits=1024)
        >>> print(fp.shape)
        (1024,)

    Note:
        For FCFP (feature-based), use `use_features=True` in the
        underlying RDKit call.
    """
```

### Class Docstrings

```python
class MoleculeDataset(Dataset):
    """PyTorch Dataset for molecular SMILES data.

    Loads molecules from CSV file and provides tokenized sequences
    for training generative models.

    Attributes:
        data_path: Path to the CSV file.
        smiles_column: Name of the SMILES column.
        transform: Optional transform to apply to each sample.

    Example:
        >>> dataset = MoleculeDataset("data.csv", smiles_column="smiles")
        >>> print(len(dataset))
        1000
        >>> sample = dataset[0]
        >>> print(sample.keys())
        dict_keys(['input_ids', 'attention_mask'])
    """

    def __init__(
        self,
        data_path: str | Path,
        smiles_column: str = "smiles",
        transform: Callable | None = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            data_path: Path to CSV file containing SMILES.
            smiles_column: Name of column containing SMILES strings.
            transform: Optional transform applied to each sample.
        """
```

## Imports Organization

```python
# Standard library (alphabetical)
import json
import logging
from pathlib import Path
from typing import Any

# Third-party (alphabetical)
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import AllChem
from torch.utils.data import DataLoader, Dataset

# Local imports (alphabetical)
from src.data import MoleculeDataset
from src.models import MoleculeEncoder
from src.utils import set_seed

# Type-only imports (if needed)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.config import TrainingConfig
```

## Naming Conventions

```python
# Classes: PascalCase
class MoleculeEncoder:
    pass

class VAETrainer:
    pass

# Functions and methods: snake_case
def compute_fingerprint(smiles: str) -> np.ndarray:
    pass

def train_epoch(model: nn.Module, loader: DataLoader) -> float:
    pass

# Variables: snake_case
learning_rate = 1e-4
batch_size = 32
hidden_dim = 256

# Constants: UPPER_SNAKE_CASE
MAX_SMILES_LENGTH = 200
DEFAULT_FINGERPRINT_RADIUS = 2
SUPPORTED_FILE_FORMATS = [".csv", ".sdf", ".smi"]

# Private: leading underscore
def _validate_smiles(smiles: str) -> bool:
    pass

class Model:
    def _init_weights(self) -> None:
        pass
```

## File Organization

### Project Structure

```
project/
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── encoder.py       # MoleculeEncoder class
│   │   ├── decoder.py       # MoleculeDecoder class
│   │   └── vae.py           # MoleculeVAE class
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py       # Dataset classes
│   │   ├── transforms.py    # Data transforms
│   │   └── collate.py       # Collate functions
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py       # Trainer class
│   │   └── callbacks.py     # Training callbacks
│   └── utils/
│       ├── __init__.py
│       ├── chemistry.py     # RDKit utilities
│       └── reproducibility.py
├── tests/
│   ├── conftest.py
│   ├── test_models/
│   ├── test_data/
│   └── fixtures/
├── configs/
│   └── default.toml
└── scripts/
    ├── train.py
    └── evaluate.py
```

### Single File Guidelines

```python
# Max 400 lines per file
# One primary class per file
# Related helpers can be in same file

# encoder.py - GOOD
class MoleculeEncoder(nn.Module):
    ...

def _create_attention_mask(seq_len: int) -> torch.Tensor:
    """Helper function for encoder."""
    ...

# encoder.py - BAD (too many classes)
class MoleculeEncoder(nn.Module): ...
class MoleculeDecoder(nn.Module): ...
class MoleculeVAE(nn.Module): ...
class Trainer: ...
```

### __init__.py Exports

```python
# src/models/__init__.py
from src.models.encoder import MoleculeEncoder
from src.models.decoder import MoleculeDecoder
from src.models.vae import MoleculeVAE

__all__ = [
    "MoleculeEncoder",
    "MoleculeDecoder",
    "MoleculeVAE",
]
```

## Modern Python Patterns

### Dataclasses

```python
from dataclasses import dataclass, field

@dataclass
class TrainingConfig:
    """Configuration for training."""

    learning_rate: float = 1e-4
    batch_size: int = 32
    epochs: int = 100
    seed: int = 42
    device: str = "cuda"

    # Mutable default with field()
    hidden_dims: list[int] = field(default_factory=lambda: [256, 128])

@dataclass(frozen=True)
class MoleculeRecord:
    """Immutable molecule record."""

    smiles: str
    name: str
    properties: dict[str, float] = field(default_factory=dict)
```

### Context Managers

```python
from contextlib import contextmanager

@contextmanager
def evaluation_mode(model: nn.Module):
    """Temporarily set model to eval mode."""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        if was_training:
            model.train()

# Usage
with evaluation_mode(model):
    predictions = model(inputs)
```

### Path Handling

```python
from pathlib import Path

# GOOD: Use pathlib
config_path = Path("configs") / "experiment.toml"
output_dir = Path.home() / "results"
output_dir.mkdir(parents=True, exist_ok=True)

# Check existence
if config_path.exists():
    config = load_config(config_path)

# Read/write
text = config_path.read_text()
config_path.write_text(new_config)

# BAD: String concatenation
config_path = "configs" + "/" + "experiment.toml"
```

### Error Handling

```python
# Specific exceptions
class InvalidSMILESError(ValueError):
    """Raised when SMILES string is invalid."""
    pass

class ModelNotTrainedError(RuntimeError):
    """Raised when using untrained model for inference."""
    pass

# Usage
def validate_smiles(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise InvalidSMILESError(f"Cannot parse SMILES: {smiles}")
    return mol

# Catching specific exceptions
try:
    mol = validate_smiles(user_input)
except InvalidSMILESError as e:
    logger.warning(f"Invalid input: {e}")
    return None
```

## Code Quality Tools

### ruff Configuration

```toml
# pyproject.toml
[tool.ruff]
target-version = "py310"
line-length = 100

[tool.ruff.lint]
select = [
    "E",    # pycodestyle errors
    "W",    # pycodestyle warnings
    "F",    # pyflakes
    "I",    # isort
    "B",    # flake8-bugbear
    "C4",   # flake8-comprehensions
    "UP",   # pyupgrade
]
ignore = [
    "E501",  # line too long (handled by formatter)
]

[tool.ruff.lint.isort]
known-first-party = ["src"]
```

### mypy Configuration

```toml
# pyproject.toml
[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
ignore_missing_imports = true

[[tool.mypy.overrides]]
module = "tests.*"
ignore_errors = true
```

### Running Tools

```bash
# Format and lint
ruff check --fix src/
ruff format src/

# Type check
mypy src/

# All checks
ruff check src/ && mypy src/ && pytest tests/
```

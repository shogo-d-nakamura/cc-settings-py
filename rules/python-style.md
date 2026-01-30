# Python Style Guide

Rules for Python code style in ML/cheminformatics projects.

## Type Hints (REQUIRED)

All functions must have type hints:

```python
# GOOD
def encode_smiles(smiles: str, max_len: int = 100) -> torch.Tensor:
    ...

# BAD
def encode_smiles(smiles, max_len=100):
    ...
```

## Docstrings (Google Style)

All public functions and classes must have docstrings:

```python
def compute_fingerprint(smiles: str, radius: int = 2) -> np.ndarray:
    """Compute Morgan fingerprint for a molecule.

    Args:
        smiles: SMILES string.
        radius: Fingerprint radius.

    Returns:
        Binary fingerprint array.

    Raises:
        ValueError: If SMILES is invalid.
    """
```

## Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Classes | PascalCase | `MoleculeEncoder` |
| Functions | snake_case | `compute_fingerprint` |
| Variables | snake_case | `learning_rate` |
| Constants | UPPER_SNAKE | `MAX_LENGTH` |
| Private | leading `_` | `_validate_input` |

## File Organization

- **Max 400 lines per file**
- **One primary class per file**
- **Group imports**: stdlib → third-party → local

```python
# Standard library
import json
from pathlib import Path

# Third-party
import torch
import numpy as np
from rdkit import Chem

# Local
from src.models import Encoder
from src.utils import set_seed
```

## Code Patterns

### No Mutable Defaults

```python
# GOOD
def process(items: list[str] | None = None):
    items = items or []

# BAD
def process(items: list[str] = []):  # Mutable default!
```

### Early Returns

```python
# GOOD
def validate(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid: {smiles}")
    return mol

# BAD
def validate(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return mol
    else:
        raise ValueError(f"Invalid: {smiles}")
```

### Context Managers

```python
# GOOD
with open(path) as f:
    data = f.read()

# BAD
f = open(path)
data = f.read()
f.close()
```

### F-strings

```python
# GOOD
name = f"model_{version}_epoch_{epoch}"

# BAD
name = "model_%s_epoch_%d" % (version, epoch)
name = "model_{}_epoch_{}".format(version, epoch)
```

## Error Handling

```python
# Use specific exceptions
class InvalidSMILESError(ValueError):
    pass

def validate(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise InvalidSMILESError(f"Cannot parse: {smiles}")
    return mol
```

## Code Quality Tools

```bash
# Format and lint
ruff check --fix src/
ruff format src/

# Type check
mypy src/
```

## Anti-Patterns to Avoid

- Magic numbers (use named constants)
- Deep nesting (>4 levels)
- Long functions (>50 lines)
- Duplicate code
- Commented-out code
- `print()` in production (use logging)

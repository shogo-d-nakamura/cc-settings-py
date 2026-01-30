---
description: Update documentation - README, docstrings, and API docs.
---

# Update Docs Command

Invokes the **doc-updater** agent to update project documentation.

## What This Command Does

1. **Update README** - Installation, usage, examples
2. **Update Docstrings** - Function and class documentation
3. **Generate API Docs** - Automated documentation
4. **Document Configs** - Configuration file documentation

## When to Use

Use `/update-docs` when:
- Adding new features
- Changing API signatures
- Updating installation steps
- Documenting configuration options

## Documentation Types

### README.md

```markdown
# Project Name

## Installation
```bash
conda env create -f environment.yml
pip install -e .
```

## Quick Start
```python
from myproject import Model
model = Model.load("checkpoint.pt")
results = model.generate(100)
```
```

### Docstrings (Google Style)

```python
def generate_molecules(
    model: nn.Module,
    num_samples: int,
    temperature: float = 1.0,
) -> list[str]:
    """Generate molecules using the trained model.

    Args:
        model: Trained generative model.
        num_samples: Number of molecules to generate.
        temperature: Sampling temperature.

    Returns:
        List of generated SMILES strings.

    Example:
        >>> molecules = generate_molecules(model, 100)
        >>> print(molecules[0])
        'CCO'
    """
```

## Example Usage

```
User: /update-docs src/models/vae.py

Agent (doc-updater):
## Documentation Update: src/models/vae.py

### Missing Docstrings Found
- `MoleculeVAE.__init__` - No docstring
- `MoleculeVAE.encode` - No docstring
- `MoleculeVAE.sample` - Incomplete docstring

### Updated Documentation

```python
class MoleculeVAE(nn.Module):
    """Variational Autoencoder for molecular SMILES.

    Encodes SMILES strings into a latent space and decodes
    samples to generate new molecules.

    Attributes:
        encoder: LSTM encoder network.
        decoder: LSTM decoder network.
        latent_dim: Dimension of latent space.

    Example:
        >>> model = MoleculeVAE(vocab_size=100, latent_dim=64)
        >>> molecules = model.sample(100)
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        latent_dim: int = 64,
    ) -> None:
        """Initialize the VAE.

        Args:
            vocab_size: Size of SMILES vocabulary.
            hidden_dim: Hidden dimension for LSTM layers.
            latent_dim: Dimension of latent space.
        """
```
```

## Documentation Commands

```bash
# Generate API docs with pdoc
pdoc --html src/ --output-dir docs/

# Generate with sphinx
sphinx-apidoc -o docs/source src/
cd docs && make html
```

## Related Agent

This command invokes the `doc-updater` agent at `agents/doc-updater.md`.

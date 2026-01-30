# CLAUDE.md - Cheminformatics & ML Research Configuration

Claude Code settings optimized for PyTorch-based deep learning, cheminformatics, and molecular generation research.

## Project Overview

This configuration provides agents, skills, commands, rules, and hooks for:
- **Deep Learning**: PyTorch model development, training, and inference
- **Cheminformatics**: RDKit-based molecular processing, SMILES handling, fingerprints
- **Molecular Generation**: VAE, GAN, diffusion, flow-matching, autoregressive, RL-based, GNN models
- **Bioinformatics**: Protein-ligand prediction, AlphaFold/Boltz integration, docking

## Core Frameworks

| Framework | Purpose |
|-----------|---------|
| PyTorch | Deep learning models and training |
| RDKit | Cheminformatics and molecular processing |
| Pydantic v2 | Configuration validation |
| pytest | Testing framework |
| TensorBoard | Experiment logging |

## Commands

### Environment
```bash
# Create/activate environment
conda env create -f environment.yml
conda activate ml-research
pip install -e .

# Check environment
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import rdkit; print(f'RDKit: {rdkit.__version__}')"
```

### Testing
```bash
# Run tests
pytest tests/ -v
pytest tests/ --cov=src --cov-report=html

# Type checking
mypy src/

# Linting
ruff check src/
ruff check --fix src/
```

### Training
```bash
# Run training
python train.py --config configs/experiment.toml

# With specific GPU
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/experiment.toml

# Resume from checkpoint
python train.py --config configs/experiment.toml --resume checkpoints/best.pt
```

### Inference/Sampling
```bash
# Generate molecules
python sample.py --config configs/sampling.toml --output results/generated.csv

# Evaluate model
python evaluate.py --config configs/eval.toml --checkpoint checkpoints/best.pt
```

## Architecture

### Directory Structure
```
project/
├── configs/              # TOML/YAML experiment configurations
├── src/
│   ├── models/          # PyTorch model definitions
│   ├── data/            # Dataset classes and transforms
│   ├── training/        # Training loops and utilities
│   ├── evaluation/      # Metrics and evaluation
│   └── utils/           # Helper functions
├── scripts/             # Entry point scripts
├── tests/               # pytest test suite
├── checkpoints/         # Model checkpoints
├── logs/                # TensorBoard logs
└── results/             # Experiment outputs
```

### Configuration Format (TOML preferred)
```toml
[experiment]
name = "molecule_generation"
seed = 42

[model]
type = "vae"
hidden_dim = 256
latent_dim = 64
num_layers = 4

[training]
epochs = 100
batch_size = 32
learning_rate = 1e-4
gradient_clip = 1.0

[data]
train_path = "data/train.csv"
val_path = "data/val.csv"
smiles_column = "smiles"
```

## Critical Constraints

### Reproducibility (MANDATORY)
```python
import random
import numpy as np
import torch

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

- Always set random seeds at script start
- Log all hyperparameters to TensorBoard/W&B
- Version control all configuration files
- Save environment.yml with each experiment

### GPU Management
```python
# Check device availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Clear cache between runs
torch.cuda.empty_cache()

# Monitor memory
print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
```

### SMILES Validation
```python
from rdkit import Chem

def validate_smiles(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

def canonicalize_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)
```

### Type Hints (REQUIRED)
```python
def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> float:
    """Train model for one epoch.

    Args:
        model: PyTorch model to train.
        loader: DataLoader with training data.
        optimizer: Optimizer instance.
        device: Device to use (cuda/cpu).

    Returns:
        Average loss for the epoch.
    """
```

## Commit Format

Use conventional commits:
- `feat:` New feature or model
- `fix:` Bug fix
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `docs:` Documentation changes
- `chore:` Maintenance tasks
- `experiment:` Experiment-related changes

Example:
```
feat: add VAE encoder with attention mechanism

- Implement self-attention in encoder
- Add skip connections between layers
- Support variable-length SMILES input
```

## Agent Usage

Use specialized agents for complex tasks:

| Agent | When to Use |
|-------|-------------|
| planner | Complex experiments, multi-step implementations |
| code-reviewer | After writing/modifying code |
| ml-trainer | Training workflow setup |
| cheminformatics-reviewer | Molecular data processing code |
| architect | Architecture decisions |
| tdd-guide | Writing tests first |

## Testing Requirements

- **Minimum coverage**: 80%
- **Critical code**: 100% (model forward pass, data processing)
- Use pytest fixtures for shared resources
- Use parametrize for multiple test cases
- Mark slow tests with `@pytest.mark.slow`

## File Organization

- Max 400 lines per file
- One model class per file
- Group imports: stdlib, third-party, local
- Use `__all__` for public APIs

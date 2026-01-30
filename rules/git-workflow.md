# Git Workflow

Rules for version control in ML projects.

## Commit Format

Use conventional commits:

```
<type>: <description>

[optional body]

[optional footer]
```

### Types

| Type | Description |
|------|-------------|
| `feat` | New feature or model |
| `fix` | Bug fix |
| `refactor` | Code refactoring |
| `test` | Adding or updating tests |
| `docs` | Documentation changes |
| `chore` | Maintenance tasks |
| `experiment` | Experiment-related changes |

### Examples

```bash
feat: add VAE encoder with attention mechanism

- Implement self-attention in encoder
- Add skip connections between layers
- Support variable-length SMILES input
```

```bash
fix: resolve CUDA memory leak in training loop

Clear GPU cache after each epoch to prevent OOM errors
on long training runs.

Fixes #123
```

```bash
experiment: try larger latent dimension

latent_dim: 64 -> 128
Results: validity +2%, diversity -1%
```

## Branch Strategy

```
main
  └── feature/vae-encoder
  └── experiment/large-latent
  └── fix/memory-leak
```

### Branch Naming

- `feature/` - New features
- `experiment/` - ML experiments
- `fix/` - Bug fixes
- `refactor/` - Code improvements

## What to Commit

### DO Commit
- Source code
- Configuration files (TOML/YAML)
- Test files
- Documentation
- Requirements/environment files

### DON'T Commit
- Model checkpoints (>100MB)
- Training data
- Generated outputs
- Credentials/API keys
- `__pycache__/`, `.pyc` files

## .gitignore

```gitignore
# Python
__pycache__/
*.py[cod]
.mypy_cache/

# ML artifacts
checkpoints/
logs/
results/
*.pt
*.ckpt

# Data
data/*.csv
data/*.sdf

# Environment
.env
*.egg-info/

# IDE
.vscode/
.idea/
```

## Pre-commit Checks

Before committing:

1. Run tests: `pytest tests/`
2. Run linter: `ruff check src/`
3. Type check: `mypy src/`

```bash
# Quick verify
pytest tests/ && ruff check src/ && mypy src/
```

## Large Files

Use Git LFS for:
- Model checkpoints
- Large datasets
- Binary files

```bash
git lfs track "*.pt"
git lfs track "data/*.csv"
```

## Experiment Tracking

For experiments, include in commit message:
- Configuration used
- Key metrics
- Comparison to baseline

```bash
experiment: train VAE with cosine scheduler

Config: configs/vae_cosine.toml
Results:
- val_loss: 0.123 (baseline: 0.145)
- validity: 0.92 (baseline: 0.88)
```

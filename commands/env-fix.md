---
description: Fix conda/pip environment issues, import errors, and dependency conflicts.
---

# Environment Fix Command

Invokes the **build-error-resolver** agent to diagnose and fix Python environment issues.

## What This Command Does

1. **Diagnose Errors** - Identify root cause of import/build failures
2. **Check Environment** - Verify correct Python/conda environment
3. **Resolve Conflicts** - Fix dependency version conflicts
4. **CUDA/GPU Issues** - Fix PyTorch CUDA compatibility

## When to Use

Use `/env-fix` when encountering:
- `ModuleNotFoundError`
- `ImportError`
- CUDA version mismatch
- Dependency conflicts
- RDKit installation issues

## Common Fixes

### Import Errors

```bash
# Check package installed
pip show torch
conda list | grep torch

# Reinstall
pip install --force-reinstall torch
```

### CUDA Issues

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Check versions
nvcc --version
python -c "import torch; print(torch.version.cuda)"

# Reinstall with correct CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### RDKit Issues

```bash
# Recommended: conda
conda install -c conda-forge rdkit

# Alternative: pip
pip install rdkit-pypi
```

### Dependency Conflicts

```bash
# Check conflicts
pip check

# Create fresh environment
conda create -n ml-research python=3.10
conda activate ml-research

# Install in order
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c conda-forge rdkit
pip install -r requirements.txt
```

## Example Usage

```
User: /env-fix ImportError: cannot import name 'Chem' from 'rdkit'

Agent (build-error-resolver):
## Diagnosis

Checking RDKit installation...

```bash
python -c "from rdkit import Chem"
# ImportError: No module named 'rdkit'
```

RDKit is not installed in current environment.

## Solution

```bash
# Check current environment
conda info --envs

# Install RDKit via conda (recommended)
conda install -c conda-forge rdkit

# Verify installation
python -c "from rdkit import Chem; print(Chem.__file__)"
```

## Verification

```bash
python -c "from rdkit import Chem; print('Success!')"
# Success!
```
```

## Related Agent

This command invokes the `build-error-resolver` agent at `agents/build-error-resolver.md`.

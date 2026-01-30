---
description: Comprehensive ML/Python code review focusing on correctness, PyTorch patterns, and reproducibility.
---

# Code Review Command

Invokes the **code-reviewer** agent for ML/Python code quality review.

## What This Command Does

1. **Python Quality** - Type hints, docstrings, error handling
2. **PyTorch Patterns** - Device management, gradients, memory
3. **ML-Specific** - Reproducibility, data leakage, train/eval modes
4. **Cheminformatics** - SMILES validation, RDKit patterns
5. **Security** - No credentials, safe loading

## When to Use

Use `/code-review` after:
- Writing new code
- Modifying existing code
- Before committing changes
- Before merging PRs

## Review Checklist

### Python
- [ ] Type hints on all functions
- [ ] Docstrings (Google style)
- [ ] No mutable default arguments
- [ ] Proper exception handling

### PyTorch
- [ ] Device management correct
- [ ] `torch.no_grad()` for inference
- [ ] Gradient clipping for RNNs/Transformers
- [ ] Memory management (del tensors, empty_cache)

### ML
- [ ] Random seeds set
- [ ] No data leakage
- [ ] `model.train()`/`model.eval()` correct
- [ ] Checkpoints save optimizer state

### Cheminformatics
- [ ] SMILES validation present
- [ ] Invalid molecules handled gracefully
- [ ] Fingerprint computation efficient

## Example Usage

```
User: /code-review src/models/vae.py

Agent (code-reviewer):
## Code Review: src/models/vae.py

### Critical Issues
- **Line 45**: Missing `torch.no_grad()` in sampling method
  - Impact: Unnecessary gradient computation during inference
  - Fix: Wrap with `@torch.no_grad()` decorator

- **Line 78**: No gradient clipping in training step
  - Impact: Potential gradient explosion
  - Fix: Add `torch.nn.utils.clip_grad_norm_()`

### Warnings
- **Line 23**: Type hint missing for return value
  - Recommendation: Add `-> tuple[torch.Tensor, torch.Tensor, torch.Tensor]`

### Positive Notes
- Good use of reparameterization trick
- Clean separation of encode/decode methods

### Checklist
- [x] Type hints present (mostly)
- [x] Docstrings complete
- [ ] Gradient clipping - MISSING
- [x] Device management correct
```

## Related Commands

- `/pytest` - Write tests before coding
- `/plan` - Plan before implementing
- `/refactor-clean` - Clean up code after review

## Related Agent

This command invokes the `code-reviewer` agent at `agents/code-reviewer.md`.

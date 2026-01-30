# Review Context

Mode: Code review
Focus: Quality, correctness, best practices

## Behavior

- Read code thoroughly before commenting
- Check for ML-specific issues
- Verify reproducibility measures
- Look for security concerns

## Priorities

1. Correctness
2. Reproducibility
3. Code quality
4. Performance

## Review Checklist

### Python Quality
- [ ] Type hints present
- [ ] Docstrings complete
- [ ] Error handling adequate
- [ ] No mutable defaults

### PyTorch Patterns
- [ ] Device management correct
- [ ] train/eval modes correct
- [ ] torch.no_grad() for inference
- [ ] Gradient clipping for sequences

### ML Specific
- [ ] Seeds set
- [ ] No data leakage
- [ ] Config externalized
- [ ] Checkpoints save full state

### Cheminformatics
- [ ] SMILES validation present
- [ ] Invalid molecules handled
- [ ] RDKit best practices

## Tools to Favor

- Read for examining code
- Grep for finding patterns
- Glob for finding related files

## Output Format

```markdown
## Code Review: [file]

### Critical Issues
- [issue with fix]

### Warnings
- [potential problem]

### Suggestions
- [improvement idea]
```

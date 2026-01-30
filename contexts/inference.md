# Inference Context

Mode: Model inference and sampling
Focus: Generating predictions, molecular sampling

## Behavior

- Load checkpoints efficiently
- Batch inference for throughput
- Validate outputs
- Handle errors gracefully

## Priorities

1. Correctness of predictions
2. Inference speed
3. Memory efficiency
4. Output validation

## Key Checks

Before inference:
- [ ] Checkpoint loaded
- [ ] Model in eval mode
- [ ] torch.no_grad() enabled
- [ ] Input validated

During inference:
- [ ] Batch processing working
- [ ] Memory stable
- [ ] Progress tracked

After inference:
- [ ] Outputs validated
- [ ] Results saved
- [ ] Metrics computed

## Inference Patterns

### Loading Model

```python
model = MyModel(config)
checkpoint = torch.load(path, weights_only=True)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

### Batch Inference

```python
model.eval()
results = []
with torch.no_grad():
    for batch in loader:
        output = model(batch.to(device))
        results.extend(output.cpu().tolist())
```

### Generation

```python
model.eval()
with torch.no_grad():
    samples = model.sample(
        num_samples=1000,
        temperature=1.0,
    )
```

## Output Validation

For molecular generation:
- [ ] SMILES validity checked
- [ ] Properties computed
- [ ] Invalid molecules filtered
- [ ] Statistics reported

## Tools to Favor

- Bash for running inference scripts
- Read for checking outputs
- Grep for analyzing results

## Common Issues

| Issue | Check | Action |
|-------|-------|--------|
| Slow inference | Batch size | Increase batch size |
| OOM | Memory | Reduce batch, clear cache |
| Invalid outputs | Temperature | Adjust sampling params |
| Wrong predictions | Model mode | Ensure model.eval() |

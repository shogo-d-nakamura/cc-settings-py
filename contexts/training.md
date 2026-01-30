# Training Context

Mode: Active model training
Focus: Training loops, optimization, monitoring

## Behavior

- Monitor loss and metrics closely
- Check for gradient issues (NaN, explosion, vanishing)
- Save checkpoints regularly
- Watch for overfitting (val loss increasing)
- Track GPU memory usage

## Priorities

1. Training stability
2. Metric improvement
3. Resource efficiency
4. Reproducibility

## Tools to Favor

- Bash for running training scripts
- Read for checking logs and outputs
- Grep for searching error messages

## Key Checks

Before training:
- [ ] Config validated
- [ ] Seed set
- [ ] GPU available
- [ ] Logging configured

During training:
- [ ] Loss decreasing
- [ ] Gradients normal
- [ ] Memory stable
- [ ] Checkpoints saving

After training:
- [ ] Best model saved
- [ ] Metrics logged
- [ ] Results documented

## Common Issues

| Issue | Check | Action |
|-------|-------|--------|
| Loss NaN | Gradient norm | Reduce LR, add clipping |
| OOM | Batch size | Reduce batch, use accumulation |
| No improvement | Learning rate | Adjust LR schedule |
| Overfitting | Val loss | Add dropout, early stopping |

---
description: Manage model checkpoints - save, load, and organize training states.
---

# Checkpoint Command

Manage model checkpoints and training states.

## What This Command Does

1. **Save Checkpoints** - Model, optimizer, scheduler states
2. **Load Checkpoints** - Resume training or inference
3. **Organize** - Keep best K checkpoints
4. **Convert** - Export for inference-only

## When to Use

Use `/checkpoint` when:
- Resuming interrupted training
- Loading models for inference
- Managing checkpoint storage
- Converting checkpoints for deployment

## Checkpoint Contents

```python
checkpoint = {
    "epoch": 50,
    "global_step": 15000,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "config": config.model_dump(),
    "metrics": {"val_loss": 0.123, "validity": 0.92},
    "best_metric": 0.123,
}
```

## Example Usage

```
User: /checkpoint save current training state

Agent:
## Checkpoint Management

### Saving Checkpoint
```python
checkpoint = {
    "epoch": 50,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "config": config,
    "metrics": {"val_loss": 0.123},
}

torch.save(checkpoint, "checkpoints/epoch_50.pt")
```

### Current Checkpoints
| File | Epoch | Val Loss | Size |
|------|-------|----------|------|
| best.pt | 45 | 0.112 | 128MB |
| epoch_50.pt | 50 | 0.123 | 128MB |
| epoch_40.pt | 40 | 0.134 | 128MB |
```

## Operations

### Save Checkpoint

```python
def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    path: Path,
) -> None:
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)
```

### Load Checkpoint

```python
def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    path: Path,
    device: torch.device,
) -> int:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"]
```

### Export for Inference

```python
# Remove optimizer state for smaller file
inference_checkpoint = {
    "model_state_dict": model.state_dict(),
    "config": config,
}
torch.save(inference_checkpoint, "model_inference.pt")
```

### Load Inference-Only

```python
checkpoint = torch.load("model_inference.pt", weights_only=True)
model.load_state_dict(checkpoint["model_state_dict"])
```

## Checkpoint Organization

```
checkpoints/
├── best.pt              # Best validation metric
├── last.pt              # Most recent
├── epoch_100.pt         # Periodic saves
├── epoch_090.pt
└── epoch_080.pt
```

### Keep Best K

```python
class CheckpointManager:
    def __init__(self, max_checkpoints: int = 5):
        self.max_checkpoints = max_checkpoints
        self.checkpoints = []  # (metric, path)

    def save_if_best(self, metric: float, checkpoint, path: Path):
        # Save if in top K
        if len(self.checkpoints) < self.max_checkpoints:
            torch.save(checkpoint, path)
            self.checkpoints.append((metric, path))
            self.checkpoints.sort()
        elif metric < self.checkpoints[-1][0]:
            # Remove worst, add new
            _, worst_path = self.checkpoints.pop()
            worst_path.unlink()
            torch.save(checkpoint, path)
            self.checkpoints.append((metric, path))
            self.checkpoints.sort()
```

## Related Commands

- `/train` - Training creates checkpoints
- `/eval` - Load checkpoint for evaluation

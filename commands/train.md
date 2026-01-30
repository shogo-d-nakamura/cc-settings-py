---
description: Set up and run training experiments with proper configuration, logging, and checkpointing.
---

# Train Command

Invokes the **ml-trainer** agent to set up and monitor training experiments.

## What This Command Does

1. **Validate Config** - Check configuration file
2. **Check Resources** - Verify GPU availability
3. **Set Up Logging** - TensorBoard/W&B configuration
4. **Initialize Checkpoints** - Create checkpoint directory
5. **Run Training** - Execute with monitoring

## When to Use

Use `/train` when:
- Starting a new training run
- Resuming from checkpoint
- Setting up distributed training
- Debugging training issues

## Training Checklist

- [ ] Configuration file validated
- [ ] Random seed set
- [ ] GPU available (if needed)
- [ ] Logging configured
- [ ] Checkpoint directory created
- [ ] Early stopping criteria defined

## Example Usage

```
User: /train --config configs/vae.toml

Agent (ml-trainer):
## Training Setup

### 1. Configuration Validation
```bash
python -c "from src.config import load_config; load_config('configs/vae.toml')"
# Config valid ✓
```

### 2. GPU Check
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
# CUDA: True ✓
# GPU: NVIDIA A100 ✓
```

### 3. Logging Setup
```bash
mkdir -p logs/vae_experiment
tensorboard --logdir logs/ --port 6006 &
```

### 4. Checkpoint Directory
```bash
mkdir -p checkpoints/vae_experiment
```

### 5. Training Command
```bash
python train.py \
    --config configs/vae.toml \
    --output-dir results/vae_experiment \
    --seed 42
```

### Monitoring
- TensorBoard: http://localhost:6006
- Checkpoints: checkpoints/vae_experiment/
```

## Configuration Template

```toml
[experiment]
name = "vae_training"
seed = 42

[model]
type = "vae"
hidden_dim = 256

[training]
epochs = 100
batch_size = 32
learning_rate = 1e-4

[data]
train_path = "data/train.csv"
val_path = "data/val.csv"
```

## Training Patterns

### Resume from Checkpoint

```bash
python train.py --config configs/vae.toml --resume checkpoints/best.pt
```

### Distributed Training

```bash
torchrun --nproc_per_node=4 train.py --config configs/vae.toml
```

### Mixed Precision

```bash
python train.py --config configs/vae.toml --mixed-precision
```

## Related Commands

- `/plan` - Plan experiment before training
- `/eval` - Evaluate trained model
- `/checkpoint` - Manage checkpoints

## Related Agent

This command invokes the `ml-trainer` agent at `agents/ml-trainer.md`.

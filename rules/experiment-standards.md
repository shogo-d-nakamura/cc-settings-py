# Experiment Standards

Rules for reproducible ML experiments.

## Configuration (MANDATORY)

Every experiment MUST have a configuration file:

```toml
# configs/experiment.toml
[experiment]
name = "my_experiment"
seed = 42

[model]
type = "vae"
hidden_dim = 256

[training]
epochs = 100
learning_rate = 1e-4

[data]
train_path = "data/train.csv"
val_path = "data/val.csv"
```

**Never hardcode hyperparameters in code.**

## Logging (MANDATORY)

Every experiment MUST log:

1. **Configuration** - Full config file
2. **Environment** - Python/PyTorch versions, CUDA
3. **Metrics** - Training and validation metrics per epoch
4. **Git info** - Commit hash, branch, dirty status

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir=f"logs/{experiment_name}")
writer.add_text("config", config_json)
writer.add_scalar("train/loss", loss, step)
```

## Checkpointing (MANDATORY)

Save complete training state:

```python
checkpoint = {
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "config": config,
    "metrics": metrics,
}
torch.save(checkpoint, path)
```

Keep at minimum:
- Best model (by validation metric)
- Last checkpoint
- Periodic checkpoints every N epochs

## Reproducibility Checklist

Before running:
- [ ] Random seed set in config
- [ ] Config file saved to output directory
- [ ] Environment exported (requirements.txt)
- [ ] Git commit hash recorded
- [ ] CUDA deterministic mode enabled

After running:
- [ ] All metrics logged
- [ ] Best checkpoint saved
- [ ] Results can be reproduced with saved config

## Directory Structure

```
results/
└── experiment_name_YYYYMMDD_HHMMSS/
    ├── config.toml           # Configuration used
    ├── environment.txt       # pip freeze output
    ├── git_info.json        # Commit hash, branch
    ├── checkpoints/
    │   ├── best.pt
    │   ├── last.pt
    │   └── epoch_050.pt
    ├── logs/                # TensorBoard logs
    ├── metrics.json         # Final metrics
    └── generated/           # Generated outputs
```

## Experiment Naming

```
{model_type}_{key_param}_{date}

Examples:
vae_latent64_20240115
transformer_layers6_20240115
```

## Results Tracking

Track experiments in a central file:

```jsonl
{"name": "vae_baseline", "val_loss": 0.145, "validity": 0.88, "date": "2024-01-15"}
{"name": "vae_larger", "val_loss": 0.123, "validity": 0.92, "date": "2024-01-16"}
```

## Comparison Protocol

When comparing models:

1. **Same data splits** - Use fixed seed for splits
2. **Same evaluation** - Identical metrics computation
3. **Multiple seeds** - Run at least 3 seeds, report mean±std
4. **Statistical tests** - Use paired tests for significance

```python
# Report format
Results (mean±std over 3 seeds):
- Validity: 0.92±0.01
- Uniqueness: 0.85±0.02
- Diversity: 0.78±0.03
```

## Version Control for Experiments

1. **Commit config changes** before running
2. **Tag significant experiments**
3. **Don't commit large checkpoints** (use Git LFS or external storage)

```bash
git tag -a "exp-vae-v1" -m "VAE baseline experiment"
```

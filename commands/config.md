---
description: Generate and validate TOML/YAML configuration files for experiments.
---

# Config Command

Generate and validate experiment configuration files.

## What This Command Does

1. **Generate Config** - Create configuration from template
2. **Validate Schema** - Check against Pydantic schema
3. **Set Hyperparameters** - Configure model and training
4. **Save to File** - Write TOML/YAML configuration

## When to Use

Use `/config` when:
- Starting a new experiment
- Creating configuration variants
- Validating existing configs
- Setting up hyperparameter sweeps

## Configuration Templates

### Basic Experiment

```toml
[experiment]
name = "my_experiment"
seed = 42
output_dir = "results/my_experiment"

[model]
type = "vae"
hidden_dim = 256
latent_dim = 64
num_layers = 4
dropout = 0.1

[training]
epochs = 100
batch_size = 32
learning_rate = 1e-4
weight_decay = 1e-5
gradient_clip = 1.0
patience = 10

[data]
train_path = "data/train.csv"
val_path = "data/val.csv"
smiles_column = "smiles"
max_length = 200
```

### With Scheduler

```toml
[training.scheduler]
type = "cosine"
warmup_steps = 1000
min_lr = 1e-6
```

### With Data Augmentation

```toml
[data.augmentation]
smiles_randomization = true
num_augmentations = 5
```

## Example Usage

```
User: /config create vae experiment with hidden_dim=512

Agent:
## Configuration Generated

```toml
# configs/vae_large.toml

[experiment]
name = "vae_large"
seed = 42
output_dir = "results/vae_large"

[model]
type = "vae"
hidden_dim = 512
latent_dim = 128
num_layers = 6
dropout = 0.1

[training]
epochs = 100
batch_size = 32
learning_rate = 1e-4
gradient_clip = 1.0

[data]
train_path = "data/train.csv"
val_path = "data/val.csv"
```

### Validation
```bash
python -c "from src.config import ExperimentConfig; ExperimentConfig.load('configs/vae_large.toml')"
# Configuration valid ✓
```
```

## Config Validation

```python
from pydantic import BaseModel, Field

class ModelConfig(BaseModel):
    type: str
    hidden_dim: int = Field(ge=32, le=2048)
    latent_dim: int = Field(ge=8, le=512)

# Validate
config = ModelConfig(**toml_data["model"])
```

## CLI Overrides

```bash
python train.py --config configs/base.toml \
    --override model.hidden_dim=512 \
    --override training.learning_rate=1e-5
```

## Grid Search Config

```toml
# configs/grid_search.toml
[experiment]
name = "grid_search"

# Grid parameters
grid_model.hidden_dim = [128, 256, 512]
grid_training.learning_rate = [1e-3, 1e-4, 1e-5]
```

## Related Commands

- `/train` - Run training with config
- `/plan` - Plan experiment before configuring

## Related Skill

See `skills/config-management/SKILL.md` for detailed patterns.

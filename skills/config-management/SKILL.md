---
name: config-management
description: Patterns for TOML/YAML configuration files, schema validation, and configuration inheritance.
---

# Configuration Management

Patterns for managing experiment configurations using TOML/YAML.

## TOML Configuration

### Basic Structure

```toml
# configs/experiment.toml

[experiment]
name = "molecule_vae"
seed = 42
output_dir = "results/vae_experiment"

[model]
type = "vae"
hidden_dim = 256
latent_dim = 64
num_layers = 4
dropout = 0.1

[model.encoder]
type = "transformer"
num_heads = 8
feedforward_dim = 1024

[model.decoder]
type = "transformer"
num_heads = 8

[training]
epochs = 100
batch_size = 32
learning_rate = 1e-4
weight_decay = 1e-5
gradient_clip = 1.0
patience = 10

[training.scheduler]
type = "cosine"
warmup_steps = 1000
min_lr = 1e-6

[data]
train_path = "data/train.csv"
val_path = "data/val.csv"
test_path = "data/test.csv"
smiles_column = "smiles"
max_length = 200

[data.augmentation]
smiles_randomization = true
num_augmentations = 5

[logging]
log_dir = "logs"
log_every = 100
save_every = 5
```

### Loading TOML

```python
import tomllib
from pathlib import Path
from typing import Any

def load_toml(path: Path) -> dict[str, Any]:
    """Load TOML configuration file."""
    with open(path, "rb") as f:
        return tomllib.load(f)

def load_config_with_defaults(
    config_path: Path,
    defaults_path: Path | None = None,
) -> dict[str, Any]:
    """Load config with optional defaults."""
    config = {}

    if defaults_path and defaults_path.exists():
        config = load_toml(defaults_path)

    user_config = load_toml(config_path)
    deep_merge(config, user_config)

    return config

def deep_merge(base: dict, override: dict) -> None:
    """Deep merge override into base."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
```

### Saving TOML

```python
import tomli_w

def save_toml(config: dict, path: Path) -> None:
    """Save configuration to TOML file."""
    with open(path, "wb") as f:
        tomli_w.dump(config, f)
```

## YAML Configuration

### Basic Structure

```yaml
# configs/experiment.yaml

experiment:
  name: molecule_vae
  seed: 42
  output_dir: results/vae_experiment

model:
  type: vae
  hidden_dim: 256
  latent_dim: 64
  num_layers: 4
  dropout: 0.1

  encoder:
    type: transformer
    num_heads: 8

training:
  epochs: 100
  batch_size: 32
  learning_rate: 1.0e-4
  weight_decay: 1.0e-5

data:
  train_path: data/train.csv
  val_path: data/val.csv
```

### Loading YAML

```python
import yaml
from pathlib import Path

def load_yaml(path: Path) -> dict:
    """Load YAML configuration file."""
    with open(path) as f:
        return yaml.safe_load(f)

def save_yaml(config: dict, path: Path) -> None:
    """Save configuration to YAML file."""
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
```

## Configuration Inheritance

### Base + Override Pattern

```toml
# configs/base.toml
[model]
hidden_dim = 256
num_layers = 4

[training]
epochs = 100
batch_size = 32
learning_rate = 1e-4
```

```toml
# configs/large_model.toml
_base_ = "base.toml"

[model]
hidden_dim = 512
num_layers = 8
```

```python
def load_config_with_inheritance(path: Path) -> dict:
    """Load config with inheritance support."""
    config = load_toml(path)

    # Handle inheritance
    if "_base_" in config:
        base_path = path.parent / config.pop("_base_")
        base_config = load_config_with_inheritance(base_path)
        deep_merge(base_config, config)
        return base_config

    return config
```

## CLI Overrides

### Argparse Integration

```python
import argparse
from typing import Any

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--override", nargs="*", default=[])
    return parser.parse_args()

def parse_overrides(overrides: list[str]) -> dict[str, Any]:
    """Parse override strings like 'training.learning_rate=1e-5'."""
    result = {}

    for override in overrides:
        key, value = override.split("=", 1)
        keys = key.split(".")

        # Parse value type
        try:
            value = eval(value)  # Handle numbers, booleans, lists
        except:
            pass  # Keep as string

        # Build nested dict
        current = result
        for k in keys[:-1]:
            current = current.setdefault(k, {})
        current[keys[-1]] = value

    return result

def main():
    args = parse_args()
    config = load_toml(args.config)

    if args.override:
        overrides = parse_overrides(args.override)
        deep_merge(config, overrides)

    # Use config...
```

### Usage

```bash
python train.py --config configs/base.toml \
    --override training.learning_rate=1e-5 \
    --override model.hidden_dim=512
```

## Environment Variable Interpolation

```python
import os
import re

def interpolate_env_vars(config: dict) -> dict:
    """Replace ${VAR} with environment variables."""
    def replace(value):
        if isinstance(value, str):
            pattern = r'\$\{(\w+)\}'
            matches = re.findall(pattern, value)
            for match in matches:
                env_value = os.environ.get(match, "")
                value = value.replace(f"${{{match}}}", env_value)
            return value
        elif isinstance(value, dict):
            return {k: replace(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [replace(v) for v in value]
        return value

    return replace(config)
```

```toml
# configs/experiment.toml
[data]
train_path = "${DATA_DIR}/train.csv"
val_path = "${DATA_DIR}/val.csv"
```

## Configuration Validation

```python
from pydantic import BaseModel, ValidationError

def validate_config(config: dict, schema: type[BaseModel]) -> BaseModel:
    """Validate configuration against Pydantic schema."""
    try:
        return schema(**config)
    except ValidationError as e:
        print("Configuration validation failed:")
        for error in e.errors():
            loc = ".".join(str(x) for x in error["loc"])
            print(f"  {loc}: {error['msg']}")
        raise
```

## Multi-Run Configuration

```python
from itertools import product

def expand_grid(config: dict) -> list[dict]:
    """Expand grid search configuration."""
    grid_params = {}
    base_config = {}

    for key, value in config.items():
        if isinstance(value, list) and key.startswith("grid_"):
            param_name = key[5:]  # Remove 'grid_' prefix
            grid_params[param_name] = value
        else:
            base_config[key] = value

    if not grid_params:
        return [base_config]

    # Generate all combinations
    keys = list(grid_params.keys())
    values = list(grid_params.values())

    configs = []
    for combo in product(*values):
        run_config = base_config.copy()
        for k, v in zip(keys, combo):
            # Handle nested keys
            set_nested(run_config, k.split("."), v)
        configs.append(run_config)

    return configs

def set_nested(d: dict, keys: list[str], value) -> None:
    """Set nested dictionary value."""
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value
```

```toml
# configs/grid_search.toml
[experiment]
name = "grid_search"

[model]
type = "vae"

# Grid parameters (prefix with grid_)
grid_model.hidden_dim = [128, 256, 512]
grid_training.learning_rate = [1e-3, 1e-4, 1e-5]
```

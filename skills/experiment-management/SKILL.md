---
name: experiment-management
description: Patterns for experiment configuration, logging, hyperparameter tracking, and results management.
---

# Experiment Management

Patterns for managing ML experiments, configurations, and results.

## Configuration with Pydantic

### Base Configuration

```python
from pydantic import BaseModel, Field, field_validator
from pathlib import Path
from typing import Literal

class ModelConfig(BaseModel):
    """Model architecture configuration."""

    model_type: Literal["vae", "autoregressive", "diffusion", "gnn"]
    hidden_dim: int = Field(ge=32, le=2048, default=256)
    latent_dim: int = Field(ge=8, le=512, default=64)
    num_layers: int = Field(ge=1, le=24, default=4)
    dropout: float = Field(ge=0.0, le=0.5, default=0.1)

class TrainingConfig(BaseModel):
    """Training hyperparameters."""

    epochs: int = Field(ge=1, default=100)
    batch_size: int = Field(ge=1, le=512, default=32)
    learning_rate: float = Field(ge=1e-6, le=1.0, default=1e-4)
    weight_decay: float = Field(ge=0.0, le=0.1, default=1e-5)
    gradient_clip: float = Field(ge=0.0, default=1.0)
    patience: int = Field(ge=1, default=10)
    warmup_steps: int = Field(ge=0, default=1000)

class DataConfig(BaseModel):
    """Data configuration."""

    train_path: Path
    val_path: Path
    test_path: Path | None = None
    smiles_column: str = "smiles"
    max_length: int = Field(ge=10, le=500, default=200)
    num_workers: int = Field(ge=0, le=16, default=4)

    @field_validator("train_path", "val_path", "test_path", mode="before")
    @classmethod
    def validate_path(cls, v):
        if v is None:
            return v
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Path does not exist: {path}")
        return path

class ExperimentConfig(BaseModel):
    """Complete experiment configuration."""

    name: str
    seed: int = 42
    device: str = "cuda"
    output_dir: Path = Path("results")

    model: ModelConfig
    training: TrainingConfig
    data: DataConfig

    def save(self, path: Path) -> None:
        """Save configuration to file."""
        import tomli_w
        with open(path, "wb") as f:
            tomli_w.dump(self.model_dump(), f)

    @classmethod
    def load(cls, path: Path) -> "ExperimentConfig":
        """Load configuration from file."""
        import tomllib
        with open(path, "rb") as f:
            data = tomllib.load(f)
        return cls(**data)
```

### Configuration Factory

```python
def create_config(
    config_path: Path | None = None,
    overrides: dict | None = None,
) -> ExperimentConfig:
    """Create configuration with optional overrides."""
    if config_path is not None:
        config = ExperimentConfig.load(config_path)
    else:
        config = ExperimentConfig(
            name="default",
            model=ModelConfig(model_type="vae"),
            training=TrainingConfig(),
            data=DataConfig(
                train_path=Path("data/train.csv"),
                val_path=Path("data/val.csv"),
            ),
        )

    if overrides:
        config_dict = config.model_dump()
        _deep_update(config_dict, overrides)
        config = ExperimentConfig(**config_dict)

    return config

def _deep_update(base: dict, update: dict) -> None:
    """Recursively update nested dict."""
    for key, value in update.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
```

## Logging with TensorBoard

```python
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

class ExperimentLogger:
    """Logger for experiment metrics."""

    def __init__(self, log_dir: Path, config: ExperimentConfig):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = log_dir / f"{config.name}_{timestamp}"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(self.log_dir)
        self.step = 0

        # Log configuration
        self._log_config(config)

    def _log_config(self, config: ExperimentConfig) -> None:
        """Log configuration as text."""
        config_str = config.model_dump_json(indent=2)
        self.writer.add_text("config", f"```json\n{config_str}\n```")

        # Save config file
        config.save(self.log_dir / "config.toml")

    def log_scalar(self, tag: str, value: float, step: int | None = None) -> None:
        """Log scalar metric."""
        step = step if step is not None else self.step
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, tag: str, values: dict[str, float], step: int | None = None) -> None:
        """Log multiple scalars."""
        step = step if step is not None else self.step
        self.writer.add_scalars(tag, values, step)

    def log_histogram(self, tag: str, values: torch.Tensor, step: int | None = None) -> None:
        """Log histogram of values."""
        step = step if step is not None else self.step
        self.writer.add_histogram(tag, values, step)

    def log_hparams(self, hparams: dict, metrics: dict) -> None:
        """Log hyperparameters with final metrics."""
        self.writer.add_hparams(hparams, metrics)

    def log_molecules(self, tag: str, smiles_list: list[str], step: int | None = None) -> None:
        """Log generated molecules as images."""
        from rdkit import Chem
        from rdkit.Chem import Draw
        import io
        from PIL import Image

        step = step if step is not None else self.step

        mols = [Chem.MolFromSmiles(s) for s in smiles_list[:16]]
        mols = [m for m in mols if m is not None]

        if mols:
            img = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(200, 200))

            # Convert to tensor
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            img_tensor = torch.tensor(
                np.array(Image.open(buf))
            ).permute(2, 0, 1)

            self.writer.add_image(tag, img_tensor, step)

    def increment_step(self, n: int = 1) -> None:
        """Increment global step."""
        self.step += n

    def close(self) -> None:
        """Close writer."""
        self.writer.close()
```

## Checkpoint Management

```python
from dataclasses import dataclass
from typing import Any

@dataclass
class Checkpoint:
    """Training checkpoint."""

    epoch: int
    global_step: int
    model_state: dict[str, Any]
    optimizer_state: dict[str, Any]
    scheduler_state: dict[str, Any] | None
    config: dict
    metrics: dict[str, float]
    best_metric: float

    def save(self, path: Path) -> None:
        """Save checkpoint to file."""
        torch.save({
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state": self.model_state,
            "optimizer_state": self.optimizer_state,
            "scheduler_state": self.scheduler_state,
            "config": self.config,
            "metrics": self.metrics,
            "best_metric": self.best_metric,
        }, path)

    @classmethod
    def load(cls, path: Path, device: torch.device) -> "Checkpoint":
        """Load checkpoint from file."""
        data = torch.load(path, map_location=device, weights_only=False)
        return cls(**data)

class CheckpointManager:
    """Manage multiple checkpoints."""

    def __init__(
        self,
        checkpoint_dir: Path,
        max_checkpoints: int = 5,
        metric_name: str = "val_loss",
        mode: Literal["min", "max"] = "min",
    ):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.metric_name = metric_name
        self.mode = mode
        self.checkpoints: list[tuple[float, Path]] = []

    def save(self, checkpoint: Checkpoint) -> bool:
        """Save checkpoint if it's good enough. Returns True if saved."""
        metric = checkpoint.metrics.get(self.metric_name, float("inf"))

        # Check if we should save
        if len(self.checkpoints) < self.max_checkpoints:
            should_save = True
        else:
            worst_metric, _ = self.checkpoints[-1]
            if self.mode == "min":
                should_save = metric < worst_metric
            else:
                should_save = metric > worst_metric

        if not should_save:
            return False

        # Save checkpoint
        path = self.checkpoint_dir / f"checkpoint_epoch{checkpoint.epoch}_metric{metric:.4f}.pt"
        checkpoint.save(path)

        # Update list
        self.checkpoints.append((metric, path))
        self.checkpoints.sort(key=lambda x: x[0], reverse=(self.mode == "max"))

        # Remove old checkpoints
        while len(self.checkpoints) > self.max_checkpoints:
            _, old_path = self.checkpoints.pop()
            old_path.unlink(missing_ok=True)

        # Save best checkpoint separately
        if self.checkpoints[0][1] == path:
            best_path = self.checkpoint_dir / "best.pt"
            checkpoint.save(best_path)

        return True

    def load_best(self, device: torch.device) -> Checkpoint | None:
        """Load best checkpoint."""
        best_path = self.checkpoint_dir / "best.pt"
        if best_path.exists():
            return Checkpoint.load(best_path, device)
        return None
```

## Results Tracking

```python
import json
from datetime import datetime

class ResultsTracker:
    """Track experiment results."""

    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results_file = self.results_dir / "results.jsonl"

    def log_result(
        self,
        experiment_name: str,
        config: dict,
        metrics: dict,
        artifacts: dict[str, Path] | None = None,
    ) -> None:
        """Log experiment result."""
        result = {
            "timestamp": datetime.now().isoformat(),
            "experiment_name": experiment_name,
            "config": config,
            "metrics": metrics,
            "artifacts": {k: str(v) for k, v in (artifacts or {}).items()},
        }

        with open(self.results_file, "a") as f:
            f.write(json.dumps(result) + "\n")

    def get_all_results(self) -> list[dict]:
        """Get all logged results."""
        results = []
        if self.results_file.exists():
            with open(self.results_file) as f:
                for line in f:
                    results.append(json.loads(line))
        return results

    def get_best_result(
        self,
        metric_name: str,
        mode: Literal["min", "max"] = "min",
    ) -> dict | None:
        """Get best result by metric."""
        results = self.get_all_results()
        if not results:
            return None

        key_fn = lambda r: r["metrics"].get(metric_name, float("inf") if mode == "min" else float("-inf"))
        return min(results, key=key_fn) if mode == "min" else max(results, key=key_fn)
```

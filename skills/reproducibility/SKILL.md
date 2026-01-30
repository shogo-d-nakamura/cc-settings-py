---
name: reproducibility
description: Patterns for ensuring reproducibility in ML experiments including seeding, environment tracking, and experiment versioning.
---

# Reproducibility Patterns

Patterns for ensuring ML experiments are reproducible.

## Random Seed Management

### Comprehensive Seed Setting

```python
import random
import numpy as np
import torch
import os

def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set all random seeds for reproducibility.

    Args:
        seed: Random seed value.
        deterministic: If True, enable deterministic algorithms (may be slower).
    """
    # Python
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    if deterministic:
        # Deterministic algorithms
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # PyTorch 1.8+
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True, warn_only=True)

        # Environment variable for CUDA
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
```

### Seed Context Manager

```python
from contextlib import contextmanager

@contextmanager
def seeded_context(seed: int):
    """Temporarily set seeds within a context."""
    # Save current states
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    # Set new seed
    set_seed(seed)

    try:
        yield
    finally:
        # Restore states
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
```

## Environment Tracking

### Environment Export

```python
import subprocess
import sys
from pathlib import Path
from datetime import datetime

def export_environment(output_dir: Path) -> dict:
    """Export complete environment information."""
    output_dir.mkdir(parents=True, exist_ok=True)

    env_info = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "platform": sys.platform,
    }

    # Conda environment
    try:
        conda_list = subprocess.check_output(
            ["conda", "list", "--export"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        (output_dir / "conda_packages.txt").write_text(conda_list)
        env_info["conda_exported"] = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        env_info["conda_exported"] = False

    # Pip packages
    pip_list = subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"],
        text=True,
    )
    (output_dir / "requirements.txt").write_text(pip_list)

    # PyTorch info
    import torch
    env_info["pytorch_version"] = torch.__version__
    env_info["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        env_info["cuda_version"] = torch.version.cuda
        env_info["cudnn_version"] = torch.backends.cudnn.version()
        env_info["gpu_name"] = torch.cuda.get_device_name(0)

    # Save environment info
    import json
    with open(output_dir / "environment.json", "w") as f:
        json.dump(env_info, f, indent=2)

    return env_info
```

### Git Information

```python
def get_git_info() -> dict:
    """Get current git repository information."""
    import subprocess

    info = {}

    try:
        # Commit hash
        info["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()

        # Branch
        info["branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()

        # Dirty status
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        info["dirty"] = len(status) > 0

        # Diff (if dirty)
        if info["dirty"]:
            diff = subprocess.check_output(
                ["git", "diff", "--stat"],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            info["diff_summary"] = diff

    except (subprocess.CalledProcessError, FileNotFoundError):
        info["error"] = "Not a git repository or git not available"

    return info
```

## Experiment Logging

### Complete Experiment Log

```python
from dataclasses import dataclass, asdict
from typing import Any
import json

@dataclass
class ExperimentLog:
    """Complete experiment log for reproducibility."""

    # Identification
    experiment_name: str
    timestamp: str
    seed: int

    # Environment
    environment: dict

    # Configuration
    config: dict

    # Git info
    git_info: dict

    # Results
    metrics: dict
    artifacts: dict[str, str]

    def save(self, path: Path) -> None:
        """Save experiment log."""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)

    @classmethod
    def load(cls, path: Path) -> "ExperimentLog":
        """Load experiment log."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

def create_experiment_log(
    name: str,
    config: dict,
    metrics: dict,
    artifacts: dict[str, Path],
    seed: int,
) -> ExperimentLog:
    """Create complete experiment log."""
    return ExperimentLog(
        experiment_name=name,
        timestamp=datetime.now().isoformat(),
        seed=seed,
        environment=export_environment(Path("/tmp/env_check")),
        config=config,
        git_info=get_git_info(),
        metrics=metrics,
        artifacts={k: str(v) for k, v in artifacts.items()},
    )
```

## Checkpoint Versioning

```python
import hashlib

def compute_config_hash(config: dict) -> str:
    """Compute hash of configuration for versioning."""
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]

def versioned_checkpoint_path(
    base_dir: Path,
    config: dict,
    epoch: int,
) -> Path:
    """Generate versioned checkpoint path."""
    config_hash = compute_config_hash(config)
    return base_dir / f"checkpoint_{config_hash}_epoch{epoch:04d}.pt"
```

## Data Versioning

```python
def compute_dataset_hash(data_path: Path) -> str:
    """Compute hash of dataset file."""
    hasher = hashlib.md5()

    with open(data_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)

    return hasher.hexdigest()

def verify_dataset(data_path: Path, expected_hash: str) -> bool:
    """Verify dataset matches expected hash."""
    actual_hash = compute_dataset_hash(data_path)
    return actual_hash == expected_hash
```

## Reproducibility Checklist

```python
def reproducibility_check(config: dict, output_dir: Path) -> dict:
    """Run reproducibility checklist."""
    checks = {}

    # 1. Seed set
    checks["seed_configured"] = "seed" in config

    # 2. Environment exported
    env_file = output_dir / "environment.json"
    checks["environment_exported"] = env_file.exists()

    # 3. Config saved
    config_file = output_dir / "config.toml"
    checks["config_saved"] = config_file.exists()

    # 4. Git info recorded
    git_file = output_dir / "git_info.json"
    checks["git_recorded"] = git_file.exists()

    # 5. Deterministic mode
    checks["deterministic_mode"] = (
        torch.backends.cudnn.deterministic and
        not torch.backends.cudnn.benchmark
    )

    # Summary
    checks["all_passed"] = all(checks.values())

    return checks
```

## Experiment Template

```python
def run_reproducible_experiment(
    config_path: Path,
    output_dir: Path,
) -> ExperimentLog:
    """Template for reproducible experiment."""

    # 1. Load and validate config
    config = load_config(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Set seed
    set_seed(config["seed"])

    # 3. Export environment
    export_environment(output_dir)

    # 4. Save git info
    git_info = get_git_info()
    with open(output_dir / "git_info.json", "w") as f:
        json.dump(git_info, f, indent=2)

    # 5. Save config
    save_config(config, output_dir / "config.toml")

    # 6. Run experiment
    metrics = train_and_evaluate(config)

    # 7. Save results
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # 8. Create experiment log
    log = create_experiment_log(
        name=config["experiment"]["name"],
        config=config,
        metrics=metrics,
        artifacts={"checkpoint": output_dir / "best.pt"},
        seed=config["seed"],
    )
    log.save(output_dir / "experiment_log.json")

    # 9. Run reproducibility check
    checks = reproducibility_check(config, output_dir)
    print(f"Reproducibility checks: {checks}")

    return log
```

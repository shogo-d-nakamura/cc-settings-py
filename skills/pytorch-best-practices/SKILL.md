---
name: pytorch-best-practices
description: PyTorch best practices for device management, memory optimization, gradient handling, and performance.
---

# PyTorch Best Practices

Essential patterns for efficient and correct PyTorch code.

## Device Management

### Device Selection

```python
import torch

def get_device(device_id: int | str | None = None) -> torch.device:
    """Get the appropriate device."""
    if device_id is not None:
        if isinstance(device_id, int):
            return torch.device(f"cuda:{device_id}")
        return torch.device(device_id)

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

# Usage
device = get_device()
model = model.to(device)
```

### Moving Data to Device

```python
# GOOD: Move entire batch at once
def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict:
    return {k: v.to(device) for k, v in batch.items()}

# GOOD: Non-blocking transfer for async
def move_batch_async(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}

# BAD: Moving in loop
for key in batch:
    batch[key] = batch[key].to(device)  # Slower
```

## Memory Management

### Clearing Memory

```python
# Delete tensors no longer needed
del large_tensor
del intermediate_result

# Clear CUDA cache
torch.cuda.empty_cache()

# Force garbage collection
import gc
gc.collect()
torch.cuda.empty_cache()
```

### Memory Monitoring

```python
def print_gpu_memory():
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

# Context manager for memory tracking
from contextlib import contextmanager

@contextmanager
def track_memory(label: str = ""):
    """Track GPU memory within a context."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start_mem = torch.cuda.memory_allocated()

    yield

    torch.cuda.synchronize()
    end_mem = torch.cuda.memory_allocated()
    peak_mem = torch.cuda.max_memory_allocated()

    print(f"{label}: Start={start_mem/1e9:.2f}GB, End={end_mem/1e9:.2f}GB, Peak={peak_mem/1e9:.2f}GB")
```

### Gradient Checkpointing

```python
from torch.utils.checkpoint import checkpoint

class MemoryEfficientModel(nn.Module):
    """Model using gradient checkpointing to save memory."""

    def __init__(self, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(hidden_dim=512)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            # Checkpoint each layer - trades compute for memory
            x = checkpoint(layer, x, use_reentrant=False)
        return x
```

## Gradient Handling

### Gradient Clipping

```python
# Clip by norm (recommended)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Clip by value
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)

# Check gradient norms
def get_gradient_norm(model: nn.Module) -> float:
    """Compute total gradient norm."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5
```

### Gradient Accumulation

```python
def train_with_accumulation(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    accumulation_steps: int = 4,
) -> float:
    """Train with gradient accumulation for larger effective batch size."""
    model.train()
    optimizer.zero_grad()
    total_loss = 0.0

    for i, batch in enumerate(loader):
        # Forward pass
        loss = model.compute_loss(batch) / accumulation_steps

        # Backward pass (accumulates gradients)
        loss.backward()

        # Update weights every accumulation_steps
        if (i + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps

    # Handle remaining batches
    if (i + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / len(loader)
```

### No Gradient Context

```python
# For inference
model.eval()
with torch.no_grad():
    predictions = model(inputs)

# For computing metrics
with torch.no_grad():
    accuracy = compute_accuracy(predictions, labels)

# Using inference_mode (faster, more restrictive)
with torch.inference_mode():
    predictions = model(inputs)
```

## Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

def train_mixed_precision(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
) -> float:
    """Training with automatic mixed precision."""
    scaler = GradScaler()
    model.train()
    total_loss = 0.0

    for batch in loader:
        optimizer.zero_grad()

        # Forward pass in mixed precision
        with autocast():
            loss = model.compute_loss(batch)

        # Backward pass with scaling
        scaler.scale(loss).backward()

        # Unscale for gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(loader)
```

## Data Loading

### Efficient DataLoader

```python
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,              # Parallel data loading
    pin_memory=True,            # Faster GPU transfer
    persistent_workers=True,    # Keep workers alive
    prefetch_factor=2,          # Batches to prefetch per worker
    drop_last=True,             # Consistent batch sizes
)
```

### Custom Sampler

```python
from torch.utils.data import Sampler

class BucketSampler(Sampler):
    """Sample batches with similar sequence lengths."""

    def __init__(self, lengths: list[int], batch_size: int):
        self.lengths = lengths
        self.batch_size = batch_size

        # Sort indices by length
        self.sorted_indices = sorted(
            range(len(lengths)),
            key=lambda i: lengths[i]
        )

    def __iter__(self):
        # Create batches of similar lengths
        batches = [
            self.sorted_indices[i:i + self.batch_size]
            for i in range(0, len(self.sorted_indices), self.batch_size)
        ]

        # Shuffle batches (not within batches)
        random.shuffle(batches)

        for batch in batches:
            yield from batch

    def __len__(self) -> int:
        return len(self.lengths)
```

## Model Modes

```python
# CRITICAL: Always set correct mode
model.train()   # Training mode (dropout active, BatchNorm updates)
model.eval()    # Evaluation mode (dropout disabled, BatchNorm frozen)

# Context manager for evaluation
@contextmanager
def eval_mode(model: nn.Module):
    """Temporarily set model to eval mode."""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        if was_training:
            model.train()
```

## Reproducibility

```python
def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make CUDA operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # For full determinism (slower)
    # torch.use_deterministic_algorithms(True)
```

## Debugging

### Detecting NaN/Inf

```python
# Enable anomaly detection (slow, use for debugging only)
torch.autograd.set_detect_anomaly(True)

# Check for NaN in tensors
def check_nan(tensor: torch.Tensor, name: str = "") -> None:
    if torch.isnan(tensor).any():
        raise ValueError(f"NaN detected in {name}")
    if torch.isinf(tensor).any():
        raise ValueError(f"Inf detected in {name}")

# Register hooks to check activations
def check_hook(module, input, output):
    if isinstance(output, torch.Tensor):
        check_nan(output, module.__class__.__name__)

for module in model.modules():
    module.register_forward_hook(check_hook)
```

### Profiling

```python
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
) as prof:
    model(inputs)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

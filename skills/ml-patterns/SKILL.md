---
name: ml-patterns
description: Common patterns for machine learning code including training loops, data loading, and model architectures.
---

# ML Patterns

Common patterns and best practices for machine learning implementations.

## Training Loop Patterns

### Basic Training Loop

```python
def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: torch.device,
) -> dict[str, list[float]]:
    """Standard training loop."""
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()
            loss = model.compute_loss(batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                loss = model.compute_loss(batch)
                val_loss += loss.item()

        # Record metrics
        history["train_loss"].append(train_loss / len(train_loader))
        history["val_loss"].append(val_loss / len(val_loader))

        print(f"Epoch {epoch}: train={history['train_loss'][-1]:.4f}, val={history['val_loss'][-1]:.4f}")

    return history
```

### Training with Callbacks

```python
from abc import ABC, abstractmethod

class Callback(ABC):
    """Base class for training callbacks."""

    def on_epoch_start(self, epoch: int, trainer: "Trainer") -> None:
        pass

    def on_epoch_end(self, epoch: int, trainer: "Trainer", metrics: dict) -> None:
        pass

    def on_batch_end(self, batch_idx: int, trainer: "Trainer", loss: float) -> None:
        pass

class EarlyStopping(Callback):
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0

    def on_epoch_end(self, epoch: int, trainer: "Trainer", metrics: dict) -> None:
        val_loss = metrics.get("val_loss", float("inf"))

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            trainer.should_stop = True
            print(f"Early stopping at epoch {epoch}")

class ModelCheckpoint(Callback):
    """Save model checkpoints."""

    def __init__(self, path: Path, save_best_only: bool = True):
        self.path = path
        self.save_best_only = save_best_only
        self.best_loss = float("inf")

    def on_epoch_end(self, epoch: int, trainer: "Trainer", metrics: dict) -> None:
        val_loss = metrics.get("val_loss", float("inf"))

        if not self.save_best_only or val_loss < self.best_loss:
            self.best_loss = val_loss
            trainer.save_checkpoint(self.path / f"epoch_{epoch}.pt")
```

## Model Patterns

### Base Model Class

```python
class BaseModel(nn.Module):
    """Base class for all models."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward pass."""
        pass

    @abstractmethod
    def compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute loss."""
        pass

    def save(self, path: Path) -> None:
        """Save model state."""
        torch.save(self.state_dict(), path)

    def load(self, path: Path, device: torch.device) -> None:
        """Load model state."""
        state_dict = torch.load(path, map_location=device, weights_only=True)
        self.load_state_dict(state_dict)
```

### Encoder-Decoder Pattern

```python
class EncoderDecoder(nn.Module):
    """Encoder-decoder architecture."""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)
```

### Model with Multiple Heads

```python
class MultiTaskModel(nn.Module):
    """Model with shared backbone and multiple task heads."""

    def __init__(
        self,
        backbone: nn.Module,
        heads: dict[str, nn.Module],
    ):
        super().__init__()
        self.backbone = backbone
        self.heads = nn.ModuleDict(heads)

    def forward(
        self,
        x: torch.Tensor,
        task: str | None = None,
    ) -> dict[str, torch.Tensor]:
        features = self.backbone(x)

        if task is not None:
            return {task: self.heads[task](features)}

        return {name: head(features) for name, head in self.heads.items()}
```

## Data Patterns

### Dataset with Lazy Loading

```python
class LazyDataset(Dataset):
    """Dataset that loads data on-demand."""

    def __init__(self, data_path: Path, transform: Callable | None = None):
        self.data_path = data_path
        self.transform = transform

        # Only load index, not full data
        with open(data_path / "index.json") as f:
            self.index = json.load(f)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        # Load data on access
        file_path = self.data_path / self.index[idx]["file"]
        data = self._load_file(file_path)

        if self.transform:
            data = self.transform(data)

        return data

    def _load_file(self, path: Path) -> dict:
        # Implement file-specific loading
        ...
```

### Collate Function Pattern

```python
def collate_molecules(
    batch: list[dict[str, Any]],
    pad_token_id: int = 0,
) -> dict[str, torch.Tensor]:
    """Collate function for variable-length molecules."""

    # Get max length in batch
    max_len = max(len(item["input_ids"]) for item in batch)

    # Pad sequences
    input_ids = []
    attention_mask = []

    for item in batch:
        seq_len = len(item["input_ids"])
        padding_len = max_len - seq_len

        input_ids.append(
            item["input_ids"] + [pad_token_id] * padding_len
        )
        attention_mask.append(
            [1] * seq_len + [0] * padding_len
        )

    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_mask),
        "labels": torch.tensor([item["label"] for item in batch]),
    }
```

## Initialization Patterns

```python
def init_weights(module: nn.Module) -> None:
    """Initialize model weights."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

# Usage
model = MyModel()
model.apply(init_weights)
```

## Loss Patterns

### Combined Loss

```python
class CombinedLoss(nn.Module):
    """Combine multiple losses with weights."""

    def __init__(
        self,
        losses: dict[str, nn.Module],
        weights: dict[str, float] | None = None,
    ):
        super().__init__()
        self.losses = nn.ModuleDict(losses)
        self.weights = weights or {name: 1.0 for name in losses}

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss_values = {}
        total_loss = 0.0

        for name, loss_fn in self.losses.items():
            loss = loss_fn(predictions[name], targets[name])
            loss_values[name] = loss
            total_loss = total_loss + self.weights[name] * loss

        return total_loss, loss_values
```

### Masked Loss

```python
def masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Cross entropy with masking for padded sequences."""
    # logits: (batch, seq_len, vocab_size)
    # targets: (batch, seq_len)
    # mask: (batch, seq_len)

    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        reduction="none",
    )

    loss = loss.view(targets.shape)
    loss = (loss * mask).sum() / mask.sum()

    return loss
```

# ML Patterns and Anti-Patterns

Rules for writing correct ML code.

## Required Patterns

### Random Seed Setting

```python
# REQUIRED at script start
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
```

### Device Management

```python
# REQUIRED: Explicit device handling
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
batch = {k: v.to(device) for k, v in batch.items()}
```

### Train/Eval Modes

```python
# REQUIRED: Set correct mode
model.train()   # During training
model.eval()    # During inference/validation

# REQUIRED: No gradients for inference
with torch.no_grad():
    predictions = model(inputs)
```

### Gradient Clipping

```python
# REQUIRED for RNNs/Transformers
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

## Anti-Patterns to Avoid

### Data Leakage

```python
# BAD: Normalize before split
scaler.fit(all_data)  # Leaks test statistics
train, test = split(all_data)

# GOOD: Normalize after split
train, test = split(all_data)
scaler.fit(train)  # Only fit on training
```

### Hardcoded Hyperparameters

```python
# BAD: Hardcoded values
lr = 0.001
hidden_dim = 256

# GOOD: Configuration file
config = load_config("experiment.toml")
lr = config.training.learning_rate
```

### Missing Validation

```python
# BAD: No SMILES validation
mol = Chem.MolFromSmiles(smiles)
fp = compute_fp(mol)  # Crashes if mol is None

# GOOD: Always validate
mol = Chem.MolFromSmiles(smiles)
if mol is None:
    raise ValueError(f"Invalid SMILES: {smiles}")
```

### Memory Leaks

```python
# BAD: Accumulating tensors with gradients
losses = []
for batch in loader:
    loss = model(batch)
    losses.append(loss)  # Keeps computation graph!

# GOOD: Detach or use item()
losses = []
for batch in loader:
    loss = model(batch)
    losses.append(loss.item())  # Just the value
```

### No Checkpoint State

```python
# BAD: Only save model
torch.save(model.state_dict(), path)

# GOOD: Save complete state
torch.save({
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "epoch": epoch,
    "config": config,
}, path)
```

## Checklist

Before running training:
- [ ] Random seed set
- [ ] Device explicitly specified
- [ ] Data splits are clean (no leakage)
- [ ] Hyperparameters in config file
- [ ] SMILES validation present
- [ ] Gradient clipping for sequences
- [ ] Checkpoint saves full state
- [ ] Logging configured

Before inference:
- [ ] Model in eval mode
- [ ] torch.no_grad() context
- [ ] Input validation

## Performance Patterns

### Efficient Data Loading

```python
loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)
```

### Mixed Precision

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    loss = model(batch)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Gradient Accumulation

```python
for i, batch in enumerate(loader):
    loss = model(batch) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

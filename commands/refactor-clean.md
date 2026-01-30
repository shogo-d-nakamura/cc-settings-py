---
description: Refactor and clean up ML code - remove dead code, improve structure, reduce duplication.
---

# Refactor Clean Command

Invokes the **refactor-cleaner** agent to improve code quality.

## What This Command Does

1. **Find Dead Code** - Unused imports, variables, functions
2. **Reduce Duplication** - Extract common patterns
3. **Simplify Functions** - Break down complex logic
4. **Modernize** - Update to modern Python patterns
5. **Organize** - Improve file structure

## When to Use

Use `/refactor-clean` when:
- Code has grown messy over time
- Duplicate code exists
- Functions are too long
- Unused code accumulates

## Refactoring Targets

### Dead Code

```python
# Find unused imports
ruff check --select F401 src/

# Find unused variables
ruff check --select F841 src/
```

### Duplicate Code

```python
# BEFORE: Duplicated
def train_epoch(model, loader):
    model.train()
    total = 0
    for batch in loader:
        loss = model(batch)
        loss.backward()
        total += loss.item()
    return total

def val_epoch(model, loader):
    model.eval()
    total = 0
    for batch in loader:
        with torch.no_grad():
            loss = model(batch)
        total += loss.item()
    return total

# AFTER: Extracted
def run_epoch(model, loader, training=True):
    model.train() if training else model.eval()
    total = 0
    ctx = torch.no_grad() if not training else nullcontext()
    with ctx:
        for batch in loader:
            loss = model(batch)
            if training:
                loss.backward()
            total += loss.item()
    return total
```

### Long Functions

```python
# BEFORE: 100+ lines
def process_data(path):
    # 100 lines of code
    ...

# AFTER: Split into smaller functions
def load_data(path):
    ...

def validate_data(df):
    ...

def transform_data(df):
    ...

def process_data(path):
    df = load_data(path)
    df = validate_data(df)
    return transform_data(df)
```

## Example Usage

```
User: /refactor-clean src/models/

Agent (refactor-cleaner):
## Refactoring Analysis: src/models/

### Dead Code Found
- `src/models/vae.py:12` - Unused import: `typing.Optional`
- `src/models/encoder.py:45` - Unused variable: `temp_buffer`

### Duplication Found
- Training loop pattern duplicated in:
  - `src/training/trainer.py:23-45`
  - `src/training/distributed.py:56-78`
  - Recommendation: Extract to shared utility

### Long Functions
- `src/models/vae.py:generate()` - 78 lines
  - Recommendation: Split into `_sample_latent()` and `_decode_samples()`

### Suggested Changes

1. Remove unused imports
2. Extract `run_epoch()` utility function
3. Split `generate()` into smaller methods

Apply changes? (requires confirmation)
```

## Refactoring Workflow

1. Run tests (baseline)
2. Apply refactoring
3. Run tests (verify no regression)
4. Commit changes

```bash
# Before refactoring
pytest tests/ -v

# After refactoring
pytest tests/ -v
# Ensure all tests still pass
```

## Related Commands

- `/code-review` - Review before refactoring
- `/pytest` - Ensure tests pass after refactoring

## Related Agent

This command invokes the `refactor-cleaner` agent at `agents/refactor-cleaner.md`.

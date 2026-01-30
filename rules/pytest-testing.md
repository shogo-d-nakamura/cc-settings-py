# Testing Requirements

Rules for testing ML and cheminformatics code.

## Coverage Requirements

- **Minimum overall**: 80%
- **Critical code (100%)**:
  - Model forward pass
  - Loss computation
  - Data transforms
  - SMILES validation
  - Fingerprint computation

## TDD Workflow (MANDATORY)

```
RED → GREEN → REFACTOR → REPEAT

1. Write failing test FIRST
2. Implement minimal code to pass
3. Refactor while keeping tests green
4. Repeat for next feature
```

**Never skip the RED phase. Never write code before tests.**

## Test Types

### Unit Tests
- Individual functions and methods
- Model components (encoder, decoder, layers)
- Data transforms
- Utility functions

### Integration Tests
- Data pipeline (load → transform → batch)
- Training step (forward → loss → backward)
- Model save/load roundtrip

### Smoke Tests
- Model training doesn't crash
- Generated SMILES are valid
- Checkpoints can be loaded

## pytest Patterns

### Fixtures for ML

```python
@pytest.fixture(scope="session")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def sample_smiles():
    return ["CCO", "c1ccccc1", "CC(=O)O"]

@pytest.fixture
def sample_batch(device):
    return {"input_ids": torch.randint(0, 100, (8, 32)).to(device)}
```

### Test Naming

```python
# GOOD: Descriptive names
def test_encoder_returns_correct_shape():
def test_invalid_smiles_raises_value_error():
def test_training_step_reduces_loss():

# BAD: Vague names
def test_encoder():
def test_smiles():
def test_training():
```

### Parametrized Tests

```python
@pytest.mark.parametrize("smiles,expected", [
    ("CCO", True),
    ("invalid", False),
])
def test_smiles_validation(smiles, expected):
    assert validate_smiles(smiles) == expected
```

## ML-Specific Tests

### Test Model Forward Pass

```python
def test_forward_shape(model, sample_batch, device):
    output = model(sample_batch)
    assert output.shape == (batch_size, hidden_dim)
```

### Test Gradient Flow

```python
def test_gradient_flow(model, sample_batch):
    model.zero_grad()
    loss = model.compute_loss(sample_batch)
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None
        assert not torch.all(param.grad == 0)
```

### Test Reproducibility

```python
def test_deterministic(model, sample_batch):
    torch.manual_seed(42)
    out1 = model(sample_batch)

    torch.manual_seed(42)
    out2 = model(sample_batch)

    torch.testing.assert_close(out1, out2)
```

## Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html --cov-fail-under=80

# Fast tests only
pytest tests/ -m "not slow"

# GPU tests
pytest tests/ -m gpu
```

## Test Markers

```python
@pytest.mark.slow          # Long-running tests
@pytest.mark.gpu           # Requires GPU
@pytest.mark.integration   # Integration tests
```

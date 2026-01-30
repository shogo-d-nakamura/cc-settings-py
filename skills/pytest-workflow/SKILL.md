---
name: pytest-workflow
description: pytest-based testing workflow for ML and cheminformatics Python code. Includes fixtures, parametrization, and ML-specific testing patterns.
---

# pytest Workflow for ML/Cheminformatics

Comprehensive testing patterns using pytest for machine learning and cheminformatics projects.

## pytest Configuration

### pyproject.toml

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_functions = ["test_*"]
addopts = [
    "-v",
    "--strict-markers",
    "-ra",
]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "gpu: marks tests requiring GPU",
    "integration: marks integration tests",
]
filterwarnings = [
    "ignore::DeprecationWarning",
]
```

### conftest.py

```python
import pytest
import torch
import numpy as np
from pathlib import Path

@pytest.fixture(scope="session")
def device():
    """Provide device for all tests."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture(autouse=True)
def set_seed():
    """Set random seed before each test for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
    yield

@pytest.fixture
def tmp_checkpoint(tmp_path):
    """Provide temporary checkpoint directory."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    return checkpoint_dir

@pytest.fixture(scope="session")
def test_data_dir():
    """Path to test data fixtures."""
    return Path(__file__).parent / "fixtures"
```

## Fixture Patterns

### Model Fixtures

```python
@pytest.fixture
def encoder(device):
    """Create encoder for testing."""
    model = MoleculeEncoder(
        vocab_size=100,
        hidden_dim=64,
        latent_dim=32,
    )
    return model.to(device).eval()

@pytest.fixture
def trained_encoder(encoder, test_data_dir):
    """Load pre-trained encoder weights."""
    checkpoint = test_data_dir / "encoder.pt"
    if checkpoint.exists():
        encoder.load_state_dict(torch.load(checkpoint, weights_only=True))
    return encoder
```

### Data Fixtures

```python
@pytest.fixture
def sample_smiles():
    """Provide sample SMILES for testing."""
    return [
        "CCO",              # Ethanol
        "c1ccccc1",         # Benzene
        "CC(=O)O",          # Acetic acid
        "CCN(CC)CC",        # Triethylamine
        "c1ccc2ccccc2c1",   # Naphthalene
    ]

@pytest.fixture
def invalid_smiles():
    """Provide invalid SMILES for error testing."""
    return [
        "invalid",
        "",
        "C(C",
        "c1ccccc",
    ]

@pytest.fixture
def sample_batch(device):
    """Create sample batch for model testing."""
    batch_size = 8
    seq_len = 32
    return {
        "input_ids": torch.randint(0, 100, (batch_size, seq_len)).to(device),
        "attention_mask": torch.ones(batch_size, seq_len).to(device),
        "labels": torch.randint(0, 2, (batch_size,)).to(device),
    }

@pytest.fixture
def sample_dataset(tmp_path):
    """Create temporary dataset for testing."""
    data_path = tmp_path / "test_data.csv"
    data_path.write_text(
        "smiles,label\n"
        "CCO,1\n"
        "c1ccccc1,0\n"
        "CC(=O)O,1\n"
    )
    return data_path
```

## Test Patterns

### Testing Model Components

```python
class TestMoleculeEncoder:
    """Test suite for MoleculeEncoder."""

    def test_forward_shape(self, encoder, sample_smiles, device):
        """Test forward pass returns correct shape."""
        embeddings = encoder.encode(sample_smiles)

        assert embeddings.shape == (len(sample_smiles), encoder.latent_dim)
        assert embeddings.device.type == device.type
        assert embeddings.dtype == torch.float32

    def test_forward_deterministic(self, encoder, sample_smiles):
        """Test forward pass is deterministic."""
        torch.manual_seed(42)
        emb1 = encoder.encode(sample_smiles)

        torch.manual_seed(42)
        emb2 = encoder.encode(sample_smiles)

        torch.testing.assert_close(emb1, emb2)

    def test_batch_independence(self, encoder, sample_smiles):
        """Test that batch order doesn't affect individual embeddings."""
        emb_full = encoder.encode(sample_smiles)
        emb_single = encoder.encode([sample_smiles[0]])

        torch.testing.assert_close(emb_full[0], emb_single[0])

    def test_invalid_smiles_raises(self, encoder, invalid_smiles):
        """Test that invalid SMILES raises appropriate error."""
        with pytest.raises(ValueError, match="Invalid SMILES"):
            encoder.encode(invalid_smiles)

    def test_empty_input_raises(self, encoder):
        """Test that empty input raises error."""
        with pytest.raises(ValueError, match="Empty"):
            encoder.encode([])
```

### Parametrized Tests

```python
@pytest.mark.parametrize("smiles,expected_valid", [
    ("CCO", True),
    ("c1ccccc1", True),
    ("invalid", False),
    ("", False),
    ("C(C", False),
])
def test_smiles_validation(smiles, expected_valid):
    """Test SMILES validation with various inputs."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    is_valid = mol is not None
    assert is_valid == expected_valid

@pytest.mark.parametrize("batch_size", [1, 8, 32])
@pytest.mark.parametrize("hidden_dim", [64, 128, 256])
def test_model_configurations(batch_size, hidden_dim, device):
    """Test model with different configurations."""
    model = MoleculeEncoder(vocab_size=100, hidden_dim=hidden_dim)
    model = model.to(device)

    x = torch.randint(0, 100, (batch_size, 32)).to(device)
    output = model(x)

    assert output.shape[0] == batch_size
```

### Testing Training Components

```python
class TestTrainer:
    """Test training functionality."""

    @pytest.fixture
    def trainer(self, device):
        model = SimpleModel().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        return Trainer(model, optimizer, device)

    def test_single_step_reduces_loss(self, trainer, sample_batch):
        """Test that training step reduces loss."""
        initial_loss = trainer.compute_loss(sample_batch)
        trainer.train_step(sample_batch)
        new_loss = trainer.compute_loss(sample_batch)

        assert new_loss < initial_loss

    def test_gradient_flow(self, trainer, sample_batch):
        """Test gradients flow through all parameters."""
        trainer.model.zero_grad()
        loss = trainer.compute_loss(sample_batch)
        loss.backward()

        for name, param in trainer.model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.all(param.grad == 0), f"Zero gradient for {name}"

    def test_checkpoint_roundtrip(self, trainer, tmp_checkpoint):
        """Test checkpoint save and load."""
        path = tmp_checkpoint / "model.pt"

        # Save
        trainer.save_checkpoint(path)
        original_state = {k: v.clone() for k, v in trainer.model.state_dict().items()}

        # Modify model
        for param in trainer.model.parameters():
            param.data.fill_(0)

        # Load
        trainer.load_checkpoint(path)

        # Verify
        for key in original_state:
            torch.testing.assert_close(
                trainer.model.state_dict()[key],
                original_state[key]
            )
```

### Testing Data Pipeline

```python
class TestDataPipeline:
    """Test data loading and preprocessing."""

    def test_dataset_length(self, sample_dataset):
        """Test dataset returns correct length."""
        dataset = MoleculeDataset(sample_dataset)
        assert len(dataset) == 3

    def test_dataset_getitem(self, sample_dataset):
        """Test dataset __getitem__."""
        dataset = MoleculeDataset(sample_dataset)
        item = dataset[0]

        assert "smiles" in item
        assert "label" in item
        assert isinstance(item["smiles"], str)

    def test_dataloader_batching(self, sample_dataset):
        """Test DataLoader creates correct batches."""
        dataset = MoleculeDataset(sample_dataset)
        loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

        batch = next(iter(loader))
        assert batch["input_ids"].shape[0] == 2

    def test_transform_applied(self, sample_dataset):
        """Test that transforms are applied."""
        transform = lambda x: x.upper()
        dataset = MoleculeDataset(sample_dataset, transform=transform)

        item = dataset[0]
        assert item["smiles"].isupper()
```

## Test Markers

```python
@pytest.mark.slow
def test_full_training_epoch():
    """Test complete training epoch (slow)."""
    # This test takes >1 minute
    ...

@pytest.mark.gpu
def test_cuda_operations():
    """Test GPU-specific operations."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    ...

@pytest.mark.integration
def test_end_to_end_pipeline():
    """Test complete data-to-prediction pipeline."""
    ...
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html --cov-fail-under=80

# Run fast tests only
pytest tests/ -m "not slow"

# Run specific test file
pytest tests/test_model.py -v

# Run specific test
pytest tests/test_model.py::TestEncoder::test_forward_shape -v

# Run with parallel execution
pytest tests/ -n auto

# Run GPU tests
pytest tests/ -m gpu

# Show print statements
pytest tests/ -s
```

## Coverage Requirements

- **Minimum overall**: 80%
- **Critical code (100%)**:
  - Model forward pass
  - Loss computation
  - Data transforms
  - SMILES validation

```bash
# Generate coverage report
pytest tests/ --cov=src --cov-report=html

# Fail if coverage below threshold
pytest tests/ --cov=src --cov-fail-under=80
```

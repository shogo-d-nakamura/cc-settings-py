---
description: Run pytest-based TDD workflow. Write tests first, then implement minimal code to pass.
---

# pytest Command

Invokes the **tdd-guide** agent to enforce test-driven development using pytest.

## What This Command Does

1. **Write Failing Tests** - Create tests before implementation (RED)
2. **Implement Minimal Code** - Just enough to pass tests (GREEN)
3. **Refactor** - Improve code while keeping tests passing (REFACTOR)
4. **Verify Coverage** - Ensure 80%+ test coverage

## TDD Cycle

```
RED → GREEN → REFACTOR → REPEAT
```

## When to Use

Use `/pytest` when:
- Implementing new functions or classes
- Fixing bugs (write test that reproduces bug first)
- Adding features to existing code
- Refactoring critical code

## Example Usage

```
User: /pytest implement a function to compute Tanimoto similarity

Agent (tdd-guide):
# TDD Session: Tanimoto Similarity

## Step 1: Write Failing Test (RED)

```python
# tests/test_similarity.py
import pytest
import numpy as np
from src.utils.similarity import tanimoto_similarity

def test_tanimoto_identical():
    fp = np.array([1, 0, 1, 1, 0])
    assert tanimoto_similarity(fp, fp) == 1.0

def test_tanimoto_orthogonal():
    fp1 = np.array([1, 0, 0])
    fp2 = np.array([0, 1, 0])
    assert tanimoto_similarity(fp1, fp2) == 0.0

def test_tanimoto_partial():
    fp1 = np.array([1, 0, 1, 1, 0])
    fp2 = np.array([1, 1, 1, 0, 0])
    # Intersection: 2, Union: 5
    assert tanimoto_similarity(fp1, fp2) == pytest.approx(0.4)
```

## Step 2: Run Tests - Verify FAIL

```bash
pytest tests/test_similarity.py -v
FAILED - ModuleNotFoundError
```

## Step 3: Implement Minimal Code (GREEN)

```python
# src/utils/similarity.py
import numpy as np

def tanimoto_similarity(fp1: np.ndarray, fp2: np.ndarray) -> float:
    intersection = np.sum(fp1 & fp2)
    union = np.sum(fp1 | fp2)
    return intersection / union if union > 0 else 0.0
```

## Step 4: Run Tests - Verify PASS

```bash
pytest tests/test_similarity.py -v
PASSED (3 tests)
```
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test
pytest tests/test_model.py::test_forward -v

# Run fast tests only
pytest tests/ -m "not slow"
```

## Coverage Requirements

- **Minimum**: 80%
- **Critical code**: 100% (model forward, loss, data transforms)

## Related Commands

- `/plan` - Plan before implementing
- `/code-review` - Review after implementing

## Related Agent

This command invokes the `tdd-guide` agent at `agents/tdd-guide.md`.

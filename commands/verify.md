---
description: Run verification loop - tests, linting, type checking, and code review.
---

# Verify Command

Run comprehensive verification of code quality.

## What This Command Does

1. **Run Tests** - pytest with coverage
2. **Lint Code** - ruff check
3. **Type Check** - mypy
4. **Code Review** - Automated review

## When to Use

Use `/verify` when:
- Before committing code
- Before merging PRs
- After major changes
- CI/CD verification

## Verification Steps

### 1. Run Tests

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

### 2. Lint Code

```bash
ruff check src/
ruff format --check src/
```

### 3. Type Check

```bash
mypy src/ --ignore-missing-imports
```

### 4. Security Check

```bash
# Check for hardcoded secrets
grep -r "password\|secret\|api_key" src/ --include="*.py"
```

## Example Usage

```
User: /verify src/

Agent:
## Verification Results

### Tests
```bash
pytest tests/ -v --cov=src
```
- Tests: 45 passed, 0 failed
- Coverage: 87% ✓

### Linting
```bash
ruff check src/
```
- Errors: 0 ✓
- Warnings: 2
  - Line too long: src/models/vae.py:123

### Type Checking
```bash
mypy src/
```
- Errors: 1 ✗
  - src/data/dataset.py:45: Missing return type

### Summary
| Check | Status |
|-------|--------|
| Tests | ✓ Pass |
| Coverage | ✓ 87% |
| Linting | ⚠ 2 warnings |
| Types | ✗ 1 error |

**Action Required**: Fix type error in dataset.py
```

## Verification Script

```bash
#!/bin/bash
# verify.sh

set -e

echo "Running tests..."
pytest tests/ -v --cov=src --cov-fail-under=80

echo "Running linter..."
ruff check src/

echo "Running type checker..."
mypy src/ --ignore-missing-imports

echo "All checks passed!"
```

## Pre-commit Hook

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: pytest
        name: pytest
        entry: pytest tests/ -v
        language: system
        pass_filenames: false

      - id: ruff
        name: ruff
        entry: ruff check
        language: system
        types: [python]

      - id: mypy
        name: mypy
        entry: mypy
        language: system
        types: [python]
```

## Related Commands

- `/pytest` - Run tests only
- `/code-review` - Detailed code review
- `/refactor-clean` - Fix issues found

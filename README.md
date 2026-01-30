# Claude Code Settings for Cheminformatics & ML Research

A comprehensive Claude Code configuration tailored for PyTorch-based deep learning, cheminformatics (RDKit), and molecular generation research.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Components](#components)
  - [Agents](#agents)
  - [Skills](#skills)
  - [Commands](#commands)
  - [Rules](#rules)
  - [Contexts](#contexts)
  - [Hooks](#hooks)
- [Usage Examples](#usage-examples)
- [Best Practices](#best-practices)

---

## Overview

### What is Claude Code?

Claude Code is Anthropic's official CLI tool that enables Claude to assist with software engineering tasks directly in your terminal or IDE. It can read files, write code, run commands, and help you build software.

### What are Claude Code Settings?

Claude Code settings are configurations that customize how Claude behaves when working on your projects. They include:

- **Agents**: Specialized sub-agents for specific tasks (e.g., code review, training)
- **Skills**: Domain knowledge and patterns Claude should follow
- **Commands**: Slash commands you can invoke (e.g., `/train`, `/pytest`)
- **Rules**: Guidelines Claude must always follow
- **Contexts**: Behavioral modes for different activities
- **Hooks**: Automated actions triggered by Claude's tool usage

### Why This Configuration?

This configuration adapts a web-focused Claude Code setup for cheminformatics and ML research:

| Original (Web) | Adapted (ML/Cheminformatics) |
|----------------|------------------------------|
| React, Next.js | PyTorch, RDKit |
| TypeScript | Python with type hints |
| Jest, Playwright | pytest |
| PostgreSQL | SMILES, molecular data |
| npm/yarn | conda/pip |

---

## Installation

### Step 1: Copy to Your Project

Copy the `cc-settings-py/` directory to your project or to `~/.claude/`:

```bash
# Option A: Copy to your project root
cp -r cc-settings-py/ /path/to/your/project/.claude/

# Option B: Copy to global Claude settings
cp -r cc-settings-py/* ~/.claude/
```

### Step 2: Verify Structure

Your directory should look like:

```
~/.claude/  (or your-project/.claude/)
├── CLAUDE.md
├── agents/
├── skills/
├── commands/
├── hooks/
├── rules/
└── contexts/
```

### Step 3: Configure Rules (Manual Step Required)

**Important**: Rules cannot be auto-loaded. You must copy them manually:

```bash
cp -r cc-settings-py/rules/* ~/.claude/rules/
```

---

## Quick Start

Once installed, you can use Claude Code with these settings:

```bash
# Start Claude Code in your ML project
cd /path/to/your/ml-project
claude

# Use a command
> /plan implement a VAE for molecular generation

# Ask Claude to review your code
> /code-review src/models/encoder.py

# Run tests with TDD workflow
> /pytest implement test for fingerprint computation
```

---

## Components

### Agents

Agents are specialized sub-agents that Claude can delegate tasks to. Each agent has specific expertise and tools.

#### Available Agents

| Agent | Description | When to Use |
|-------|-------------|-------------|
| `planner` | Creates detailed implementation plans for ML experiments | Starting new features, complex implementations |
| `code-reviewer` | Reviews Python/PyTorch code for quality and correctness | After writing code, before commits |
| `architect` | Designs ML architectures and system structure | Architecture decisions, model design |
| `tdd-guide` | Guides test-driven development with pytest | Implementing new features with tests |
| `build-error-resolver` | Fixes Python environment and import errors | Import errors, conda/pip issues |
| `ml-trainer` | Specializes in training loops and optimization | Training setup, debugging training |
| `cheminformatics-reviewer` | Reviews molecular data processing code | RDKit code, SMILES handling |
| `refactor-cleaner` | Removes dead code and improves structure | Code cleanup, reducing duplication |
| `doc-updater` | Updates documentation and docstrings | README updates, API documentation |

#### How Agents Work

When you ask Claude a complex question, it may automatically use an agent:

```
You: I need to implement a VAE encoder for SMILES

Claude: I'll use the planner agent to create an implementation plan...
[Claude delegates to planner agent]
[planner agent analyzes requirements and creates step-by-step plan]
[Claude returns the plan to you]
```

You can also explicitly request an agent by describing the task:

```
You: Please review my training code for PyTorch best practices
Claude: [Automatically uses code-reviewer agent]
```

#### Agent File Location

Agents are defined in `agents/*.md`. Each file contains:

```markdown
---
name: agent-name
description: What this agent does
tools: ["Read", "Grep", "Glob", "Bash"]
model: opus
---

[Instructions for the agent]
```

---

### Skills

Skills are collections of domain knowledge and code patterns. They teach Claude how to write code following best practices for specific domains.

#### Available Skills

| Skill | Description |
|-------|-------------|
| `pytest-workflow` | pytest fixtures, parametrization, ML testing patterns |
| `python-standards` | PEP8, type hints, docstrings, file organization |
| `ml-patterns` | Training loops, callbacks, loss functions |
| `pytorch-best-practices` | Device management, gradients, mixed precision |
| `cheminformatics-patterns` | RDKit, SMILES, fingerprints, molecular descriptors |
| `molgen-patterns` | VAE, diffusion, GAN, RL-based molecular generation |
| `experiment-management` | Configs, logging, checkpoints |
| `config-management` | TOML/YAML configuration patterns |
| `data-pipeline` | Dataset classes, transforms, DataLoaders |
| `model-evaluation` | Metrics for generation and prediction |
| `reproducibility` | Seeds, environment tracking, versioning |

#### How Skills Work

Skills are automatically loaded when relevant. For example, when you ask about molecular fingerprints, Claude uses knowledge from `cheminformatics-patterns`:

```
You: How should I compute Morgan fingerprints for a dataset?

Claude: [Uses cheminformatics-patterns skill]
Based on best practices, here's how to compute fingerprints:

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import numpy as np

def compute_fingerprints(smiles_list: list[str], radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Compute Morgan fingerprints for a list of SMILES."""
    # Create generator once for efficiency
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)

    fps = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue  # Skip invalid SMILES
        arr = fpgen.GetFingerprintAsNumPy(mol)
        fps.append(arr)
    return np.array(fps)
```

#### Skill File Location

Skills are defined in `skills/*/SKILL.md`. Each skill contains code examples and patterns.

---

### Commands

Commands are slash commands you can type to invoke specific workflows. They're shortcuts for common tasks.

#### Available Commands

| Command | Description | Example |
|---------|-------------|---------|
| `/plan` | Create implementation plan | `/plan add attention to encoder` |
| `/pytest` | Run TDD workflow | `/pytest implement similarity function` |
| `/code-review` | Review code quality | `/code-review src/models/vae.py` |
| `/env-fix` | Fix environment issues | `/env-fix ImportError: torch` |
| `/train` | Set up training | `/train --config configs/vae.toml` |
| `/eval` | Evaluate model | `/eval checkpoints/best.pt` |
| `/config` | Generate config file | `/config create VAE experiment` |
| `/sample` | Generate molecules | `/sample 1000 molecules` |
| `/checkpoint` | Manage checkpoints | `/checkpoint save training state` |
| `/refactor-clean` | Clean up code | `/refactor-clean src/models/` |
| `/update-docs` | Update documentation | `/update-docs src/models/vae.py` |
| `/verify` | Run all checks | `/verify src/` |

#### How to Use Commands

Simply type the command in Claude Code:

```
You: /train --config configs/experiment.toml

Claude: ## Training Setup

### 1. Configuration Validation
Checking configs/experiment.toml...
Config valid ✓

### 2. GPU Check
CUDA available: True
GPU: NVIDIA A100 ✓

### 3. Training Command
python train.py --config configs/experiment.toml --seed 42
```

#### Command File Location

Commands are defined in `commands/*.md`. Each file describes what the command does.

---

### Rules

Rules are guidelines that Claude ALWAYS follows. They ensure consistent code quality and practices.

#### Available Rules

| Rule | Description |
|------|-------------|
| `agents.md` | When to use which agent |
| `pytest-testing.md` | Testing requirements (80% coverage, TDD) |
| `python-style.md` | Code style (type hints, docstrings) |
| `git-workflow.md` | Commit conventions |
| `ml-patterns.md` | Required ML patterns (seeds, device management) |
| `experiment-standards.md` | Reproducibility requirements |
| `hooks.md` | Hook usage guidelines |

#### Key Rules Enforced

**Always Set Random Seeds**
```python
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

**Always Use Type Hints**
```python
# Required
def encode(smiles: str, max_len: int = 100) -> torch.Tensor:
    ...

# Not allowed
def encode(smiles, max_len=100):
    ...
```

**Always Validate SMILES**
```python
mol = Chem.MolFromSmiles(smiles)
if mol is None:
    raise ValueError(f"Invalid SMILES: {smiles}")
```

#### Rule File Location

Rules are in `rules/*.md`. **You must manually copy them to `~/.claude/rules/`** for them to work.

---

### Contexts

Contexts define behavioral modes for different activities. They change how Claude prioritizes tasks.

#### Available Contexts

| Context | Mode | Focus |
|---------|------|-------|
| `training.md` | Active training | Loss monitoring, checkpoints, GPU |
| `review.md` | Code review | Quality, correctness, best practices |
| `exploration.md` | Research | Understanding code, investigating options |
| `inference.md` | Model inference | Speed, memory, output validation |

#### How Contexts Work

Contexts are activated based on what you're doing:

- Running `/train` → Training context activated
- Running `/code-review` → Review context activated
- Asking "how does this work?" → Exploration context activated

Each context has different priorities. For example:

**Training Context Priorities:**
1. Training stability
2. Metric improvement
3. Resource efficiency

**Review Context Priorities:**
1. Correctness
2. Reproducibility
3. Code quality

#### Context File Location

Contexts are in `contexts/*.md`.

---

### Hooks

Hooks are automated actions that run when Claude uses certain tools. They provide automated checks and formatting.

#### Available Hooks

| Hook | Trigger | Action |
|------|---------|--------|
| GPU Check | Before `python train.py` | Shows GPU availability |
| Ruff Auto-fix | After editing `.py` files | Runs `ruff check --fix` |
| Mypy Check | After editing `.py` files | Runs type checking |
| Print Warning | After editing `.py` files | Warns about `print()` statements |
| Session Start | When starting Claude | Shows Python/PyTorch/RDKit versions |

#### Example Hook Output

When Claude edits a Python file:

```
[Hook] Running ruff check --fix...
[Hook] Running mypy...
src/models/vae.py:45: error: Missing return type annotation
[Hook] WARNING: print() at line 23: print(f"Debug: {value}")
```

When Claude runs a training command:

```
[Hook] GPU available: True
[Hook] Device count: 2
[Hook] GPU 0: NVIDIA A100
[Hook] GPU 1: NVIDIA A100
```

#### Hook File Location

Hooks are defined in `hooks/hooks.json`.

---

## Usage Examples

### Example 1: Starting a New ML Project

```
You: I want to implement a VAE for molecular generation using SMILES

Claude: I'll create a comprehensive implementation plan.

/plan VAE for molecular generation

[Claude uses planner agent to create detailed steps]

## Implementation Plan: Molecular VAE

### Phase 1: Data Pipeline
1. Create MoleculeDataset (src/data/dataset.py)
2. Implement SMILES tokenizer (src/data/tokenizer.py)
3. Create DataLoader with padding (src/data/loader.py)

### Phase 2: Model
4. Implement LSTM Encoder (src/models/encoder.py)
5. Implement LSTM Decoder (src/models/decoder.py)
6. Create VAE combining both (src/models/vae.py)

### Phase 3: Training
7. Implement training loop (src/training/trainer.py)
8. Add checkpointing and logging

Shall I proceed with implementation?
```

### Example 2: Writing Tests First (TDD)

```
You: /pytest implement Tanimoto similarity function

Claude: # TDD Session: Tanimoto Similarity

## Step 1: Write Failing Test (RED)

# tests/test_similarity.py
import pytest
import numpy as np
from src.utils.similarity import tanimoto_similarity

def test_tanimoto_identical():
    fp = np.array([1, 0, 1, 1])
    assert tanimoto_similarity(fp, fp) == 1.0

def test_tanimoto_orthogonal():
    fp1 = np.array([1, 0, 0])
    fp2 = np.array([0, 1, 0])
    assert tanimoto_similarity(fp1, fp2) == 0.0

## Step 2: Run Tests - Verify FAIL
pytest tests/test_similarity.py -v
# FAILED - ModuleNotFoundError

## Step 3: Implement (GREEN)

# src/utils/similarity.py
import numpy as np

def tanimoto_similarity(fp1: np.ndarray, fp2: np.ndarray) -> float:
    """Compute Tanimoto similarity between fingerprints."""
    intersection = np.sum(fp1 & fp2)
    union = np.sum(fp1 | fp2)
    return intersection / union if union > 0 else 0.0

## Step 4: Run Tests - PASS
pytest tests/test_similarity.py -v
# PASSED
```

### Example 3: Code Review

```
You: /code-review src/training/trainer.py

Claude: ## Code Review: src/training/trainer.py

### Critical Issues

- **Line 45**: Missing `torch.no_grad()` in validation
  - Impact: Unnecessary memory usage
  - Fix: Add `with torch.no_grad():` wrapper

- **Line 78**: No gradient clipping
  - Impact: Potential gradient explosion
  - Fix: Add `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`

### Warnings

- **Line 12**: Missing type hints on `train_epoch` function
- **Line 34**: Hardcoded learning rate (should be in config)

### Checklist
- [x] Device management correct
- [ ] Gradient clipping - MISSING
- [x] Checkpoints save optimizer state
- [ ] Type hints complete - PARTIAL
```

### Example 4: Fixing Environment Issues

```
You: /env-fix ImportError: cannot import name 'Chem' from 'rdkit'

Claude: ## Environment Diagnosis

### Checking RDKit Installation
python -c "from rdkit import Chem"
# ImportError: No module named 'rdkit'

RDKit is not installed.

### Solution

# Install via conda (recommended)
conda install -c conda-forge rdkit

# Verify
python -c "from rdkit import Chem; print('Success!')"
# Success!
```

### Example 5: Generating Molecules

```
You: /sample 1000 molecules from checkpoints/vae_best.pt

Claude: ## Molecular Sampling

### Loading Model
model = load_checkpoint("checkpoints/vae_best.pt")
model.eval()

### Generating Samples
Generated: 1000 molecules

### Results
| Metric | Value |
|--------|-------|
| Valid | 923 (92.3%) |
| Unique | 891 (96.5%) |
| Diversity | 0.78 |

### Sample Molecules
1. CCOc1ccc(NC(=O)c2ccccc2)cc1
2. CN1CCN(c2ccccc2)CC1
3. Cc1ccc(C(=O)NC2CCCCC2)cc1

Saved to: generated_molecules.csv
```

---

## Best Practices

### 1. Always Plan Before Implementing

Use `/plan` for any non-trivial task:

```
You: /plan add attention mechanism to encoder
```

### 2. Write Tests First

Use `/pytest` to follow TDD:

```
You: /pytest implement molecular fingerprint caching
```

### 3. Review Code Before Committing

Use `/code-review` after making changes:

```
You: /code-review src/models/
```

### 4. Keep Experiments Reproducible

Always include in your code:
- Random seed setting
- Configuration files (TOML)
- Environment export

### 5. Validate Molecular Data

Always check SMILES validity:

```python
mol = Chem.MolFromSmiles(smiles)
if mol is None:
    raise ValueError(f"Invalid SMILES: {smiles}")
```

### 6. Use Proper Device Management

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
data = data.to(device)
```

### 7. Set Train/Eval Modes

```python
model.train()  # During training
model.eval()   # During inference
with torch.no_grad():  # No gradients for inference
    output = model(input)
```

---

## Directory Structure

```
cc-settings-py/
├── CLAUDE.md                           # Main guidance file
├── README.md                           # This file
├── agents/                             # Specialized agents
│   ├── planner.md                      # Experiment planning
│   ├── code-reviewer.md                # Code quality review
│   ├── architect.md                    # Architecture design
│   ├── tdd-guide.md                    # Test-driven development
│   ├── build-error-resolver.md         # Environment fixes
│   ├── ml-trainer.md                   # Training workflows
│   ├── cheminformatics-reviewer.md     # Molecular code review
│   ├── refactor-cleaner.md             # Code cleanup
│   └── doc-updater.md                  # Documentation
├── skills/                             # Domain knowledge
│   ├── pytest-workflow/SKILL.md
│   ├── python-standards/SKILL.md
│   ├── ml-patterns/SKILL.md
│   ├── pytorch-best-practices/SKILL.md
│   ├── cheminformatics-patterns/SKILL.md
│   ├── molgen-patterns/SKILL.md
│   ├── experiment-management/SKILL.md
│   ├── config-management/SKILL.md
│   ├── data-pipeline/SKILL.md
│   ├── model-evaluation/SKILL.md
│   └── reproducibility/SKILL.md
├── commands/                           # Slash commands
│   ├── plan.md
│   ├── pytest.md
│   ├── code-review.md
│   ├── env-fix.md
│   ├── train.md
│   ├── eval.md
│   ├── config.md
│   ├── sample.md
│   ├── checkpoint.md
│   ├── refactor-clean.md
│   ├── update-docs.md
│   └── verify.md
├── hooks/
│   └── hooks.json                      # Automated actions
├── rules/                              # Always-follow guidelines
│   ├── agents.md
│   ├── pytest-testing.md
│   ├── python-style.md
│   ├── git-workflow.md
│   ├── ml-patterns.md
│   ├── experiment-standards.md
│   └── hooks.md
└── contexts/                           # Behavioral modes
    ├── training.md
    ├── review.md
    ├── exploration.md
    └── inference.md
```

---

## Troubleshooting

### Commands Not Working

Make sure you've installed the settings in the correct location:

```bash
# Check if CLAUDE.md exists
ls ~/.claude/CLAUDE.md

# Or in your project
ls /path/to/project/.claude/CLAUDE.md
```

### Rules Not Applied

Rules must be manually copied:

```bash
cp -r cc-settings-py/rules/* ~/.claude/rules/
```

### Hooks Not Running

Check that `hooks.json` is valid:

```bash
python -c "import json; json.load(open('hooks/hooks.json'))"
```

---

## Customization

### Adding New Commands

Create a new file in `commands/`:

```markdown
---
description: My custom command description
---

# My Command

What this command does and how to use it.
```

### Adding New Skills

Create a new directory in `skills/` with a `SKILL.md` file:

```markdown
---
name: my-skill
description: What this skill teaches
---

# My Skill

Code patterns and examples...
```

### Modifying Hooks

Edit `hooks/hooks.json` to add or modify hooks:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "tool == \"Edit\" && tool_input.file_path matches \"\\.py$\"",
        "hooks": [
          {
            "type": "command",
            "command": "your-command-here"
          }
        ]
      }
    ]
  }
}
```

---

## License

This configuration is provided as-is for research and educational purposes.

## Acknowledgments

Adapted from [everything-claude-code](https://github.com/anthropics/everything-claude-code) for cheminformatics and ML research workflows.
# cc-settings-py

---
description: Plan ML experiments and implementations. Creates detailed step-by-step plans before coding.
---

# Plan Command

Invokes the **planner** agent to create comprehensive implementation plans for ML experiments and features.

## What This Command Does

1. **Analyze Requirements** - Understand experiment goals and constraints
2. **Review Codebase** - Identify relevant existing code and patterns
3. **Create Step-by-Step Plan** - Detailed implementation steps with dependencies
4. **Identify Risks** - Potential issues and mitigations
5. **Define Success Criteria** - How to verify the implementation

## When to Use

Use `/plan` when:
- Starting a new ML experiment
- Implementing a new model architecture
- Adding a significant feature to the codebase
- Refactoring data pipelines
- Setting up training infrastructure

## Example Usage

```
User: /plan I need to implement a VAE for molecular generation

Agent (planner):
# Implementation Plan: Molecular VAE

## Overview
Implement a Variational Autoencoder for SMILES-based molecular generation
with RNN encoder/decoder and latent space sampling.

## Requirements
- Dataset: SMILES strings (train/val/test splits)
- Model: VAE with LSTM encoder and decoder
- Metrics: Validity, uniqueness, diversity, reconstruction accuracy

## Implementation Steps

### Phase 1: Data Pipeline
1. **Create MoleculeDataset** (src/data/dataset.py)
   - Load SMILES from CSV
   - Tokenize and pad sequences
   - Dependencies: None

2. **Create DataLoader** (src/data/loader.py)
   - Batch collation with padding
   - Dependencies: Step 1

### Phase 2: Model
3. **Implement Encoder** (src/models/encoder.py)
   - LSTM with attention
   - Output: mu, logvar
   - Dependencies: None

4. **Implement Decoder** (src/models/decoder.py)
   - LSTM with teacher forcing
   - Dependencies: None

5. **Implement VAE** (src/models/vae.py)
   - Combine encoder/decoder
   - Reparameterization trick
   - Dependencies: Steps 3, 4

### Phase 3: Training
6. **Training Loop** (src/training/trainer.py)
   - VAE loss (recon + KL)
   - Checkpointing
   - Dependencies: Steps 1, 2, 5

## Success Criteria
- [ ] Model trains without errors
- [ ] Validation loss decreases
- [ ] Generated SMILES >80% valid
```

## Related Commands

- `/train` - Execute training after planning
- `/pytest` - Write tests for planned components
- `/code-review` - Review implemented code

## Related Agent

This command invokes the `planner` agent at `agents/planner.md`.

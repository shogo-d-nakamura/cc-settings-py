---
description: Evaluate trained models with comprehensive metrics and analysis.
---

# Eval Command

Run model evaluation with generation metrics, property distributions, and analysis.

## What This Command Does

1. **Load Checkpoint** - Load trained model
2. **Run Evaluation** - Generate samples or predictions
3. **Compute Metrics** - Validity, uniqueness, diversity, etc.
4. **Generate Report** - Summary of results

## When to Use

Use `/eval` when:
- Evaluating a trained model
- Comparing model versions
- Generating final results
- Running benchmark comparisons

## Evaluation Types

### Generation Evaluation

```bash
python evaluate.py \
    --checkpoint checkpoints/best.pt \
    --config configs/eval.toml \
    --num-samples 10000 \
    --output results/eval_results.json
```

### Property Prediction

```bash
python evaluate.py \
    --checkpoint checkpoints/best.pt \
    --test-data data/test.csv \
    --task regression \
    --output results/predictions.csv
```

## Example Usage

```
User: /eval checkpoints/vae_best.pt

Agent:
## Model Evaluation

### Loading Model
```bash
python -c "import torch; m = torch.load('checkpoints/vae_best.pt'); print('Loaded')"
# Loaded ✓
```

### Generating Samples
Generating 10,000 molecules...

### Generation Metrics
| Metric | Value |
|--------|-------|
| Validity | 0.92 |
| Uniqueness | 0.85 |
| Diversity | 0.78 |
| Novelty | 0.95 |

### Property Distributions
| Property | Generated (mean±std) | Reference (mean±std) |
|----------|---------------------|---------------------|
| MW | 324.5±85.2 | 318.2±82.1 |
| LogP | 2.8±1.5 | 2.6±1.4 |
| TPSA | 68.2±28.4 | 65.8±26.9 |

### Sample Generated Molecules
1. CCOc1ccc(NC(=O)c2ccccc2)cc1
2. CN1CCN(c2ccccc2)CC1
3. Cc1ccc(C(=O)NC2CCCCC2)cc1
```

## Metrics Computed

### Generation Metrics
- **Validity**: % valid SMILES
- **Uniqueness**: % unique among valid
- **Diversity**: 1 - avg pairwise Tanimoto
- **Novelty**: % not in training set

### Property Metrics
- **KL Divergence**: Property distribution match
- **Wasserstein Distance**: Distribution similarity

### Prediction Metrics (Regression)
- RMSE, MAE, R², Pearson correlation

### Prediction Metrics (Classification)
- Accuracy, Precision, Recall, F1, AUROC

## Related Commands

- `/train` - Train model first
- `/sample` - Generate samples only
- `/code-review` - Review evaluation code

## Related Skill

See `skills/model-evaluation/SKILL.md` for detailed metrics implementations.

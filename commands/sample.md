---
description: Generate molecular samples from trained models.
---

# Sample Command

Generate molecules using trained generative models.

## What This Command Does

1. **Load Model** - Load trained checkpoint
2. **Configure Sampling** - Set temperature, num_samples
3. **Generate** - Run sampling/generation
4. **Validate** - Check validity of generated molecules
5. **Save Results** - Export to CSV/SDF

## When to Use

Use `/sample` when:
- Generating molecules for screening
- Testing model outputs
- Creating datasets from trained models
- Visualizing generated molecules

## Sampling Options

```bash
# Basic sampling
python sample.py \
    --checkpoint checkpoints/best.pt \
    --num-samples 1000 \
    --output generated.csv

# With temperature
python sample.py \
    --checkpoint checkpoints/best.pt \
    --num-samples 1000 \
    --temperature 1.2 \
    --output generated.csv

# Beam search (deterministic)
python sample.py \
    --checkpoint checkpoints/best.pt \
    --num-samples 100 \
    --beam-size 5 \
    --output generated.csv
```

## Example Usage

```
User: /sample 1000 molecules from checkpoints/vae.pt

Agent:
## Molecular Sampling

### Loading Model
```python
model = load_checkpoint("checkpoints/vae.pt")
model.eval()
```

### Generating Samples
```python
with torch.no_grad():
    samples = model.sample(num_samples=1000, temperature=1.0)
```

### Results
- Total generated: 1000
- Valid: 923 (92.3%)
- Unique: 891 (96.5% of valid)

### Sample Molecules
| # | SMILES | MW | LogP |
|---|--------|----|----- |
| 1 | CCOc1ccc(NC(=O)c2ccccc2)cc1 | 255.3 | 2.8 |
| 2 | CN1CCN(c2ccccc2)CC1 | 176.3 | 1.2 |
| 3 | Cc1ccc(C(=O)NC2CCCCC2)cc1 | 217.3 | 3.1 |

### Output
Saved to: generated_molecules.csv
```

## Sampling Strategies

### Temperature Sampling
- **temperature=1.0**: Standard sampling
- **temperature<1.0**: More conservative, higher validity
- **temperature>1.0**: More diverse, lower validity

### Latent Space Sampling (VAE)
```python
# Sample from prior
z = torch.randn(num_samples, latent_dim)
molecules = model.decode(z)

# Interpolate between molecules
z1 = model.encode(mol1)
z2 = model.encode(mol2)
z_interp = torch.lerp(z1, z2, torch.linspace(0, 1, 10))
```

### Conditional Generation
```python
# Generate with target property
molecules = model.sample(
    num_samples=100,
    target_property={"logp": 2.5},
)
```

## Output Formats

### CSV
```csv
smiles,mw,logp,validity
CCO,46.07,−0.14,True
c1ccccc1,78.11,1.90,True
```

### SDF (with 3D coordinates)
```bash
python sample.py --output generated.sdf --generate-3d
```

## Related Commands

- `/train` - Train model first
- `/eval` - Full evaluation with metrics

## Related Skills

See `skills/molgen-patterns/SKILL.md` for generation patterns.

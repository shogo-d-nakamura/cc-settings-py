---
name: model-evaluation
description: Patterns for model evaluation, metrics computation, and validation strategies for molecular ML models.
---

# Model Evaluation Patterns

Patterns for evaluating machine learning models in molecular generation and property prediction.

## Generation Metrics

### Core Generation Metrics

```python
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
import numpy as np
from collections import Counter

def compute_generation_metrics(
    generated_smiles: list[str],
    reference_smiles: list[str] | None = None,
) -> dict[str, float]:
    """Compute comprehensive generation metrics."""

    # Parse molecules
    generated_mols = []
    for smiles in generated_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            generated_mols.append(mol)

    # Validity
    validity = len(generated_mols) / len(generated_smiles) if generated_smiles else 0

    # Uniqueness
    canonical_smiles = [Chem.MolToSmiles(m, canonical=True) for m in generated_mols]
    unique_smiles = set(canonical_smiles)
    uniqueness = len(unique_smiles) / len(generated_mols) if generated_mols else 0

    # Internal Diversity
    diversity = compute_internal_diversity(generated_mols)

    metrics = {
        "validity": validity,
        "uniqueness": uniqueness,
        "diversity": diversity,
        "valid_unique": validity * uniqueness,
    }

    # Novelty (requires reference)
    if reference_smiles:
        reference_set = set(reference_smiles)
        novel_smiles = [s for s in unique_smiles if s not in reference_set]
        novelty = len(novel_smiles) / len(unique_smiles) if unique_smiles else 0
        metrics["novelty"] = novelty

    return metrics

def compute_internal_diversity(mols: list[Chem.Mol], n_samples: int = 1000) -> float:
    """Compute internal diversity via average pairwise Tanimoto distance."""
    if len(mols) < 2:
        return 0.0

    # Compute fingerprints using generator
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [fpgen.GetFingerprint(m) for m in mols]

    # Sample pairs if too many molecules
    if len(fps) > n_samples:
        indices = np.random.choice(len(fps), n_samples, replace=False)
        fps = [fps[i] for i in indices]

    # Compute pairwise similarities
    similarities = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            similarities.append(sim)

    return 1 - np.mean(similarities)
```

### Distribution Metrics

```python
from scipy import stats

def compute_distribution_metrics(
    generated_smiles: list[str],
    reference_smiles: list[str],
    properties: list[str] = ["mw", "logp", "tpsa", "qed"],
) -> dict[str, float]:
    """Compare property distributions between generated and reference."""
    from rdkit.Chem import Descriptors, QED

    property_fns = {
        "mw": Descriptors.MolWt,
        "logp": Descriptors.MolLogP,
        "tpsa": Descriptors.TPSA,
        "qed": QED.qed,
        "hbd": Descriptors.NumHDonors,
        "hba": Descriptors.NumHAcceptors,
    }

    def get_property_values(smiles_list: list[str], prop: str) -> np.ndarray:
        values = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                values.append(property_fns[prop](mol))
        return np.array(values)

    metrics = {}

    for prop in properties:
        gen_values = get_property_values(generated_smiles, prop)
        ref_values = get_property_values(reference_smiles, prop)

        if len(gen_values) == 0 or len(ref_values) == 0:
            continue

        # KL divergence (via histogram approximation)
        hist_gen, bins = np.histogram(gen_values, bins=50, density=True)
        hist_ref, _ = np.histogram(ref_values, bins=bins, density=True)

        # Add small epsilon to avoid log(0)
        hist_gen = hist_gen + 1e-10
        hist_ref = hist_ref + 1e-10

        kl_div = stats.entropy(hist_gen, hist_ref)
        metrics[f"{prop}_kl"] = kl_div

        # Wasserstein distance
        wd = stats.wasserstein_distance(gen_values, ref_values)
        metrics[f"{prop}_wd"] = wd

    return metrics
```

### Frechet ChemNet Distance

```python
def compute_fcd(
    generated_smiles: list[str],
    reference_smiles: list[str],
    model_path: str | None = None,
) -> float:
    """Compute Frechet ChemNet Distance."""
    try:
        from fcd import get_fcd, load_ref_model, canonical_smiles
    except ImportError:
        raise ImportError("Install fcd: pip install fcd")

    # Filter valid SMILES
    gen_valid = [s for s in generated_smiles if Chem.MolFromSmiles(s) is not None]
    ref_valid = [s for s in reference_smiles if Chem.MolFromSmiles(s) is not None]

    # Compute FCD
    fcd_score = get_fcd(gen_valid, ref_valid)

    return fcd_score
```

## Property Prediction Metrics

### Regression Metrics

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute regression metrics."""
    return {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "pearson": np.corrcoef(y_true, y_pred)[0, 1],
    }
```

### Classification Metrics

```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute classification metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "f1": f1_score(y_true, y_pred, average="weighted"),
    }

    if y_prob is not None:
        if len(np.unique(y_true)) == 2:  # Binary classification
            metrics["auroc"] = roc_auc_score(y_true, y_prob)
            metrics["auprc"] = average_precision_score(y_true, y_prob)

    return metrics
```

## Evaluation Pipeline

### Complete Evaluation

```python
@dataclass
class EvaluationResults:
    """Container for evaluation results."""

    metrics: dict[str, float]
    generated_smiles: list[str] | None = None
    predictions: np.ndarray | None = None
    metadata: dict | None = None

    def save(self, path: Path) -> None:
        """Save results to JSON."""
        import json
        with open(path, "w") as f:
            json.dump({
                "metrics": self.metrics,
                "metadata": self.metadata,
            }, f, indent=2)

def evaluate_generator(
    model: nn.Module,
    num_samples: int,
    reference_smiles: list[str],
    device: torch.device,
) -> EvaluationResults:
    """Evaluate generative model."""
    model.eval()

    with torch.no_grad():
        generated = model.generate(num_samples, device)

    # Decode to SMILES
    generated_smiles = decode_sequences(generated)

    # Compute metrics
    metrics = compute_generation_metrics(generated_smiles, reference_smiles)
    dist_metrics = compute_distribution_metrics(generated_smiles, reference_smiles)
    metrics.update(dist_metrics)

    return EvaluationResults(
        metrics=metrics,
        generated_smiles=generated_smiles,
    )

def evaluate_predictor(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    task: str = "regression",
) -> EvaluationResults:
    """Evaluate predictive model."""
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(batch)
            predictions = outputs["predictions"]

            if task == "classification":
                probs = torch.softmax(predictions, dim=-1)
                preds = predictions.argmax(dim=-1)
                all_probs.extend(probs[:, 1].cpu().numpy())
            else:
                preds = predictions.squeeze()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    if task == "regression":
        metrics = compute_regression_metrics(y_true, y_pred)
    else:
        y_prob = np.array(all_probs) if all_probs else None
        metrics = compute_classification_metrics(y_true, y_pred, y_prob)

    return EvaluationResults(
        metrics=metrics,
        predictions=y_pred,
    )
```

## Cross-Validation

```python
from sklearn.model_selection import KFold

def cross_validate(
    dataset: Dataset,
    model_fn: Callable[[], nn.Module],
    train_fn: Callable,
    evaluate_fn: Callable,
    n_folds: int = 5,
    seed: int = 42,
) -> dict[str, list[float]]:
    """K-fold cross-validation."""
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_metrics = {}

    for fold, (train_idx, val_idx) in enumerate(kfold.split(range(len(dataset)))):
        print(f"Fold {fold + 1}/{n_folds}")

        # Create data subsets
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        # Create fresh model
        model = model_fn()

        # Train
        train_fn(model, train_subset)

        # Evaluate
        results = evaluate_fn(model, val_subset)

        # Record metrics
        for key, value in results.metrics.items():
            if key not in fold_metrics:
                fold_metrics[key] = []
            fold_metrics[key].append(value)

    # Compute mean and std
    summary = {}
    for key, values in fold_metrics.items():
        summary[f"{key}_mean"] = np.mean(values)
        summary[f"{key}_std"] = np.std(values)

    return summary
```

## Scaffold Split Evaluation

```python
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict

def scaffold_split(
    smiles_list: list[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> tuple[list[int], list[int], list[int]]:
    """Split by Murcko scaffolds for rigorous evaluation."""

    # Group by scaffold
    scaffold_to_indices = defaultdict(list)

    for idx, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        scaffold_to_indices[scaffold].append(idx)

    # Sort scaffolds by size
    scaffolds = list(scaffold_to_indices.keys())
    scaffolds.sort(key=lambda s: len(scaffold_to_indices[s]), reverse=True)

    # Assign scaffolds to splits
    train_idx, val_idx, test_idx = [], [], []
    train_cutoff = train_ratio
    val_cutoff = train_ratio + val_ratio

    current_ratio = 0
    total = len(smiles_list)

    for scaffold in scaffolds:
        indices = scaffold_to_indices[scaffold]

        if current_ratio < train_cutoff:
            train_idx.extend(indices)
        elif current_ratio < val_cutoff:
            val_idx.extend(indices)
        else:
            test_idx.extend(indices)

        current_ratio += len(indices) / total

    return train_idx, val_idx, test_idx
```

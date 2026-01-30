---
name: cheminformatics-patterns
description: RDKit patterns for molecular processing, SMILES handling, fingerprints, and chemical property calculations.
---

# Cheminformatics Patterns

Common patterns for working with molecular data using RDKit.

## SMILES Processing

### Validation and Canonicalization

```python
from rdkit import Chem

def validate_smiles(smiles: str) -> bool:
    """Check if SMILES is valid."""
    return Chem.MolFromSmiles(smiles) is not None

def canonicalize_smiles(smiles: str) -> str | None:
    """Convert SMILES to canonical form."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)

def canonicalize_with_stereo(smiles: str) -> str | None:
    """Canonicalize preserving stereochemistry."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
```

### Batch Processing

```python
def process_smiles_batch(
    smiles_list: list[str],
    canonicalize: bool = True,
) -> tuple[list[str], list[int]]:
    """Process batch of SMILES, returning valid ones and failed indices."""
    valid_smiles = []
    failed_indices = []

    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            failed_indices.append(i)
            continue

        if canonicalize:
            smiles = Chem.MolToSmiles(mol, canonical=True)

        valid_smiles.append(smiles)

    return valid_smiles, failed_indices
```

### SMILES Augmentation

```python
from rdkit import Chem

def augment_smiles(
    smiles: str,
    num_augmentations: int = 10,
    random_seed: int | None = None,
) -> list[str]:
    """Generate randomized SMILES representations."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    if random_seed is None:
        random_smiles = Chem.MolToRandomSmilesVect(mol, numSmiles=num_augmentations)
    else:
        random_smiles = Chem.MolToRandomSmilesVect(
            mol,
            numSmiles=num_augmentations,
            randomSeed=random_seed,
        )

    return list(set(random_smiles))  # Remove duplicates
```

## Fingerprints

### Morgan (ECFP) Fingerprints

```python
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs
import numpy as np

# Create generators once (reusable for efficiency)
def get_morgan_generator(radius: int = 2, n_bits: int = 2048):
    """Create Morgan fingerprint generator."""
    return rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=n_bits,
    )

def get_fcfp_generator(radius: int = 2, n_bits: int = 2048):
    """Create FCFP (feature-based) fingerprint generator."""
    return rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=n_bits,
        atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen(),
    )

def compute_morgan_fingerprint(
    smiles: str,
    radius: int = 2,
    n_bits: int = 2048,
) -> np.ndarray:
    """Compute Morgan fingerprint as numpy array."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    fpgen = get_morgan_generator(radius, n_bits)
    return fpgen.GetFingerprintAsNumPy(mol)

def compute_fcfp_fingerprint(
    smiles: str,
    radius: int = 2,
    n_bits: int = 2048,
) -> np.ndarray:
    """Compute feature-based circular fingerprint (FCFP)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    fpgen = get_fcfp_generator(radius, n_bits)
    return fpgen.GetFingerprintAsNumPy(mol)

# Different output types from generators
def compute_morgan_variants(smiles: str, fpgen):
    """Show different fingerprint output types."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    return {
        "bit_vector": fpgen.GetFingerprint(mol),           # ExplicitBitVect
        "sparse": fpgen.GetSparseFingerprint(mol),         # SparseBitVect
        "count": fpgen.GetCountFingerprint(mol),           # UIntSparseIntVect
        "sparse_count": fpgen.GetSparseCountFingerprint(mol),
        "numpy": fpgen.GetFingerprintAsNumPy(mol),         # np.ndarray
        "numpy_count": fpgen.GetCountFingerprintAsNumPy(mol),
    }
```

### MACCS Keys

```python
from rdkit.Chem import MACCSkeys

def compute_maccs_keys(smiles: str) -> np.ndarray:
    """Compute MACCS keys (166 bits)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    fp = MACCSkeys.GenMACCSKeys(mol)

    arr = np.zeros(167, dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr
```

### Other Fingerprint Types

```python
# RDKit topological fingerprints
rdkgen = rdFingerprintGenerator.GetRDKitFPGenerator(
    maxPath=7,
    fpSize=2048,
)

# Atom pair fingerprints
apgen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=2048)

# Topological torsion fingerprints
ttgen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=2048)
```

### Batch Fingerprint Computation

```python
def compute_fingerprints_batch(
    smiles_list: list[str],
    fp_type: str = "morgan",
    radius: int = 2,
    n_bits: int = 2048,
) -> np.ndarray:
    """Compute fingerprints for a batch of SMILES."""
    # Create generator once for efficiency
    generators = {
        "morgan": get_morgan_generator(radius, n_bits),
        "fcfp": get_fcfp_generator(radius, n_bits),
    }

    if fp_type in generators:
        fpgen = generators[fp_type]
        fingerprints = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fingerprints.append(fpgen.GetFingerprintAsNumPy(mol))
        return np.array(fingerprints)

    elif fp_type == "maccs":
        fingerprints = []
        for smiles in smiles_list:
            try:
                fp = compute_maccs_keys(smiles)
                fingerprints.append(fp)
            except ValueError:
                continue
        return np.array(fingerprints)

    raise ValueError(f"Unknown fingerprint type: {fp_type}")
```

### Fingerprint Bit Information

```python
from rdkit.Chem import AllChem

def get_morgan_bit_info(smiles: str, radius: int = 2) -> dict:
    """Get detailed information about fingerprint bits."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius)
    ao = rdFingerprintGenerator.AdditionalOutput()
    ao.CollectBitInfoMap()

    _ = fpgen.GetSparseCountFingerprint(mol, additionalOutput=ao)

    # Returns {bit_id: [(atom_idx, radius), ...]}
    return ao.GetBitInfoMap()
```

## Similarity Calculations

```python
from rdkit import DataStructs

def tanimoto_similarity_rdkit(fp1, fp2) -> float:
    """Compute Tanimoto similarity using RDKit (for fingerprint objects)."""
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def dice_similarity_rdkit(fp1, fp2) -> float:
    """Compute Dice similarity using RDKit."""
    return DataStructs.DiceSimilarity(fp1, fp2)

def bulk_tanimoto_similarity_rdkit(query_fp, database_fps: list) -> list[float]:
    """Compute Tanimoto similarity against database using RDKit."""
    return DataStructs.BulkTanimotoSimilarity(query_fp, database_fps)

# NumPy-based similarity (for numpy array fingerprints)
def tanimoto_similarity(fp1: np.ndarray, fp2: np.ndarray) -> float:
    """Compute Tanimoto similarity between numpy fingerprints."""
    intersection = np.sum(fp1 & fp2)
    union = np.sum(fp1 | fp2)
    return intersection / union if union > 0 else 0.0

def bulk_tanimoto_similarity(
    query_fp: np.ndarray,
    database_fps: np.ndarray,
) -> np.ndarray:
    """Compute Tanimoto similarity against database (numpy)."""
    # Efficient vectorized computation
    intersection = np.sum(query_fp & database_fps, axis=1)
    union = np.sum(query_fp | database_fps, axis=1)
    return np.divide(intersection, union, where=union > 0, out=np.zeros_like(union, dtype=float))

# Example: similarity search with generators
def similarity_search(
    query_smiles: str,
    database_smiles: list[str],
    threshold: float = 0.7,
    radius: int = 2,
    n_bits: int = 2048,
) -> list[tuple[str, float]]:
    """Find similar molecules above threshold."""
    fpgen = get_morgan_generator(radius, n_bits)

    query_mol = Chem.MolFromSmiles(query_smiles)
    if query_mol is None:
        raise ValueError(f"Invalid query SMILES: {query_smiles}")

    query_fp = fpgen.GetFingerprint(query_mol)

    results = []
    for smiles in database_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        fp = fpgen.GetFingerprint(mol)
        sim = DataStructs.TanimotoSimilarity(query_fp, fp)
        if sim >= threshold:
            results.append((smiles, sim))

    return sorted(results, key=lambda x: x[1], reverse=True)
```

## Molecular Descriptors

```python
from rdkit.Chem import Descriptors

DESCRIPTOR_FUNCS = {
    "mw": Descriptors.MolWt,
    "logp": Descriptors.MolLogP,
    "tpsa": Descriptors.TPSA,
    "hbd": Descriptors.NumHDonors,
    "hba": Descriptors.NumHAcceptors,
    "rotatable_bonds": Descriptors.NumRotatableBonds,
    "rings": Descriptors.RingCount,
    "aromatic_rings": Descriptors.NumAromaticRings,
    "heavy_atoms": Descriptors.HeavyAtomCount,
    "fraction_sp3": Descriptors.FractionCSP3,
}

def compute_descriptors(smiles: str) -> dict[str, float]:
    """Compute molecular descriptors."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    return {name: func(mol) for name, func in DESCRIPTOR_FUNCS.items()}

def compute_descriptors_batch(smiles_list: list[str]) -> pd.DataFrame:
    """Compute descriptors for batch of molecules."""
    records = []
    for smiles in smiles_list:
        try:
            desc = compute_descriptors(smiles)
            desc["smiles"] = smiles
            records.append(desc)
        except ValueError:
            continue
    return pd.DataFrame(records)
```

## Drug-likeness Filters

```python
def lipinski_filter(smiles: str) -> bool:
    """Check Lipinski's Rule of Five."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False

    return (
        Descriptors.MolWt(mol) <= 500 and
        Descriptors.MolLogP(mol) <= 5 and
        Descriptors.NumHDonors(mol) <= 5 and
        Descriptors.NumHAcceptors(mol) <= 10
    )

def veber_filter(smiles: str) -> bool:
    """Check Veber's rules for oral bioavailability."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False

    return (
        Descriptors.NumRotatableBonds(mol) <= 10 and
        Descriptors.TPSA(mol) <= 140
    )
```

## Scaffold Analysis

```python
from rdkit.Chem.Scaffolds import MurckoScaffold

def get_murcko_scaffold(smiles: str) -> str | None:
    """Extract Murcko scaffold."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold)

def get_generic_scaffold(smiles: str) -> str | None:
    """Extract generic scaffold (heteroatoms → C)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    generic = MurckoScaffold.MakeScaffoldGeneric(scaffold)
    return Chem.MolToSmiles(generic)

def group_by_scaffold(smiles_list: list[str]) -> dict[str, list[str]]:
    """Group molecules by Murcko scaffold."""
    scaffold_groups = defaultdict(list)

    for smiles in smiles_list:
        scaffold = get_murcko_scaffold(smiles)
        if scaffold:
            scaffold_groups[scaffold].append(smiles)

    return dict(scaffold_groups)
```

## Substructure Search

```python
def has_substructure(smiles: str, smarts: str) -> bool:
    """Check if molecule contains substructure."""
    mol = Chem.MolFromSmiles(smiles)
    pattern = Chem.MolFromSmarts(smarts)

    if mol is None or pattern is None:
        return False

    return mol.HasSubstructMatch(pattern)

# Common SMARTS patterns
FUNCTIONAL_GROUPS = {
    "hydroxyl": "[OH]",
    "carboxylic_acid": "[CX3](=O)[OX2H1]",
    "amine": "[NX3;H2,H1;!$(NC=O)]",
    "amide": "[NX3][CX3](=[OX1])[#6]",
    "ester": "[#6][CX3](=O)[OX2H0][#6]",
    "ketone": "[#6][CX3](=O)[#6]",
    "aldehyde": "[CX3H1](=O)[#6]",
    "sulfonamide": "[SX4](=[OX1])(=[OX1])([NX3])",
}

def count_functional_groups(smiles: str) -> dict[str, int]:
    """Count functional groups in molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}

    counts = {}
    for name, smarts in FUNCTIONAL_GROUPS.items():
        pattern = Chem.MolFromSmarts(smarts)
        matches = mol.GetSubstructMatches(pattern)
        counts[name] = len(matches)

    return counts
```

## 3D Conformer Generation

```python
def generate_conformers(
    smiles: str,
    num_conformers: int = 10,
    optimize: bool = True,
) -> Chem.Mol:
    """Generate 3D conformers."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    # Add hydrogens for 3D
    mol = Chem.AddHs(mol)

    # Generate conformers
    AllChem.EmbedMultipleConfs(
        mol,
        numConfs=num_conformers,
        randomSeed=42,
        pruneRmsThresh=0.5,
    )

    if optimize:
        for conf_id in range(mol.GetNumConformers()):
            AllChem.MMFFOptimizeMolecule(mol, confId=conf_id)

    return mol
```

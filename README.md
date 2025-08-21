# DrugReflector

A minimal implementation for compound ranking predictions from gene expression signatures using ensemble neural network models.

## Overview

This package provides tools for virtual drug screening using transcriptional signatures. It includes:

- **DrugReflector**: Ensemble neural network models for compound ranking predictions
- **Signature Refinement**: Tools to refine transcriptional signatures using experimental data
- **V-score Computation**: Fast vectorized v-score calculations for differential expression analysis
- **Data Utilities**: Functions for preprocessing and handling gene expression data

## Installation

### Prerequisites

- Python 3.8+
- PyTorch
- NumPy
- Pandas
- SciPy
- AnnData
- Scanpy (for signature refinement)

### Setup

1. Clone this repository:
```bash
git clone <repository-url>
cd deep-virtual-screening
```

Or install via pip:
```bash
pip install drugreflector
```

2. Install dependencies:
```bash
pip install torch numpy pandas scipy anndata scanpy
```

3. Download model checkpoints from Zenodo:
   - **DOI: 10.5281/zenodo.16912445**
   - Download all checkpoint files and place them in the `checkpoints/` directory
   - The checkpoints directory is empty in the git repository - you must download the models separately

## Model Checkpoints

The trained model checkpoints are available on Zenodo at DOI **10.5281/zenodo.16912445**.

After downloading, your directory structure should look like:
```
deep-virtual-screening/
├── checkpoints/
│   ├── model_fold_0.pt
│   ├── model_fold_1.pt
│   └── model_fold_2.pt
├── drugreflector.py
├── utils.py
└── ...
```

## Quick Start

### Basic Drug Screening

```python
import pandas as pd
import scanpy as sc
import drugreflector as dr

# All major classes available at top level
# dr.DrugReflector, dr.SignatureRefinement, dr.compute_vscores_adata, etc.

# Step 1: Load PBMC 3k dataset with cell type annotations
adata = sc.datasets.pbmc3k()  # Unfiltered dataset with more genes
annots = sc.datasets.pbmc3k_processed().obs  # Cell type annotations

# Merge annotations
adata.obs = pd.merge(adata.obs, annots, how='left', left_index=True, right_index=True)

# Step 2: Compute v-scores between two monocyte populations
vscores = dr.compute_vscores_adata(
    adata, 
    group_col='louvain',
    group1_value='CD14+ Monocytes',    # Classical monocytes
    group2_value='FCGR3A+ Monocytes'   # Non-classical monocytes
)

print(f"V-score comparison: {vscores.name}")
print(f"Computed v-scores for {len(vscores)} genes")
print(f"Top upregulated genes in FCGR3A+ vs CD14+ monocytes:")
print(vscores.nlargest(10))

# Step 3: Initialize DrugReflector with model checkpoints
model_paths = [
    'checkpoints/model_fold_0.pt',
    'checkpoints/model_fold_1.pt', 
    'checkpoints/model_fold_2.pt'
]

model = dr.DrugReflector(checkpoint_paths=model_paths)

# Step 4: Make predictions using v-scores
# DrugReflector will automatically preprocess gene names to HGNC format
predictions = model.predict(vscores, n_top=50)
print(f"Prediction results shape: {predictions.shape}")
print(f"Columns: {predictions.columns.names}")

# Access different metrics
print("\nTop 10 predicted compounds by rank:")
rank_col = ('rank', vscores.name)  # Uses informative name: 'louvain:CD14+ Monocytes->FCGR3A+ Monocytes'
print(predictions[rank_col].nsmallest(10))

print("\nTop 10 compounds by probability:")
prob_col = ('prob', vscores.name)  
print(predictions[prob_col].nlargest(10))

print(f"\nAvailable columns: {list(predictions.columns)}")
```


### Input Formats for DrugReflector

DrugReflector accepts v-score data in three formats:

```python
# 1. Pandas Series (single v-score vector)
# The series name will be used as the transition identifier in outputs
vscore_series = pd.Series([1.2, -0.8, 0.5, ...], index=['GENE1', 'GENE2', 'GENE3', ...], 
                         name='treatment:control->drug')
predictions = model.predict(vscore_series)
# Columns will be: ('rank', 'treatment:control->drug'), ('logit', 'treatment:control->drug'), etc.

# 2. Pandas DataFrame (multiple transitions/signatures)
vscores_df = pd.DataFrame({
    'GENE1': [1.2, 0.8],
    'GENE2': [-0.8, 1.1], 
    'GENE3': [0.5, -0.3]
}, index=['treatment_A', 'treatment_B'])
predictions = model.predict(vscores_df)

# 3. AnnData (v-scores in .X)
vscores_adata = AnnData(
    X=vscores_df.values,
    var=pd.DataFrame(index=vscores_df.columns),
    obs=pd.DataFrame(index=vscores_df.index)
)
predictions = model.predict(vscores_adata)
```

## Gene Symbol Requirements

**CRITICAL**: DrugReflector requires gene names in HGNC (HUGO Gene Nomenclature Committee) format for accurate predictions.

### Automatic Gene Name Preprocessing

DrugReflector automatically preprocesses gene names to be HGNC-compatible:

```python
# Example of automatic preprocessing
import pandas as pd
from drugreflector import DrugReflector

# Input with mixed gene name formats
mixed_genes = ['tp53', 'EGFR', 'ENSG00000141510.11', 'CDKN1A_at', 'il6.v2']
vscores = pd.Series([1.2, -0.8, 0.5, 2.1, -1.1], index=mixed_genes)

model = DrugReflector(checkpoint_paths=model_paths)
predictions = model.transform(vscores)

# Output shows preprocessing:
# Preprocessing gene names to HGNC format...
# Preprocessed 4/5 gene names for HGNC compatibility
# Examples of changes:
#   tp53 -> TP53
#   ENSG00000141510.11 -> ENSG00000141510
#   CDKN1A_at -> CDKN1A
#   il6.v2 -> IL6
```

### HGNC Format Rules Applied

1. **Uppercase conversion**: `tp53` → `TP53`
2. **Remove Ensembl versions**: `ENSG00000141510.11` → `ENSG00000141510`  
3. **Remove Affymetrix suffixes**: `CDKN1A_at` → `CDKN1A`
4. **Remove version numbers**: `IL6.v2` → `IL6`
5. **Clean non-standard characters**: Keep only `A-Z`, `0-9`, and `-`

### Best Practices

- **Preferred**: Use official HGNC symbols (`TP53`, `EGFR`, `CDKN1A`)
- **Acceptable**: Mixed case, common prefixes/suffixes (automatically cleaned)
- **Check coverage**: Ensure your genes overlap with the 978 landmark genes used by the model

## Signature Refinement

Refine transcriptional signatures using paired transcriptional + phenotypic data:

```python
import drugreflector as dr
import pandas as pd

# Starting signature (pandas Series with gene names as index)
starting_signature = pd.Series([1.2, -0.8, 0.5, ...], 
                              index=['GENE1', 'GENE2', 'GENE3', ...])

# Initialize signature refinement (available at top level)
refiner = dr.SignatureRefinement(starting_signature)

# Load experimental data (AnnData with compound treatments)
# adata should have:
# - Gene expression data in .X or layers
# - Compound IDs in .obs (e.g., 'compound_id' column)
# - Sample IDs in .obs (e.g., 'sample_id' column) 
refiner.load_counts_data(
    adata, 
    compound_id_obs_col='compound_id',
    sample_id_obs_cols=['sample_id'],
    layer='raw_counts'  # or None to use .X
)

# Load phenotypic readouts
readouts = pd.Series([0.8, -1.2, 0.3, ...], 
                    index=['compound_A', 'compound_B', 'compound_C', ...])
refiner.load_phenotypic_readouts(readouts)

# Compute learned signatures using correlation analysis
refiner.compute_learned_signatures(corr_method='pearson')

# Generate refined signatures (interpolation between starting and learned)
refiner.compute_refined_signatures(
    learning_rate=0.5,      # 0.5 = equal weight to starting and learned
    scale_learned_sig=True  # Scale learned signature to match starting signature std
)

# Access results
refined_signatures = refiner.refined_signatures  # AnnData object
learned_signatures = refiner.learned_signatures   # AnnData object
```

### Signature Refinement with Multiple Conditions

```python
# For multiple experimental conditions, specify signature_id_obs_cols
refiner.load_counts_data(
    adata,
    compound_id_obs_col='compound_id',
    sample_id_obs_cols=['sample_id'],
    signature_id_obs_cols=['treatment_type', 'timepoint'],  # Creates separate signatures
    layer='raw_counts'
)

# This will create one learned/refined signature for each unique combination
# of values in signature_id_obs_cols
```

## V-score Computation

Fast vectorized v-score calculations for differential expression analysis:

```python
from drugreflector import compute_vscores_adata, compute_vscore_two_groups

# Compute v-scores between two cell populations
vscores = compute_vscores_adata(
    adata, 
    group_col='cell_type',      # Column identifying groups
    group1_value='control',     # Reference group
    group2_value='treatment',   # Comparison group
    layer=None                  # Use .X, or specify layer name
)

# vscores is a pandas Series with gene names as index and informative name
print(f"V-score comparison: {vscores.name}")  # e.g., "cell_type:control->treatment"
print(f"Top upregulated genes:")
print(vscores.nlargest(10))
print(f"Top downregulated genes:")
print(vscores.nsmallest(10))

# For two arrays directly
group1_values = [1.2, 0.8, 1.5, 0.9]  # Reference/control
group2_values = [2.1, 1.9, 2.3, 2.0]  # Treatment/comparison
vscore = compute_vscore_two_groups(group1_values, group2_values)
```

## Data Utilities

### Loading and Preprocessing

```python
from drugreflector import load_h5ad_file, pseudobulk_adata

# Load H5AD file with preprocessing
adata = load_h5ad_file('data.h5ad')

# Pseudobulk single-cell data
pseudobulked = pseudobulk_adata(
    adata,
    sample_id_obs_cols=['donor_id', 'condition'],  # Columns defining samples
    method='sum'  # or 'mean'
)
```

### V-score Integration with Existing Workflow

```python
from drugreflector import compute_vscores

# Use v-scores in existing workflow
transitions = {
    'group_col': 'cell_type',
    'group1_value': 'control',
    'group2_value': 'treatment'
}

vscores_adata = compute_vscores(adata, transitions=transitions)
# Returns AnnData object with v-scores as .X
```

## Command Line Usage

```bash
# Example command line usage (if predict.py exists)
python drugreflector/predict.py input.h5ad \
    --model1 checkpoints/model_fold_0.pt \
    --model2 checkpoints/model_fold_1.pt \
    --model3 checkpoints/model_fold_2.pt \
    --output results.csv
```

## API Reference

### DrugReflector Class

#### `DrugReflector(checkpoint_paths, device='auto')`
- **checkpoint_paths**: List of paths to model checkpoint files (.pt)
- **device**: PyTorch device ('cuda', 'cpu', or 'auto')

#### Methods
- `predict(data, n_top=None)`: Get compound predictions with ranks, scores, probabilities
- `get_top_compounds(data, n_top=10)`: Get top N compounds as separate DataFrames  
- `predict_top_compounds(data, n_top=50)`: Alias for get_top_compounds
- `check_gene_coverage(gene_names)`: Check how many genes are recognized by the model

### SignatureRefinement Class

#### `SignatureRefinement(starting_signature)`
- **starting_signature**: pandas Series or AnnData with initial signature

#### Methods
- `load_counts_data(adata, compound_id_obs_col, layer=None, ...)`: Load raw count data
- `load_normalized_data(adata, compound_id_obs_col, layer=None, ...)`: Load normalized data  
- `load_phenotypic_readouts(readouts, readout_col=None, ...)`: Load phenotypic data
- `compute_learned_signatures(corr_method='pearson')`: Compute signatures from data
- `compute_refined_signatures(learning_rate=0.5, scale_learned_sig=True)`: Generate refined signatures

### Utility Functions

- `compute_vscore_two_groups(group0, group1)`: V-score between two arrays
- `compute_vscores_adata(adata, group_col, group0_value, group1_value, layer=None)`: V-scores from AnnData
- `pseudobulk_adata(adata, sample_id_obs_cols, method='sum')`: Pseudobulk expression data
- `load_h5ad_file(filepath)`: Load and preprocess H5AD files
- `create_synthetic_gene_expression(n_obs, n_vars, ...)`: Generate synthetic data for testing

## Input Data Requirements

### Gene Expression Data
- **Format**: AnnData objects (.h5ad files)
- **Genes**: Must include the 978 landmark genes used by the model
- **Gene Symbols**: **CRITICAL** - Gene names must be in HGNC (HUGO Gene Nomenclature Committee) format
  - Examples: `TP53`, `EGFR`, `CDKN1A`, `IL6`
  - DrugReflector automatically converts gene names to HGNC-compatible format (uppercase, removes prefixes/suffixes)
  - Supported input formats: Ensembl IDs, Affymetrix probe IDs, mixed case symbols
- **Samples**: Expression profiles for compounds/treatments of interest  
- **Preprocessing**: Log-transformed, normalized expression values

### Model Checkpoints
- **Source**: Zenodo DOI 10.5281/zenodo.16912445
- **Format**: PyTorch .pt files
- **Count**: 3 model files (ensemble of 3-fold cross-validation)

### For Signature Refinement
- **Expression Data**: Raw counts or normalized expression in AnnData format
- **Metadata**: Compound IDs, sample IDs, and experimental conditions in `.obs`
- **Phenotypic Readouts**: Numeric values (e.g., viability, efficacy scores) as pandas Series

## Citation

If you use this package, please cite:

```
[TBD]
```

Model checkpoints: DOI 10.5281/zenodo.16912445

## License

[TBD]

## Troubleshooting

### Scikit-learn Version Warning

If you see a warning about scikit-learn version mismatch:
```
InconsistentVersionWarning: Trying to unpickle estimator LabelEncoder from version 1.2.2 when using version 1.5.1
```

This occurs because the model checkpoints were trained with scikit-learn 1.2.2. The warning is generally harmless and does not affect functionality, but indicates a version difference between training and inference environments.

### Gene Symbol Issues

If you get unexpected results or low prediction scores:

1. **Check gene name format**: Ensure genes are in HGNC format or compatible
2. **Verify gene coverage**: Check how many of your genes overlap with the 978 landmark genes
3. **Review preprocessing output**: DrugReflector shows which genes were modified during preprocessing

```python
# Check gene coverage using built-in function
coverage = model.check_gene_coverage(your_data.var_names)
print(f"Gene coverage: {coverage['total_found']}/{coverage['total_input']} ({coverage['coverage_percent']:.1f}%)")

# See which genes were not found
if coverage['missing_genes']:
    print(f"Missing genes: {coverage['missing_genes'][:10]}")  # Show first 10

# See preprocessing changes
for gene_info in coverage['gene_mapping'][:5]:  # Show first 5
    if gene_info['original'] != gene_info['processed']:
        print(f"  {gene_info['original']} -> {gene_info['processed']} ({'found' if gene_info['found'] else 'not found'})")
```

## Support

For issues and questions, please use the GitHub issue tracker.
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
import numpy as np
from drugreflector import DrugReflector, create_synthetic_gene_expression

# Load your gene expression data (AnnData format)
# adata = load_your_data()  # Replace with your data loading

# For demonstration, create synthetic data
adata = create_synthetic_gene_expression(
    n_obs=3,
    n_vars=978,  # 978 landmark genes
    obs_names=['sample_A', 'sample_B', 'sample_C']
)

# Initialize DrugReflector with model checkpoints
model_paths = [
    'checkpoints/model_fold_0.pt',
    'checkpoints/model_fold_1.pt', 
    'checkpoints/model_fold_2.pt'
]

model = DrugReflector(checkpoint_paths=model_paths)

# Make predictions (get top 50 compounds for each sample)
predictions = model.predict_ranks_on_adata(adata, n_top=50)

# Get top compounds
top_compounds = model.get_top_compounds(adata, n_top=10)
print("Top 10 compounds for each sample:")
print(top_compounds)
```

### Computing P-values

```python
# Compute background distribution for p-value calculation
model.compute_background_distribution(n_samples=1000)

# Get predictions with p-values
predictions_with_pvals = model.predict_ranks_on_adata(
    adata, 
    compute_pvalues=True, 
    n_top=50
)
```

## Signature Refinement

Refine transcriptional signatures using paired transcriptional + phenotypic data:

```python
from drugreflector.signature_refinement import SignatureRefinement
import pandas as pd

# Starting signature (pandas Series with gene names as index)
starting_signature = pd.Series([1.2, -0.8, 0.5, ...], 
                              index=['GENE1', 'GENE2', 'GENE3', ...])

# Initialize signature refinement
refiner = SignatureRefinement(starting_signature)

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
    group0_value='control',     # Reference group
    group1_value='treatment',   # Comparison group
    layer=None                  # Use .X, or specify layer name
)

# vscores is a pandas Series with gene names as index
print(f"Top upregulated genes:")
print(vscores.nlargest(10))
print(f"Top downregulated genes:")
print(vscores.nsmallest(10))

# For two arrays directly
group0_values = [1.2, 0.8, 1.5, 0.9]
group1_values = [2.1, 1.9, 2.3, 2.0]
vscore = compute_vscore_two_groups(group0_values, group1_values)
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
    'group0_value': 'control',
    'group1_value': 'treatment'
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
- `predict_ranks_on_adata(adata, n_top=50, compute_pvalues=False)`: Get ranked compound predictions
- `get_top_compounds(adata, n_top=10)`: Get top N compounds for each sample
- `compute_background_distribution(n_samples=1000)`: Compute background for p-values

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

## Support

For issues and questions, please use the GitHub issue tracker.
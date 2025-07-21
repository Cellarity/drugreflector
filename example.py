#!/usr/bin/env python3
"""
Example usage of DrugReflector V3.5 for compound ranking predictions.

This script demonstrates how to use the DrugReflectorV35 class to make
predictions from gene expression data.
"""

import numpy as np
from drug_reflector import DrugReflectorV35
from utils import create_synthetic_gene_expression


def main():
    print("DrugReflector V3.5 Example")
    print("=" * 30)
    
    # Create synthetic gene expression data for demonstration
    print("Creating synthetic gene expression data...")
    adata = create_synthetic_gene_expression(
        n_obs=3,
        n_vars=978,  # Landmark genes
        obs_names=['sample_A', 'sample_B', 'sample_C']
    )
    print(f"Created data with {adata.n_obs} samples and {adata.n_vars} genes")
    
    # Note: In real usage, you would load your trained model checkpoints
    model_paths = [
        'models/model_fold_0.pt',
        'models/model_fold_1.pt', 
        'models/model_fold_2.pt'
    ]
    
    print("\nTo use DrugReflector with your trained models:")
    print("1. Place your 3 model checkpoints (.pt files) in the models/ directory")
    print("2. Update the model_paths list above with the correct paths")
    print("3. Run this example with real data")
    
    print("\nExample code structure:")
    print("""
# Initialize DrugReflector with trained models
model = DrugReflectorV35(checkpoint_paths=model_paths)

# Make predictions
predictions = model.predict_ranks_on_adata(adata, n_top=50)

# Get top compounds for each sample
top_compounds = model.get_top_compounds(adata, n_top=10)

# Optionally compute p-values
model.compute_background_distribution(n_samples=1000)
predictions_with_pvals = model.predict_ranks_on_adata(
    adata, compute_pvalues=True, n_top=50
)
    """)
    
    print("\nCommand-line usage:")
    print("python predict.py input.h5ad --model1 models/model1.pt --model2 models/model2.pt --model3 models/model3.pt")


if __name__ == "__main__":
    main()
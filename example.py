#!/usr/bin/env python3
"""
Example usage of DrugReflector for compound ranking predictions.

This script demonstrates how to use the DrugReflector class to make
predictions from gene expression data.
"""

import numpy as np
from drugreflector import DrugReflector
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
        'checkpoints/model_fold_0.pt',
        'checkpoints/model_fold_1.pt', 
        'checkpoints/model_fold_2.pt'
    ]
    
    print("\nTo use DrugReflector with your trained models:")
    print("1. Place your 3 model checkpoints (.pt files) in the checkpoints/ directory")
    print("2. Update the model_paths list above with the correct paths")
    print("3. Run this example with real data")
    
    print("\nExample code structure:")
    print("""
# Initialize DrugReflector with trained models
model = DrugReflector(checkpoint_paths=model_paths)

# Make predictions
predictions = model.predict(adata, n_top=50)

# Get top compounds for each sample
top_compounds = model.get_top_compounds(adata, n_top=10)
    """)
    
    print("\nCommand-line usage:")
    print("python drugreflector/predict.py input.h5ad --model1 checkpoints/model_fold_0.pt --model2 checkpoints/model_fold_1.pt --model3 checkpoints/model_fold_2.pt")


if __name__ == "__main__":
    main()
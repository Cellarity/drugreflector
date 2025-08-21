#!/usr/bin/env python3
"""
Command-line interface for DrugReflector.
"""

import argparse
import sys
from pathlib import Path
from drugreflector import DrugReflector, load_h5ad_file


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="DrugReflector: Compound ranking predictions from gene expression signatures"
    )
    
    parser.add_argument(
        "input", 
        type=str,
        help="Input H5AD file with gene expression data"
    )
    
    parser.add_argument(
        "--model1", 
        type=str,
        default="checkpoints/model_fold_0.pt",
        help="Path to first model checkpoint (default: checkpoints/model_fold_0.pt)"
    )
    
    parser.add_argument(
        "--model2", 
        type=str,
        default="checkpoints/model_fold_1.pt",
        help="Path to second model checkpoint (default: checkpoints/model_fold_1.pt)"
    )
    
    parser.add_argument(
        "--model3", 
        type=str,
        default="checkpoints/model_fold_2.pt",
        help="Path to third model checkpoint (default: checkpoints/model_fold_2.pt)"
    )
    
    parser.add_argument(
        "--output", 
        type=str,
        default="predictions.csv",
        help="Output CSV file for predictions (default: predictions.csv)"
    )
    
    parser.add_argument(
        "--n-top", 
        type=int,
        default=50,
        help="Number of top compounds to return (default: 50)"
    )
    
    
    parser.add_argument(
        "--device", 
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for computation (default: auto)"
    )
    
    args = parser.parse_args()
    
    # Check input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file '{args.input}' not found", file=sys.stderr)
        sys.exit(1)
    
    # Check model checkpoints exist
    model_paths = [args.model1, args.model2, args.model3]
    for model_path in model_paths:
        if not Path(model_path).exists():
            print(f"Error: Model checkpoint '{model_path}' not found", file=sys.stderr)
            print("Please download model checkpoints from Zenodo DOI: 10.5281/zenodo.16912445", file=sys.stderr)
            sys.exit(1)
    
    try:
        # Load data
        print(f"Loading data from {args.input}...")
        adata = load_h5ad_file(args.input)
        print(f"Loaded {adata.n_obs} samples with {adata.n_vars} genes")
        
        # Initialize model
        print("Initializing DrugReflector model...")
        model = DrugReflector(checkpoint_paths=model_paths)
        
        # Note: This CLI assumes v-score data, not raw gene expression
        print("Warning: CLI expects v-score data, not raw gene expression.")
        print("In a real workflow, compute v-scores first using drugreflector.compute_vscores_adata")
        
        # Make predictions (assuming input is v-scores)
        print(f"Making predictions for top {args.n_top} compounds...")
        predictions = model.predict(adata, n_top=args.n_top)
        
        # Save results
        print(f"Saving results to {args.output}...")
        predictions.to_csv(args.output)
        
        print("Done!")
        print(f"Results saved to: {args.output}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
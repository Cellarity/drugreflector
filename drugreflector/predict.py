#!/usr/bin/env python3
"""
Command-line interface for DrugReflector V3.5 predictions.

This script provides a simple interface for making compound ranking
predictions using trained DrugReflector models.
"""

import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path to allow running as script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drugreflector.drug_reflector import DrugReflector
from utils import load_h5ad_file


def main():
    parser = argparse.ArgumentParser(
        description="DrugReflector V3.5 Compound Ranking Predictions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to input H5AD file containing gene expression data"
    )
    
    parser.add_argument(
        "--model1",
        type=str,
        required=True,
        help="Path to first model checkpoint (.pt file)"
    )
    
    parser.add_argument(
        "--model2", 
        type=str,
        required=True,
        help="Path to second model checkpoint (.pt file)"
    )
    
    parser.add_argument(
        "--model3",
        type=str,
        required=True,
        help="Path to third model checkpoint (.pt file)"
    )
    
    # Optional arguments
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="predictions.csv",
        help="Output CSV file path"
    )
    
    parser.add_argument(
        "--top-n",
        type=int,
        default=100,
        help="Number of top compounds to include in output"
    )
    
    parser.add_argument(
        "--background-samples",
        type=int,
        default=1000,
        help="Number of background samples for p-value computation"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Validate input files exist
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found")
        sys.exit(1)
    
    for i, model_path in enumerate([args.model1, args.model2, args.model3], 1):
        if not os.path.exists(model_path):
            print(f"Error: Model {i} checkpoint '{model_path}' not found")
            sys.exit(1)
    
    if args.verbose:
        print("Loading input data...")
    
    # Load input data
    try:
        adata = load_h5ad_file(args.input_file)
        print(f"Loaded data with {adata.n_obs} observations and {adata.n_vars} genes")
    except Exception as e:
        print(f"Error loading input file: {e}")
        sys.exit(1)
    
    if args.verbose:
        print("Initializing DrugReflector model...")
    
    # Initialize model
    try:
        model = DrugReflector(
            checkpoint_paths=[args.model1, args.model2, args.model3]
        )
        print(f"Model initialized with {model.n_compounds} compounds")
    except Exception as e:
        print(f"Error initializing model: {e}")
        sys.exit(1)
    
    if args.verbose:
        print("Making predictions...")
    
    # Make predictions
    try:
        results = model.predict(
            adata,
            n_top=args.top_n
        )
        print(f"Generated predictions for {len(adata.obs_names)} observations")
    except Exception as e:
        print(f"Error making predictions: {e}")
        sys.exit(1)
    
    # Save results
    if args.verbose:
        print(f"Saving results to {args.output}...")
    
    try:
        results.to_csv(args.output)
        print(f"Results saved to {args.output}")
    except Exception as e:
        print(f"Error saving results: {e}")
        sys.exit(1)
    
    # Print summary
    print("\nPrediction Summary:")
    print(f"  Input observations: {adata.n_obs}")
    print(f"  Input genes: {adata.n_vars}")
    print(f"  Output compounds: {len(results.index)}")
    print(f"  Top compounds per observation: {args.top_n}")

    # Show sample of results
    if args.verbose and len(results) > 0:
        print("\nSample predictions (first observation, top 5 compounds):")
        first_obs = adata.obs_names[0]
        if ('rank', first_obs) in results.columns:
            sample_results = results[('rank', first_obs)].nsmallest(5)
            for compound, rank in sample_results.items():
                logit = results.loc[compound, ('logit', first_obs)]
                prob = results.loc[compound, ('prob', first_obs)]
                print(f"  {compound}: rank={rank}, logit={logit:.4f}, prob={prob:.4f}")


if __name__ == "__main__":
    main()
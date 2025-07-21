"""
DrugReflector V3.5 implementation for deep virtual screening.

This module provides the DrugReflector class for compound ranking
predictions from gene expression signatures.
"""

import numpy as np
import pandas as pd
import torch
from typing import Optional, Dict, Any, List
from anndata import AnnData
import warnings
import scipy.stats as stats
from scipy.special import softmax

from .ensemble_model import EnsembleModel
from .utils import compute_vscores, clip_rescale_rows


class DrugReflector:
    """
    DrugReflector V3.5 ensemble model for compound ranking predictions.
    
    This class provides compound ranking predictions from gene expression
    signatures using an ensemble of 3 trained neural network models.
    
    Parameters
    ----------
    checkpoint_paths : List[str]
        List of paths to the 3 trained model checkpoints
    model_class : str, default='pert_id'
        Model class (only 'pert_id' supported)
    ensemble : bool, default=True
        Whether to use ensemble predictions
    """
    
    def __init__(self, checkpoint_paths: List[str], model_class: str = 'pert_id', 
                 ensemble: bool = True):
        if model_class != 'pert_id':
            raise ValueError(f"Only 'pert_id' model class is supported, got {model_class}")
        
        if not ensemble:
            raise ValueError("Only ensemble=True is supported")
        
        if len(checkpoint_paths) != 3:
            raise ValueError("Exactly 3 checkpoint paths are required for the ensemble")
        
        self.model_class = model_class
        self.ensemble = ensemble
        self.checkpoint_paths = checkpoint_paths
        
        # Load ensemble model
        self.model = EnsembleModel(checkpoint_paths)
        self.background_distribution = None
        
        # Store compound information
        self.compound_names = self.model.dimensions['output_names']
        self.n_compounds = len(self.compound_names)
    
    def transform(self, adata: AnnData, transitions=None, ranks: bool = False) -> AnnData:
        """
        Transform input data through the ensemble model.
        
        Parameters
        ----------
        adata : AnnData
            Input gene expression data
        transitions : optional
            Transition specifications for v-score computation
        ranks : bool, default=False
            Whether to compute ranks for predictions
            
        Returns
        -------
        AnnData
            Prediction scores with compounds as variables
        """
        # Compute v-scores and apply preprocessing
        vscores = compute_vscores(adata, transitions)

        # Clip and rescale rows
        clip_rescale_rows(vscores.X, clip=2, target_std=1)

        # Standardize gene names
        vscores.var_names = vscores.var_names.str.upper()
        vscores.var_names_make_unique()

        self._X_landmarks = vscores
        self._weights_x = None
        
        # Get ensemble predictions
        scores = self.model.transform(self._X_landmarks, ranks=ranks)
        
        return scores
    
    def predict_ranks_on_adata(self, adata: AnnData, transitions=None, compute_pvalues: bool = False, 
                               n_top: int = None) -> pd.DataFrame:
        """
        Predict compound ranks for input gene expression data.
        
        Parameters
        ----------
        adata : AnnData
            Input gene expression data
        transitions : optional
            Transition specifications for v-score computation
        compute_pvalues : bool, default=False
            Whether to compute p-values using background distribution
        n_top : int, optional
            Number of top compounds to include in output
            
        Returns
        -------
        pd.DataFrame
            Multi-index DataFrame with ranks, scores, probabilities, and optionally p-values
        """
        # Get predictions
        predictions = self.transform(adata, transitions=transitions, ranks=True)
        
        # Extract ranks and scores
        ranks = predictions.layers['ranks']
        scores = predictions.X
        probs = softmax(scores, axis=1)
        
        # Prepare output data
        output_data = {}
        
        # Add ranks
        for i, obs_name in enumerate(adata.obs_names):
            output_data[('rank', obs_name)] = ranks[i]
        
        # Add logits (scores)
        for i, obs_name in enumerate(adata.obs_names):
            output_data[('logit', obs_name)] = scores[i]
            
        # Add probabilities
        for i, obs_name in enumerate(adata.obs_names):
            output_data[('prob', obs_name)] = probs[i]
        
        # Add p-values if requested
        if compute_pvalues:
            if self.background_distribution is None:
                raise ValueError("Background distribution not computed. Call compute_background_distribution() first.")
            
            pvalues = self._compute_pvalues(scores)
            for i, obs_name in enumerate(adata.obs_names):
                output_data[('pvalue', obs_name)] = pvalues[i]
        
        # Create DataFrame
        df = pd.DataFrame(output_data, index=self.compound_names)
        
        # Filter to top compounds if requested
        if n_top is not None and n_top > 0:
            # Get compounds that are in top n_top for any observation
            top_compounds = set()
            for obs_name in adata.obs_names:
                top_ranks = df[('rank', obs_name)].nsmallest(n_top)
                top_compounds.update(top_ranks.index)
            
            df = df.loc[list(top_compounds)]
        
        return df.transpose()
    
    def compute_background_distribution(self, n_samples: int = 1000, 
                                       random_state: int = 42):
        """
        Compute background distribution for p-value calculation.
        
        Parameters
        ----------
        n_samples : int, default=1000
            Number of random samples to generate
        random_state : int, default=42
            Random seed for reproducibility
        """
        np.random.seed(random_state)
        
        # Get input dimensions from first model
        input_size = None
        for var_names in self.model.dimensions['var_names']:
            if input_size is None:
                input_size = len(var_names)
            elif len(var_names) != input_size:
                warnings.warn("Models have different input sizes, using first model's size")
                break
        
        # Generate random gene expression data
        random_data = np.random.normal(0, 1, size=(n_samples, input_size))
        
        # Create AnnData with dummy gene names
        dummy_genes = [f"gene_{i}" for i in range(input_size)]
        dummy_obs = [f"sample_{i}" for i in range(n_samples)]
        
        background_adata = AnnData(
            X=random_data,
            obs=pd.DataFrame(index=dummy_obs),
            var=pd.DataFrame(index=dummy_genes)
        )
        
        # Get predictions for background data
        background_predictions = self.transform(background_adata, ranks=False)
        
        # Store background distribution
        self.background_distribution = background_predictions.X
        
        print(f"Background distribution computed with {n_samples} samples")
    
    def _compute_pvalues(self, scores: np.ndarray) -> np.ndarray:
        """
        Compute p-values using background distribution.
        
        Parameters
        ----------
        scores : np.ndarray
            Prediction scores of shape (n_obs, n_compounds)
            
        Returns
        -------
        np.ndarray
            P-values of shape (n_obs, n_compounds)
        """
        if self.background_distribution is None:
            raise ValueError("Background distribution not computed")
        
        pvalues = np.zeros_like(scores)
        
        # Compute p-values for each compound
        for i in range(self.n_compounds):
            background_scores = self.background_distribution[:, i]
            
            for j in range(scores.shape[0]):
                # P-value is the fraction of background scores >= observed score
                pvalue = (background_scores >= scores[j, i]).mean()
                pvalues[j, i] = pvalue
        
        return pvalues
    
    def get_top_compounds(self, adata: AnnData, n_top: int = 20, 
                         compute_pvalues: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Get top-ranked compounds for each observation.
        
        Parameters
        ----------
        adata : AnnData
            Input gene expression data
        n_top : int, default=20
            Number of top compounds to return
        compute_pvalues : bool, default=False
            Whether to include p-values
            
        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary with observation names as keys and top compounds as values
        """
        # Get full predictions
        full_results = self.predict_ranks_on_adata(adata, compute_pvalues=compute_pvalues)
        
        results = {}
        for obs_name in adata.obs_names:
            # Get data for this observation
            ranks = full_results[('rank', obs_name)]
            logits = full_results[('logit', obs_name)]
            probs = full_results[('prob', obs_name)]
            
            # Create DataFrame with top compounds
            df_data = {
                'compound': ranks.index,
                'rank': ranks.values,
                'logit': logits.values,
                'prob': probs.values
            }
            
            if compute_pvalues:
                pvalues = full_results[('pvalue', obs_name)]
                df_data['pvalue'] = pvalues.values
            
            df = pd.DataFrame(df_data)
            df = df.sort_values('rank').head(n_top)
            
            results[obs_name] = df
        
        return results
    
    def predict(self, adata: AnnData, n_top: int = 50) -> Dict[str, pd.DataFrame]:
        """
        Make predictions and return top-ranked compounds.
        
        Parameters
        ----------
        adata : AnnData
            Input gene expression data
        n_top : int, default=50
            Number of top compounds to return per observation
            
        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary with observation names as keys and ranked predictions as values
        """
        return self.get_top_compounds(adata, n_top=n_top, compute_pvalues=False)
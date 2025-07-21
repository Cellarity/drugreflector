"""
Utility functions for deep virtual screening with DrugReflector.

This module contains core functions for data preprocessing including
v-score computation and clipping/rescaling operations.
"""

import numpy as np
import pandas as pd
import warnings
from scipy.optimize import minimize_scalar
from anndata import AnnData


def _norm_func(data, clip, target_std):
    """Helper function called in `clip_rescale_rows`."""
    return lambda norm: (np.std(np.clip(data / norm, -clip, clip)) - target_std) ** 2


def clip_rescale_rows(X, clip, target_std, bounds=(0, 1e3)):
    """For each row of X, rescale and then clip the values such that the
    resulting standard deviation is as close to possible as to target_std. This
    function mutates X itself.

    Params
    ------
    X : 2d ndarray
    clip : float
    target_std : float
    bounds : (float, float)
        bounds of optimization for finding scaling constant
    """
    n = X.shape[0]
    norm = np.zeros(n)
    for i in range(n):
        ms = minimize_scalar(_norm_func(X[i], clip, target_std), method='Bounded', bounds=bounds)
        assert ms.success, f'Unable to rescale row {i}.'
        norm[i] = ms.x

    np.divide(X, norm[:, np.newaxis], out=X)
    np.clip(X, -clip, clip, out=X)


def compute_vscores(adata, transitions=None, mute=False):
    """
    Compute v-scores for gene expression data.
    
    Parameters
    ----------
    adata : AnnData
        Gene expression data
    transitions : dict, optional
        Transition specifications for v-score computation (not yet implemented)
    mute : bool, default=False
        Whether to suppress warnings
        
    Returns
    -------
    AnnData
        Processed v-scores
    """
    if transitions is not None:
        raise ValueError('cannnot do transitions yet')
        # TODO: implement v-score computation for transitions
    else:
        if not mute:
            warnings.warn('Assuming passed representation is v-score.', stacklevel=1)
        vscores = adata.copy()

    # Standardize gene names
    vscores.var_names = vscores.var_names.str.upper()
    vscores.var_names_make_unique()
    
    # Apply clipping and rescaling
    clip_rescale_rows(X=vscores.X, clip=2, target_std=1)
    
    return vscores


def load_h5ad_file(filepath: str) -> AnnData:
    """
    Load an H5AD file and perform basic preprocessing.
    
    Parameters
    ----------
    filepath : str
        Path to the H5AD file
        
    Returns
    -------
    AnnData
        Loaded and preprocessed AnnData object
    """
    # Load the H5AD file
    adata = AnnData.read_h5ad(filepath)
    
    # Basic validation
    if adata.X is None:
        raise ValueError("AnnData object has no X matrix")
    
    # Ensure X is a numpy array
    if hasattr(adata.X, 'toarray'):
        adata.X = adata.X.toarray()
    
    # Check for infinite or NaN values
    if not np.isfinite(adata.X).all():
        print("Warning: Found infinite or NaN values in data, replacing with 0")
        adata.X = np.nan_to_num(adata.X, nan=0.0, posinf=0.0, neginf=0.0)
    
    return adata


def create_synthetic_gene_expression(n_obs: int, n_vars: int, 
                                   obs_names: list = None,
                                   var_names: list = None,
                                   random_state: int = 42) -> AnnData:
    """
    Create synthetic gene expression data for testing.
    
    Parameters
    ----------
    n_obs : int
        Number of observations (samples)
    n_vars : int
        Number of variables (genes)
    obs_names : list, optional
        Names for observations
    var_names : list, optional
        Names for variables
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns
    -------
    AnnData
        Synthetic gene expression data
    """
    np.random.seed(random_state)
    
    # Generate synthetic expression data (normally distributed)
    X = np.random.normal(0, 1, size=(n_obs, n_vars))
    
    # Create observation names
    if obs_names is None:
        obs_names = [f"sample_{i}" for i in range(n_obs)]
    
    # Create variable names
    if var_names is None:
        var_names = [f"gene_{i}" for i in range(n_vars)]
    
    # Create AnnData object
    adata = AnnData(
        X=X,
        obs=pd.DataFrame(index=obs_names),
        var=pd.DataFrame(index=var_names)
    )
    
    return adata
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
        Transition specifications for v-score computation. Should be a dict with
        'group_col', 'group0_value', 'group1_value' keys for computing v-scores
        between two populations.
    mute : bool, default=False
        Whether to suppress warnings
        
    Returns
    -------
    AnnData
        AnnData object with v-scores as .X if transitions specified,
        otherwise returns copy of input with warning
    """
    if transitions is not None:
        # Extract transition parameters
        if not isinstance(transitions, dict):
            raise ValueError('transitions must be a dict with group_col, group0_value, group1_value keys')
        
        required_keys = ['group_col', 'group0_value', 'group1_value']
        for key in required_keys:
            if key not in transitions:
                raise ValueError(f'transitions dict must contain key: {key}')
        
        group_col = transitions['group_col']
        group0_value = transitions['group0_value']
        group1_value = transitions['group1_value']
        layer = transitions.get('layer', None)
        
        # Compute v-scores between the two groups
        vscores_series = compute_vscores_adata(adata, group_col, group0_value, group1_value, layer=layer)
        
        # Create new AnnData with v-scores
        vscores = AnnData(
            X=vscores_series.values.reshape(1, -1),
            var=pd.DataFrame(index=vscores_series.index),
            obs=pd.DataFrame(index=[f'{group1_value}_vs_{group0_value}'])
        )
        
        return vscores
    else:
        if not mute:
            warnings.warn('Assuming passed representation is v-score.', stacklevel=1)
        vscores = adata.copy()
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


def pseudobulk_adata(
    adata,
    sample_id_obs_cols,
    sample_metadata_obs_cols='auto',
    layer=None,
    method='sum',
):
    """
    'Pseudobulks' an AnnData object by computing mean or total expression across each sample.
    Samples are identified by unique combinations of values in .obs columns specified
    by sample_id_obs_cols. The values in the .obs columns in sample_metadata_obs_cols for
    each sample are copied over into the AnnData object returned by this function.
    
    Parameters
    ----------
    adata : AnnData
        Input AnnData object
    sample_id_obs_cols : list of str
        Columns in adata.obs whose combination of values uniquely identifies samples.
    sample_metadata_obs_cols : str or list of str, default='auto'
        If a list is passed, it will be interpreted as the columns that are unique to each
        sample and that will be copied over into `.obs` of the pseudobulked AnnData. If 'auto'
        is passed, all the columns that have one-to-one mapping with each combination of values
        in `sample_id_obs_cols` will be copied over.
    layer : str, optional
        Layer to use for pseudobulking. If None, uses .X
    method : str, default='sum'
        Whether to compute total ('sum') or average ('mean') expression when pseudobulking.
        
    Returns
    -------
    AnnData
        Pseudobulked AnnData object with sparse layers support
    """
    import scipy.sparse as sp
    
    # Make a copy to avoid modifying original
    adata_temp = adata.copy()
    
    # Use specified layer or X
    if layer:
        if layer not in adata_temp.layers:
            raise ValueError(f"Layer '{layer}' not found in AnnData object")
        X_data = adata_temp.layers[layer]
    else:
        X_data = adata_temp.X
    
    # Handle sparse matrices
    is_sparse = sp.issparse(X_data)
    if is_sparse:
        X_data = X_data.tocsr()
    
    # Create temporary index for grouping
    adata_obs_cols = set(adata_temp.obs.columns)
    adata_obs_cols -= set(sample_id_obs_cols)
    
    adata_temp.obs['_TempIndex'] = adata_temp.obs.apply(
        lambda row: '_'.join([f'{row[id_col]}' for id_col in sample_id_obs_cols]), axis='columns'
    )
    
    # Get unique group indices
    group_names = adata_temp.obs['_TempIndex'].unique()
    n_groups = len(group_names)
    n_genes = adata_temp.n_vars
    
    # Initialize result matrix
    if is_sparse:
        bulk_X = sp.lil_matrix((n_groups, n_genes), dtype=X_data.dtype)
    else:
        bulk_X = np.zeros((n_groups, n_genes), dtype=X_data.dtype)
    
    # Perform groupby operation
    for i, group_name in enumerate(group_names):
        group_mask = adata_temp.obs['_TempIndex'] == group_name
        group_data = X_data[group_mask]
        
        if method == 'sum':
            if is_sparse:
                bulk_X[i, :] = group_data.sum(axis=0)
            else:
                bulk_X[i, :] = group_data.sum(axis=0)
        elif method == 'mean':
            if is_sparse:
                bulk_X[i, :] = group_data.mean(axis=0)
            else:
                bulk_X[i, :] = group_data.mean(axis=0)
        else:
            raise ValueError("method parameter must be 'sum' or 'mean'")
    
    # Convert back to appropriate sparse format
    if is_sparse:
        bulk_X = bulk_X.tocsr()
    
    # Create new AnnData object
    bulk_adata = AnnData(
        X=bulk_X,
        var=adata_temp.var.copy(),
        dtype=X_data.dtype,
    )
    
    # Set observation names
    bulk_adata.obs.index = group_names
    
    # Handle metadata columns
    if sample_metadata_obs_cols == 'auto':
        sample_metadata_obs_cols = []
        for obs_col in adata_obs_cols:
            if adata_temp.obs.groupby('_TempIndex')[obs_col].nunique().max() == 1:
                sample_metadata_obs_cols.append(obs_col)
    
    # Create metadata mapping
    if sample_metadata_obs_cols or sample_id_obs_cols:
        try:
            cols_to_use = ['_TempIndex'] + list(sample_id_obs_cols)
            if sample_metadata_obs_cols:
                cols_to_use.extend(sample_metadata_obs_cols)
            
            metadata_mapping = (
                adata_temp.obs[cols_to_use]
                .drop_duplicates()
                .set_index('_TempIndex', verify_integrity=True)
            )
        except ValueError as e:
            raise ValueError(
                'The combination of values in sample_metadata_cols of adata.obs must be '
                'unique for each value in sample_id_col.'
            ) from e
        
        # Merge metadata
        bulk_adata.obs = bulk_adata.obs.merge(
            metadata_mapping,
            left_index=True,
            right_index=True,
            how='left',
        )
    
    # Copy layers if they exist
    if hasattr(adata_temp, 'layers') and adata_temp.layers:
        for layer_name, layer_data in adata_temp.layers.items():
            if layer_name == layer:
                continue  # Skip the layer we used for X
                
            # Pseudobulk each layer
            is_layer_sparse = sp.issparse(layer_data)
            if is_layer_sparse:
                layer_data = layer_data.tocsr()
            
            if is_layer_sparse:
                layer_bulk = sp.lil_matrix((n_groups, n_genes), dtype=layer_data.dtype)
            else:
                layer_bulk = np.zeros((n_groups, n_genes), dtype=layer_data.dtype)
            
            for i, group_name in enumerate(group_names):
                group_mask = adata_temp.obs['_TempIndex'] == group_name
                group_layer_data = layer_data[group_mask]
                
                if method == 'sum':
                    if is_layer_sparse:
                        layer_bulk[i, :] = group_layer_data.sum(axis=0)
                    else:
                        layer_bulk[i, :] = group_layer_data.sum(axis=0)
                elif method == 'mean':
                    if is_layer_sparse:
                        layer_bulk[i, :] = group_layer_data.mean(axis=0)
                    else:
                        layer_bulk[i, :] = group_layer_data.mean(axis=0)
            
            if is_layer_sparse:
                layer_bulk = layer_bulk.tocsr()
            
            bulk_adata.layers[layer_name] = layer_bulk
    
    return bulk_adata


def compute_vscore_two_groups(group0, group1):
    """
    Compute v-score between two groups of numbers.
    
    V-score is defined as (mean1 - mean0) / (sqrt(var0 + var1) + (var0 + var1 == 0))
    This is the difference in means normalized by the sum of standard deviations.
    
    Parameters
    ----------
    group0 : array-like
        First group of values
    group1 : array-like
        Second group of values
        
    Returns
    -------
    float
        V-score between the two groups
    """
    group0 = np.asarray(group0)
    group1 = np.asarray(group1)
    
    mean0 = np.mean(group0)
    mean1 = np.mean(group1)
    var0 = np.var(group0, ddof=0)
    var1 = np.var(group1, ddof=0)
    
    # V-score formula from cifra
    denominator = np.sqrt(var0 + var1) + (var0 + var1 == 0)
    vscore = (mean1 - mean0) / denominator
    
    return vscore


def compute_vscores_adata(adata, group_col, group0_value, group1_value, layer=None):
    """
    Compute v-scores between two populations in an AnnData object.
    
    Parameters
    ----------
    adata : AnnData
        Input AnnData object
    group_col : str
        Column in adata.obs that identifies the groups
    group0_value : str or numeric
        Value in group_col that identifies the first group (control/reference)
    group1_value : str or numeric
        Value in group_col that identifies the second group (treatment/comparison)
    layer : str, optional
        Layer to use for computation. If None, uses .X
        
    Returns
    -------
    pd.Series
        V-scores for each gene, indexed by gene names
    """
    import scipy.sparse as sp
    
    # Check that group column exists
    if group_col not in adata.obs.columns:
        raise ValueError(f"Column '{group_col}' not found in adata.obs")
    
    # Get masks for each group
    group0_mask = adata.obs[group_col] == group0_value
    group1_mask = adata.obs[group_col] == group1_value
    
    # Check that both groups exist
    if not group0_mask.any():
        raise ValueError(f"No samples found with {group_col}='{group0_value}'")
    if not group1_mask.any():
        raise ValueError(f"No samples found with {group_col}='{group1_value}'")
    
    # Get expression data
    if layer and layer in adata.layers:
        X_data = adata.layers[layer]
    else:
        X_data = adata.X
    
    # Handle sparse matrices
    if sp.issparse(X_data):
        X_data = X_data.toarray()
    
    # Extract data for each group
    group0_data = X_data[group0_mask, :]  # Shape: (n_samples_group0, n_genes)
    group1_data = X_data[group1_mask, :]  # Shape: (n_samples_group1, n_genes)
    
    # Vectorized v-score computation across all genes
    mean0 = np.mean(group0_data, axis=0)  # Shape: (n_genes,)
    mean1 = np.mean(group1_data, axis=0)  # Shape: (n_genes,)
    var0 = np.var(group0_data, axis=0, ddof=0)  # Shape: (n_genes,)
    var1 = np.var(group1_data, axis=0, ddof=0)  # Shape: (n_genes,)
    
    # V-score formula from cifra (vectorized)
    denominator = np.sqrt(var0 + var1) + (var0 + var1 == 0)
    vscores = (mean1 - mean0) / denominator
    
    # Return as pandas Series with gene names as index
    return pd.Series(vscores, index=adata.var_names, name='vscore')




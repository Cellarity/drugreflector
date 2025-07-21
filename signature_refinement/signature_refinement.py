"""
Signature refinement for transcriptional signatures using experimental data.

This module provides functionality to refine transcriptional signatures based on
paired transcriptional + phenotypic data using correlation analysis.
"""

import numpy as np
import pandas as pd
import warnings
from typing import Optional, Union, List
from anndata import AnnData, concat
from scipy.stats import pearsonr
import scipy.sparse as sp
import sys
import os

# Add parent directory to path to access utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import pseudobulk_adata


class SignatureRefinement:
    """
    Refine transcriptional signatures using paired transcriptional + phenotypic data.
    
    This class allows updating a starting transcriptional signature based on 
    experimental data that pairs gene expression with phenotypic readouts.
    
    Parameters
    ----------
    starting_signature : AnnData or pd.Series
        Starting signature values. If AnnData, should have genes in columns and 1 row.
        If Series, should be keyed by genes.
    """

    def __init__(self, starting_signature: Union[AnnData, pd.Series]):
        # Handle different input types for starting signature
        if isinstance(starting_signature, AnnData):
            if starting_signature.n_obs != 1:
                raise ValueError("AnnData starting signature must have exactly 1 observation (row)")
            # Convert to Series
            self.starting_signature = pd.Series(
                starting_signature.X.flatten(),
                index=starting_signature.var_names,
                name='starting_signature'
            )
        elif isinstance(starting_signature, pd.Series):
            self.starting_signature = starting_signature.copy()
        else:
            raise ValueError("starting_signature must be AnnData or pd.Series")
        
        # Initialize other attributes
        self.expr = None
        self.readouts = None
        self.learned_signatures = None
        self.refined_signatures = None
        
        # Keep track of column names
        self._compound_id_obs_col = None
        self._sample_id_obs_cols = None
        self._signature_id_obs_cols = None

    def load_counts_data(self, adata: AnnData, compound_id_obs_col: str, layer: Optional[str] = None, 
                        sample_id_obs_cols: Optional[List[str]] = None,
                        signature_id_obs_cols: Optional[List[str]] = None):
        """
        Load raw counts data and perform pseudobulking.
        
        Parameters
        ----------
        adata : AnnData
            Input AnnData with raw counts
        compound_id_obs_col : str
            Column identifying compounds
        layer : str, optional
            Layer containing raw counts. If None, uses .X
        sample_id_obs_cols : list of str, optional
            Columns identifying samples for pseudobulking
        signature_id_obs_cols : list of str, optional
            Columns uniquely identifying signatures
        """
        import scanpy as sc
        
        # Check for counts data
        if layer and layer in adata.layers:
            counts_data = adata.layers[layer]
        else:
            counts_data = adata.X
            
        # Check if data looks like counts (non-negative integers)
        if hasattr(counts_data, 'data'):  # sparse matrix
            sample_values = counts_data.data[:1000]  # Check first 1000 values
        else:
            sample_values = counts_data.flatten()[:1000]
            
        if not np.all(sample_values >= 0):
            warnings.warn("Data contains negative values - may not be raw counts!")
        elif not np.allclose(sample_values, sample_values.astype(int)):
            warnings.warn("Data contains non-integer values - may not be raw counts!")
        
        # Set up pseudobulking columns
        if sample_id_obs_cols is None:
            sample_id_obs_cols = []
        if signature_id_obs_cols is None:
            signature_id_obs_cols = []
            
        # Step 1: Pseudobulk by compound + sample_id columns (sum for counts)
        sample_pseudobulk_cols = [compound_id_obs_col] + sample_id_obs_cols + signature_id_obs_cols
        pb_adata = pseudobulk_adata(
            adata, 
            sample_id_obs_cols=sample_pseudobulk_cols,
            layer=layer,
            method='sum'
        )
        
        # Store pseudobulked counts in layers
        if not hasattr(pb_adata, 'layers'):
            pb_adata.layers = {}
        pb_adata.layers['pseudobulked_counts'] = pb_adata.X.copy()
        
        # Convert to log(TPM) for .X
        # TPM = (counts / total_counts_per_sample) * 1e6
        # log(TPM) = log(1 + (counts / total_counts_per_sample) * 1e6)

        sc.pp.normalize_total(pb_adata, target_sum=1e6)
        sc.pp.log1p(pb_adata)
        
    
        # Store results
        self.expr = pb_adata
        self._compound_id_obs_col = compound_id_obs_col
        self._sample_id_obs_cols = sample_id_obs_cols if sample_id_obs_cols else []
        self._signature_id_obs_cols = signature_id_obs_cols if signature_id_obs_cols else []
    
    def load_normalized_data(self, adata: AnnData, compound_id_obs_col: str, layer: Optional[str] = None,
                            sample_id_obs_cols: Optional[List[str]] = None,
                            signature_id_obs_cols: Optional[List[str]] = None):
        """
        Load normalized transcriptional data.
        
        Parameters
        ----------
        adata : AnnData
            Input AnnData with normalized expression data
        compound_id_obs_col : str
            Column identifying compounds
        layer : str, optional
            Layer containing normalized data (e.g., log(TPM)). If None, uses .X
        sample_id_obs_cols : list of str, optional
            Columns identifying samples for pseudobulking. Different samples can have
            the same compound (e.g., replicates) and will be treated as separate
            measurements in learned signature computation.
        signature_id_obs_cols : list of str, optional
            Columns uniquely identifying different experimental conditions that will
            yield separate learned/refined signatures
        """
        # Use specified layer or .X
        if layer and layer in adata.layers:
            # Create temporary adata with layer as .X for pseudobulking
            temp_adata = adata.copy()
            temp_adata.X = temp_adata.layers[layer]
        else:
            temp_adata = adata.copy()
        
        # Set up pseudobulking columns
        if sample_id_obs_cols is None:
            sample_id_obs_cols = []
        if signature_id_obs_cols is None:
            signature_id_obs_cols = []
            
        # Step 1: Pseudobulk by compound + sample_id columns (mean for normalized data)
        sample_pseudobulk_cols = [compound_id_obs_col] + sample_id_obs_cols + signature_id_obs_cols
        pb_adata = pseudobulk_adata(
            temp_adata,
            sample_id_obs_cols=sample_pseudobulk_cols,
            method='mean'
        )
        
        # Store results
        self.expr = pb_adata
        self._compound_id_obs_col = compound_id_obs_col
        self._sample_id_obs_cols = sample_id_obs_cols if sample_id_obs_cols else []
        self._signature_id_obs_cols = signature_id_obs_cols if signature_id_obs_cols else []

    def load_phenotypic_readouts(self, readouts: Union[pd.DataFrame, pd.Series], 
                               readout_col: Optional[str] = None, 
                               compound_id_col: Optional[str] = None):
        """
        Load phenotypic readout data.
        
        Parameters
        ----------
        readouts : pd.DataFrame or pd.Series
            Phenotypic readout data
        readout_col : str, optional
            Column containing readout values (required if readouts is DataFrame)
        compound_id_col : str, optional
            Column containing compound IDs. If None, uses index
            
        Returns
        -------
        pd.Series
            Series of phenotypic readouts indexed by compound ID
        """
        if isinstance(readouts, pd.Series):
            readout_series = readouts.copy()
        elif isinstance(readouts, pd.DataFrame):
            if readout_col is None:
                raise ValueError("readout_col must be specified when readouts is a DataFrame")
            
            # Extract compound IDs and readout values
            if compound_id_col is None:
                compound_ids = readouts.index
            else:
                compound_ids = readouts[compound_id_col]
            
            readout_series = pd.Series(readouts[readout_col].values, index=compound_ids)
        else:
            raise ValueError("readouts must be a pandas DataFrame or Series")
        
        # Drop NaN values
        initial_size = len(readout_series)
        readout_series = readout_series.dropna()
        if len(readout_series) < initial_size:
            warnings.warn(f"Dropped {initial_size - len(readout_series)} NaN values from readouts")
        
        # Handle duplicate compound IDs
        if readout_series.index.duplicated().any():
            # Check if duplicates have different values
            duplicate_compounds = readout_series.index[readout_series.index.duplicated()].unique()
            for compound in duplicate_compounds:
                compound_values = readout_series.loc[compound]
                if not np.allclose(compound_values, compound_values.iloc[0], equal_nan=True):
                    warnings.warn(f"Compound {compound} has multiple distinct readout values - taking mean")
            
            # Take mean of duplicate readouts
            readout_series = readout_series.groupby(readout_series.index).mean()
        
        self.readouts = readout_series
        return readout_series


    def load_paired_readouts(self, adata: AnnData, compound_id_obs_col: str, 
                           readouts: Union[pd.DataFrame, pd.Series],
                           normalized_counts_layer: Optional[str] = None, 
                           raw_counts_layer: Optional[str] = None,
                           sample_id_obs_cols: Optional[List[str]] = None, 
                           signature_id_obs_cols: Optional[List[str]] = None,
                           readout_col: Optional[str] = None,
                           compound_id_col: Optional[str] = None):
        """
        Load both transcriptional data and phenotypic readouts together.
        
        Parameters
        ----------
        adata : AnnData
            Input AnnData object
        compound_id_obs_col : str
            Column identifying compounds in adata
        readouts : pd.DataFrame or pd.Series
            Phenotypic readout data
        normalized_counts_layer : str, optional
            Layer with normalized data (mutually exclusive with raw_counts_layer)
        raw_counts_layer : str, optional
            Layer with raw counts data (mutually exclusive with normalized_counts_layer)
        sample_id_obs_cols : list of str, optional
            Columns identifying samples (for raw counts only)
        signature_id_obs_cols : list of str, optional
            Columns uniquely identifying signatures
        readout_col : str, optional
            Column containing readout values in readouts DataFrame
        compound_id_col : str, optional
            Column containing compound IDs in readouts DataFrame
        """
        if normalized_counts_layer and raw_counts_layer:
            raise ValueError("Cannot specify both normalized_counts_layer and raw_counts_layer")
        
        if not normalized_counts_layer and not raw_counts_layer:
            raise ValueError("Must specify either normalized_counts_layer or raw_counts_layer")
        
        # Load transcriptional data
        if normalized_counts_layer:
            self.load_normalized_data(
                adata, compound_id_obs_col, layer=normalized_counts_layer,
                sample_id_obs_cols=sample_id_obs_cols,
                signature_id_obs_cols=signature_id_obs_cols
            )
        elif raw_counts_layer:
            self.load_counts_data(
                adata, compound_id_obs_col, layer=raw_counts_layer,
                sample_id_obs_cols=sample_id_obs_cols,
                signature_id_obs_cols=signature_id_obs_cols
            )
        
        # Load phenotypic readouts
        self.load_phenotypic_readouts(readouts, readout_col, compound_id_col)
    
    def _learned_signature(self, expr: np.ndarray, readouts: np.ndarray, 
                          include_stats: bool = False, corr_method: str = 'pearson'):
        """
        Compute learned signature using correlation analysis.
        
        Parameters
        ----------
        expr : np.ndarray
            Gene expression data (samples x genes)
        readouts : np.ndarray
            Phenotypic readouts for each sample
        include_stats : bool, default=False
            Whether to return additional statistics
        corr_method : str, default='pearson'
            Correlation method to use
            
        Returns
        -------
        dict or np.ndarray
            If include_stats=True, returns dict with scores, pvals, and statistics.
            Otherwise returns scores array.
        """
        if corr_method != 'pearson':
            raise ValueError("Only 'pearson' correlation method is currently supported")
        
        # Compute correlations for each gene
        pearson_results = [pearsonr(readouts, expr[:, i]) for i in range(expr.shape[1])]
        pearson_stats = [x.statistic for x in pearson_results]
        pearson_scores = np.array([-np.log10(x.pvalue) * np.sign(x.statistic) for x in pearson_results])
        pearson_pvals = np.array([x.pvalue for x in pearson_results])
        
        # Handle NaN values
        pearson_scores[np.isnan(pearson_scores)] = 0.
        pearson_pvals[np.isnan(pearson_pvals)] = 1.
        
        if include_stats:
            return {
                'scores': pearson_scores, 
                'pval': pearson_pvals,
                'statistic': pearson_stats
            }
        else:
            return pearson_scores
    
    def compute_learned_signatures(self, corr_method: str = 'pearson'):
        """
        Compute learned signatures for all signature combinations.
        
        Parameters
        ----------
        corr_method : str, default='pearson'
            Correlation method to use
        """
        if self.expr is None:
            raise ValueError("No expression data loaded. Call load_counts_data or load_normalized_data first.")
        
        if self.readouts is None:
            raise ValueError("No readout data loaded. Call load_phenotypic_readouts first.")
        
        # # Get expression data (handle sparse matrices)
        # import scipy.sparse as sp
        # if sp.issparse(self.expr.X):
        #     expr_data = self.expr.X.toarray()
        # else:
        #     expr_data = self.expr.X
        

        # Match compounds between expression and readouts
        expr_compounds = self.expr.obs[self._compound_id_obs_col] if self._compound_id_obs_col in self.expr.obs.columns else self.expr.obs.index
        readout_compounds = self.readouts.index
        
        # Find common compounds
        common_compounds = pd.Index(expr_compounds).intersection(readout_compounds)

        if len(common_compounds) < 2:
            raise ValueError("Not enough common compounds found between expression data and readouts")

        # Filter to common compounds
        expr_mask = expr_compounds.isin(common_compounds)
        readout_mask = readout_compounds.isin(common_compounds)
        
        filtered_expr = self.expr[expr_mask,:].copy() # expr with compounds
        
        # If we have signature ID columns, compute signatures for each combination
        if self._signature_id_obs_cols:
            expr_groups = filtered_expr.obs.groupby(self._signature_id_obs_cols, observed=True).groups
            learned_sigs = {}
            
            for signame, group in expr_groups.items():
                group_expr = filtered_expr[group,:].copy()
                if sp.issparse(group_expr.X):
                    group_expr.X = group_expr.X.toarray()
                
                if group_expr.shape[0] < 2:
                    warnings.warn('Not enough samples with readout in group {}; skipping'.format(signame))
                    continue

                
                group_readouts = self.readouts[group_expr.obs[self._compound_id_obs_col].values]

                signature_scores = pd.DataFrame(self._learned_signature(group_expr.X,group_readouts, corr_method=corr_method, include_stats=True))
                signature_scores.index = group_expr.var_names
                
                learned_sigs[signame] = signature_scores

            learned_sig_adatas = []
            for signame, sig in learned_sigs.items():
                print(sig)
                sigdata = AnnData(sig['scores'].values.reshape(1,-1), var = pd.DataFrame(index=sig.index),
                                     obs=pd.DataFrame(np.array(list(signame)).reshape(1,-1), columns=self._signature_id_obs_cols)
                                    )
                for col in sig.columns:
                    sigdata.layers[col] = sig[col].values.reshape(1,-1)
                learned_sig_adatas.append(sigdata)
            
            
            self.learned_signatures = concat(learned_sig_adatas, axis=0)
                                              
        else:
            # Single signature for all data
            signature_readouts = self.readouts[filtered_expr.obs[self._compound_id_obs_col].values]
            learned_sig = pd.DataFrame(self._learned_signature(filtered_expr.X, signature_readouts, corr_method=corr_method,include_stats=True),
                                      index=filtered_expr.var_names)
            
            sigdata = AnnData(learned_sig['scores'].values.reshape(1,-1), var = pd.DataFrame(index=learned_sig.index))
            
            for col in learned_sig.columns:
                sigdata.layers[col] = learned_sig[col].values.reshape(1,-1)
            self.learned_signatures = sigdata
    
    def compute_refined_signatures(self, learning_rate: float = 0.5, scale_learned_sig=True):
        """
        Compute refined signatures by combining starting and learned signatures.
        
        The refined signatures will have the same gene set as the initial signature.
        For genes not in the learned signature, they are left unchanged from the starting signature.
        For genes in the learned signature, interpolation is performed.
        
        Parameters
        ----------
        learning_rate : float, default=0.5
            Weight for learned signatures (0=only starting, 1=only learned)
        scale_learned_sig : bool, default=True
            Whether to scale learned signatures to have same std as starting signature
        """
        if self.learned_signatures is None:
            raise ValueError("No learned signatures available. Call compute_learned_signatures first.")
        
        # Find common genes between starting signature and learned signatures
        common_genes = self.starting_signature.index.intersection(self.learned_signatures.var_names)
        if len(common_genes) == 0:
            raise ValueError("No common genes between starting and learned signatures")
        
        if len(common_genes) < len(self.starting_signature.index):
            warnings.warn(f"Only {len(common_genes)}/{len(self.starting_signature.index)} genes shared between starting and learned signatures")
        
        # Create list to store refined signatures as pandas Series
        refined_signatures = []
        
        # Process each learned signature
        for i in range(self.learned_signatures.n_obs):
            # 1. Initialize refined signature as copy of starting signature
            refined_sig = self.starting_signature.copy()
            
            # 2. Get learned signature for this row as pandas Series
            learned_sig = pd.Series(
                self.learned_signatures.X[i, :],
                index=self.learned_signatures.var_names
            )
            
            # 3. Get learned scores for common genes using pandas indexing
            learned_common = learned_sig.loc[common_genes]
            starting_common = self.starting_signature.loc[common_genes]
            
            # 4. Scale learned signature if requested
            if scale_learned_sig:
                starting_sig_std = starting_common.std()
                learned_sig_std = learned_common.std()
                
                if learned_sig_std == 0:
                    # Set learned scores to zero (no contribution)
                    learned_common = pd.Series(0, index=common_genes)
                else:
                    # Scale to match starting signature std
                    learned_common = learned_common * (starting_sig_std / learned_sig_std)
            
            # 5. Perform interpolation for common genes using pandas indexing
            refined_sig.loc[common_genes] = (
                (1 - learning_rate) * starting_common + 
                learning_rate * learned_common
            )
            
            refined_signatures.append(refined_sig)
        
        # Convert refined signatures to AnnData format
        if len(refined_signatures) > 0:
            # Stack all refined signatures into a matrix
            refined_matrix = np.vstack([sig.values for sig in refined_signatures])
            
            # Create AnnData with full starting signature gene set
            refined_adata = AnnData(
                X=refined_matrix,
                obs=self.learned_signatures.obs.copy(),
                var=pd.DataFrame(index=self.starting_signature.index)
            )
            
            # Copy layers structure from learned signatures (but expand to full gene set)
            if hasattr(self.learned_signatures, 'layers') and self.learned_signatures.layers:
                for layer_name, layer_data in self.learned_signatures.layers.items():
                    # Initialize layer with zeros for all genes
                    full_layer = np.zeros((len(refined_signatures), len(self.starting_signature)))
                    
                    # Copy layer data for common genes only
                    for i in range(len(refined_signatures)):
                        learned_layer_sig = pd.Series(
                            layer_data[i, :],
                            index=self.learned_signatures.var_names
                        )
                        # Use pandas indexing to align common genes
                        for gene in common_genes:
                            gene_idx = self.starting_signature.index.get_loc(gene)
                            full_layer[i, gene_idx] = learned_layer_sig.loc[gene]
                    
                    refined_adata.layers[layer_name] = full_layer
            
            # Store the refined signatures
            self.refined_signatures = refined_adata
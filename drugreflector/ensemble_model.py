"""
Ensemble model implementation for DrugReflector.

This module provides the EnsembleModel class that loads and averages
predictions from multiple trained models.
"""

import numpy as np
import pandas as pd
import torch
from typing import List, Union
from anndata import AnnData
import warnings
import scipy.stats as stats

from .models import nnFC


class EnsembleModel:
    """
    Ensemble model for averaging predictions from multiple trained models.
    
    This class loads 3 trained neural network models and averages their
    predictions to provide compound rankings for gene expression signatures.
    
    Parameters
    ----------
    checkpoint_paths : List[str]
        List of paths to the 3 trained model checkpoints
    """
    
    def __init__(self, checkpoint_paths: List[str]):
        if len(checkpoint_paths) != 3:
            raise ValueError("Exactly 3 checkpoint paths are required for the ensemble")
            
        self.checkpoint_paths = checkpoint_paths
        self.dimensions = {}
        self._loaded_models = []
        self._load_models()
    
    def _load_models(self):
        """
        Load all models from checkpoint paths and extract dimensions.

        Uses weights_only=True for secure loading. Checkpoints must be cleaned
        to remove cifra/sklearn dependencies (see clean_checkpoint.ipynb).
        """
        var_names = []
        output_names = []

        for ckpt_path in self.checkpoint_paths:
            # Load checkpoint with weights_only=True for security
            checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=True)
            
            # Extract model state dict and dimensions
            model_state_dict = checkpoint['model_state_dict']
            input_size = checkpoint['dimensions']['input_size']
            output_size = checkpoint['dimensions']['output_size']
            
            # Extract model architecture parameters from the correct location
            torch_init_params = checkpoint['params_init']['model_init_params']['torch_init_params']
            
            # Create model with exact same architecture as original
            model = nnFC(
                input_dim=input_size,
                output_dim=output_size,
                hidden_dims=torch_init_params.get('hidden_dims', [1024, 1024]),
                dropout_p=torch_init_params.get('dropout_p', 0.2),
                activation=torch_init_params.get('activation', 'ReLU'),
                batch_norm=torch_init_params.get('batch_norm', True),
                order=torch_init_params.get('order', 'act-drop-bn'),
                final_layer_bias=torch_init_params.get('final_layer_bias', True)
            )
            
            # Filter model state dict to remove 'model.' prefix if present
            filtered_state_dict = {}
            for key, value in model_state_dict.items():
                if key.startswith('model.'):
                    filtered_state_dict[key[6:]] = value
                else:
                    filtered_state_dict[key] = value
            
            model.load_state_dict(filtered_state_dict)
            model.eval()
            
            self._loaded_models.append(model)
            
            # Store dimensions (now lists instead of pandas Index)
            var_names.append(checkpoint['dimensions']['input_names'])
            output_names.append(checkpoint['dimensions']['output_names'])

        # Validate that all models have the same output classes
        # Compare lists element-wise since output_names is now list of lists
        for i in range(1, len(output_names)):
            if output_names[i] != output_names[0]:
                raise ValueError(f"Model {i} has different output classes than model 0")

        self.dimensions['output_names'] = output_names[0]
        self.dimensions['var_names'] = var_names
    
    @property
    def loaded_models(self):
        """Core PyTorch models used for prediction."""
        return self._loaded_models
    
    def _check_and_get_X(self, data: Union[np.ndarray, AnnData]) -> torch.Tensor:
        """Extract and validate input data."""
        if isinstance(data, AnnData):
            X = data.X
        else:
            X = data
        
        if not np.isfinite(np.abs(X).max()):
            raise ValueError('Data must not have nan or inf values.')
        
        return torch.from_numpy(X.astype(np.float32))
    
    @torch.no_grad()
    def get_predictions(self, data: List[Union[np.ndarray, AnnData]], average: bool = True):
        """
        Get predictions using ensemble of models.
        
        Parameters
        ----------
        data : List[Union[np.ndarray, AnnData]]
            List of formatted data arrays, one per model
        average : bool, default=True
            If True, averages predictions across models
            
        Returns
        -------
        np.ndarray
            Prediction matrix of shape (n_obs, n_classes)
        """
        scores = []
        
        for idx, model_data in enumerate(data):
            X = self._check_and_get_X(model_data)
            
            # Cast to same dtype as model parameters
            model_dtype = next(self.loaded_models[idx].parameters()).dtype
            X = X.to(dtype=model_dtype)
            
            # Get predictions
            pred = self.loaded_models[idx](X).cpu().detach()
            scores.append(pred)
        
        scores = torch.stack(scores)
        
        if average:
            scores = scores.mean(dim=0)
        
        return scores.numpy()
    
    def format_vscores(self, data: AnnData) -> List[AnnData]:
        """
        Format input data for each model in the ensemble.
        
        Parameters
        ----------
        data : AnnData
            Input data with gene expression values
            
        Returns
        -------
        List[AnnData]
            List of formatted AnnData objects, one per model
        """
        formatted_data = []
        
        for idx, model_var_names in enumerate(self.dimensions['var_names']):
            # Select genes required by this model
            model_genes = pd.Index(model_var_names)
            data_genes = data.var_names
            
            # Find intersection and missing genes
            common_genes = model_genes.intersection(data_genes)
            missing_genes = model_genes.difference(data_genes)
            
            if len(missing_genes) > 0:
                warnings.warn(f"Model {idx} missing {len(missing_genes)} genes: {missing_genes[:10].tolist()}")
            
            # Create subset with available genes
            gene_mask = data.var_names.isin(common_genes)
            subset_data = data[:, gene_mask].copy()
            
            # Reorder to match model's expected gene order
            if len(common_genes) > 0:
                gene_positions = [subset_data.var_names.get_loc(gene) for gene in common_genes 
                                if gene in subset_data.var_names]
                subset_data = subset_data[:, gene_positions]
            
            # Handle missing genes by padding with zeros
            if len(missing_genes) > 0:
                # Create zero matrix for missing genes
                missing_data = np.zeros((subset_data.n_obs, len(missing_genes)))
                
                # Combine available and missing gene data
                combined_X = np.hstack([subset_data.X, missing_data])
                
                # Create new var_names with missing genes
                combined_var_names = list(subset_data.var_names) + list(missing_genes)
                
                # Create new AnnData with combined data
                combined_data = AnnData(
                    X=combined_X,
                    obs=subset_data.obs.copy(),
                    var=pd.DataFrame(index=combined_var_names)
                )
                
                # Reorder to match model's expected gene order
                reorder_idx = [combined_data.var_names.get_loc(gene) for gene in model_genes]
                subset_data = combined_data[:, reorder_idx]
            
            formatted_data.append(subset_data)
        
        return formatted_data
    
    def transform(self, data: AnnData, ranks: bool = False, average: bool = True, 
                  return_anndata: bool = True) -> Union[np.ndarray, AnnData]:
        """
        Transform input data through the ensemble model.
        
        Parameters
        ----------
        data : AnnData
            Input gene expression data
        ranks : bool, default=False
            Whether to compute ranks for predictions
        average : bool, default=True
            Whether to average predictions across models
        return_anndata : bool, default=True
            Whether to return AnnData object (vs numpy array)
            
        Returns
        -------
        Union[np.ndarray, AnnData]
            Prediction scores or AnnData with predictions
        """
        # Format data for each model
        formatted_data = self.format_vscores(data)
        
        # Get ensemble predictions
        scores = self.get_predictions(formatted_data, average=average)
        
        if not return_anndata:
            return scores
        
        # Create AnnData with predictions
        result = AnnData(
            X=scores,
            obs=data.obs.copy(),
            var=pd.DataFrame(index=self.dimensions['output_names'])
        )
        
        if ranks:
            self._add_ranks_layer(result)
        
        return result
    
    def _add_ranks_layer(self, scores: AnnData):
        """Add ranks layer to AnnData object."""
        ranks = stats.rankdata(-scores.X, axis=1) - 1
        scores.layers['ranks'] = ranks.astype(int)
        scores.var['ranks_min'] = ranks.min(axis=0)
        scores.var['ranks_mean'] = ranks.mean(axis=0)
        scores.var['ranks_sd'] = ranks.std(axis=0)
        scores.var['n_obs_top_50'] = (ranks < 50).sum(axis=0)
        scores.var['frac_obs_top_50'] = scores.var['n_obs_top_50'] / scores.n_obs
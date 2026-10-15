"""
DrugReflector

A minimal implementation for compound ranking predictions from gene expression signatures
using ensemble neural network models.
"""

from .drug_reflector import DrugReflector
from .ensemble_model import EnsembleModel
from .models import nnFC
from .utils import (
    load_h5ad_file,
    create_synthetic_gene_expression,
    compute_vscores,
    compute_vscore_two_groups,
    compute_vscores_adata,
    pseudobulk_adata,
)
from .signature_refinement import SignatureRefinement

from importlib.metadata import version as _version

__version__ = _version("drugreflector")
__all__ = [
    "DrugReflector",
    "EnsembleModel",
    "nnFC",
    "load_h5ad_file",
    "create_synthetic_gene_expression",
    "compute_vscores",
    "compute_vscore_two_groups",
    "compute_vscores_adata",
    "pseudobulk_adata",
    "SignatureRefinement",
]
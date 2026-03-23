"""
Utility functions for DrugReflector.

This module re-exports functions from drugreflector.utils for backwards compatibility.
"""

from drugreflector.utils import (
    _norm_func,
    clip_rescale_rows,
    compute_vscore_two_groups,
    compute_vscores_adata,
    compute_vscores,
    load_h5ad_file,
    create_synthetic_gene_expression,
    pseudobulk_adata,
)

__all__ = [
    "clip_rescale_rows",
    "compute_vscore_two_groups",
    "compute_vscores_adata",
    "compute_vscores",
    "load_h5ad_file",
    "create_synthetic_gene_expression",
    "pseudobulk_adata",
]

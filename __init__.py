"""
Deep Virtual Screening with DrugReflector V3.5

A minimal implementation for compound ranking predictions from gene expression signatures
using ensemble neural network models.
"""

from .drug_reflector import DrugReflectorV35
from .ensemble_model import EnsembleModel
from .models import nnFC
from .utils import load_h5ad_file, compute_vscores, clip_rescale_rows

__version__ = "1.0.0"
__all__ = [
    "DrugReflectorV35",
    "EnsembleModel", 
    "nnFC",
    "load_h5ad_file",
    "compute_vscores",
    "clip_rescale_rows"
]
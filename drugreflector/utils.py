"""
Utility functions for DrugReflector preprocessing.

This module contains functions for v-score computation and data preprocessing
that are specific to DrugReflector functionality.
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



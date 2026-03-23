"""Shared fixtures for drugreflector test suite."""

import os
import pytest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from anndata import AnnData

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(REPO_ROOT, "checkpoints")
CHECKPOINT_PATHS = [
    os.path.join(CHECKPOINT_DIR, "model_fold_0.pt"),
    os.path.join(CHECKPOINT_DIR, "model_fold_1.pt"),
    os.path.join(CHECKPOINT_DIR, "model_fold_2.pt"),
]


# ---------------------------------------------------------------------------
# Session-scoped: extract gene names from first checkpoint (loaded once)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def real_gene_names():
    """978 input gene names from the first checkpoint."""
    ckpt = torch.load(CHECKPOINT_PATHS[0], map_location="cpu", weights_only=True)
    return ckpt["dimensions"]["input_names"]


@pytest.fixture(scope="session")
def real_output_names():
    """9597 output compound IDs from the first checkpoint."""
    ckpt = torch.load(CHECKPOINT_PATHS[0], map_location="cpu", weights_only=True)
    return ckpt["dimensions"]["output_names"]


@pytest.fixture(scope="session")
def checkpoint_paths():
    """List of 3 real checkpoint paths."""
    return CHECKPOINT_PATHS


# ---------------------------------------------------------------------------
# Session-scoped integration fixtures (load models once)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def ensemble_model_real():
    """Real EnsembleModel with all 3 checkpoints."""
    from drugreflector.ensemble_model import EnsembleModel
    return EnsembleModel(CHECKPOINT_PATHS)


@pytest.fixture(scope="session")
def drug_reflector_real():
    """Real DrugReflector with all 3 checkpoints."""
    from drugreflector.drug_reflector import DrugReflector
    return DrugReflector(checkpoint_paths=CHECKPOINT_PATHS)


# ---------------------------------------------------------------------------
# Synthetic data fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def synthetic_adata_small():
    """10 obs x 100 genes with a 'group' column (control/treatment)."""
    np.random.seed(42)
    n_obs, n_vars = 10, 100
    X = np.random.normal(0, 1, (n_obs, n_vars))
    obs = pd.DataFrame(
        {"group": (["control"] * 5) + (["treatment"] * 5)},
        index=[f"sample_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(index=[f"GENE{i}" for i in range(n_vars)])
    return AnnData(X=X, obs=obs, var=var)


@pytest.fixture
def synthetic_adata_landmark(real_gene_names):
    """3 obs x 978 genes using real landmark gene names."""
    np.random.seed(99)
    n_obs = 3
    X = np.random.normal(0, 1, (n_obs, len(real_gene_names))).astype(np.float64)
    obs = pd.DataFrame(index=[f"transition_{i}" for i in range(n_obs)])
    var = pd.DataFrame(index=real_gene_names)
    return AnnData(X=X, obs=obs, var=var)


@pytest.fixture
def vscore_series_landmark(real_gene_names):
    """Single v-score pd.Series with 978 real gene names."""
    np.random.seed(7)
    return pd.Series(
        np.random.normal(0, 1, len(real_gene_names)),
        index=real_gene_names,
        name="test_transition",
    )


@pytest.fixture
def vscore_dataframe_landmark(real_gene_names):
    """DataFrame with 2 transitions x 978 real genes."""
    np.random.seed(8)
    return pd.DataFrame(
        np.random.normal(0, 1, (2, len(real_gene_names))),
        columns=real_gene_names,
        index=["transition_A", "transition_B"],
    )


# ---------------------------------------------------------------------------
# Mock checkpoint factory
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_checkpoint_factory():
    """Factory that creates tiny mock checkpoints matching real structure."""
    def _make(input_size=20, output_size=10, hidden_dims=None,
              input_names=None, output_names=None):
        if hidden_dims is None:
            hidden_dims = [32]
        if input_names is None:
            input_names = [f"GENE{i}" for i in range(input_size)]
        if output_names is None:
            output_names = [f"BRD-{i:08d}" for i in range(output_size)]

        from drugreflector.models import nnFC
        model = nnFC(
            input_dim=input_size,
            output_dim=output_size,
            hidden_dims=hidden_dims,
            dropout_p=0.5,
            activation="ReLU",
            batch_norm=True,
            order="bn-act-drop",
            final_layer_bias=True,
        )
        state_dict = {f"model.{k}": v for k, v in model.state_dict().items()}

        return {
            "model_state_dict": state_dict,
            "dimensions": {
                "input_size": input_size,
                "output_size": output_size,
                "input_names": input_names,
                "output_names": output_names,
            },
            "params_init": {
                "model_init_params": {
                    "torch_init_params": {
                        "hidden_dims": hidden_dims,
                        "dropout_p": 0.5,
                        "activation": "ReLU",
                        "batch_norm": True,
                        "final_layer_bias": True,
                        "order": "bn-act-drop",
                    }
                }
            },
        }
    return _make

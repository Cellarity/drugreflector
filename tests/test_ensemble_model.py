"""Tests for EnsembleModel (ensemble_model.py)."""

import pytest
import numpy as np
import pandas as pd
import torch
import warnings
from unittest.mock import patch, MagicMock
from anndata import AnnData

from drugreflector.ensemble_model import EnsembleModel


# ---------------------------------------------------------------------------
# Helper to build a mock ensemble
# ---------------------------------------------------------------------------
def _build_mock_ensemble(mock_checkpoint_factory, input_size=20, output_size=10,
                         hidden_dims=None, input_names=None, output_names=None):
    """Patch torch.load and construct an EnsembleModel from mock checkpoints."""
    ckpt = mock_checkpoint_factory(
        input_size=input_size, output_size=output_size,
        hidden_dims=hidden_dims, input_names=input_names,
        output_names=output_names,
    )
    with patch("torch.load", return_value=ckpt):
        return EnsembleModel(["f0.pt", "f1.pt", "f2.pt"])


# ---------------------------------------------------------------------------
# Unit tests (mock checkpoints, fast)
# ---------------------------------------------------------------------------
class TestEnsembleModelInit:
    def test_wrong_checkpoint_count_2(self):
        with pytest.raises(ValueError, match="Exactly 3"):
            EnsembleModel(["a.pt", "b.pt"])

    def test_wrong_checkpoint_count_4(self):
        with pytest.raises(ValueError, match="Exactly 3"):
            EnsembleModel(["a.pt", "b.pt", "c.pt", "d.pt"])

    def test_model_prefix_stripping(self, mock_checkpoint_factory):
        em = _build_mock_ensemble(mock_checkpoint_factory)
        assert len(em.loaded_models) == 3
        for m in em.loaded_models:
            assert not m.training  # eval mode


class TestFormatVscores:
    def test_all_genes_present(self, mock_checkpoint_factory):
        em = _build_mock_ensemble(mock_checkpoint_factory, input_size=20, output_size=10)
        gene_names = [f"GENE{i}" for i in range(20)]
        adata = AnnData(
            X=np.random.normal(0, 1, (3, 20)),
            var=pd.DataFrame(index=gene_names),
        )
        formatted = em.format_vscores(adata)
        assert len(formatted) == 3
        for f in formatted:
            assert f.shape == (3, 20)

    def test_missing_genes_padded(self, mock_checkpoint_factory):
        em = _build_mock_ensemble(mock_checkpoint_factory, input_size=20, output_size=10)
        # Only provide first 10 genes
        adata = AnnData(
            X=np.random.normal(0, 1, (2, 10)),
            var=pd.DataFrame(index=[f"GENE{i}" for i in range(10)]),
        )
        with pytest.warns(UserWarning, match="missing"):
            formatted = em.format_vscores(adata)
        for f in formatted:
            assert f.shape == (2, 20)
            # Last 10 genes should be zero-padded
            missing_mask = ~pd.Index(f.var_names).isin([f"GENE{i}" for i in range(10)])
            # Zero-padded values
            assert np.all(f.X[:, f.var_names.isin([f"GENE{i}" for i in range(10, 20)])] == 0)

    def test_extra_genes_dropped(self, mock_checkpoint_factory):
        em = _build_mock_ensemble(mock_checkpoint_factory, input_size=20, output_size=10)
        # 30 genes (20 model + 10 extra)
        gene_names = [f"GENE{i}" for i in range(30)]
        adata = AnnData(
            X=np.random.normal(0, 1, (2, 30)),
            var=pd.DataFrame(index=gene_names),
        )
        formatted = em.format_vscores(adata)
        for f in formatted:
            assert f.shape == (2, 20)

    def test_reorders_genes(self, mock_checkpoint_factory):
        em = _build_mock_ensemble(mock_checkpoint_factory, input_size=20, output_size=10)
        # Reversed gene order
        gene_names = [f"GENE{i}" for i in range(19, -1, -1)]
        X = np.arange(40).reshape(2, 20).astype(float)
        adata = AnnData(X=X, var=pd.DataFrame(index=gene_names))
        formatted = em.format_vscores(adata)
        # Should be reordered to GENE0..GENE19
        expected_order = [f"GENE{i}" for i in range(20)]
        for f in formatted:
            assert list(f.var_names) == expected_order


class TestCheckAndGetX:
    def test_valid_array(self, mock_checkpoint_factory):
        em = _build_mock_ensemble(mock_checkpoint_factory)
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        tensor = em._check_and_get_X(X)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (2, 2)

    def test_anndata_input(self, mock_checkpoint_factory):
        em = _build_mock_ensemble(mock_checkpoint_factory)
        adata = AnnData(X=np.array([[1.0, 2.0]]))
        tensor = em._check_and_get_X(adata)
        assert isinstance(tensor, torch.Tensor)

    def test_nan_raises(self, mock_checkpoint_factory):
        em = _build_mock_ensemble(mock_checkpoint_factory)
        X = np.array([[1.0, np.nan]])
        with pytest.raises(ValueError, match="nan or inf"):
            em._check_and_get_X(X)

    def test_inf_raises(self, mock_checkpoint_factory):
        em = _build_mock_ensemble(mock_checkpoint_factory)
        X = np.array([[1.0, np.inf]])
        with pytest.raises(ValueError, match="nan or inf"):
            em._check_and_get_X(X)


class TestGetPredictions:
    def test_output_shape(self, mock_checkpoint_factory):
        em = _build_mock_ensemble(mock_checkpoint_factory, input_size=20, output_size=10)
        data = [np.random.normal(0, 1, (2, 20)).astype(np.float32) for _ in range(3)]
        result = em.get_predictions(data, average=True)
        assert result.shape == (2, 10)

    def test_no_average(self, mock_checkpoint_factory):
        em = _build_mock_ensemble(mock_checkpoint_factory, input_size=20, output_size=10)
        data = [np.random.normal(0, 1, (2, 20)).astype(np.float32) for _ in range(3)]
        result = em.get_predictions(data, average=False)
        assert result.shape == (3, 2, 10)


class TestTransform:
    def test_returns_anndata(self, mock_checkpoint_factory):
        em = _build_mock_ensemble(mock_checkpoint_factory, input_size=20, output_size=10)
        gene_names = [f"GENE{i}" for i in range(20)]
        adata = AnnData(
            X=np.random.normal(0, 1, (2, 20)).astype(np.float32),
            var=pd.DataFrame(index=gene_names),
        )
        result = em.transform(adata, return_anndata=True)
        assert isinstance(result, AnnData)
        assert result.shape == (2, 10)

    def test_returns_ndarray(self, mock_checkpoint_factory):
        em = _build_mock_ensemble(mock_checkpoint_factory, input_size=20, output_size=10)
        gene_names = [f"GENE{i}" for i in range(20)]
        adata = AnnData(
            X=np.random.normal(0, 1, (2, 20)).astype(np.float32),
            var=pd.DataFrame(index=gene_names),
        )
        result = em.transform(adata, return_anndata=False)
        assert isinstance(result, np.ndarray)

    def test_with_ranks(self, mock_checkpoint_factory):
        em = _build_mock_ensemble(mock_checkpoint_factory, input_size=20, output_size=10)
        gene_names = [f"GENE{i}" for i in range(20)]
        adata = AnnData(
            X=np.random.normal(0, 1, (2, 20)).astype(np.float32),
            var=pd.DataFrame(index=gene_names),
        )
        result = em.transform(adata, ranks=True)
        assert "ranks" in result.layers
        assert "ranks_min" in result.var.columns
        assert "ranks_mean" in result.var.columns
        assert "ranks_sd" in result.var.columns
        assert "n_obs_top_50" in result.var.columns
        assert "frac_obs_top_50" in result.var.columns


class TestAddRanksLayer:
    def test_rank_ordering(self, mock_checkpoint_factory):
        em = _build_mock_ensemble(mock_checkpoint_factory, input_size=20, output_size=5)
        # Known scores: higher score = lower rank (rankdata on -scores)
        scores = AnnData(
            X=np.array([[10.0, 5.0, 8.0, 3.0, 1.0]]),
            var=pd.DataFrame(index=[f"C{i}" for i in range(5)]),
        )
        em._add_ranks_layer(scores)
        ranks = scores.layers["ranks"][0]
        # Score 10 -> rank 0, 8 -> rank 1, 5 -> rank 2, 3 -> rank 3, 1 -> rank 4
        assert ranks[0] == 0  # highest score
        assert ranks[2] == 1
        assert ranks[1] == 2
        assert ranks[3] == 3
        assert ranks[4] == 4  # lowest score

    def test_rank_stats(self, mock_checkpoint_factory):
        em = _build_mock_ensemble(mock_checkpoint_factory, input_size=20, output_size=5)
        scores = AnnData(
            X=np.array([[10.0, 5.0, 8.0, 3.0, 1.0],
                        [1.0, 10.0, 5.0, 8.0, 3.0]]),
            var=pd.DataFrame(index=[f"C{i}" for i in range(5)]),
        )
        em._add_ranks_layer(scores)
        assert "ranks_min" in scores.var.columns
        assert "ranks_mean" in scores.var.columns
        assert "frac_obs_top_50" in scores.var.columns


# ---------------------------------------------------------------------------
# Integration tests (real checkpoints)
# ---------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.integration
class TestEnsembleModelIntegration:
    def test_real_checkpoint_loading(self, ensemble_model_real):
        assert len(ensemble_model_real.loaded_models) == 3
        for m in ensemble_model_real.loaded_models:
            assert not m.training

    def test_real_dimensions(self, ensemble_model_real):
        assert len(ensemble_model_real.dimensions["output_names"]) == 9597
        assert len(ensemble_model_real.dimensions["var_names"]) == 3
        for vn in ensemble_model_real.dimensions["var_names"]:
            assert len(vn) == 978

    def test_real_forward_pass(self, ensemble_model_real, synthetic_adata_landmark):
        result = ensemble_model_real.transform(synthetic_adata_landmark)
        assert isinstance(result, AnnData)
        assert result.shape == (3, 9597)

    def test_real_predictions_finite(self, ensemble_model_real, synthetic_adata_landmark):
        result = ensemble_model_real.transform(synthetic_adata_landmark)
        assert np.isfinite(result.X).all()

    def test_real_predictions_deterministic(self, ensemble_model_real, synthetic_adata_landmark):
        r1 = ensemble_model_real.transform(synthetic_adata_landmark)
        r2 = ensemble_model_real.transform(synthetic_adata_landmark)
        np.testing.assert_array_equal(r1.X, r2.X)

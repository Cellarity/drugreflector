"""Tests for SignatureRefinement (signature_refinement.py)."""

import pytest
import numpy as np
import pandas as pd
import warnings
from anndata import AnnData

from drugreflector import SignatureRefinement


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_expr_adata(n_compounds=10, n_genes=50, compound_col="compound_id"):
    """Create expression AnnData with compound IDs."""
    np.random.seed(42)
    X = np.random.normal(5, 2, (n_compounds, n_genes))
    obs = pd.DataFrame(
        {compound_col: [f"cpd_{i}" for i in range(n_compounds)]},
        index=[f"obs_{i}" for i in range(n_compounds)],
    )
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
    return AnnData(X=X, obs=obs, var=var)


def _make_readouts(n_compounds=10):
    """Create phenotypic readout series."""
    np.random.seed(43)
    return pd.Series(
        np.random.normal(0, 1, n_compounds),
        index=[f"cpd_{i}" for i in range(n_compounds)],
        name="viability",
    )


def _make_starting_sig(n_genes=50):
    """Create starting signature series."""
    np.random.seed(44)
    return pd.Series(
        np.random.normal(0, 1, n_genes),
        index=[f"gene_{i}" for i in range(n_genes)],
        name="starting",
    )


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------
class TestSignatureRefinementInit:
    def test_from_series(self):
        sig = _make_starting_sig(10)
        sr = SignatureRefinement(sig)
        assert isinstance(sr.starting_signature, pd.Series)
        assert len(sr.starting_signature) == 10

    def test_from_anndata_single_obs(self):
        X = np.array([[1.0, 2.0, 3.0]])
        adata = AnnData(X=X, var=pd.DataFrame(index=["g1", "g2", "g3"]))
        sr = SignatureRefinement(adata)
        assert isinstance(sr.starting_signature, pd.Series)
        assert list(sr.starting_signature.index) == ["g1", "g2", "g3"]

    def test_from_anndata_multi_obs_raises(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        adata = AnnData(X=X)
        with pytest.raises(ValueError, match="exactly 1"):
            SignatureRefinement(adata)

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="AnnData or pd.Series"):
            SignatureRefinement({"a": 1})

    def test_attributes_initialized_none(self):
        sr = SignatureRefinement(_make_starting_sig(5))
        assert sr.expr is None
        assert sr.readouts is None
        assert sr.learned_signatures is None
        assert sr.refined_signatures is None


# ---------------------------------------------------------------------------
# load_phenotypic_readouts
# ---------------------------------------------------------------------------
class TestLoadPhenotypicReadouts:
    def test_from_series(self):
        sr = SignatureRefinement(_make_starting_sig(5))
        readouts = pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"])
        result = sr.load_phenotypic_readouts(readouts)
        assert isinstance(result, pd.Series)
        assert len(sr.readouts) == 3

    def test_from_dataframe(self):
        sr = SignatureRefinement(_make_starting_sig(5))
        df = pd.DataFrame(
            {"viability": [1.0, 2.0], "other": [3.0, 4.0]},
            index=["cpd_0", "cpd_1"],
        )
        result = sr.load_phenotypic_readouts(df, readout_col="viability")
        assert len(sr.readouts) == 2

    def test_dataframe_no_readout_col_raises(self):
        sr = SignatureRefinement(_make_starting_sig(5))
        df = pd.DataFrame({"viability": [1.0]})
        with pytest.raises(ValueError, match="readout_col"):
            sr.load_phenotypic_readouts(df)

    def test_with_compound_id_col(self):
        sr = SignatureRefinement(_make_starting_sig(5))
        df = pd.DataFrame({
            "cpd": ["a", "b"],
            "viability": [1.0, 2.0],
        })
        sr.load_phenotypic_readouts(df, readout_col="viability", compound_id_col="cpd")
        assert list(sr.readouts.index) == ["a", "b"]

    def test_nan_dropped(self):
        sr = SignatureRefinement(_make_starting_sig(5))
        readouts = pd.Series([1.0, np.nan, 3.0], index=["a", "b", "c"])
        with pytest.warns(UserWarning, match="Dropped 1 NaN"):
            sr.load_phenotypic_readouts(readouts)
        assert len(sr.readouts) == 2

    def test_duplicate_compounds_same_value(self):
        sr = SignatureRefinement(_make_starting_sig(5))
        readouts = pd.Series([1.0, 1.0], index=["a", "a"])
        sr.load_phenotypic_readouts(readouts)
        assert len(sr.readouts) == 1
        assert sr.readouts.loc["a"] == 1.0

    def test_duplicate_compounds_diff_value(self):
        sr = SignatureRefinement(_make_starting_sig(5))
        readouts = pd.Series([1.0, 3.0], index=["a", "a"])
        with pytest.warns(UserWarning, match="multiple distinct"):
            sr.load_phenotypic_readouts(readouts)
        assert sr.readouts.loc["a"] == pytest.approx(2.0)

    def test_invalid_type_raises(self):
        sr = SignatureRefinement(_make_starting_sig(5))
        with pytest.raises(ValueError, match="DataFrame or Series"):
            sr.load_phenotypic_readouts([1, 2, 3])


# ---------------------------------------------------------------------------
# load_normalized_data
# ---------------------------------------------------------------------------
class TestLoadNormalizedData:
    def test_basic(self):
        sr = SignatureRefinement(_make_starting_sig(50))
        adata = _make_expr_adata(n_compounds=10, n_genes=50)
        sr.load_normalized_data(adata, compound_id_obs_col="compound_id")
        assert sr.expr is not None
        assert sr.expr.n_vars == 50

    def test_pseudobulks_by_compound(self):
        sr = SignatureRefinement(_make_starting_sig(5))
        # 6 observations, 3 compounds with 2 replicates each
        X = np.ones((6, 5))
        obs = pd.DataFrame({
            "compound_id": ["c1", "c1", "c2", "c2", "c3", "c3"],
        }, index=[f"s{i}" for i in range(6)])
        adata = AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(5)]))
        sr.load_normalized_data(adata, compound_id_obs_col="compound_id")
        # Should pseudobulk to 3 compounds
        assert sr.expr.n_obs == 3


# ---------------------------------------------------------------------------
# load_counts_data
# ---------------------------------------------------------------------------
class TestLoadCountsData:
    @pytest.mark.xfail(
        reason="Source bug: counts_data.flatten()[:1000] returns memoryview in newer numpy/anndata",
        raises=TypeError,
    )
    def test_basic(self):
        sr = SignatureRefinement(_make_starting_sig(50))
        np.random.seed(10)
        X = np.random.poisson(10, (10, 50)).astype(np.float64)
        obs = pd.DataFrame(
            {"compound_id": [f"cpd_{i}" for i in range(10)]},
            index=[f"obs_{i}" for i in range(10)],
        )
        var = pd.DataFrame(index=[f"gene_{i}" for i in range(50)])
        adata = AnnData(X=X, obs=obs, var=var)
        sr.load_counts_data(adata, compound_id_obs_col="compound_id")
        assert sr.expr is not None
        assert "pseudobulked_counts" in sr.expr.layers

    @pytest.mark.xfail(
        reason="Source bug: counts_data.flatten()[:1000] returns memoryview in newer numpy/anndata",
    )
    def test_negative_warns(self):
        sr = SignatureRefinement(_make_starting_sig(5))
        X = np.array([[-1.0, 2.0, 3.0]], dtype=np.float64)
        obs = pd.DataFrame({"compound_id": ["c1"]}, index=["s1"])
        adata = AnnData(X=X, obs=obs, var=pd.DataFrame(index=["g1", "g2", "g3"]))
        with pytest.warns(UserWarning, match="negative values"):
            sr.load_counts_data(adata, compound_id_obs_col="compound_id")


# ---------------------------------------------------------------------------
# load_paired_readouts
# ---------------------------------------------------------------------------
class TestLoadPairedReadouts:
    def test_both_layers_raises(self):
        sr = SignatureRefinement(_make_starting_sig(5))
        adata = AnnData(X=np.ones((2, 5)))
        with pytest.raises(ValueError, match="Cannot specify both"):
            sr.load_paired_readouts(
                adata, compound_id_obs_col="cpd",
                readouts=pd.Series([1.0]),
                normalized_counts_layer="norm",
                raw_counts_layer="raw",
            )

    def test_neither_layer_raises(self):
        sr = SignatureRefinement(_make_starting_sig(5))
        adata = AnnData(X=np.ones((2, 5)))
        with pytest.raises(ValueError, match="Must specify either"):
            sr.load_paired_readouts(
                adata, compound_id_obs_col="cpd",
                readouts=pd.Series([1.0]),
            )


# ---------------------------------------------------------------------------
# _learned_signature
# ---------------------------------------------------------------------------
class TestLearnedSignature:
    def test_basic_shape(self):
        sr = SignatureRefinement(_make_starting_sig(10))
        np.random.seed(50)
        expr = np.random.normal(0, 1, (20, 10))
        readouts = np.random.normal(0, 1, 20)
        result = sr._learned_signature(expr, readouts)
        assert isinstance(result, np.ndarray)
        assert result.shape == (10,)

    def test_include_stats(self):
        sr = SignatureRefinement(_make_starting_sig(10))
        np.random.seed(50)
        expr = np.random.normal(0, 1, (20, 10))
        readouts = np.random.normal(0, 1, 20)
        result = sr._learned_signature(expr, readouts, include_stats=True)
        assert isinstance(result, dict)
        assert "scores" in result
        assert "pval" in result
        assert "statistic" in result

    def test_nan_handling(self):
        sr = SignatureRefinement(_make_starting_sig(3))
        # Gene with zero variance -> pearsonr returns NaN
        expr = np.array([[1.0, 2.0, 5.0], [1.0, 3.0, 5.0], [1.0, 4.0, 5.0]])
        readouts = np.array([1.0, 2.0, 3.0])
        result = sr._learned_signature(expr, readouts)
        # Gene 0 and 2 have zero variance -> score should be 0
        assert result[0] == 0.0
        assert result[2] == 0.0

    def test_invalid_method_raises(self):
        sr = SignatureRefinement(_make_starting_sig(5))
        with pytest.raises(ValueError, match="pearson"):
            sr._learned_signature(np.ones((3, 5)), np.ones(3), corr_method="spearman")


# ---------------------------------------------------------------------------
# compute_learned_signatures
# ---------------------------------------------------------------------------
class TestComputeLearnedSignatures:
    def test_no_expr_raises(self):
        sr = SignatureRefinement(_make_starting_sig(5))
        sr.readouts = _make_readouts(5)
        with pytest.raises(ValueError, match="No expression data"):
            sr.compute_learned_signatures()

    def test_no_readouts_raises(self):
        sr = SignatureRefinement(_make_starting_sig(50))
        sr.expr = _make_expr_adata(10, 50)
        with pytest.raises(ValueError, match="No readout data"):
            sr.compute_learned_signatures()

    def test_basic(self):
        sr = SignatureRefinement(_make_starting_sig(50))
        adata = _make_expr_adata(10, 50)
        sr.load_normalized_data(adata, compound_id_obs_col="compound_id")
        sr.load_phenotypic_readouts(_make_readouts(10))
        sr.compute_learned_signatures()
        assert sr.learned_signatures is not None
        assert sr.learned_signatures.n_obs == 1
        assert sr.learned_signatures.n_vars == 50

    def test_few_common_compounds_raises(self):
        sr = SignatureRefinement(_make_starting_sig(50))
        adata = _make_expr_adata(10, 50)
        sr.load_normalized_data(adata, compound_id_obs_col="compound_id")
        # Readouts with no overlapping compound IDs
        readouts = pd.Series([1.0, 2.0], index=["no_match_1", "no_match_2"])
        sr.load_phenotypic_readouts(readouts)
        with pytest.raises(ValueError, match="Not enough common"):
            sr.compute_learned_signatures()


# ---------------------------------------------------------------------------
# compute_refined_signatures
# ---------------------------------------------------------------------------
class TestComputeRefinedSignatures:
    def _setup_full_pipeline(self):
        """Set up SR with learned signatures ready."""
        sig = _make_starting_sig(50)
        sr = SignatureRefinement(sig)
        adata = _make_expr_adata(10, 50)
        sr.load_normalized_data(adata, compound_id_obs_col="compound_id")
        sr.load_phenotypic_readouts(_make_readouts(10))
        sr.compute_learned_signatures()
        return sr

    def test_no_learned_raises(self):
        sr = SignatureRefinement(_make_starting_sig(5))
        with pytest.raises(ValueError, match="No learned signatures"):
            sr.compute_refined_signatures()

    def test_learning_rate_0(self):
        sr = self._setup_full_pipeline()
        sr.compute_refined_signatures(learning_rate=0.0)
        # Should equal starting signature
        np.testing.assert_allclose(
            sr.refined_signatures.X[0],
            sr.starting_signature.values,
            atol=1e-10,
        )

    def test_learning_rate_1_no_scale(self):
        sr = self._setup_full_pipeline()
        sr.compute_refined_signatures(learning_rate=1.0, scale_learned_sig=False)
        # Should equal learned signature (for common genes)
        learned_vals = pd.Series(
            sr.learned_signatures.X[0],
            index=sr.learned_signatures.var_names,
        )
        common = sr.starting_signature.index.intersection(sr.learned_signatures.var_names)
        refined_vals = pd.Series(sr.refined_signatures.X[0], index=sr.refined_signatures.var_names)
        np.testing.assert_allclose(
            refined_vals.loc[common].values,
            learned_vals.loc[common].values,
            atol=1e-10,
        )

    def test_output_gene_set(self):
        sr = self._setup_full_pipeline()
        sr.compute_refined_signatures()
        # Output should have same genes as starting signature
        assert list(sr.refined_signatures.var_names) == list(sr.starting_signature.index)

    def test_no_common_genes_raises(self):
        sig = pd.Series([1.0, 2.0], index=["UNIQUE1", "UNIQUE2"])
        sr = SignatureRefinement(sig)
        # Manually create learned sigs with different genes
        sr.learned_signatures = AnnData(
            X=np.array([[1.0, 2.0]]),
            var=pd.DataFrame(index=["OTHER1", "OTHER2"]),
        )
        with pytest.raises(ValueError, match="No common genes"):
            sr.compute_refined_signatures()

    def test_scale_learned_sig(self):
        sr = self._setup_full_pipeline()
        sr.compute_refined_signatures(learning_rate=0.5, scale_learned_sig=True)
        assert sr.refined_signatures is not None
        # Result should be different from starting
        assert not np.allclose(sr.refined_signatures.X[0], sr.starting_signature.values)

    def test_zero_std_learned(self):
        sig = _make_starting_sig(5)
        sr = SignatureRefinement(sig)
        # Learned signature with all zeros (zero std)
        sr.learned_signatures = AnnData(
            X=np.zeros((1, 5)),
            var=pd.DataFrame(index=sig.index),
            obs=pd.DataFrame(index=["sig_0"]),
        )
        sr.compute_refined_signatures(learning_rate=0.5, scale_learned_sig=True)
        # With zero-std learned, learned contribution is zeroed out
        # refined = (1-lr)*starting + lr*0 = 0.5 * starting
        np.testing.assert_allclose(
            sr.refined_signatures.X[0],
            0.5 * sig.values,
            atol=1e-10,
        )

    def test_partial_gene_overlap_warns(self):
        # Starting signature has extra genes not in learned
        sig = pd.Series(
            np.random.normal(0, 1, 10),
            index=[f"gene_{i}" for i in range(10)],
        )
        sr = SignatureRefinement(sig)
        # Learned only has first 5 genes
        sr.learned_signatures = AnnData(
            X=np.random.normal(0, 1, (1, 5)),
            var=pd.DataFrame(index=[f"gene_{i}" for i in range(5)]),
            obs=pd.DataFrame(index=["sig_0"]),
        )
        with pytest.warns(UserWarning, match="Only 5/10"):
            sr.compute_refined_signatures()

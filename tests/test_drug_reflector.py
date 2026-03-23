"""Tests for DrugReflector class (drug_reflector.py)."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from anndata import AnnData

from drugreflector.drug_reflector import DrugReflector


# ---------------------------------------------------------------------------
# Helper: build a mock DrugReflector with tiny models
# ---------------------------------------------------------------------------
def _build_mock_dr(mock_checkpoint_factory, input_size=20, output_size=10,
                   input_names=None, output_names=None):
    """Build DrugReflector with patched tiny checkpoints."""
    ckpt = mock_checkpoint_factory(
        input_size=input_size, output_size=output_size,
        input_names=input_names, output_names=output_names,
    )
    with patch("torch.load", return_value=ckpt):
        return DrugReflector(checkpoint_paths=["f0.pt", "f1.pt", "f2.pt"])


# ---------------------------------------------------------------------------
# Init validation
# ---------------------------------------------------------------------------
class TestDrugReflectorInit:
    def test_invalid_model_class(self, mock_checkpoint_factory):
        ckpt = mock_checkpoint_factory()
        with patch("torch.load", return_value=ckpt):
            with pytest.raises(ValueError, match="pert_id"):
                DrugReflector(["a.pt", "b.pt", "c.pt"], model_class="other")

    def test_ensemble_false(self, mock_checkpoint_factory):
        ckpt = mock_checkpoint_factory()
        with patch("torch.load", return_value=ckpt):
            with pytest.raises(ValueError, match="ensemble=True"):
                DrugReflector(["a.pt", "b.pt", "c.pt"], ensemble=False)

    def test_wrong_checkpoint_count(self):
        with pytest.raises(ValueError, match="Exactly 3"):
            DrugReflector(["a.pt", "b.pt"])


# ---------------------------------------------------------------------------
# _prepare_vscores
# ---------------------------------------------------------------------------
class TestPrepareVscores:
    def test_from_series(self, mock_checkpoint_factory):
        dr = _build_mock_dr(mock_checkpoint_factory)
        s = pd.Series([1.0, 2.0, 3.0], index=["A", "B", "C"], name="my_transition")
        result = dr._prepare_vscores(s)
        assert isinstance(result, AnnData)
        assert result.shape == (1, 3)
        assert result.obs_names[0] == "my_transition"

    def test_from_series_no_name(self, mock_checkpoint_factory):
        dr = _build_mock_dr(mock_checkpoint_factory)
        s = pd.Series([1.0, 2.0], index=["A", "B"])
        result = dr._prepare_vscores(s)
        assert result.obs_names[0] == "vscore"

    def test_from_dataframe(self, mock_checkpoint_factory):
        dr = _build_mock_dr(mock_checkpoint_factory)
        df = pd.DataFrame(
            [[1.0, 2.0], [3.0, 4.0]],
            index=["t1", "t2"],
            columns=["G1", "G2"],
        )
        result = dr._prepare_vscores(df)
        assert result.shape == (2, 2)
        assert list(result.obs_names) == ["t1", "t2"]
        assert list(result.var_names) == ["G1", "G2"]

    def test_from_anndata(self, mock_checkpoint_factory):
        dr = _build_mock_dr(mock_checkpoint_factory)
        adata = AnnData(X=np.array([[1.0, 2.0]]))
        result = dr._prepare_vscores(adata)
        assert isinstance(result, AnnData)
        # Should be a copy
        assert result is not adata

    def test_invalid_type(self, mock_checkpoint_factory):
        dr = _build_mock_dr(mock_checkpoint_factory)
        with pytest.raises(ValueError, match="Series, DataFrame, or AnnData"):
            dr._prepare_vscores([1, 2, 3])


# ---------------------------------------------------------------------------
# Gene name preprocessing (tested via transform)
# ---------------------------------------------------------------------------
class TestGenePreprocessing:
    def test_uppercase(self, mock_checkpoint_factory):
        gene_names = ["tp53", "egfr", "brca1"]
        dr = _build_mock_dr(
            mock_checkpoint_factory, input_size=3,
            input_names=["TP53", "EGFR", "BRCA1"],
        )
        s = pd.Series([1.0, 2.0, 3.0], index=gene_names, name="test")
        result = dr.transform(s)
        # Should have preprocessed to uppercase and matched model genes

    def test_affymetrix_suffix(self, mock_checkpoint_factory):
        dr = _build_mock_dr(
            mock_checkpoint_factory, input_size=2,
            input_names=["CDKN1A", "TP53"],
        )
        s = pd.Series([1.0, 2.0], index=["CDKN1A_AT", "TP53"], name="test")
        result = dr.transform(s)
        # CDKN1A_AT -> uppercase -> CDKN1A_AT -> remove _AT -> CDKN1A

    def test_version_dot_suffix(self, mock_checkpoint_factory):
        dr = _build_mock_dr(
            mock_checkpoint_factory, input_size=2,
            input_names=["IL6", "TP53"],
        )
        s = pd.Series([1.0, 2.0], index=["IL6.v2", "TP53"], name="test")
        result = dr.transform(s)
        # IL6.v2 -> IL6.V2 -> .V2 removed -> IL6

    def test_special_chars_removed(self, mock_checkpoint_factory):
        dr = _build_mock_dr(
            mock_checkpoint_factory, input_size=2,
            input_names=["GENE1", "IL-6"],
        )
        s = pd.Series([1.0, 2.0], index=["GENE@1", "IL-6"], name="test")
        result = dr.transform(s)
        # GENE@1 -> GENE@1 -> uppercase GENE@1 -> remove @ -> GENE1
        # IL-6 -> IL-6 (hyphen kept)


# ---------------------------------------------------------------------------
# check_gene_coverage
# ---------------------------------------------------------------------------
class TestCheckGeneCoverage:
    def test_full_coverage(self, mock_checkpoint_factory):
        gene_names = [f"GENE{i}" for i in range(20)]
        dr = _build_mock_dr(mock_checkpoint_factory, input_size=20, input_names=gene_names)
        result = dr.check_gene_coverage(gene_names)
        assert result["coverage_percent"] == 100.0
        assert result["total_found"] == 20
        assert len(result["missing_genes"]) == 0

    def test_partial_coverage(self, mock_checkpoint_factory):
        model_genes = [f"GENE{i}" for i in range(20)]
        dr = _build_mock_dr(mock_checkpoint_factory, input_size=20, input_names=model_genes)
        # Half model genes + 5 unknown
        input_genes = [f"GENE{i}" for i in range(10)] + [f"UNKNOWN{i}" for i in range(5)]
        result = dr.check_gene_coverage(input_genes)
        assert result["total_input"] == 15
        assert result["total_found"] == 10
        assert len(result["missing_genes"]) == 5


# ---------------------------------------------------------------------------
# predict output structure (mock)
# ---------------------------------------------------------------------------
class TestPredictStructure:
    def test_output_columns(self, mock_checkpoint_factory):
        dr = _build_mock_dr(mock_checkpoint_factory, input_size=20, output_size=10)
        gene_names = [f"GENE{i}" for i in range(20)]
        s = pd.Series(np.random.normal(0, 1, 20), index=gene_names, name="test_obs")
        result = dr.predict(s)

        assert isinstance(result, pd.DataFrame)
        # Should have MultiIndex columns: (rank, obs), (logit, obs), (prob, obs)
        assert ("rank", "test_obs") in result.columns
        assert ("logit", "test_obs") in result.columns
        assert ("prob", "test_obs") in result.columns

    def test_probabilities_sum_to_one(self, mock_checkpoint_factory):
        dr = _build_mock_dr(mock_checkpoint_factory, input_size=20, output_size=10)
        gene_names = [f"GENE{i}" for i in range(20)]
        s = pd.Series(np.random.normal(0, 1, 20), index=gene_names, name="test_obs")
        result = dr.predict(s)
        probs = result[("prob", "test_obs")]
        assert abs(probs.sum() - 1.0) < 1e-5

    def test_ranks_valid(self, mock_checkpoint_factory):
        dr = _build_mock_dr(mock_checkpoint_factory, input_size=20, output_size=10)
        gene_names = [f"GENE{i}" for i in range(20)]
        s = pd.Series(np.random.normal(0, 1, 20), index=gene_names, name="test_obs")
        result = dr.predict(s)
        ranks = result[("rank", "test_obs")].values
        # Ranks should be 0..9 (0-indexed from rankdata -1)
        assert set(ranks) == set(range(10))

    def test_n_top_filters(self, mock_checkpoint_factory):
        dr = _build_mock_dr(mock_checkpoint_factory, input_size=20, output_size=10)
        gene_names = [f"GENE{i}" for i in range(20)]
        s = pd.Series(np.random.normal(0, 1, 20), index=gene_names, name="test_obs")
        result = dr.predict(s, n_top=5)
        assert len(result) <= 5


class TestGetTopCompounds:
    def test_returns_dict(self, mock_checkpoint_factory):
        dr = _build_mock_dr(mock_checkpoint_factory, input_size=20, output_size=10)
        gene_names = [f"GENE{i}" for i in range(20)]
        s = pd.Series(np.random.normal(0, 1, 20), index=gene_names, name="test_obs")
        result = dr.get_top_compounds(s, n_top=5)
        assert isinstance(result, dict)
        assert "test_obs" in result

    def test_n_top_respected(self, mock_checkpoint_factory):
        dr = _build_mock_dr(mock_checkpoint_factory, input_size=20, output_size=10)
        gene_names = [f"GENE{i}" for i in range(20)]
        s = pd.Series(np.random.normal(0, 1, 20), index=gene_names, name="test_obs")
        result = dr.get_top_compounds(s, n_top=3)
        assert len(result["test_obs"]) == 3

    def test_sorted_by_rank(self, mock_checkpoint_factory):
        dr = _build_mock_dr(mock_checkpoint_factory, input_size=20, output_size=10)
        gene_names = [f"GENE{i}" for i in range(20)]
        s = pd.Series(np.random.normal(0, 1, 20), index=gene_names, name="test_obs")
        result = dr.get_top_compounds(s, n_top=5)
        ranks = result["test_obs"]["rank"].values
        assert list(ranks) == sorted(ranks)

    def test_dataframe_columns(self, mock_checkpoint_factory):
        dr = _build_mock_dr(mock_checkpoint_factory, input_size=20, output_size=10)
        gene_names = [f"GENE{i}" for i in range(20)]
        s = pd.Series(np.random.normal(0, 1, 20), index=gene_names, name="test_obs")
        result = dr.get_top_compounds(s, n_top=3)
        df = result["test_obs"]
        assert "compound" in df.columns
        assert "rank" in df.columns
        assert "logit" in df.columns
        assert "prob" in df.columns

    def test_predict_top_compounds_is_alias(self, mock_checkpoint_factory):
        dr = _build_mock_dr(mock_checkpoint_factory, input_size=20, output_size=10)
        gene_names = [f"GENE{i}" for i in range(20)]
        s = pd.Series(np.random.normal(0, 1, 20), index=gene_names, name="test_obs")
        r1 = dr.get_top_compounds(s, n_top=5)
        r2 = dr.predict_top_compounds(s, n_top=5)
        pd.testing.assert_frame_equal(r1["test_obs"].reset_index(drop=True),
                                      r2["test_obs"].reset_index(drop=True),
                                      atol=1e-5)


# ---------------------------------------------------------------------------
# Integration tests (real checkpoints)
# ---------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.integration
class TestDrugReflectorIntegration:
    def test_real_transform_from_series(self, drug_reflector_real, vscore_series_landmark):
        result = drug_reflector_real.transform(vscore_series_landmark)
        assert isinstance(result, AnnData)
        assert result.shape == (1, 9597)

    def test_real_transform_from_dataframe(self, drug_reflector_real, vscore_dataframe_landmark):
        result = drug_reflector_real.transform(vscore_dataframe_landmark)
        assert isinstance(result, AnnData)
        assert result.shape == (2, 9597)

    def test_real_transform_from_anndata(self, drug_reflector_real, synthetic_adata_landmark):
        result = drug_reflector_real.transform(synthetic_adata_landmark)
        assert isinstance(result, AnnData)
        assert result.shape == (3, 9597)

    def test_real_predict(self, drug_reflector_real, vscore_series_landmark):
        result = drug_reflector_real.predict(vscore_series_landmark)
        assert isinstance(result, pd.DataFrame)
        assert ("rank", "test_transition") in result.columns
        assert ("logit", "test_transition") in result.columns
        assert ("prob", "test_transition") in result.columns
        # All values should be finite
        assert np.isfinite(result.values).all()

    def test_real_get_top_compounds(self, drug_reflector_real, vscore_series_landmark):
        result = drug_reflector_real.get_top_compounds(vscore_series_landmark, n_top=20)
        assert isinstance(result, dict)
        assert "test_transition" in result
        assert len(result["test_transition"]) == 20

    def test_real_check_gene_coverage_full(self, drug_reflector_real, real_gene_names):
        result = drug_reflector_real.check_gene_coverage(real_gene_names)
        assert result["coverage_percent"] == 100.0
        assert result["total_found"] == 978

    def test_real_check_gene_coverage_partial(self, drug_reflector_real, real_gene_names):
        mixed = list(real_gene_names[:500]) + ["FAKEGENE1", "FAKEGENE2"]
        result = drug_reflector_real.check_gene_coverage(mixed)
        assert result["total_input"] == 502
        assert result["total_found"] == 500
        assert len(result["missing_genes"]) == 2

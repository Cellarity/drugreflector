"""Tests for package imports and API surface."""


def test_top_level_public_api():
    """The full public API is importable from the drugreflector package."""
    from drugreflector import (
        DrugReflector,
        EnsembleModel,
        nnFC,
        SignatureRefinement,
        load_h5ad_file,
        compute_vscores_adata,
        compute_vscore_two_groups,
        compute_vscores,
        pseudobulk_adata,
        create_synthetic_gene_expression,
    )
    for obj in [DrugReflector, EnsembleModel, nnFC, SignatureRefinement,
                load_h5ad_file, compute_vscores_adata, compute_vscore_two_groups,
                compute_vscores, pseudobulk_adata, create_synthetic_gene_expression]:
        assert callable(obj)


def test_version():
    import drugreflector
    assert drugreflector.__version__ == "1.0.0"


def test_subpackage_classes():
    from drugreflector.drug_reflector import DrugReflector
    from drugreflector.ensemble_model import EnsembleModel
    from drugreflector.models import nnFC
    assert callable(DrugReflector)
    assert callable(EnsembleModel)
    assert callable(nnFC)


def test_subpackage_utils():
    from drugreflector.utils import (
        load_h5ad_file,
        compute_vscores_adata,
        compute_vscore_two_groups,
        compute_vscores,
        pseudobulk_adata,
        create_synthetic_gene_expression,
        clip_rescale_rows,
    )
    for fn in [load_h5ad_file, compute_vscores_adata, compute_vscore_two_groups,
               compute_vscores, pseudobulk_adata, create_synthetic_gene_expression,
               clip_rescale_rows]:
        assert callable(fn)


def test_signature_refinement_import():
    from drugreflector import SignatureRefinement
    assert callable(SignatureRefinement)
    assert hasattr(SignatureRefinement, "compute_refined_signatures")
    assert hasattr(SignatureRefinement, "compute_learned_signatures")
    assert hasattr(SignatureRefinement, "load_phenotypic_readouts")

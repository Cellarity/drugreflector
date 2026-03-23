"""Tests for package imports and API surface.

Note: The editable install maps 'drugreflector' to the repo root, but
signature_refinement has a circular import via utils.py. We test the
subpackage imports directly (drugreflector.* resolves to drugreflector/ subdir).
"""


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
    from signature_refinement.signature_refinement import SignatureRefinement
    assert callable(SignatureRefinement)
    assert hasattr(SignatureRefinement, "compute_refined_signatures")
    assert hasattr(SignatureRefinement, "compute_learned_signatures")
    assert hasattr(SignatureRefinement, "load_phenotypic_readouts")

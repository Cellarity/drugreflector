"""Tests for CLI entry points (cli.py)."""

import os
import pytest
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from anndata import AnnData

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(REPO_ROOT, "checkpoints")
CHECKPOINT_PATHS = [
    os.path.join(CHECKPOINT_DIR, "model_fold_0.pt"),
    os.path.join(CHECKPOINT_DIR, "model_fold_1.pt"),
    os.path.join(CHECKPOINT_DIR, "model_fold_2.pt"),
]


# ---------------------------------------------------------------------------
# cli.py argument parsing
# ---------------------------------------------------------------------------
class TestCliParser:
    def test_missing_input_file_exits(self, tmp_path, monkeypatch):
        """CLI exits with error when input file doesn't exist."""
        monkeypatch.setattr(
            sys, "argv",
            ["drugreflector", str(tmp_path / "nonexistent.h5ad")],
        )
        from cli import main
        with pytest.raises(SystemExit):
            main()

    def test_missing_checkpoint_exits(self, tmp_path, monkeypatch):
        """CLI exits with error when checkpoint doesn't exist."""
        # Create a valid h5ad file
        adata = AnnData(X=np.ones((2, 5)))
        input_path = str(tmp_path / "input.h5ad")
        adata.write_h5ad(input_path)
        monkeypatch.setattr(
            sys, "argv",
            ["drugreflector", input_path,
             "--model1", "/nonexistent/model.pt",
             "--model2", "/nonexistent/model.pt",
             "--model3", "/nonexistent/model.pt"],
        )
        from cli import main
        with pytest.raises(SystemExit):
            main()


# ---------------------------------------------------------------------------
# Integration: end-to-end CLI
# ---------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.integration
class TestCliEndToEnd:
    def test_full_pipeline(self, tmp_path, real_gene_names, monkeypatch):
        """Run CLI with synthetic data and real checkpoints."""
        # Create synthetic v-score data with real gene names
        np.random.seed(123)
        X = np.random.normal(0, 1, (1, len(real_gene_names)))
        adata = AnnData(
            X=X,
            obs=pd.DataFrame(index=["test_sample"]),
            var=pd.DataFrame(index=real_gene_names),
        )
        input_path = str(tmp_path / "input.h5ad")
        output_path = str(tmp_path / "output.csv")
        adata.write_h5ad(input_path)

        monkeypatch.setattr(
            sys, "argv",
            ["drugreflector", input_path,
             "--model1", CHECKPOINT_PATHS[0],
             "--model2", CHECKPOINT_PATHS[1],
             "--model3", CHECKPOINT_PATHS[2],
             "--output", output_path,
             "--n-top", "20"],
        )
        from cli import main
        main()

        # Verify output
        assert Path(output_path).exists()
        result = pd.read_csv(output_path, index_col=0, header=[0, 1])
        assert len(result) > 0
        assert len(result) <= 20

import numpy as np
import pandas as pd

from sccont import load_expression_matrix, preprocess


def _toy_counts(n_genes=120, n_cells=80, seed=0):
    rng = np.random.RandomState(seed)
    counts = rng.poisson(lam=2.0, size=(n_genes, n_cells)).astype(float)
    genes = [f"G{i}" for i in range(n_genes - 2)] + ["MT-CO1", "MT-ND1"]
    cells = [f"V{i}_T{i % 3}" for i in range(n_cells)]
    return pd.DataFrame(counts, index=genes, columns=cells)


def test_load_expression_matrix_roundtrip(tmp_path):
    df = _toy_counts()
    path = tmp_path / "expr.tsv"
    df.rename_axis("Gene.names").reset_index().to_csv(path, sep="\t", index=False)
    loaded = load_expression_matrix(path)
    pd.testing.assert_frame_equal(loaded, df, check_names=False)


def test_preprocess_dataframe_input_returns_anndata():
    df = _toy_counts()
    out = preprocess(
        df,
        min_genes=10, max_genes=None, max_pct_mito=None, min_cells=3,
        remove_doublets=False, n_top_genes=50,
        regress_mito=False, regress_cell_cycle=False, verbose=False,
    )
    assert out.shape[1] == 50          # HVG subset -> genes as variables
    assert out.shape[0] <= df.shape[1]  # cells as observations
    assert set(out.obs_names) <= set(df.columns)


def test_preprocess_simulated_mode_returns_genes_by_cells():
    df = _toy_counts()
    out = preprocess(
        df,
        min_genes=10,
        max_genes=None,
        max_pct_mito=None,
        min_cells=3,
        remove_doublets=False,
        n_top_genes=50,
        regress_mito=False,
        regress_cell_cycle=False,
        verbose=False,
        as_dataframe=True,
    )
    assert isinstance(out, pd.DataFrame)
    assert out.shape[0] == 50            # HVG subset -> genes on rows
    assert out.shape[1] <= df.shape[1]   # cells on columns
    assert set(out.columns) <= set(df.columns)
    assert np.isfinite(out.to_numpy()).all()
    assert out.to_numpy().max() <= 10 + 1e-6


def test_preprocess_all_steps_off_is_identity():
    df = _toy_counts()
    out = preprocess(
        df,
        min_genes=0,
        max_genes=None,
        max_pct_mito=None,
        min_cells=0,
        remove_doublets=False,
        normalize_total=False,
        log1p=False,
        n_top_genes=None,
        regress_mito=False,
        regress_counts=False,
        regress_cell_cycle=False,
        scale=False,
        verbose=False,
        as_dataframe=True,
    )
    assert out.shape == df.shape
    np.testing.assert_allclose(out.to_numpy(), df.to_numpy())


def test_preprocess_return_anndata():
    df = _toy_counts()
    adata = preprocess(
        df,
        min_genes=10,
        max_genes=None,
        max_pct_mito=None,
        remove_doublets=False,
        n_top_genes=None,
        regress_mito=True,
        regress_cell_cycle=False,
        verbose=False,
    )
    assert adata.n_vars == df.shape[0]
    assert {"n_genes", "n_counts", "percent_mito"} <= set(adata.obs.columns)

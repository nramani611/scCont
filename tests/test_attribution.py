import numpy as np
import pandas as pd
import pytest

from sccont import (
    Encoder,
    cluster_gene_sets,
    collect_enriched_genes,
    compute_shap_values,
    select_top_genes_by_zscore,
    top_genes_per_latent,
)

pytest.importorskip("shap")

LATENT_DIM = 4


@pytest.fixture(scope="module")
def shap_values(toy_dataset):
    encoder = Encoder(input_dim=toy_dataset.n_genes, latent_dim=LATENT_DIM).eval()
    X = toy_dataset.data.numpy()[:40]
    return compute_shap_values(encoder, X, n_background=10, seed=0, device="cpu")


def test_shap_shape(shap_values, toy_dataset):
    assert shap_values.shape == (40, toy_dataset.n_genes, LATENT_DIM)
    assert np.isfinite(shap_values).all()


def test_shap_rejects_wrong_orientation(toy_dataset):
    encoder = Encoder(input_dim=toy_dataset.n_genes, latent_dim=LATENT_DIM).eval()
    with pytest.raises(ValueError):
        compute_shap_values(encoder, toy_dataset.data.numpy().T, device="cpu")


def test_select_top_genes_by_zscore():
    rng = np.random.RandomState(0)
    sv = rng.normal(scale=0.01, size=(30, 20, 2))
    sv[:, 3, 0] += 5.0  # gene 3 dominates latent 0
    genes = [f"G{i}" for i in range(20)]
    df = select_top_genes_by_zscore(sv, genes, 0, zscore_threshold=2.57)
    assert list(df.columns) == ["Gene", "Mean_SHAP_Value", "Z_Score"]
    assert df["Gene"].tolist() == ["G3"]


def test_top_genes_per_latent(shap_values, toy_matrix):
    genes = list(toy_matrix.index)
    top = top_genes_per_latent(shap_values, genes, n=5)
    assert set(top) == set(range(LATENT_DIM))
    assert all(len(v) == 5 for v in top.values())


def test_cluster_gene_sets():
    rng = np.random.RandomState(0)
    sv = rng.normal(scale=0.01, size=(30, 20, 3))
    sv[:, 3, 0] += 5.0
    sv[:, 7, 1] += 5.0
    sv[:, 11, 2] += 5.0
    genes = [f"G{i}" for i in range(20)]
    sets = cluster_gene_sets(sv, genes, np.array([0, 0, 1]))
    assert sets == [["G3", "G7"], ["G11"]]


def test_collect_enriched_genes():
    df = pd.DataFrame({"intersections": [["A", "B"], None, ["B", "C"]]})
    assert collect_enriched_genes([df, pd.DataFrame({"x": [1]})]) == {"A", "B", "C"}

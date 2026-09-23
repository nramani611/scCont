"""End-to-end test of the AnnData-centric API with arbitrary obs labels."""

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import pytest

import sccont

matplotlib.use("Agg")
pytest.importorskip("shap")

LATENT_DIM = 8


@pytest.fixture(scope="module")
def adata(toy_matrix):
    obs = pd.DataFrame(
        {
            "timepoint": [c.split("_")[1] for c in toy_matrix.columns],
            "condition": np.where(np.arange(toy_matrix.shape[1]) % 2 == 0, "ctrl", "treated"),
            "score": np.linspace(0, 1, toy_matrix.shape[1]),
        },
        index=toy_matrix.columns,
    )
    return sccont.to_anndata(toy_matrix, obs=obs)


def test_to_anndata_orientation_and_labels(adata, toy_matrix):
    assert adata.shape == (toy_matrix.shape[1], toy_matrix.shape[0])  # cells x genes
    assert list(adata.var_names) == list(toy_matrix.index)
    assert set(adata.obs.columns) == {"timepoint", "condition", "score"}
    np.testing.assert_allclose(adata.X[0], toy_matrix.iloc[:, 0].to_numpy())


def test_to_anndata_kwargs_and_alignment(toy_matrix):
    a = sccont.to_anndata(toy_matrix, batch=["b"] * toy_matrix.shape[1])
    assert (a.obs["batch"] == "b").all()
    shuffled = pd.DataFrame({"x": range(toy_matrix.shape[1])}, index=toy_matrix.columns).iloc[::-1]
    b = sccont.to_anndata(toy_matrix, obs=shuffled)
    assert b.obs.loc[toy_matrix.columns[0], "x"] == 0  # aligned by name, not position
    with pytest.raises(ValueError):
        sccont.to_anndata(toy_matrix, obs=shuffled.iloc[:5])
    with pytest.raises(ValueError):
        sccont.to_anndata(toy_matrix, batch=["b"] * 3)


def test_preprocess_keeps_obs_labels(adata):
    out = sccont.preprocess(
        adata,
        min_genes=0, max_genes=None, max_pct_mito=None, min_cells=0,
        remove_doublets=False, normalize_total=False, log1p=False, n_top_genes=None,
        regress_mito=False, regress_counts=False, regress_cell_cycle=False,
        verbose=False,
    )
    assert isinstance(out, ad.AnnData)
    assert out is not adata  # input untouched
    for col in ("timepoint", "condition", "score"):
        assert (out.obs[col].to_numpy() == adata.obs[col].to_numpy()).all()
    assert {"n_genes", "n_counts", "percent_mito"} <= set(out.obs.columns)


@pytest.fixture(scope="module")
def trained(adata):
    pairs = sccont.get_knn_pairs(adata, k=3, n_pcs=20, random_state=0)
    encoder, _, losses = sccont.train_contrastive(
        adata, pairs, latent_dim=LATENT_DIM, proj_dim=4, epochs=40, batch_size=64,
        device="cpu", seed=42, verbose=False,
    )
    return encoder, pairs, losses


def test_pairs_from_anndata_match_array(adata, trained):
    _, pairs, _ = trained
    assert pairs == sccont.get_knn_pairs(adata.X, k=3, n_pcs=20, random_state=0)
    assert len(pairs) == adata.n_obs * 3


def test_embed_writes_obsm(adata, trained):
    encoder, _, _ = trained
    out = sccont.embed(encoder, adata, device="cpu")
    assert out is adata  # in place
    assert adata.obsm["X_sccont"].shape == (adata.n_obs, LATENT_DIM)
    assert adata.obsm["X_sccont_pca"].shape[0] == adata.n_obs
    assert adata.uns["sccont"]["latent_key"] == "X_sccont"
    assert adata.uns["sccont"]["latent_dim"] == LATENT_DIM
    # labels untouched
    assert set(adata.obs.columns) == {"timepoint", "condition", "score"}


def test_embed_copy_and_custom_key(adata, trained):
    encoder, _, _ = trained
    out = sccont.embed(encoder, adata, device="cpu", key_added="X_lat", copy=True)
    assert out is not adata
    assert "X_lat" in out.obsm and "X_lat" not in adata.obsm


def test_latent_anndata_and_spatial_grouping(adata, trained):
    encoder, _, _ = trained
    sccont.embed(encoder, adata, device="cpu")

    lat = sccont.latent_anndata(adata)
    assert lat.shape == (adata.n_obs, LATENT_DIM)
    assert list(lat.var_names) == [str(i) for i in range(LATENT_DIM)]
    assert list(lat.obs["condition"]) == list(adata.obs["condition"])
    np.testing.assert_allclose(lat.obsm["X_pca"], adata.obsm["X_sccont_pca"])

    g_full, d_full = sccont.group_spatially_similar_latents(adata, n_groups=2, bins=10)
    g_lat, d_lat = sccont.group_spatially_similar_latents(lat, n_groups=2, bins=10)
    np.testing.assert_allclose(d_full, d_lat)
    assert (g_full == g_lat).all()
    assert (adata.uns["sccont"]["latent_groups"] == g_full).all()

    ks, inertias = sccont.elbow_plot_for_clusters(adata, max_k=3, bins=10, show=False)
    assert ks == [1, 2, 3] and len(inertias) == 3


def test_shap_from_anndata_stores_varm(adata, trained):
    encoder, _, _ = trained
    sub = adata[:30].copy()
    sv = sccont.compute_shap_values(encoder, sub, n_background=5, device="cpu")
    assert sv.shape == (30, adata.n_vars, LATENT_DIM)
    assert sub.varm[sccont.SHAP_VARM_KEY].shape == (adata.n_vars, LATENT_DIM)
    np.testing.assert_allclose(sub.varm[sccont.SHAP_VARM_KEY], np.abs(sv).mean(axis=0))


def test_legacy_dataset_still_works(toy_matrix, trained):
    encoder, _, _ = trained
    ds = sccont.TemporalSingleCellDataset(toy_matrix.to_numpy(), [c.split("_")[1] for c in toy_matrix.columns])
    lat = sccont.embed(encoder, ds, device="cpu")
    assert lat.shape == (ds.n_cells, LATENT_DIM)
    assert "timepoint" in lat.obs

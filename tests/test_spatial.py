import anndata as ad
import matplotlib
import numpy as np
import pytest
import scanpy as sc

from sccont import (
    elbow_plot_for_clusters,
    group_spatially_similar_latents,
    latents_in_group,
    spatial_map,
)

matplotlib.use("Agg")

LATENT_DIM = 12


@pytest.fixture(scope="module")
def adata_latent():
    rng = np.random.RandomState(1)
    coords = rng.normal(size=(300, 2))
    # Three families of latent features, each a different spatial pattern + noise.
    patterns = [coords[:, 0], coords[:, 1], coords[:, 0] * coords[:, 1]]
    X = np.stack([patterns[i % 3] + 0.1 * rng.normal(size=300) for i in range(LATENT_DIM)], axis=1)
    adata = ad.AnnData(X.astype(np.float32))
    adata.var_names = [str(i) for i in range(LATENT_DIM)]
    sc.tl.pca(adata, svd_solver="arpack")
    return adata


def test_spatial_map_shape(adata_latent):
    maps = spatial_map(adata_latent.X, adata_latent.obsm["X_pca"][:, :2], bins=10)
    assert maps.shape == (LATENT_DIM, 100)


def test_grouping_labels_and_distance(adata_latent):
    labels, dist = group_spatially_similar_latents(adata_latent, n_groups=3, bins=20)
    assert labels.shape == (LATENT_DIM,)
    assert set(labels) == {0, 1, 2}
    assert dist.shape == (LATENT_DIM, LATENT_DIM)
    np.testing.assert_allclose(np.diag(dist), 0, atol=1e-8)
    # Features built from the same pattern should land in the same group.
    for offset in range(3):
        assert len({labels[i] for i in range(offset, LATENT_DIM, 3)}) == 1


def test_grouping_requires_pca():
    adata = ad.AnnData(np.zeros((10, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        group_spatially_similar_latents(adata, n_groups=2)


def test_elbow_returns_monotone_inertias(adata_latent):
    ks, inertias = elbow_plot_for_clusters(adata_latent, max_k=5, bins=20, show=False)
    assert ks == [1, 2, 3, 4, 5]
    assert len(inertias) == 5
    assert all(inertias[i] >= inertias[i + 1] - 1e-9 for i in range(4))


def test_latents_in_group():
    assert latents_in_group(np.array([0, 1, 0, 2]), 0) == ["0", "2"]

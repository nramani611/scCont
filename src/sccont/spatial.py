"""Stage C: signal identification via spatial clustering of latent features."""

from __future__ import annotations

import anndata as ad
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering


def spatial_map(features: np.ndarray, coords: np.ndarray, bins: int = 40) -> np.ndarray:
    """Bin each latent feature onto a 2-D grid over ``coords``.

    Parameters
    ----------
    features
        ``(n_cells, n_features)`` latent activations.
    coords
        ``(n_cells, 2)`` embedding coordinates (typically the first two PCs).
    bins
        Grid resolution per axis.

    Returns
    -------
    ndarray of shape ``(n_features, bins * bins)``
        Summed activation of each feature per grid cell.
    """
    maps = []
    for i in range(features.shape[1]):
        heatmap, _, _ = np.histogram2d(coords[:, 0], coords[:, 1], bins=bins, weights=features[:, i])
        maps.append(heatmap.flatten())
    return np.array(maps)


def _latent_distance_matrix(adata_latent: ad.AnnData, bins: int) -> np.ndarray:
    if "X_pca" not in adata_latent.obsm:
        raise ValueError("adata_latent.obsm['X_pca'] missing; run scanpy.tl.pca first "
                         "(sccont.embed does this by default)")
    X = np.asarray(adata_latent.X)
    coords = np.asarray(adata_latent.obsm["X_pca"][:, :2])
    maps = spatial_map(X, coords, bins=bins)
    return squareform(pdist(maps, metric="correlation"))


def group_spatially_similar_latents(
    adata_latent: ad.AnnData,
    n_groups: int = 3,
    bins: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster latent features by the correlation of their spatial activation maps.

    Parameters
    ----------
    adata_latent
        Output of :func:`~sccont.training.embed` (needs ``obsm['X_pca']``).
    n_groups
        Number of clusters (choose with :func:`elbow_plot_for_clusters`).
    bins
        Grid resolution passed to :func:`spatial_map`.

    Returns
    -------
    labels, dist_matrix
        Cluster label per latent feature (length ``latent_dim``) and the
        ``(latent_dim, latent_dim)`` correlation-distance matrix.
    """
    dist_matrix = _latent_distance_matrix(adata_latent, bins)
    clustering = AgglomerativeClustering(n_clusters=n_groups, metric="precomputed", linkage="average")
    labels = clustering.fit_predict(dist_matrix)
    return labels, dist_matrix


def elbow_plot_for_clusters(
    adata_latent: ad.AnnData,
    max_k: int = 10,
    bins: int = 40,
    ax=None,
    show: bool = True,
) -> tuple[list[int], list[float]]:
    """Within-cluster sum of distances for ``k = 1..max_k`` spatial clusters.

    Returns ``(k_values, inertias)`` and draws an elbow plot unless ``show=False``
    and ``ax`` is None.
    """
    dist_matrix = _latent_distance_matrix(adata_latent, bins)

    inertias: list[float] = []
    k_range = list(range(1, max_k + 1))
    for n_clusters in k_range:
        if n_clusters == 1:
            labels = np.zeros(dist_matrix.shape[0], dtype=int)
        else:
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters, metric="precomputed", linkage="average"
            )
            labels = clustering.fit_predict(dist_matrix)

        inertia = 0.0
        for label in np.unique(labels):
            idx = np.where(labels == label)[0]
            if len(idx) > 1:
                inertia += np.sum(dist_matrix[np.ix_(idx, idx)]) / 2
        inertias.append(inertia)

    if ax is not None or show:
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 4))
        ax.plot(k_range, inertias, marker="o")
        ax.set_xlabel("Number of clusters")
        ax.set_ylabel("Within-cluster sum of distances")
        ax.set_title("Elbow Method for Optimal Clusters")
        if show:
            plt.show()

    return k_range, inertias


def latents_in_group(labels: np.ndarray, group: int) -> list[str]:
    """String var_names of the latent features assigned to ``group``."""
    return [str(i) for i in range(len(labels)) if labels[i] == group]

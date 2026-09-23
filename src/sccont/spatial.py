"""Stage C: signal identification via spatial clustering of latent features."""

from __future__ import annotations

import anndata as ad
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

from .training import LATENT_KEY


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


def _latents_and_coords(adata: ad.AnnData, use_rep: str | None) -> tuple[np.ndarray, np.ndarray]:
    """Return (latents, 2-D coords) from either the full AnnData or a latent AnnData.

    * Full ``AnnData`` after :func:`sccont.embed`: latents in ``obsm[use_rep]``
      (default ``uns['sccont']['latent_key']`` or ``'X_sccont'``) and coordinates
      in ``obsm[f"{use_rep}_pca"]`` (computed on the fly if missing).
    * Latent ``AnnData`` (from :func:`sccont.latent_anndata` or the legacy path):
      latents in ``X`` and coordinates in ``obsm['X_pca']``.
    """
    if use_rep is None:
        use_rep = adata.uns.get("sccont", {}).get("latent_key", LATENT_KEY)
        if use_rep not in adata.obsm:
            use_rep = None

    if use_rep is not None:
        X = np.asarray(adata.obsm[use_rep])
        pca_key = f"{use_rep}_pca"
        if pca_key in adata.obsm:
            coords = np.asarray(adata.obsm[pca_key])[:, :2]
        else:
            coords = PCA(n_components=2).fit_transform(X)
        return X, coords

    if "X_pca" not in adata.obsm:
        raise ValueError(
            "No latent representation found: pass an AnnData processed by sccont.embed "
            "(obsm['X_sccont']) or a latent AnnData with obsm['X_pca']"
        )
    return np.asarray(adata.X), np.asarray(adata.obsm["X_pca"])[:, :2]


def _latent_distance_matrix(adata: ad.AnnData, bins: int, use_rep: str | None = None) -> np.ndarray:
    X, coords = _latents_and_coords(adata, use_rep)
    maps = spatial_map(X, coords, bins=bins)
    return squareform(pdist(maps, metric="correlation"))


def group_spatially_similar_latents(
    adata: ad.AnnData,
    n_groups: int = 3,
    bins: int = 40,
    use_rep: str | None = None,
    key_added: str = "latent_groups",
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster latent features by the correlation of their spatial activation maps.

    Parameters
    ----------
    adata
        Either the full ``AnnData`` after :func:`sccont.embed`, or a latent
        ``AnnData`` (:func:`sccont.latent_anndata` / legacy ``embed`` output).
    n_groups
        Number of clusters (choose with :func:`elbow_plot_for_clusters`).
    bins
        Grid resolution passed to :func:`spatial_map`.
    use_rep
        ``obsm`` key of the latents in the full-AnnData case (auto-detected).
    key_added
        The labels are also stored in ``adata.uns['sccont'][key_added]``.

    Returns
    -------
    labels, dist_matrix
        Cluster label per latent feature (length ``latent_dim``) and the
        ``(latent_dim, latent_dim)`` correlation-distance matrix.
    """
    dist_matrix = _latent_distance_matrix(adata, bins, use_rep)
    clustering = AgglomerativeClustering(n_clusters=n_groups, metric="precomputed", linkage="average")
    labels = clustering.fit_predict(dist_matrix)
    adata.uns["sccont"] = {**adata.uns.get("sccont", {}), key_added: labels}
    return labels, dist_matrix


def elbow_plot_for_clusters(
    adata: ad.AnnData,
    max_k: int = 10,
    bins: int = 40,
    ax=None,
    show: bool = True,
    use_rep: str | None = None,
) -> tuple[list[int], list[float]]:
    """Within-cluster sum of distances for ``k = 1..max_k`` spatial clusters.

    Returns ``(k_values, inertias)`` and draws an elbow plot unless ``show=False``
    and ``ax`` is None.
    """
    dist_matrix = _latent_distance_matrix(adata, bins, use_rep)

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

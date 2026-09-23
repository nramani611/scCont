"""Stage A: k-nearest-neighbour positive-pair selection."""

from __future__ import annotations

import anndata as ad
import numpy as np
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA

from .data import SingleCellDataset, _dense


def _cells_by_genes(data, layer: str | None, use_rep: str | None) -> np.ndarray:
    if isinstance(data, ad.AnnData):
        if use_rep is not None:
            return np.asarray(data.obsm[use_rep])
        return _dense(data.layers[layer] if layer is not None else data.X)
    if isinstance(data, SingleCellDataset):
        return data.data.numpy()
    if hasattr(data, "numpy"):
        return data.numpy()
    return _dense(data)


def get_knn_pairs(
    data,
    k: int = 3,
    max_distance: float | None = None,
    n_pcs: int | None = 50,
    random_state: int | None = None,
    layer: str | None = None,
    use_rep: str | None = None,
) -> list[tuple[int, int]]:
    """Return kNN positive pairs in PCA space. Labels are ignored (unsupervised).

    Parameters
    ----------
    data
        An ``AnnData`` (cells x genes), a :class:`~sccont.data.SingleCellDataset`,
        or an array of shape ``(n_cells, n_genes)``.
    k
        Number of nearest neighbours per cell (the paper uses ``k=3``).
    max_distance
        Optional maximum Euclidean distance (in PCA space) for a pair to be kept.
    n_pcs
        Number of principal components used to denoise before the kNN search
        (capped at ``min(n_cells, n_genes)``). ``None`` searches in the raw space.
    random_state
        Passed to :class:`sklearn.decomposition.PCA`.
    layer
        ``adata.layers`` key to use instead of ``adata.X``.
    use_rep
        ``adata.obsm`` key holding a precomputed representation (e.g. ``'X_pca'``).
        When given, no further PCA is applied.

    Returns
    -------
    list of (i, j)
        ``j`` is one of the ``k`` nearest neighbours of ``i``; self-pairs are excluded.
    """
    X = _cells_by_genes(data, layer, use_rep)
    n_cells, n_features = X.shape

    if use_rep is None and n_pcs is not None:
        n_components = min(n_pcs, n_cells, n_features)
        X = PCA(n_components=n_components, random_state=random_state).fit_transform(X)

    tree = cKDTree(X)
    distances, indices = tree.query(X, k=k + 1)  # k+1 includes self

    pairs: list[tuple[int, int]] = []
    for i in range(n_cells):
        for j, dist in zip(indices[i][1:], distances[i][1:]):
            if max_distance is None or dist <= max_distance:
                pairs.append((i, int(j)))
    return pairs


# Backwards-compatible alias matching the notebook's original function name.
get_knn_pairs_agnostic = get_knn_pairs

"""Stage A: k-nearest-neighbour positive-pair selection."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA

from .data import TemporalSingleCellDataset


def _as_cells_by_genes(dataset_or_array) -> np.ndarray:
    if isinstance(dataset_or_array, TemporalSingleCellDataset):
        return dataset_or_array.data.numpy()
    if hasattr(dataset_or_array, "numpy"):
        return dataset_or_array.numpy()
    return np.asarray(dataset_or_array)


def get_knn_pairs(
    dataset,
    k: int = 3,
    max_distance: float | None = None,
    n_pcs: int = 50,
    random_state: int | None = None,
) -> list[tuple[int, int]]:
    """Return kNN positive pairs in PCA space, ignoring timepoints.

    Parameters
    ----------
    dataset
        A :class:`~sccont.data.TemporalSingleCellDataset`, or any array of
        shape ``(n_cells, n_genes)``.
    k
        Number of nearest neighbours per cell (the paper uses ``k=3``).
    max_distance
        Optional maximum Euclidean distance (in PCA space) for a pair to be kept.
    n_pcs
        Number of principal components used to denoise before the kNN search.
        Capped at ``min(n_cells, n_genes)``.
    random_state
        Passed to :class:`sklearn.decomposition.PCA`.

    Returns
    -------
    list of (i, j)
        ``j`` is one of the ``k`` nearest neighbours of ``i``; self-pairs are excluded.
    """
    data_np = _as_cells_by_genes(dataset)
    n_cells, n_genes = data_np.shape
    n_components = min(n_pcs, n_cells, n_genes)

    pca = PCA(n_components=n_components, random_state=random_state)
    data_reduced = pca.fit_transform(data_np)

    tree = cKDTree(data_reduced)
    distances, indices = tree.query(data_reduced, k=k + 1)  # k+1 includes self

    pairs: list[tuple[int, int]] = []
    for i in range(n_cells):
        for j, dist in zip(indices[i][1:], distances[i][1:]):
            if max_distance is None or dist <= max_distance:
                pairs.append((i, int(j)))
    return pairs


# Backwards-compatible alias matching the notebook's original function name.
get_knn_pairs_agnostic = get_knn_pairs

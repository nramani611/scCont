import numpy as np
import pytest

from sccont import TemporalSingleCellDataset, get_knn_pairs, timepoints_from_columns


def test_dataset_shapes(toy_dataset, toy_matrix):
    assert toy_dataset.n_cells == toy_matrix.shape[1]
    assert toy_dataset.n_genes == toy_matrix.shape[0]
    x, t = toy_dataset[0]
    assert x.shape == (toy_matrix.shape[0],)
    assert t == "T0"
    assert toy_dataset.unique_timepoints == ["T0", "T1", "T2", "T3"]


def test_dataset_rejects_mismatched_timepoints():
    with pytest.raises(ValueError):
        TemporalSingleCellDataset(np.zeros((5, 4)), ["a", "b"])


def test_timepoints_from_columns():
    assert timepoints_from_columns(["V1_T0", "V2_T3"]) == ["T0", "T3"]


def test_knn_pairs_count_and_no_self(toy_dataset):
    k = 3
    pairs = get_knn_pairs(toy_dataset, k=k, n_pcs=20, random_state=0)
    assert len(pairs) == toy_dataset.n_cells * k
    assert all(i != j for i, j in pairs)
    assert all(0 <= j < toy_dataset.n_cells for _, j in pairs)


def test_knn_pairs_accepts_array(toy_dataset):
    arr = toy_dataset.data.numpy()
    pairs_a = get_knn_pairs(arr, k=2, n_pcs=10, random_state=0)
    pairs_b = get_knn_pairs(toy_dataset, k=2, n_pcs=10, random_state=0)
    assert pairs_a == pairs_b


def test_knn_pairs_max_distance(toy_dataset):
    pairs = get_knn_pairs(toy_dataset, k=3, max_distance=1e-6)
    assert pairs == []

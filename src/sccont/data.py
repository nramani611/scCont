"""Dataset container for time-course single-cell expression matrices."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


class TemporalSingleCellDataset(Dataset):
    """Hold a (genes x cells) expression matrix and per-cell timepoint labels.

    Parameters
    ----------
    data
        Gene expression matrix of shape ``(n_genes, n_cells)``.
    timepoints
        Sequence of ``n_cells`` timepoint labels (e.g. ``['T0', 'T0', 'T1', ...]``).

    Attributes
    ----------
    data : torch.Tensor
        Float32 tensor of shape ``(n_cells, n_genes)``.
    timepoints : np.ndarray
        Array of timepoint labels, one per cell.
    unique_timepoints : list
        Sorted unique timepoint labels.
    """

    def __init__(self, data, timepoints: Sequence):
        data = np.asarray(data)
        if data.ndim != 2:
            raise ValueError(f"data must be 2-D (genes x cells), got shape {data.shape}")
        if data.shape[1] != len(timepoints):
            raise ValueError(
                f"data has {data.shape[1]} cells but {len(timepoints)} timepoints were given"
            )
        self.data = torch.tensor(data.T, dtype=torch.float32)  # (n_cells, n_genes)
        self.timepoints = np.array(timepoints)
        self.unique_timepoints = sorted(np.unique(self.timepoints))

    @property
    def n_cells(self) -> int:
        return self.data.shape[0]

    @property
    def n_genes(self) -> int:
        return self.data.shape[1]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.timepoints[idx]

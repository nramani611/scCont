"""Dataset containers. The primary input to scCont is an ``AnnData`` (cells x genes)."""

from __future__ import annotations

from collections.abc import Sequence

import anndata as ad
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _dense(X) -> np.ndarray:
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)


class SingleCellDataset(Dataset):
    """Torch view of an ``AnnData`` used by the training loop.

    Any per-cell annotation lives in ``adata.obs`` and is carried through the
    pipeline untouched; scCont is unsupervised and never reads it for training.

    Parameters
    ----------
    adata
        ``AnnData`` with cells as observations and genes as variables.
    layer
        Optional ``adata.layers`` key to train on instead of ``adata.X``.

    Attributes
    ----------
    data : torch.Tensor
        Float32 tensor of shape ``(n_cells, n_genes)``.
    obs : pandas.DataFrame
        ``adata.obs`` (same object, not a copy).
    var_names : list of str
        Gene names.
    """

    def __init__(self, adata: ad.AnnData, layer: str | None = None):
        if not isinstance(adata, ad.AnnData):
            raise TypeError("SingleCellDataset expects an AnnData; use from_array() for raw matrices")
        X = adata.layers[layer] if layer is not None else adata.X
        self.adata = adata
        self.data = torch.tensor(_dense(X), dtype=torch.float32)
        self.obs = adata.obs
        self.var_names = list(adata.var_names)

    @classmethod
    def from_array(
        cls,
        X,
        obs: pd.DataFrame | None = None,
        var_names: Sequence[str] | None = None,
        cells_by_genes: bool = True,
    ) -> "SingleCellDataset":
        """Build a dataset from a plain matrix.

        Parameters
        ----------
        X
            Expression matrix. ``(n_cells, n_genes)`` by default; pass
            ``cells_by_genes=False`` for a genes x cells matrix.
        obs
            Optional per-cell annotations (index = cell names).
        var_names
            Optional gene names.
        """
        X = _dense(X)
        if not cells_by_genes:
            X = X.T
        adata = ad.AnnData(np.asarray(X, dtype=np.float32))
        if obs is not None:
            if len(obs) != adata.n_obs:
                raise ValueError(f"obs has {len(obs)} rows but X has {adata.n_obs} cells")
            adata.obs = obs.copy()
            adata.obs_names = obs.index.astype(str)
        if var_names is not None:
            adata.var_names = [str(v) for v in var_names]
        return cls(adata)

    @property
    def n_cells(self) -> int:
        return self.data.shape[0]

    @property
    def n_genes(self) -> int:
        return self.data.shape[1]

    def __len__(self) -> int:
        return self.n_cells

    def __getitem__(self, idx):
        return self.data[idx], idx


class TemporalSingleCellDataset(SingleCellDataset):
    """Legacy container: a genes x cells matrix plus one label per cell.

    Kept for backwards compatibility with the original notebook. New code should
    build an ``AnnData`` with labels in ``obs`` and pass it directly to the
    pipeline functions.

    Parameters
    ----------
    data
        Gene expression matrix of shape ``(n_genes, n_cells)``.
    timepoints
        Sequence of ``n_cells`` labels; stored in ``obs['timepoint']``.
    """

    def __init__(self, data, timepoints: Sequence):
        data = np.asarray(data)
        if data.ndim != 2:
            raise ValueError(f"data must be 2-D (genes x cells), got shape {data.shape}")
        if data.shape[1] != len(timepoints):
            raise ValueError(
                f"data has {data.shape[1]} cells but {len(timepoints)} timepoints were given"
            )
        adata = ad.AnnData(np.asarray(data.T, dtype=np.float32))
        adata.obs["timepoint"] = np.asarray(timepoints)
        super().__init__(adata)
        self.timepoints = np.asarray(timepoints)
        self.unique_timepoints = sorted(np.unique(self.timepoints))

    def __getitem__(self, idx):
        return self.data[idx], self.timepoints[idx]


def as_dataset(data, layer: str | None = None) -> SingleCellDataset:
    """Coerce an ``AnnData``, a :class:`SingleCellDataset` or a cells x genes array to a dataset."""
    if isinstance(data, SingleCellDataset):
        return data
    if isinstance(data, ad.AnnData):
        return SingleCellDataset(data, layer=layer)
    return SingleCellDataset.from_array(data)

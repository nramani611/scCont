"""Stage D: network attribution analysis with SHAP."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
from scipy import stats

from .models import Encoder
from .utils import get_device


class _EncoderWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model.forward(x)


def _as_cells_by_genes(X) -> np.ndarray:
    if hasattr(X, "toarray"):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


def compute_shap_values(
    encoder: Encoder,
    X: np.ndarray,
    n_background: int = 100,
    seed: int | None = 42,
    device: str | torch.device | None = None,
) -> np.ndarray:
    """Gene-to-latent SHAP attributions using ``shap.DeepExplainer``.

    Parameters
    ----------
    encoder
        Trained encoder.
    X
        Expression matrix of shape ``(n_cells, n_genes)`` — i.e. the transpose of
        the genes x cells matrix used for training.
    n_background
        Number of randomly chosen cells used as the DeepExplainer background.
    seed
        Seed for the background sample (paper: 42).
    device
        Device to run on; defaults to the encoder's device.

    Returns
    -------
    ndarray of shape ``(n_cells, n_genes, latent_dim)``
    """
    import shap

    X = _as_cells_by_genes(X)
    if X.shape[1] != encoder.input_dim:
        raise ValueError(
            f"X has {X.shape[1]} genes but the encoder expects {encoder.input_dim}; "
            "pass the matrix as cells x genes"
        )
    if device is None:
        device = next(encoder.parameters()).device
    device = get_device(device)

    rng = np.random.RandomState(seed)
    background_idx = rng.choice(X.shape[0], min(n_background, X.shape[0]), replace=False)
    background = torch.tensor(X[background_idx], dtype=torch.float32).to(device)

    wrapper = _EncoderWrapper(encoder).to(device)
    wrapper.eval()

    explainer = shap.DeepExplainer(wrapper, background)
    test_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    shap_values = explainer.shap_values(test_tensor)

    # Older shap versions return a list with one (cells, genes) array per output.
    if isinstance(shap_values, list):
        shap_values = np.stack(shap_values, axis=-1)
    return np.asarray(shap_values)


def select_top_genes_by_zscore(
    shap_values: np.ndarray,
    gene_names: Sequence[str],
    latent_feature_index: int,
    zscore_threshold: float = 2.57,
) -> pd.DataFrame:
    """Genes whose mean |SHAP| for one latent feature is an outlier by z-score.

    ``zscore_threshold=2.57`` corresponds to p < 0.01 (one-sided).

    Returns a DataFrame with columns ``Gene``, ``Mean_SHAP_Value``, ``Z_Score``
    sorted by mean |SHAP| descending.
    """
    feature_shap = np.abs(shap_values[:, :, latent_feature_index]).mean(axis=0)
    z_scores = stats.zscore(feature_shap)
    top_indices = np.where(z_scores >= zscore_threshold)[0]

    return (
        pd.DataFrame(
            {
                "Gene": [gene_names[i] for i in top_indices],
                "Mean_SHAP_Value": feature_shap[top_indices],
                "Z_Score": z_scores[top_indices],
            }
        )
        .sort_values("Mean_SHAP_Value", ascending=False)
        .reset_index(drop=True)
    )


def top_genes_per_latent(
    shap_values: np.ndarray,
    gene_names: Sequence[str],
    n: int = 50,
) -> dict[int, list[str]]:
    """Top-``n`` genes by mean |SHAP| for every latent feature."""
    mean_abs_shap = np.abs(np.moveaxis(shap_values, 2, 0)).mean(axis=1)  # (latent, genes)
    out: dict[int, list[str]] = {}
    for latent_idx in range(mean_abs_shap.shape[0]):
        top_idx = mean_abs_shap[latent_idx].argsort()[::-1][:n]
        out[latent_idx] = [gene_names[i] for i in top_idx]
    return out

"""Stage B: contrastive training and latent embedding."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
import torch
from tqdm.auto import tqdm

from .data import TemporalSingleCellDataset
from .losses import InfoNCELoss
from .models import Encoder, Projector
from .utils import get_device, set_seeds


def train_contrastive(
    dataset: TemporalSingleCellDataset,
    pairs: Sequence[tuple[int, int]],
    *,
    latent_dim: int = 32,
    proj_dim: int = 16,
    epochs: int = 20000,
    batch_size: int = 256,
    lr: float = 1e-4,
    temperature: float = 0.20,
    device: str | torch.device | None = None,
    seed: int | None = 42,
    verbose: bool = True,
) -> tuple[Encoder, Projector, list[float]]:
    """Train the scCont encoder with InfoNCE on kNN positive pairs.

    Each epoch samples ``batch_size`` pairs without replacement, encodes and
    projects both members of every pair and minimises
    :class:`~sccont.losses.InfoNCELoss`.

    Parameters
    ----------
    dataset
        Dataset holding the ``(n_cells, n_genes)`` expression tensor.
    pairs
        Positive pairs from :func:`~sccont.pairs.get_knn_pairs`.
    latent_dim, proj_dim
        Encoder latent size and projector output size.
    epochs
        Number of training steps (one batch per epoch). The paper uses 20 000;
        expect training time to scale with dataset size.
    batch_size
        Number of pairs per step. Must be ``<= len(pairs)``.
    lr
        Adam learning rate.
    temperature
        InfoNCE temperature.
    device
        ``'cuda'``, ``'cpu'`` or None (auto).
    seed
        If not None, :func:`~sccont.utils.set_seeds` is called first. Results in
        the paper used ``seed=42``.
    verbose
        Show a tqdm progress bar.

    Returns
    -------
    encoder, projector, losses
        Trained networks (in eval mode) and the per-epoch loss history.
    """
    if seed is not None:
        set_seeds(seed)
    if batch_size > len(pairs):
        raise ValueError(
            f"batch_size={batch_size} exceeds the number of pairs ({len(pairs)}); "
            "increase k or lower batch_size"
        )

    device = get_device(device)
    encoder = Encoder(input_dim=dataset.n_genes, latent_dim=latent_dim).to(device)
    projector = Projector(latent_dim=latent_dim, proj_dim=proj_dim).to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(projector.parameters()), lr=lr)
    loss_fn = InfoNCELoss(temperature=temperature)

    encoder.train()
    projector.train()
    losses: list[float] = []
    iterator = tqdm(range(epochs), disable=not verbose, desc="scCont training")
    for _ in iterator:
        batch_indices = np.random.choice(len(pairs), size=batch_size, replace=False)
        batch_pairs = [pairs[i] for i in batch_indices]

        x1 = torch.stack([dataset.data[i] for i, _ in batch_pairs]).to(device)
        x2 = torch.stack([dataset.data[j] for _, j in batch_pairs]).to(device)

        z1 = encoder(x1)
        z2 = encoder(x2)
        p1 = projector(z1)
        p2 = projector(z2)

        loss = loss_fn(p1, p2)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    encoder.eval()
    projector.eval()
    return encoder, projector, losses


@torch.no_grad()
def embed(
    encoder: Encoder,
    data,
    timepoints: Sequence | None = None,
    device: str | torch.device | None = None,
    compute_pca: bool = True,
) -> ad.AnnData:
    """Encode all cells and return an ``AnnData`` of latent features.

    Parameters
    ----------
    encoder
        Trained :class:`~sccont.models.Encoder`.
    data
        A :class:`~sccont.data.TemporalSingleCellDataset` or an array of shape
        ``(n_genes, n_cells)`` (the same orientation as the input matrix).
    timepoints
        Per-cell labels stored in ``adata.obs['timepoint']``. Taken from the
        dataset when ``data`` is a dataset and this is None.
    device
        Device to run the encoder on; defaults to the encoder's own device.
    compute_pca
        Run ``scanpy.tl.pca(svd_solver='arpack')`` on the latent matrix.

    Returns
    -------
    AnnData
        ``n_cells x latent_dim``; ``var_names`` are ``'0', '1', ...`` so
        latent features can be selected by string index.
    """
    if isinstance(data, TemporalSingleCellDataset):
        x = data.data
        if timepoints is None:
            timepoints = data.timepoints
    else:
        x = torch.tensor(np.asarray(data).T, dtype=torch.float32)

    if device is None:
        device = next(encoder.parameters()).device
    device = get_device(device)

    was_training = encoder.training
    encoder.eval()
    latent = encoder(x.to(device)).cpu().numpy()
    if was_training:
        encoder.train()

    adata_latent = ad.AnnData(latent)
    adata_latent.var_names = [str(i) for i in range(latent.shape[1])]
    if timepoints is not None:
        adata_latent.obs["timepoint"] = np.asarray(timepoints)
    if compute_pca:
        sc.tl.pca(adata_latent, svd_solver="arpack")
    return adata_latent


def save_encoder(encoder: Encoder, path: str | Path) -> None:
    """Save the encoder's ``state_dict`` (compatible with the repo's ``encoder.pth`` files)."""
    torch.save(encoder.state_dict(), Path(path))


def load_encoder(
    path: str | Path,
    input_dim: int,
    latent_dim: int = 32,
    device: str | torch.device | None = None,
) -> Encoder:
    """Load an encoder saved with :func:`save_encoder` (or a repo ``encoder.pth``).

    ``input_dim`` must equal the number of genes the encoder was trained on.
    """
    device = get_device(device)
    encoder = Encoder(input_dim=input_dim, latent_dim=latent_dim)
    state = torch.load(Path(path), map_location=device)
    if not isinstance(state, dict) or "net.0.weight" not in state:
        # Whole-module pickles are also accepted.
        if isinstance(state, Encoder):
            return state.to(device).eval()
        raise ValueError(f"{path} does not look like an Encoder state_dict")
    encoder.load_state_dict(state)
    return encoder.to(device).eval()

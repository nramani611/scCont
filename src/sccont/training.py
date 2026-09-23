"""Stage B: contrastive training and latent embedding."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
import torch
from tqdm.auto import tqdm

from .data import SingleCellDataset, as_dataset
from .losses import InfoNCELoss
from .models import Encoder, Projector
from .utils import get_device, set_seeds

LATENT_KEY = "X_sccont"


def train_contrastive(
    data,
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
    layer: str | None = None,
) -> tuple[Encoder, Projector, list[float]]:
    """Train the scCont encoder with InfoNCE on kNN positive pairs.

    Training is unsupervised: nothing in ``adata.obs`` is used.

    Parameters
    ----------
    data
        An ``AnnData`` (cells x genes), a :class:`~sccont.data.SingleCellDataset`,
        or an array of shape ``(n_cells, n_genes)``.
    pairs
        Positive pairs from :func:`~sccont.pairs.get_knn_pairs`.
    latent_dim, proj_dim
        Encoder latent size and projector output size.
    epochs
        Number of training steps (one batch per epoch). The paper uses 20 000.
    batch_size
        Number of pairs per step. Must be ``<= len(pairs)``.
    lr
        Adam learning rate.
    temperature
        InfoNCE temperature.
    device
        ``'cuda'``, ``'cpu'`` or None (auto).
    seed
        If not None, :func:`~sccont.utils.set_seeds` is called first (paper: 42).
    verbose
        Show a tqdm progress bar.
    layer
        ``adata.layers`` key to train on instead of ``adata.X``.

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

    dataset = as_dataset(data, layer=layer)
    device = get_device(device)
    encoder = Encoder(input_dim=dataset.n_genes, latent_dim=latent_dim).to(device)
    projector = Projector(latent_dim=latent_dim, proj_dim=proj_dim).to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(projector.parameters()), lr=lr)
    loss_fn = InfoNCELoss(temperature=temperature)

    encoder.train()
    projector.train()
    losses: list[float] = []
    for _ in tqdm(range(epochs), disable=not verbose, desc="scCont training"):
        batch_indices = np.random.choice(len(pairs), size=batch_size, replace=False)
        batch_pairs = [pairs[i] for i in batch_indices]

        x1 = torch.stack([dataset.data[i] for i, _ in batch_pairs]).to(device)
        x2 = torch.stack([dataset.data[j] for _, j in batch_pairs]).to(device)

        p1 = projector(encoder(x1))
        p2 = projector(encoder(x2))

        loss = loss_fn(p1, p2)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    encoder.eval()
    projector.eval()
    return encoder, projector, losses


@torch.no_grad()
def _encode(encoder: Encoder, x: torch.Tensor, device) -> np.ndarray:
    if device is None:
        device = next(encoder.parameters()).device
    device = get_device(device)
    was_training = encoder.training
    encoder.eval()
    latent = encoder(x.to(device)).cpu().numpy()
    if was_training:
        encoder.train()
    return latent


def _latent_pca(latent: np.ndarray) -> np.ndarray:
    tmp = ad.AnnData(np.asarray(latent, dtype=np.float32))
    sc.tl.pca(tmp, svd_solver="arpack")
    return np.asarray(tmp.obsm["X_pca"])


def embed(
    encoder: Encoder,
    data,
    timepoints: Sequence | None = None,
    device: str | torch.device | None = None,
    compute_pca: bool = True,
    key_added: str = LATENT_KEY,
    layer: str | None = None,
    copy: bool = False,
):
    """Encode all cells into scCont latent features.

    With an ``AnnData`` input the latents are written back onto that object:

    * ``adata.obsm[key_added]`` – latent features, ``(n_cells, latent_dim)``
    * ``adata.obsm[f"{key_added}_pca"]`` – PCA of the latent space (if ``compute_pca``)
    * ``adata.uns["sccont"]`` – ``{"latent_key": key_added, "latent_dim": ...}``

    and the (possibly copied) ``AnnData`` is returned. Use :func:`latent_anndata`
    to obtain a latent-features-as-variables view for stage C.

    With a dataset or a genes x cells array (legacy path) a new latent
    ``AnnData`` is returned instead, with ``var_names`` ``'0', '1', ...`` and
    ``obs['timepoint']`` if labels are available.

    Parameters
    ----------
    encoder
        Trained :class:`~sccont.models.Encoder`.
    data
        ``AnnData`` (cells x genes), :class:`~sccont.data.SingleCellDataset`,
        or a genes x cells array.
    timepoints
        Legacy path only: labels stored in ``obs['timepoint']``.
    device
        Device to run the encoder on; defaults to the encoder's own device.
    compute_pca
        Also compute PCA of the latent space (``svd_solver='arpack'``).
    key_added
        ``obsm`` key for the latents (AnnData path).
    layer
        ``adata.layers`` key to encode instead of ``adata.X``.
    copy
        AnnData path: return a modified copy instead of writing in place.
    """
    if isinstance(data, ad.AnnData):
        adata = data.copy() if copy else data
        dataset = SingleCellDataset(adata, layer=layer)
        latent = _encode(encoder, dataset.data, device)
        adata.obsm[key_added] = latent
        if compute_pca:
            adata.obsm[f"{key_added}_pca"] = _latent_pca(latent)
        adata.uns["sccont"] = {**adata.uns.get("sccont", {}),
                               "latent_key": key_added, "latent_dim": int(latent.shape[1])}
        return adata

    # Legacy path: dataset or genes x cells array -> latent AnnData
    if isinstance(data, SingleCellDataset):
        x = data.data
        obs = data.obs.copy()
    else:
        x = torch.tensor(np.asarray(data).T, dtype=torch.float32)
        obs = None
    latent = _encode(encoder, x, device)

    adata_latent = ad.AnnData(latent)
    adata_latent.var_names = [str(i) for i in range(latent.shape[1])]
    if obs is not None and len(obs) == latent.shape[0]:
        adata_latent.obs = obs
    if timepoints is not None:
        adata_latent.obs["timepoint"] = np.asarray(timepoints)
    if compute_pca:
        sc.tl.pca(adata_latent, svd_solver="arpack")
    return adata_latent


def latent_anndata(adata: ad.AnnData, use_rep: str | None = None, compute_pca: bool = True) -> ad.AnnData:
    """Build an ``AnnData`` whose variables are the scCont latent features.

    ``X`` = ``adata.obsm[use_rep]``, ``obs`` = a copy of ``adata.obs`` (all labels
    preserved), ``var_names`` = ``'0', '1', ...`` and ``obsm['X_pca']`` = the latent
    PCA (taken from ``adata.obsm[f"{use_rep}_pca"]`` when present). This is the
    object stage C (:mod:`sccont.spatial`) and the notebook figures work on.
    """
    if use_rep is None:
        use_rep = adata.uns.get("sccont", {}).get("latent_key", LATENT_KEY)
    if use_rep not in adata.obsm:
        raise KeyError(f"adata.obsm['{use_rep}'] not found; run sccont.embed first")
    latent = np.asarray(adata.obsm[use_rep])
    out = ad.AnnData(latent, obs=adata.obs.copy())
    out.var_names = [str(i) for i in range(latent.shape[1])]
    pca_key = f"{use_rep}_pca"
    if pca_key in adata.obsm:
        out.obsm["X_pca"] = np.asarray(adata.obsm[pca_key])
    elif compute_pca:
        out.obsm["X_pca"] = _latent_pca(latent)
    return out


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
        if isinstance(state, Encoder):
            return state.to(device).eval()
        raise ValueError(f"{path} does not look like an Encoder state_dict")
    encoder.load_state_dict(state)
    return encoder.to(device).eval()

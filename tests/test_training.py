import numpy as np
import pytest
import torch

from sccont import (
    Encoder,
    InfoNCELoss,
    embed,
    get_knn_pairs,
    load_encoder,
    save_encoder,
    train_contrastive,
)

LATENT_DIM = 8


@pytest.fixture(scope="module")
def trained(toy_dataset):
    pairs = get_knn_pairs(toy_dataset, k=3, n_pcs=20, random_state=0)
    encoder, projector, losses = train_contrastive(
        toy_dataset,
        pairs,
        latent_dim=LATENT_DIM,
        proj_dim=4,
        epochs=60,
        batch_size=64,
        device="cpu",
        seed=42,
        verbose=False,
    )
    return encoder, projector, losses


def test_infonce_prefers_matching_pairs():
    torch.manual_seed(0)
    loss_fn = InfoNCELoss(temperature=0.2)
    z = torch.randn(16, 4)
    matched = loss_fn(z, z + 1e-3 * torch.randn_like(z))
    shuffled = loss_fn(z, z[torch.randperm(16)])
    assert matched < shuffled


def test_training_runs_and_loss_decreases(trained):
    encoder, projector, losses = trained
    assert len(losses) == 60
    assert np.mean(losses[-10:]) < np.mean(losses[:10])
    assert not encoder.training and not projector.training


def test_batch_size_larger_than_pairs_raises(toy_dataset):
    with pytest.raises(ValueError):
        train_contrastive(toy_dataset, [(0, 1)], epochs=1, batch_size=2, device="cpu", verbose=False)


def test_embed_shapes_and_pca(trained, toy_dataset):
    encoder, _, _ = trained
    adata = embed(encoder, toy_dataset)
    assert adata.shape == (toy_dataset.n_cells, LATENT_DIM)
    assert list(adata.var_names) == [str(i) for i in range(LATENT_DIM)]
    assert "timepoint" in adata.obs
    assert adata.obsm["X_pca"].shape[0] == toy_dataset.n_cells


def test_embed_accepts_genes_by_cells_array(trained, toy_dataset, toy_matrix):
    encoder, _, _ = trained
    a = embed(encoder, toy_dataset, compute_pca=False)
    b = embed(encoder, toy_matrix.to_numpy(), timepoints=None, compute_pca=False)
    np.testing.assert_allclose(a.X, b.X, atol=1e-5)
    assert "timepoint" not in b.obs


def test_save_and_load_encoder(trained, toy_dataset, tmp_path):
    encoder, _, _ = trained
    path = tmp_path / "encoder.pth"
    save_encoder(encoder, path)
    loaded = load_encoder(path, input_dim=toy_dataset.n_genes, latent_dim=LATENT_DIM, device="cpu")
    assert isinstance(loaded, Encoder)
    with torch.no_grad():
        np.testing.assert_allclose(
            encoder(toy_dataset.data).numpy(), loaded(toy_dataset.data).numpy(), atol=1e-6
        )


def test_seed_reproducibility(toy_dataset):
    pairs = get_knn_pairs(toy_dataset, k=3, n_pcs=20, random_state=0)
    kw = dict(latent_dim=4, proj_dim=4, epochs=5, batch_size=32, device="cpu", seed=7, verbose=False)
    _, _, l1 = train_contrastive(toy_dataset, pairs, **kw)
    _, _, l2 = train_contrastive(toy_dataset, pairs, **kw)
    np.testing.assert_allclose(l1, l2)

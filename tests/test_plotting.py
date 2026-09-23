"""Smoke tests for sccont.pl: every function draws without error and returns axes."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import sccont  # noqa: E402

pytest.importorskip("shap")
LATENT_DIM = 8


@pytest.fixture(scope="module")
def fitted(toy_matrix):
    obs = pd.DataFrame(
        {
            "timepoint": [c.split("_")[1] for c in toy_matrix.columns],
            "condition": np.where(np.arange(toy_matrix.shape[1]) % 2 == 0, "ctrl", "treated"),
            "pseudotime": np.linspace(0, 1, toy_matrix.shape[1]),
        },
        index=toy_matrix.columns,
    )
    adata = sccont.to_anndata(toy_matrix, obs=obs)
    pairs = sccont.get_knn_pairs(adata, k=3, n_pcs=20, random_state=0)
    encoder, _, losses = sccont.train_contrastive(
        adata, pairs, latent_dim=LATENT_DIM, proj_dim=4, epochs=30, batch_size=64,
        device="cpu", seed=42, verbose=False,
    )
    sccont.embed(encoder, adata, device="cpu")
    groups, _ = sccont.group_spatially_similar_latents(adata, n_groups=3, bins=10)
    sv = sccont.compute_shap_values(encoder, adata, n_background=5, device="cpu")
    return adata, encoder, losses, groups, sv


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_palette_helpers():
    assert len(sccont.pl.CATEGORICAL) == 8
    cols = sccont.pl.category_colors(["a", "b"])
    assert cols["a"] == sccont.pl.CATEGORICAL[0] and cols["b"] == sccont.pl.CATEGORICAL[1]
    ordered = sccont.pl.category_colors(["T0", "T1", "T2"], ordered=True)
    assert len(ordered) == 3
    many = sccont.pl.category_colors([str(i) for i in range(12)])
    assert len(many) == 12
    assert sccont.pl.diverging_cmap()(0.5)[:3] == pytest.approx(
        matplotlib.colors.to_rgb("#f0efec"), abs=0.02)


def test_training_loss(fitted):
    ax = sccont.pl.training_loss(fitted[2], smooth=5, show=False)
    assert len(ax.lines) == 2


def test_elbow(fitted):
    ax = sccont.pl.elbow(fitted[0], max_k=4, bins=10, show=False)
    assert ax.get_xlabel()


@pytest.mark.parametrize("color", [None, "timepoint", "condition", "pseudotime", 3, "3"])
def test_latent_embedding_colors(fitted, color):
    ax = sccont.pl.latent_embedding(fitted[0], color=color, show=False)
    assert len(ax.collections) >= 1


def test_latent_embedding_custom_order_and_basis(fitted):
    adata = fitted[0]
    ax = sccont.pl.latent_embedding(adata, color="timepoint", order=["T3", "T2", "T1", "T0"],
                                    basis="X_sccont_pca", ordered=False, show=False)
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == ["T3", "T2", "T1", "T0"]


def test_latent_grid(fitted):
    axes = sccont.pl.latent_grid(fitted[0], ncols=4, show=False)
    assert axes.size >= LATENT_DIM
    axes2 = sccont.pl.latent_grid(fitted[0], latents=[0, 1], show=False)
    assert axes2.size == 2


def test_latent_distance_heatmap(fitted):
    ax = sccont.pl.latent_distance_heatmap(fitted[0], bins=10, show=False)
    assert len(ax.images) == 1
    assert ax.images[0].get_array().shape == (LATENT_DIM, LATENT_DIM)


@pytest.mark.parametrize("kind", ["violin", "box"])
def test_latent_by_label(fitted, kind):
    ax = sccont.pl.latent_by_label(fitted[0], 2, "timepoint", kind=kind, show=False)
    assert [t.get_text() for t in ax.get_xticklabels()] == ["T0", "T1", "T2", "T3"]
    with pytest.raises(ValueError):
        sccont.pl.latent_by_label(fitted[0], 2, "timepoint", kind="pie", show=False)


def test_latent_label_association_categorical_and_numeric(fitted):
    adata = fitted[0]
    ax = sccont.pl.latent_label_association(adata, "timepoint", show=False)
    mat = adata.uns["sccont"]["label_association"]
    assert mat.shape == (LATENT_DIM, 4)
    assert np.allclose(mat.mean(axis=1), 0, atol=1e-6)  # z-scored rows
    ax2 = sccont.pl.latent_label_association(adata, "pseudotime", show=False)
    corr = adata.uns["sccont"]["label_association"]
    assert len(corr) == LATENT_DIM and np.all(np.abs(corr) <= 1)
    assert len(ax2.patches) == LATENT_DIM


def test_top_genes_from_varm_and_with_direction(fitted):
    adata, _, _, _, sv = fitted
    ax = sccont.pl.top_genes(adata, latent=1, n=7, show=False)
    assert len(ax.patches) == 7
    ax = sccont.pl.top_genes(adata, latent=1, n=5, shap_values=sv, show=False)
    assert ax.get_legend() is not None
    names = [f"G{i}" for i in range(adata.n_vars)]
    ax = sccont.pl.top_genes(latent=1, n=4, shap_values=sv, gene_names=names, show=False)
    assert len(ax.patches) == 4
    with pytest.raises(ValueError):
        sccont.pl.top_genes(latent=0, show=False)


def test_go_enrichment_dotplot():
    df = pd.DataFrame({
        "name": [f"term {i}" for i in range(15)],
        "p_value": np.logspace(-8, -2, 15),
        "intersection_size": np.arange(15) + 1,
        "source": ["GO:BP", "GO:CC", "GO:MF"] * 5,
    })
    axes = sccont.pl.go_enrichment([df, pd.DataFrame()], top_n=6, show=False)
    assert axes.size == 2
    assert len(axes.flat[0].get_yticklabels()) == 6


def test_group_trajectories(fitted):
    adata = fitted[0]
    axes = sccont.pl.group_trajectories(adata, groupby="timepoint", show=False)
    assert axes.size >= 3
    axes = sccont.pl.group_trajectories(adata, groupby="timepoint", split_after="T1", ncols=2, show=False)
    assert axes.size >= 3


def test_gene_set_shap_embedding(fitted):
    adata, _, _, _, sv = fitted
    genes = list(adata.var_names[:5])
    ax = sccont.pl.gene_set_shap_embedding(adata, sv, genes, latent=0, show=False)
    assert "shap_sum_LF0" in adata.obs
    np.testing.assert_allclose(adata.obs["shap_sum_LF0"], sv[:, :5, 0].sum(axis=1))
    with pytest.raises(ValueError):
        sccont.pl.gene_set_shap_embedding(adata, sv, ["nope"], latent=0, show=False)


def test_shap_beeswarm(fitted):
    adata, _, _, _, sv = fitted
    ax = sccont.pl.shap_beeswarm(adata, sv, latent=0, n=5, show=False)
    assert ax is not None


def test_assign_branches_and_functional_group_heatmap(fitted):
    adata, _, _, _, sv = fitted
    branch = sccont.pl.assign_branches(adata, sv, latent=0)
    assert set(branch.unique()) <= {"Branch_A", "Branch_B"} and "sccont_branch" in adata.obs

    fg = {"grpA": list(adata.var_names[:5]), "grpB": list(adata.var_names[5:9]), "missing": ["zzz"]}
    g2l = {"grpA": 0, "grpB": 3, "missing": 1}
    scores = sccont.pl.group_shap_scores(adata, sv, fg, g2l, invert_shap={3: True})
    assert list(scores.columns) == ["grpA", "grpB"]
    np.testing.assert_allclose(scores["grpB"], -sv[:, 5:9, 3].sum(axis=1))

    ax = sccont.pl.functional_group_heatmap(adata, sv, fg, g2l, groupby="timepoint", show=False)
    mat = adata.uns["sccont"]["functional_group_matrix"]
    assert mat.shape == (2, 4)

    ax = sccont.pl.functional_group_heatmap(adata, sv, fg, g2l, groupby="timepoint",
                                            branch_key="sccont_branch", show=False)
    assert "root" in " ".join(t.get_text() for t in ax.get_xticklabels())
    with pytest.raises(ValueError):
        sccont.pl.functional_group_heatmap(adata, sv, fg, g2l, groupby="timepoint", branch_key="timepoint", show=False)


def test_latent_trajectory(fitted):
    adata, _, _, _, sv = fitted
    gene_sets = [list(adata.var_names[:5]), list(adata.var_names[5:10])]
    ax = sccont.pl.latent_trajectory(adata, sv, gene_sets, latents=[0, 1], groupby="timepoint",
                                     bifurcation_thresholds=0.5, invert=[False, True], show=False)
    assert ax.get_legend() is not None
    with pytest.raises(ValueError):
        sccont.pl.latent_trajectory(adata, sv, gene_sets, latents=[0], groupby="timepoint", show=False)


def test_save(fitted, tmp_path):
    out = tmp_path / "loss.png"
    sccont.pl.training_loss(fitted[2], show=False, save=out)
    assert out.exists() and out.stat().st_size > 0

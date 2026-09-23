"""Plotting (``sccont.pl``).

All functions read from the ``AnnData`` produced by the pipeline
(``obsm['X_sccont']``, ``obsm['X_sccont_pca']``, ``varm['sccont_shap_mean_abs']``,
``uns['sccont']``) and colour by any ``obs`` column. Each takes ``ax=None``,
``show=True`` and ``save=None`` and returns the matplotlib ``Axes`` (or array of
axes) so figures can be composed further.

Colour conventions
------------------
* categorical labels: a fixed-order palette (never cycled); more than eight
  unordered categories fall back to ``tab20``
* ordered labels (time, stage, pseudotime bins): a single-hue blue ramp
* magnitudes (mean |SHAP|, distances): the same blue ramp
* signed quantities (latent activation, z-scores, SHAP sums): blue - grey - red,
  always centred on zero
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from .attribution import SHAP_VARM_KEY
from .spatial import _latent_distance_matrix
from .training import LATENT_KEY

# --------------------------------------------------------------------------- #
# Palette
# --------------------------------------------------------------------------- #
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQUENTIAL_STEPS = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
DIVERGING = ("#2a78d6", "#f0efec", "#e34948")
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
       "grid": "#e1e0d9", "axis": "#c3c2b7", "surface": "#fcfcfb"}
MARKERS = ["o", "^", "s", "D", "P", "X", "v", "*"]


def sequential_cmap() -> LinearSegmentedColormap:
    """Single-hue blue ramp for magnitudes and ordered labels."""
    return LinearSegmentedColormap.from_list("sccont_sequential", SEQUENTIAL_STEPS)


def diverging_cmap() -> LinearSegmentedColormap:
    """Blue - neutral grey - red, for signed quantities centred on zero."""
    return LinearSegmentedColormap.from_list("sccont_diverging", list(DIVERGING))


def _natural_key(value):
    parts = re.split(r"(\d+\.?\d*)", str(value))
    return [float(p) if re.fullmatch(r"\d+\.?\d*", p) else p for p in parts]


def _categories(series: pd.Series, order: Sequence | None = None) -> list:
    if order is not None:
        return list(order)
    if isinstance(series.dtype, pd.CategoricalDtype):
        cats = [c for c in series.cat.categories if c in set(series)]
        return cats if series.cat.ordered else sorted(cats, key=_natural_key)
    return sorted(pd.unique(series.dropna()), key=_natural_key)


def _is_ordered(series: pd.Series, categories: Sequence, ordered: bool | None) -> bool:
    if ordered is not None:
        return ordered
    if isinstance(series.dtype, pd.CategoricalDtype) and series.cat.ordered:
        return True
    if len(categories) > len(CATEGORICAL):
        return True
    return all(re.search(r"\d", str(c)) for c in categories) and len(categories) > 2


def category_colors(categories: Sequence, ordered: bool = False) -> dict:
    """Colour per category: fixed palette, or the blue ramp when ``ordered``."""
    n = len(categories)
    if ordered:
        cmap = sequential_cmap()
        vals = np.linspace(0.25, 1.0, n) if n > 1 else [0.8]
        return {c: cmap(v) for c, v in zip(categories, vals)}
    if n <= len(CATEGORICAL):
        return {c: CATEGORICAL[i] for i, c in enumerate(categories)}
    tab = mpl.colormaps["tab20"]
    return {c: tab(i % 20) for i, c in enumerate(categories)}


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #
def _style(ax, grid: bool = False) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK["axis"])
    ax.tick_params(colors=INK["secondary"], labelsize=8)
    ax.xaxis.label.set_color(INK["secondary"])
    ax.yaxis.label.set_color(INK["secondary"])
    ax.title.set_color(INK["primary"])
    if grid:
        ax.grid(True, color=INK["grid"], linewidth=0.6)
        ax.set_axisbelow(True)


def _finish(fig, axes, show: bool, save):
    if save:
        fig.savefig(save, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return axes


def _new_axes(ax, figsize):
    if ax is not None:
        return ax.figure, ax
    return plt.subplots(figsize=figsize)


def _resolve_rep(adata: ad.AnnData, use_rep: str | None) -> tuple[np.ndarray, str | None]:
    """Latent matrix and the obsm key it came from (None for a latent AnnData)."""
    if use_rep is None:
        use_rep = adata.uns.get("sccont", {}).get("latent_key", LATENT_KEY)
        if use_rep not in adata.obsm:
            use_rep = None
    if use_rep is not None:
        return np.asarray(adata.obsm[use_rep]), use_rep
    return np.asarray(adata.X), None


def _coords(adata: ad.AnnData, basis: str | None, use_rep: str | None, components=(0, 1)) -> np.ndarray:
    """2-D coordinates: ``obsm[basis]`` or the latent PCA (computed if absent)."""
    latents, key = _resolve_rep(adata, use_rep)
    if basis is None:
        basis = f"{key}_pca" if key is not None else "X_pca"
    if basis in adata.obsm:
        coords = np.asarray(adata.obsm[basis])
    else:
        coords = PCA(n_components=max(components) + 1).fit_transform(latents)
    return coords[:, list(components)]


def _groups(adata: ad.AnnData, groups) -> np.ndarray:
    if groups is None:
        groups = adata.uns.get("sccont", {}).get("latent_groups")
        if groups is None:
            raise ValueError("No latent groups found; run group_spatially_similar_latents or pass groups=")
    return np.asarray(groups)


def _latent_index(latent) -> int:
    return int(latent)


def _symmetric_norm(values: np.ndarray, percentile: float = 99.0) -> TwoSlopeNorm:
    lim = float(np.nanpercentile(np.abs(values), percentile)) or 1e-9
    return TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)


def _categorical_legend(ax, colors: Mapping, title: str | None, marker="o",
                        bbox_to_anchor=(1.02, 1.0), **kw):
    handles = [Line2D([], [], marker=marker, linestyle="None", markersize=7,
                      markerfacecolor=c, markeredgecolor="none", label=str(k))
               for k, c in colors.items()]
    return ax.legend(handles=handles, title=title, loc="upper left", bbox_to_anchor=bbox_to_anchor,
                     frameon=False, fontsize=8, title_fontsize=8, **kw)


# --------------------------------------------------------------------------- #
# Tier 1
# --------------------------------------------------------------------------- #
def training_loss(losses: Sequence[float], smooth: int | None = 50, ax=None, show=True, save=None):
    """InfoNCE loss per training step, with an optional moving-average overlay."""
    fig, ax = _new_axes(ax, (6, 3.5))
    losses = np.asarray(losses, dtype=float)
    ax.plot(losses, color=CATEGORICAL[0], alpha=0.35 if smooth else 1.0, linewidth=1.0, label="loss")
    if smooth and len(losses) > smooth:
        kernel = np.ones(smooth) / smooth
        ma = np.convolve(losses, kernel, mode="valid")
        ax.plot(np.arange(smooth - 1, len(losses)), ma, color=CATEGORICAL[0], linewidth=2.0,
                label=f"{smooth}-step mean")
        ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel("Training step")
    ax.set_ylabel("InfoNCE loss")
    ax.set_title("Training loss")
    _style(ax, grid=True)
    return _finish(fig, ax, show, save)


def elbow(adata: ad.AnnData, max_k: int = 10, bins: int = 40, use_rep=None, ax=None, show=True, save=None):
    """Elbow plot for the number of spatial latent clusters (stage C)."""
    from .spatial import elbow_plot_for_clusters

    fig, ax = _new_axes(ax, (5, 3.5))
    elbow_plot_for_clusters(adata, max_k=max_k, bins=bins, ax=ax, show=False, use_rep=use_rep)
    ax.lines[-1].set_color(CATEGORICAL[0])
    _style(ax, grid=True)
    return _finish(fig, ax, show, save)


def latent_embedding(
    adata: ad.AnnData,
    color: str | int | None = None,
    basis: str | None = None,
    use_rep: str | None = None,
    components=(0, 1),
    order: Sequence | None = None,
    ordered: bool | None = None,
    size: float | None = None,
    alpha: float = 0.8,
    cmap=None,
    title: str | None = None,
    legend: bool = True,
    ax=None,
    show=True,
    save=None,
):
    """Cells in the latent PCA (or any ``obsm`` basis) coloured by a label or a latent feature.

    Parameters
    ----------
    color
        An ``obs`` column (categorical or numeric) or a latent feature index
        (``7`` / ``'7'``) whose activation colours the cells. ``None`` for plain grey.
    basis
        ``obsm`` key of the 2-D coordinates; defaults to the latent PCA.
    order / ordered
        Category order and whether to treat categories as ordered (blue ramp)
        instead of distinct hues. Auto-detected when omitted.
    """
    fig, ax = _new_axes(ax, (5.5, 4.5))
    xy = _coords(adata, basis, use_rep, components)
    n = xy.shape[0]
    s = size if size is not None else max(2.0, min(20.0, 4000.0 / n))

    if color is None:
        ax.scatter(xy[:, 0], xy[:, 1], s=s, c=INK["muted"], alpha=alpha, linewidths=0)
        ttl = title or ""
    elif isinstance(color, (int, np.integer)) or (isinstance(color, str) and color.isdigit() and color not in adata.obs):
        latents, _ = _resolve_rep(adata, use_rep)
        values = latents[:, _latent_index(color)]
        sc_ = ax.scatter(xy[:, 0], xy[:, 1], s=s, c=values, cmap=cmap or diverging_cmap(),
                         norm=_symmetric_norm(values), alpha=alpha, linewidths=0)
        cb = fig.colorbar(sc_, ax=ax, shrink=0.7, pad=0.02)
        cb.set_label(f"LF {int(color)} activation", color=INK["secondary"], fontsize=8)
        cb.ax.tick_params(labelsize=7, colors=INK["secondary"])
        ttl = title or f"Latent feature {int(color)}"
    else:
        series = adata.obs[color]
        if pd.api.types.is_numeric_dtype(series) and not isinstance(series.dtype, pd.CategoricalDtype):
            sc_ = ax.scatter(xy[:, 0], xy[:, 1], s=s, c=series.to_numpy(), cmap=cmap or sequential_cmap(),
                             alpha=alpha, linewidths=0)
            cb = fig.colorbar(sc_, ax=ax, shrink=0.7, pad=0.02)
            cb.set_label(color, color=INK["secondary"], fontsize=8)
            cb.ax.tick_params(labelsize=7, colors=INK["secondary"])
        else:
            cats = _categories(series, order)
            colors = category_colors(cats, _is_ordered(series, cats, ordered))
            for cat in cats:
                m = (series == cat).to_numpy()
                ax.scatter(xy[m, 0], xy[m, 1], s=s, color=colors[cat], alpha=alpha, linewidths=0, label=str(cat))
            if legend:
                _categorical_legend(ax, colors, color)
        ttl = title or str(color)

    ax.set_title(ttl)
    ax.set_xlabel(f"{basis or 'latent PCA'} {components[0] + 1}")
    ax.set_ylabel(f"{basis or 'latent PCA'} {components[1] + 1}")
    ax.set_xticks([])
    ax.set_yticks([])
    _style(ax)
    return _finish(fig, ax, show, save)


def latent_grid(
    adata: ad.AnnData,
    latents: Sequence[int] | None = None,
    groups=None,
    basis: str | None = None,
    use_rep: str | None = None,
    ncols: int = 6,
    panel_size: float = 2.2,
    size: float | None = None,
    show=True,
    save=None,
):
    """Small multiples: every latent feature's activation over the embedding.

    Panels are ordered by spatial group (from ``uns['sccont']['latent_groups']``
    or ``groups=``) when available; the group is shown in each title.
    """
    latent_mat, key = _resolve_rep(adata, use_rep)
    xy = _coords(adata, basis, use_rep)
    try:
        grp = _groups(adata, groups)
    except ValueError:
        grp = None
    if latents is None:
        latents = list(range(latent_mat.shape[1]))
        if grp is not None:
            latents = sorted(latents, key=lambda i: (grp[i], i))
    latents = [int(i) for i in latents]

    n = len(latents)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_size * ncols, panel_size * nrows), squeeze=False)
    s = size if size is not None else max(2.5, min(8.0, 4000.0 / xy.shape[0]))
    cmap = diverging_cmap()
    for k, i in enumerate(latents):
        ax = axes.flat[k]
        v = latent_mat[:, i]
        ax.scatter(xy[:, 0], xy[:, 1], c=v, s=s, cmap=cmap, norm=_symmetric_norm(v), linewidths=0)
        ttl = f"LF {i}" + (f"  (group {grp[i]})" if grp is not None else "")
        ax.set_title(ttl, fontsize=8, color=INK["primary"])
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(INK["grid"])
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.suptitle("Latent feature activation (blue < 0 < red)", fontsize=9, color=INK["secondary"])
    fig.tight_layout()
    return _finish(fig, axes, show, save)


def latent_distance_heatmap(
    adata: ad.AnnData,
    groups=None,
    bins: int = 40,
    use_rep: str | None = None,
    ax=None,
    show=True,
    save=None,
):
    """Correlation-distance matrix between latent spatial maps, ordered by group.

    Latents are sorted by spatial group and, within a group, by hierarchical
    clustering of the distances; a colour strip marks the groups.
    """
    dist = _latent_distance_matrix(adata, bins, use_rep)
    grp = _groups(adata, groups)
    order = []
    for g in np.unique(grp):
        idx = np.where(grp == g)[0]
        if len(idx) > 2:
            sub = dist[np.ix_(idx, idx)]
            idx = idx[leaves_list(linkage(squareform(sub, checks=False), method="average"))]
        order.extend(idx.tolist())
    order = np.array(order)

    fig, ax = _new_axes(ax, (6.5, 5.5))
    im = ax.imshow(dist[np.ix_(order, order)], cmap=sequential_cmap(), vmin=0, vmax=float(dist.max()) or 1)
    ax.set_xticks(range(len(order))); ax.set_yticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=7, rotation=90); ax.set_yticklabels(order, fontsize=7)
    ax.tick_params(length=0, colors=INK["secondary"])
    for sp in ax.spines.values():
        sp.set_visible(False)

    ugroups = list(np.unique(grp))
    gcolors = category_colors(ugroups)
    # group strip on the left
    for pos, i in enumerate(order):
        ax.add_patch(mpl.patches.Rectangle((-1.6, pos - 0.5), 0.8, 1.0, color=gcolors[grp[i]], clip_on=False))
    # separators between groups
    bounds = np.where(np.diff(grp[order]) != 0)[0] + 0.5
    for b in bounds:
        ax.axhline(b, color=INK["surface"], linewidth=2.0)
        ax.axvline(b, color=INK["surface"], linewidth=2.0)
    ax.set_xlim(-0.5, len(order) - 0.5)
    cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label("correlation distance", color=INK["secondary"], fontsize=8)
    cb.ax.tick_params(labelsize=7, colors=INK["secondary"])
    _categorical_legend(ax, {f"group {g}": c for g, c in gcolors.items()}, None, marker="s",
                        bbox_to_anchor=(1.25, 1.0))
    ax.set_title("Spatial similarity of latent features")
    return _finish(fig, ax, show, save)


def latent_by_label(
    adata: ad.AnnData,
    latent: int | str,
    groupby: str,
    kind: str = "violin",
    order: Sequence | None = None,
    ordered: bool | None = None,
    use_rep: str | None = None,
    ax=None,
    show=True,
    save=None,
):
    """Distribution of one latent feature's activation per category of ``groupby``."""
    latents, _ = _resolve_rep(adata, use_rep)
    i = _latent_index(latent)
    series = adata.obs[groupby]
    cats = _categories(series, order)
    colors = category_colors(cats, _is_ordered(series, cats, ordered))
    data = [latents[(series == c).to_numpy(), i] for c in cats]

    fig, ax = _new_axes(ax, (max(4, 0.7 * len(cats) + 1.5), 3.5))
    pos = np.arange(len(cats))
    if kind == "violin":
        parts = ax.violinplot(data, positions=pos, showmedians=True, showextrema=False, widths=0.8)
        for body, c in zip(parts["bodies"], cats):
            body.set_facecolor(colors[c]); body.set_edgecolor("none"); body.set_alpha(0.85)
        parts["cmedians"].set_color(INK["primary"]); parts["cmedians"].set_linewidth(1.2)
    elif kind == "box":
        bp = ax.boxplot(data, positions=pos, widths=0.6, patch_artist=True, showfliers=False,
                        medianprops=dict(color=INK["primary"], linewidth=1.2))
        for patch, c in zip(bp["boxes"], cats):
            patch.set_facecolor(colors[c]); patch.set_edgecolor("none")
        for key in ("whiskers", "caps"):
            for line in bp[key]:
                line.set_color(INK["axis"])
    else:
        raise ValueError("kind must be 'violin' or 'box'")
    ax.axhline(0, color=INK["axis"], linewidth=0.8, zorder=0)
    ax.set_xticks(pos); ax.set_xticklabels([str(c) for c in cats], rotation=45 if len(cats) > 6 else 0, ha="right" if len(cats) > 6 else "center")
    ax.set_xlabel(groupby)
    ax.set_ylabel(f"LF {i} activation")
    ax.set_title(f"Latent feature {i} by {groupby}")
    _style(ax, grid=True)
    ax.grid(False, axis="x")
    return _finish(fig, ax, show, save)


def latent_label_association(
    adata: ad.AnnData,
    groupby: str,
    latents: Sequence[int] | None = None,
    order: Sequence | None = None,
    zscore: bool = True,
    use_rep: str | None = None,
    groups=None,
    ax=None,
    show=True,
    save=None,
):
    """Which latent features track a label.

    * Categorical ``groupby``: heatmap of mean activation per category
      (z-scored across categories per latent when ``zscore``), latents ordered
      by spatial group when available.
    * Numeric ``groupby`` (e.g. pseudotime): horizontal bars of the Pearson
      correlation between each latent and the label.

    Returns the axes; the underlying matrix / series is stored in
    ``adata.uns['sccont']['label_association']``.
    """
    latent_mat, _ = _resolve_rep(adata, use_rep)
    if latents is None:
        latents = list(range(latent_mat.shape[1]))
        try:
            grp = _groups(adata, groups)
            latents = sorted(latents, key=lambda i: (grp[i], i))
        except ValueError:
            grp = None
    else:
        grp = None
    latents = [int(i) for i in latents]
    series = adata.obs[groupby]

    if pd.api.types.is_numeric_dtype(series) and not isinstance(series.dtype, pd.CategoricalDtype):
        y = series.to_numpy(dtype=float)
        ok = np.isfinite(y)
        corr = pd.Series([pearsonr(latent_mat[ok, i], y[ok])[0] for i in latents],
                         index=[f"LF {i}" for i in latents], name=f"r({groupby})")
        fig, ax = _new_axes(ax, (5, 0.28 * len(latents) + 1.2))
        colors = [CATEGORICAL[0] if v >= 0 else CATEGORICAL[7] for v in corr]
        ax.barh(np.arange(len(corr)), corr.values, color=colors, height=0.7)
        ax.set_yticks(np.arange(len(corr))); ax.set_yticklabels(corr.index, fontsize=7)
        ax.invert_yaxis()
        ax.axvline(0, color=INK["axis"], linewidth=0.8)
        ax.set_xlim(-1, 1)
        ax.set_xlabel(f"Pearson r with {groupby}")
        ax.set_title(f"Latent features vs {groupby}")
        _style(ax, grid=True); ax.grid(False, axis="y")
        result = corr
    else:
        cats = _categories(series, order)
        mat = pd.DataFrame(
            {c: latent_mat[(series == c).to_numpy()][:, latents].mean(axis=0) for c in cats},
            index=[f"LF {i}" for i in latents],
        )
        if zscore:
            std = mat.std(axis=1).replace(0, np.nan)
            mat = ((mat.T - mat.mean(axis=1)) / std).T.fillna(0)
        fig, ax = _new_axes(ax, (0.45 * len(cats) + 2.5, 0.28 * len(latents) + 1.2))
        vals = mat.to_numpy()
        im = ax.imshow(vals, cmap=diverging_cmap(), norm=_symmetric_norm(vals, 100), aspect="auto")
        ax.set_xticks(range(len(cats))); ax.set_xticklabels([str(c) for c in cats], rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(latents))); ax.set_yticklabels(mat.index, fontsize=7)
        ax.tick_params(length=0, colors=INK["secondary"])
        for sp in ax.spines.values():
            sp.set_visible(False)
        if grp is not None:
            for b in np.where(np.diff(grp[latents]) != 0)[0] + 0.5:
                ax.axhline(b, color=INK["surface"], linewidth=2.0)
        cb = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
        cb.set_label("mean activation" + (" (z-score)" if zscore else ""), color=INK["secondary"], fontsize=8)
        cb.ax.tick_params(labelsize=7, colors=INK["secondary"])
        ax.set_xlabel(groupby)
        ax.set_title(f"Latent features by {groupby}")
        result = mat
    adata.uns["sccont"] = {**adata.uns.get("sccont", {}), "label_association": result}
    return _finish(fig, ax, show, save)


def _direction_colors(adata, shap_values, latent, gene_idx):
    """Sign of corr(expression, SHAP) per gene -> colour and label."""
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    colors, signs = [], []
    for g in gene_idx:
        r = pearsonr(X[:, g], shap_values[:, g, latent])[0] if np.std(X[:, g]) > 0 else 0.0
        signs.append(r)
        colors.append(CATEGORICAL[0] if r >= 0 else CATEGORICAL[1])
    return colors, np.array(signs)


def top_genes(
    adata: ad.AnnData | None = None,
    latent: int | str = 0,
    n: int = 20,
    shap_values: np.ndarray | None = None,
    gene_names: Sequence[str] | None = None,
    varm_key: str = SHAP_VARM_KEY,
    direction: bool = True,
    ax=None,
    show=True,
    save=None,
):
    """Top-``n`` genes of a latent feature by mean |SHAP| as horizontal bars.

    Reads ``adata.varm[varm_key]`` (written by :func:`sccont.compute_shap_values`)
    so the full SHAP tensor is not needed. If ``shap_values`` and ``adata`` are
    both given and ``direction`` is True, bars are coloured by whether high
    expression activates (blue) or suppresses (orange) the feature, from the
    sign of the expression-SHAP correlation.
    """
    i = _latent_index(latent)
    if adata is not None and varm_key in adata.varm:
        mean_abs = np.asarray(adata.varm[varm_key])[:, i]
        names = np.asarray(adata.var_names)
    elif shap_values is not None:
        mean_abs = np.abs(shap_values[:, :, i]).mean(axis=0)
        names = np.asarray(gene_names if gene_names is not None else
                           (adata.var_names if adata is not None else range(len(mean_abs))))
    else:
        raise ValueError("Pass an AnnData with varm['%s'] or shap_values" % varm_key)

    idx = np.argsort(mean_abs)[::-1][:n]
    fig, ax = _new_axes(ax, (5, 0.28 * len(idx) + 1.2))
    legend = None
    if direction and shap_values is not None and adata is not None:
        colors, _ = _direction_colors(adata, shap_values, i, idx)
        legend = {"high expression activates": CATEGORICAL[0], "high expression suppresses": CATEGORICAL[1]}
    else:
        colors = CATEGORICAL[0]
    ax.barh(np.arange(len(idx)), mean_abs[idx], color=colors, height=0.7)
    ax.set_yticks(np.arange(len(idx))); ax.set_yticklabels(names[idx], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("mean |SHAP|")
    ax.set_title(f"Top genes for latent feature {i}")
    _style(ax, grid=True); ax.grid(False, axis="y")
    if legend:
        _categorical_legend(ax, legend, None, marker="s", bbox_to_anchor=(1.02, 0.5))
    return _finish(fig, ax, show, save)


def go_enrichment(
    results: Sequence[pd.DataFrame],
    top_n: int = 10,
    clusters: Sequence[int] | None = None,
    p_col: str = "p_value",
    term_col: str = "name",
    size_col: str = "intersection_size",
    source_col: str = "source",
    max_label: int = 55,
    show=True,
    save=None,
):
    """Dot plot of the top GO terms per spatial cluster (one panel per cluster).

    x = -log10 p, dot size = intersection size, colour = GO source (BP / CC / MF).
    ``results`` is the list returned by :func:`sccont.enrich_latent_clusters`.
    """
    clusters = list(range(len(results))) if clusters is None else list(clusters)
    fig, axes = plt.subplots(len(clusters), 1, figsize=(7, 0.32 * top_n * len(clusters) + 1.0 * len(clusters)),
                             squeeze=False, sharex=True)
    sources = sorted({s for c in clusters for s in results[c].get(source_col, pd.Series(dtype=str)).unique()})
    colors = category_colors(sources)
    for ax, c in zip(axes.flat, clusters):
        df = results[c]
        if df is None or len(df) == 0 or p_col not in df:
            ax.text(0.5, 0.5, "no enriched terms", ha="center", va="center", color=INK["muted"], transform=ax.transAxes)
            ax.set_yticks([]); _style(ax); ax.set_title(f"Cluster {c}", loc="left", fontsize=9)
            continue
        df = df.sort_values(p_col).head(top_n).iloc[::-1]
        x = -np.log10(df[p_col].to_numpy(dtype=float))
        sizes = df[size_col].to_numpy(dtype=float) if size_col in df else np.full(len(df), 20.0)
        s = 20 + 120 * sizes / max(sizes.max(), 1)
        col = [colors.get(v, CATEGORICAL[0]) for v in df[source_col]] if source_col in df else CATEGORICAL[0]
        y = np.arange(len(df))
        ax.hlines(y, 0, x, color=INK["grid"], linewidth=1.0, zorder=1)
        ax.scatter(x, y, s=s, c=col, edgecolors=INK["surface"], linewidths=0.8, zorder=2)
        labels = [t if len(str(t)) <= max_label else str(t)[: max_label - 1] + "…" for t in df[term_col]]
        ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7)
        ax.set_title(f"Cluster {c}", loc="left", fontsize=9)
        _style(ax, grid=True); ax.grid(False, axis="y")
    axes.flat[-1].set_xlabel("-log10(p)")
    if sources:
        _categorical_legend(axes.flat[0], colors, "source")
    fig.tight_layout()
    return _finish(fig, axes, show, save)


# --------------------------------------------------------------------------- #
# Tier 2: paper figures, generalised to any label
# --------------------------------------------------------------------------- #
def _codes(series: pd.Series, order):
    cats = _categories(series, order)
    lookup = {c: k for k, c in enumerate(cats)}
    codes = series.map(lookup).to_numpy()
    return cats, codes


def group_trajectories(
    adata: ad.AnnData,
    groupby: str,
    groups=None,
    order: Sequence | None = None,
    split_after=None,
    use_rep: str | None = None,
    ncols: int = 3,
    panel_size: float = 4.0,
    size: float | None = None,
    show_titles: bool = True,
    show=True,
    save=None,
):
    """Per spatial cluster: PCA of that cluster's latents with a trajectory through label centroids.

    Generalises the manuscript's Figure 2. For each spatial group the cells are
    projected onto the first two PCs of that group's latent features, coloured
    by ``groupby`` (an ordered label such as time), and the per-category
    centroids are joined in category order. With ``split_after=<category>``, the
    categories after it are split into two branches by k-means (k = 2), branch
    assignment avoiding crossings, so a bifurcation is drawn as two paths.
    """
    latents, _ = _resolve_rep(adata, use_rep)
    grp = _groups(adata, groups)
    cats, codes = _codes(adata.obs[groupby], order)
    colors = category_colors(cats, True)
    split_idx = cats.index(split_after) if split_after is not None else None

    ugroups = list(np.unique(grp))
    ncols = min(ncols, len(ugroups))
    nrows = int(np.ceil(len(ugroups) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_size * ncols, panel_size * nrows), squeeze=False)
    s = size if size is not None else max(2.0, min(12.0, 3000.0 / adata.n_obs))

    for k, g in enumerate(ugroups):
        ax = axes.flat[k]
        idx = np.where(grp == g)[0]
        xy = PCA(n_components=2).fit_transform(latents[:, idx]) if len(idx) > 1 else \
            np.c_[latents[:, idx[0]], np.zeros(adata.n_obs)]
        for ci, cat in enumerate(cats):
            m = codes == ci
            ax.scatter(xy[m, 0], xy[m, 1], s=s, color=colors[cat], alpha=0.35, linewidths=0, zorder=1)

        node_x, node_y, node_c = [], [], []
        trunk, b1, b2 = [], [], []
        for ci, cat in enumerate(cats):
            pts = xy[codes == ci]
            if len(pts) == 0:
                continue
            if split_idx is None or ci <= split_idx or len(pts) < 2:
                c = pts.mean(axis=0)
                trunk.append(c); node_x.append(c[0]); node_y.append(c[1]); node_c.append(colors[cat])
            else:
                km = KMeans(n_clusters=2, random_state=42, n_init=10).fit(pts)
                c1, c2 = km.cluster_centers_
                if not b1 and trunk:
                    b1.append(trunk[-1]); b2.append(trunk[-1])
                if b1:
                    straight = np.linalg.norm(b1[-1] - c1) + np.linalg.norm(b2[-1] - c2)
                    crossed = np.linalg.norm(b1[-1] - c2) + np.linalg.norm(b2[-1] - c1)
                    if crossed < straight:
                        c1, c2 = c2, c1
                b1.append(c1); b2.append(c2)
                node_x += [c1[0], c2[0]]; node_y += [c1[1], c2[1]]; node_c += [colors[cat]] * 2
        for path in (trunk, b1, b2):
            if len(path) > 1:
                p = np.array(path)
                ax.plot(p[:, 0], p[:, 1], color=INK["primary"], linewidth=1.8, zorder=2)
        ax.scatter(node_x, node_y, s=110, c=node_c, marker=MARKERS[k % len(MARKERS)],
                   edgecolors=INK["primary"], linewidths=1.2, zorder=3)
        if show_titles:
            ax.set_title(f"Spatial cluster {g}  ({len(idx)} LFs)", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(INK["grid"])
    for ax in axes.flat[len(ugroups):]:
        ax.axis("off")
    _categorical_legend(axes.flat[min(ncols, len(ugroups)) - 1], colors, groupby)
    fig.tight_layout()
    return _finish(fig, axes, show, save)


def gene_set_shap_embedding(
    adata: ad.AnnData,
    shap_values: np.ndarray,
    genes: Sequence[str],
    latent: int | str,
    invert: bool = False,
    basis: str | None = None,
    use_rep: str | None = None,
    key_added: str | None = None,
    size: float | None = None,
    ax=None,
    show=True,
    save=None,
):
    """Cells coloured by the summed SHAP of a gene set for one latent feature.

    The colour scale is centred on zero. The per-cell sum is stored in
    ``adata.obs[key_added]`` (default ``f"shap_sum_LF{latent}"``).
    """
    i = _latent_index(latent)
    gene_idx = [adata.var_names.get_loc(g) for g in genes if g in adata.var_names]
    if not gene_idx:
        raise ValueError("none of the genes are in adata.var_names")
    total = shap_values[:, gene_idx, i].sum(axis=1)
    if invert:
        total = -total
    key_added = key_added or f"shap_sum_LF{i}"
    adata.obs[key_added] = total

    fig, ax = _new_axes(ax, (5.5, 4.5))
    xy = _coords(adata, basis, use_rep)
    s = size if size is not None else max(2.0, min(20.0, 4000.0 / adata.n_obs))
    sc_ = ax.scatter(xy[:, 0], xy[:, 1], c=total, s=s, cmap=diverging_cmap(), norm=_symmetric_norm(total, 100), linewidths=0)
    cb = fig.colorbar(sc_, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label(f"sum SHAP, LF {i} ({len(gene_idx)} genes)", color=INK["secondary"], fontsize=8)
    cb.ax.tick_params(labelsize=7, colors=INK["secondary"])
    ax.set_title(f"Gene-set attribution, latent feature {i}")
    ax.set_xticks([]); ax.set_yticks([])
    _style(ax)
    return _finish(fig, ax, show, save)


def shap_beeswarm(
    adata: ad.AnnData,
    shap_values: np.ndarray,
    latent: int | str,
    n: int = 20,
    genes: Sequence[str] | None = None,
    show=True,
    save=None,
):
    """``shap.summary_plot`` (beeswarm) for one latent feature, restricted to its top genes."""
    import shap

    i = _latent_index(latent)
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    if genes is None:
        mean_abs = np.abs(shap_values[:, :, i]).mean(axis=0)
        idx = np.argsort(mean_abs)[::-1][:n]
    else:
        idx = [adata.var_names.get_loc(g) for g in genes if g in adata.var_names]
    names = [adata.var_names[j] for j in idx]
    fig = plt.figure(figsize=(7, 0.3 * len(idx) + 1.5))   # shap draws on the current figure
    shap.summary_plot(shap_values[:, idx, i], features=X[:, idx], feature_names=names, show=False,
                      cmap=diverging_cmap(), plot_size=None)
    fig.axes[0].set_title(f"Latent feature {i}", color=INK["primary"])
    return _finish(fig, fig.axes[0], show, save)


def assign_branches(
    adata: ad.AnnData,
    shap_values: np.ndarray,
    latent: int | str,
    genes: Sequence[str] | None = None,
    invert: bool = False,
    key_added: str = "sccont_branch",
    labels: tuple[str, str] = ("Branch_A", "Branch_B"),
) -> pd.Series:
    """Split cells into two branches with a 2-component Gaussian mixture on a latent's SHAP signal.

    The signal is the summed SHAP of ``genes`` (default: all genes) for ``latent``,
    negated if ``invert``. The high-signal component gets ``labels[0]``. The
    result is stored in ``adata.obs[key_added]`` and returned.
    """
    i = _latent_index(latent)
    if genes is None:
        signal = shap_values[:, :, i].sum(axis=1)
    else:
        gene_idx = [adata.var_names.get_loc(g) for g in genes if g in adata.var_names]
        signal = shap_values[:, gene_idx, i].sum(axis=1)
    if invert:
        signal = -signal
    gmm = GaussianMixture(n_components=2, random_state=42).fit(signal.reshape(-1, 1))
    comp = gmm.predict(signal.reshape(-1, 1))
    high = int(np.argmax(gmm.means_.ravel()))
    branch = pd.Series(np.where(comp == high, labels[0], labels[1]), index=adata.obs_names, name=key_added)
    adata.obs[key_added] = branch
    return branch


def group_shap_scores(
    adata: ad.AnnData,
    shap_values: np.ndarray,
    functional_groups: Mapping[str, Sequence[str]],
    group_to_latent: Mapping[str, int],
    invert_shap: Mapping[int, bool] | None = None,
) -> pd.DataFrame:
    """Per-cell score of each functional group: summed SHAP of its genes for its latent."""
    invert_shap = invert_shap or {}
    pos = {g: k for k, g in enumerate(adata.var_names)}
    scores = pd.DataFrame(index=adata.obs_names)
    for grp, genes in functional_groups.items():
        if grp not in group_to_latent:
            continue
        lat = int(group_to_latent[grp])
        idx = [pos[g] for g in genes if g in pos]
        if idx:
            val = shap_values[:, idx, lat].sum(axis=1)
            if invert_shap.get(lat, False):
                val = -val
            scores[grp] = val
    return scores


def functional_group_heatmap(
    adata: ad.AnnData,
    shap_values: np.ndarray,
    functional_groups: Mapping[str, Sequence[str]],
    group_to_latent: Mapping[str, int],
    groupby: str,
    invert_shap: Mapping[int, bool] | None = None,
    order: Sequence | None = None,
    branch_key: str | None = None,
    zscore: bool = True,
    title: str | None = None,
    ax=None,
    show=True,
    save=None,
):
    """Functional-group activity across the categories of ``groupby``.

    Linear layout (default): rows are functional groups sorted by the category
    where they peak; columns are the ordered categories.

    Bifurcating "beam" layout (``branch_key`` given, an ``obs`` column with two
    values, e.g. from :func:`assign_branches`): the first category is the root in
    the centre, one branch runs leftwards and the other rightwards through the
    remaining categories; rows are sorted by branch asymmetry.

    Returns the axes; the plotted matrix is in ``adata.uns['sccont']['functional_group_matrix']``.
    """
    scores = group_shap_scores(adata, shap_values, functional_groups, group_to_latent, invert_shap)
    if scores.shape[1] == 0:
        raise ValueError("no functional group had genes present in adata.var_names")
    series = adata.obs[groupby]
    cats = _categories(series, order)

    if branch_key is None:
        mat = pd.DataFrame({c: scores[(series == c).to_numpy()].mean() for c in cats})
        peak = mat.idxmax(axis=1).map({c: k for k, c in enumerate(cats)})
        mat = mat.loc[peak.sort_values().index]
        cols = [str(c) for c in cats]
        center = None
        xlabel = groupby
    else:
        branches = _categories(adata.obs[branch_key])
        if len(branches) != 2:
            raise ValueError(f"obs['{branch_key}'] must have exactly two values, got {branches}")
        bA, bB = branches
        root = cats[0]
        cols, vals = [], []
        for c in reversed(cats[1:]):
            m = ((series == c) & (adata.obs[branch_key] == bA)).to_numpy()
            if m.any():
                cols.append(f"{c} {bA}"); vals.append(scores[m].mean())
        cols.append(f"{root} root"); vals.append(scores[(series == root).to_numpy()].mean())
        center = len(cols) - 1
        for c in cats[1:]:
            m = ((series == c) & (adata.obs[branch_key] == bB)).to_numpy()
            if m.any():
                cols.append(f"{c} {bB}"); vals.append(scores[m].mean())
        mat = pd.DataFrame(vals, index=cols).T
        left = mat.iloc[:, :center].to_numpy()[:, ::-1]
        right = mat.iloc[:, center + 1:].to_numpy()
        k = min(left.shape[1], right.shape[1])
        asym = np.abs(left[:, :k] - right[:, :k]).sum(axis=1)
        mat = mat.iloc[np.argsort(-asym)]
        xlabel = f"← {bA}          {groupby}          {bB} →"

    plot = mat.copy()
    if zscore:
        std = plot.std(axis=1).replace(0, np.nan)
        plot = ((plot.T - plot.mean(axis=1)) / std).T.fillna(0)

    fig, ax = _new_axes(ax, (0.5 * plot.shape[1] + 3.0, 0.3 * plot.shape[0] + 1.6))
    vals_arr = plot.to_numpy()
    im = ax.imshow(vals_arr, cmap=diverging_cmap(), norm=_symmetric_norm(vals_arr, 100), aspect="auto")
    ax.set_xticks(range(plot.shape[1])); ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(plot.shape[0])); ax.set_yticklabels(plot.index, fontsize=7)
    ax.tick_params(length=0, colors=INK["secondary"])
    for sp in ax.spines.values():
        sp.set_visible(False)
    # cell gaps
    for x in np.arange(0.5, plot.shape[1] - 0.5):
        ax.axvline(x, color=INK["surface"], linewidth=1.0)
    for y in np.arange(0.5, plot.shape[0] - 0.5):
        ax.axhline(y, color=INK["surface"], linewidth=1.0)
    if center is not None:
        ax.axvline(center - 0.5, color=INK["primary"], linewidth=1.0, linestyle="--")
        ax.axvline(center + 0.5, color=INK["primary"], linewidth=1.0, linestyle="--")
    cb = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cb.set_label("latent activity" + (" (z-score)" if zscore else ""), color=INK["secondary"], fontsize=8)
    cb.ax.tick_params(labelsize=7, colors=INK["secondary"])
    ax.set_xlabel(xlabel)
    ax.set_title(title or ("Bifurcating functional groups" if branch_key else "Functional groups"))
    adata.uns["sccont"] = {**adata.uns.get("sccont", {}), "functional_group_matrix": mat}
    return _finish(fig, ax, show, save)


def latent_trajectory(
    adata: ad.AnnData,
    shap_values: np.ndarray,
    gene_sets: Sequence[Sequence[str]],
    latents: Sequence[int],
    groupby: str,
    bifurcation_thresholds: Sequence[float] | float = 0.75,
    invert: Sequence[bool] | bool = False,
    order: Sequence | None = None,
    basis: str | None = None,
    use_rep: str | None = None,
    robust_percentiles=(1, 99),
    offset_threshold: float | None = None,
    offset_magnitude: float | None = None,
    background_alpha: float = 0.5,
    ax=None,
    show=True,
    save=None,
):
    """Trajectories of several latent features through the categories of ``groupby``.

    Port of the manuscript's multi-latent trajectory figure. For each latent the
    per-cell signal is the summed SHAP of its gene set; per category a
    SHAP-weighted centroid is placed (split into a high and a low branch by a
    Gaussian mixture when the signal correlates with the axis orthogonal to the
    overall flow above ``bifurcation_thresholds``). Nodes of different latents at
    the same spot are offset so all stay visible. Cells are coloured by whichever
    latent's signal deviates most from neutral.
    """
    latents = [int(i) for i in latents]
    n = len(latents)
    if np.isscalar(bifurcation_thresholds):
        bifurcation_thresholds = [float(bifurcation_thresholds)] * n
    if isinstance(invert, (bool, np.bool_)):
        invert = [bool(invert)] * n
    if not (len(gene_sets) == len(bifurcation_thresholds) == len(invert) == n):
        raise ValueError("gene_sets, latents, bifurcation_thresholds and invert must have equal length")

    xy = _coords(adata, basis, use_rep)
    cats, codes = _codes(adata.obs[groupby], order)
    scale = max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]))
    offset_threshold = offset_threshold if offset_threshold is not None else scale * 0.02
    offset_magnitude = offset_magnitude if offset_magnitude is not None else scale * 0.01

    c_start = xy[codes == 0].mean(axis=0)
    c_end = xy[codes == len(cats) - 1].mean(axis=0)
    flow = c_end - c_start
    flow = flow / np.linalg.norm(flow) if np.linalg.norm(flow) > 0 else np.array([1.0, 0.0])
    orth_pos = (xy - c_start) @ np.array([-flow[1], flow[0]])

    cmap = diverging_cmap()
    norms, all_norm_vals, trajectories = {}, np.zeros((adata.n_obs, n)), {}
    for k, (lat, genes, thr, inv) in enumerate(zip(latents, gene_sets, bifurcation_thresholds, invert)):
        gene_idx = [adata.var_names.get_loc(g) for g in genes if g in adata.var_names]
        signal = shap_values[:, gene_idx, lat].sum(axis=1) if gene_idx else np.zeros(adata.n_obs)
        if inv:
            signal = -signal
        vmin, vmax = np.percentile(signal, robust_percentiles)
        if vmin >= vmax:
            vmin, vmax = -1e-3, 1e-3
        norm = TwoSlopeNorm(vmin=min(vmin, -1e-9), vcenter=0.0, vmax=max(vmax, 1e-9))
        norms[lat] = norm
        all_norm_vals[:, k] = norm(signal)

        gmm = GaussianMixture(n_components=2, random_state=42).fit(signal.reshape(-1, 1))
        high_c = int(np.argmax(gmm.means_.ravel()))

        def centroid(idx, vals):
            w = np.abs(vals)
            return xy[idx].mean(axis=0) if w.sum() < 1e-9 else np.average(xy[idx], axis=0, weights=w)

        pts = {}
        for ci in range(len(cats)):
            idx = np.where(codes == ci)[0]
            if len(idx) == 0:
                pts[ci] = []
                continue
            vals = signal[idx]
            if ci == 0 or len(idx) < 10 or np.std(orth_pos[idx]) < 1e-5:
                corr = 0.0
            else:
                corr = abs(pearsonr(orth_pos[idx], vals)[0])
            nodes = []
            if corr <= thr:
                nodes.append(dict(coords=centroid(idx, vals), mean=vals.mean(), lat=lat, marker=MARKERS[k % len(MARKERS)]))
            else:
                lab = gmm.predict(vals.reshape(-1, 1))
                for sel in (lab == high_c, lab != high_c):
                    if sel.any():
                        nodes.append(dict(coords=centroid(idx[sel], vals[sel]), mean=vals[sel].mean(), lat=lat,
                                          marker=MARKERS[k % len(MARKERS)]))
            pts[ci] = nodes
        trajectories[lat] = pts

    dominant = np.argmax(np.abs(all_norm_vals - 0.5), axis=1)
    cell_colors = cmap(all_norm_vals[np.arange(adata.n_obs), dominant])

    # offset coincident nodes
    by_cat = {}
    for ci in range(len(cats)):
        nodes = [p for lat in latents for p in trajectories[lat][ci]]
        done, clusters = set(), []
        for a in range(len(nodes)):
            if a in done:
                continue
            cl = [a]; done.add(a)
            for b in range(a + 1, len(nodes)):
                if b not in done and np.linalg.norm(nodes[a]["coords"] - nodes[b]["coords"]) < offset_threshold:
                    cl.append(b); done.add(b)
            clusters.append(cl)
        out = []
        for cl in clusters:
            group = sorted((nodes[a] for a in cl), key=lambda p: p["lat"])
            if len(group) > 1:
                centre = np.mean([p["coords"] for p in group], axis=0)
                for j, p in enumerate(group):
                    ang = 2 * np.pi * j / len(group)
                    p["coords"] = centre + offset_magnitude * np.array([np.cos(ang), np.sin(ang)])
            out.extend(group)
        by_cat[ci] = out

    fig, ax = _new_axes(ax, (9, 8))
    ax.scatter(xy[:, 0], xy[:, 1], c=cell_colors, s=18, alpha=background_alpha, linewidths=0)
    for ci in range(len(cats) - 1):
        for p2 in by_cat[ci + 1]:
            cands = [p1 for p1 in by_cat[ci] if p1["lat"] == p2["lat"]]
            if not cands:
                continue
            p1 = min(cands, key=lambda p: np.linalg.norm(p["coords"] - p2["coords"]))
            ax.plot([p1["coords"][0], p2["coords"][0]], [p1["coords"][1], p2["coords"][1]],
                    color=INK["primary"], linewidth=3.0, zorder=10)
            mid = (p1["coords"] + p2["coords"]) / 2
            if np.linalg.norm(p2["coords"] - p1["coords"]) > 0.1:
                ax.annotate("", xy=mid, xytext=p1["coords"], zorder=15,
                            arrowprops=dict(arrowstyle="->", color=INK["primary"], lw=1.5, shrinkA=0, shrinkB=0))
    for ci in range(len(cats)):
        for p in by_cat[ci]:
            v = norms[p["lat"]](p["mean"])
            rgba = cmap(v)
            bright = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            face = "white" if (0.4 < v < 0.6) or bright > 0.9 else rgba
            ax.scatter(*p["coords"], color=face, s=320, marker=p["marker"], zorder=30,
                       edgecolors=INK["primary"], linewidths=1.2)
    handles = [Line2D([], [], color=INK["primary"], marker=MARKERS[k % len(MARKERS)], linestyle="None",
                      markersize=9, label=f"LF {lat}") for k, lat in enumerate(latents)]
    ax.legend(handles=handles, loc="best", frameon=False, fontsize=8)
    ax.set_title(f"Latent trajectories across {groupby}")
    ax.axis("off")
    return _finish(fig, ax, show, save)


__all__ = [
    "CATEGORICAL", "sequential_cmap", "diverging_cmap", "category_colors",
    "training_loss", "elbow", "latent_embedding", "latent_grid", "latent_distance_heatmap",
    "latent_by_label", "latent_label_association", "top_genes", "go_enrichment",
    "group_trajectories", "gene_set_shap_embedding", "shap_beeswarm", "assign_branches",
    "group_shap_scores", "functional_group_heatmap", "latent_trajectory",
]

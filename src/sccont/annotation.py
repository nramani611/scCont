"""Stage E (cluster level): Gene Ontology enrichment of spatial latent clusters."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from .attribution import select_top_genes_by_zscore

DEFAULT_GO_SOURCES = ("GO:BP", "GO:CC", "GO:MF")


def run_go_enrichment(
    genes: Sequence[str],
    background: Sequence[str],
    organism: str = "hsapiens",
    sources: Sequence[str] = DEFAULT_GO_SOURCES,
) -> pd.DataFrame:
    """Query g:Profiler (network call) for GO enrichment of ``genes`` against ``background``.

    Requires the ``gprofiler-official`` package (``pip install "sccont[annotation]"``).
    """
    try:
        from gprofiler import GProfiler
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "GO enrichment needs gprofiler-official; install with `pip install \"sccont[annotation]\"`"
        ) from exc

    gp = GProfiler(return_dataframe=True)
    return gp.profile(
        organism=organism,
        query=list(genes),
        background=list(background),
        sources=list(sources),
        no_evidences=False,
    )


def cluster_gene_sets(
    shap_values: np.ndarray,
    gene_names: Sequence[str],
    groups: np.ndarray,
    zscore_threshold: float = 2.57,
) -> list[list[str]]:
    """Union of z-score-selected top genes over the latent features in each spatial cluster.

    Returns one gene list per cluster label ``0 .. groups.max()``.
    """
    groups = np.asarray(groups)
    gene_sets: list[list[str]] = []
    for g in range(int(groups.max()) + 1):
        latents = np.where(groups == g)[0]
        genes: set[str] = set()
        for latent_idx in latents:
            df = select_top_genes_by_zscore(shap_values, gene_names, int(latent_idx), zscore_threshold)
            genes.update(df["Gene"].tolist())
        gene_sets.append(sorted(genes))
    return gene_sets


def enrich_latent_clusters(
    shap_values: np.ndarray,
    gene_names: Sequence[str],
    groups: np.ndarray,
    zscore_threshold: float = 2.57,
    organism: str = "hsapiens",
    sources: Sequence[str] = DEFAULT_GO_SOURCES,
) -> list[pd.DataFrame]:
    """Run GO enrichment for every spatial latent cluster.

    Parameters
    ----------
    shap_values
        ``(n_cells, n_genes, latent_dim)`` from :func:`~sccont.attribution.compute_shap_values`.
    gene_names
        Gene names matching axis 1 of ``shap_values``; also used as the background.
    groups
        Cluster label per latent feature from
        :func:`~sccont.spatial.group_spatially_similar_latents`.

    Returns
    -------
    list of DataFrame
        One g:Profiler result table per cluster, in cluster-label order.
    """
    gene_sets = cluster_gene_sets(shap_values, gene_names, groups, zscore_threshold)
    return [run_go_enrichment(gs, list(gene_names), organism, sources) for gs in gene_sets]


def write_go_results_excel(results: Sequence[pd.DataFrame], path: str | Path) -> Path:
    """Write one sheet per cluster (``'Cluster 0'``, ``'Cluster 1'``, ...). Needs ``openpyxl``."""
    path = Path(path)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for i, df in enumerate(results):
            df.to_excel(writer, sheet_name=f"Cluster {i}", index=False)
    return path


def collect_enriched_genes(results: Sequence[pd.DataFrame]) -> set[str]:
    """All genes appearing in the ``intersections`` column of any enrichment result."""
    genes: set[str] = set()
    for df in results:
        if "intersections" in df.columns:
            for gene_group in df["intersections"].dropna():
                genes.update(gene_group)
    return genes

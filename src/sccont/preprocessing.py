"""Loading and normalising time-course scRNA-seq count matrices."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

# Cell-cycle marker lists (Tirosh et al. 2016) used for S / G2M scoring.
S_GENES = [
    "MCM5", "PCNA", "TYMS", "FEN1", "MCM2", "MCM4", "RRM1", "UNG", "GINS2", "MCM6",
    "CDCA7", "DTL", "PRIM1", "UHRF1", "HELLS", "RFC2", "RPA2", "NASP", "RAD51AP1",
    "GMNN", "WDR76", "SLBP", "CCNE2", "UBR7", "POLD3", "MSH2", "ATAD2", "RAD51", "RRM2",
    "CDC45", "CDC6", "EXO1", "TIPIN", "DSCC1", "BLM", "CASP8AP2", "USP1", "CLSPN", "POLA1",
    "CHAF1B", "BRIP1", "E2F8",
]

G2M_GENES = [
    "HMGB2", "CDK1", "NUSAP1", "UBE2C", "BIRC5", "TPX2", "TOP2A", "NDC80", "CKS2",
    "NUF2", "CKS1B", "MKI67", "TMPO", "CENPF", "TACC3", "FAM64A", "SMC4", "CCNB2",
    "CKAP2L", "CKAP2", "AURKB", "BUB1", "KIF11", "ANP32E", "TUBB4B", "GTSE1", "KIF20B",
    "HJURP", "CDCA3", "SGOL1", "CCNA2", "DSN1", "CDC20", "TTK", "CDC25C", "KIF2C",
    "RANGAP1", "NCAPD2", "DLGAP5", "CDCA2", "CDCA8", "ECT2", "KIF23", "HMMR", "AURKA",
    "PSRC1", "ANLN", "LBR", "CENPE", "CTCF", "NEK2", "G2E3", "GAS2L3", "CBX5",
    "CENPA",
]


def load_expression_matrix(
    path: str | Path,
    gene_col: str = "Gene.names",
    sep: str = "\t",
    **read_csv_kwargs,
) -> pd.DataFrame:
    """Read a genes x cells table where one column holds gene names.

    Works for the GSE147405 ``*_TimeCourse.tsv.gz`` and GSE200981 files shipped
    with the repository. Any extra keyword arguments go to :func:`pandas.read_csv`.
    """
    df = pd.read_csv(path, sep=sep, **read_csv_kwargs)
    if gene_col in df.columns:
        df.index = df[gene_col]
        df = df.drop(gene_col, axis=1)
    df.index.name = None
    return df


def timepoints_from_columns(columns: Sequence[str], sep: str = "_", index: int = 1) -> list[str]:
    """Extract timepoint labels from cell names such as ``V12_T3`` -> ``'T3'``."""
    return [str(c).split(sep)[index] for c in columns]


def _dense(X) -> np.ndarray:
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)


def preprocess(
    counts: pd.DataFrame,
    *,
    min_genes: int = 200,
    max_genes: int | None = 6000,
    max_pct_mito: float | None = 10.0,
    min_cells: int = 3,
    remove_doublets: bool = True,
    target_sum: float = 1e4,
    n_top_genes: int | None = 3000,
    regress_mito: bool = True,
    regress_counts: bool = True,
    regress_cell_cycle: bool = True,
    max_value: float | None = 10.0,
    plot_qc: bool = False,
    verbose: bool = True,
    return_anndata: bool = False,
) -> pd.DataFrame | ad.AnnData:
    """QC-filter, normalise, select HVGs, regress covariates and scale a count matrix.

    This reproduces the notebook's ``normalize = True`` block. For simulated data
    with no mitochondrial or cell-cycle signal use
    ``remove_doublets=False, regress_mito=False, regress_cell_cycle=False,
    max_pct_mito=None``.

    Parameters
    ----------
    counts
        Raw counts, genes x cells (as returned by :func:`load_expression_matrix`).
    min_genes, max_genes
        Keep cells with ``min_genes <= n_genes < max_genes``. ``None`` disables the bound.
    max_pct_mito
        Drop cells with a higher percentage of ``MT-`` counts. ``None`` disables.
    min_cells
        Keep genes detected in at least this many cells.
    remove_doublets
        Run Scrublet and drop predicted doublets (needs ``pip install "sccont[preprocess]"``).
    target_sum
        ``scanpy.pp.normalize_total`` target.
    n_top_genes
        Number of highly variable genes to keep (``flavor='seurat'``). ``None`` keeps all.
    regress_mito, regress_counts, regress_cell_cycle
        Covariates for ``scanpy.pp.regress_out`` (``percent_mito``, ``n_counts``,
        ``S_score``/``G2M_score``).
    max_value
        Clip value for ``scanpy.pp.scale``.
    plot_qc
        Show the scanpy QC violin plot and the Scrublet histogram.
    return_anndata
        Return the processed ``AnnData`` (cells x genes) instead of a DataFrame.

    Returns
    -------
    DataFrame (genes x cells) of scaled expression, or AnnData if ``return_anndata``.
    """
    adata = ad.AnnData(counts.T.astype(np.float32))  # cells x genes
    adata.var_names_make_unique()

    X = adata.X
    adata.obs["n_genes"] = np.asarray((X > 0).sum(axis=1)).ravel()
    adata.obs["n_counts"] = np.asarray(X.sum(axis=1)).ravel()

    mt_mask = np.array([str(g).upper().startswith("MT-") for g in adata.var_names])
    mito_counts = _dense(adata[:, mt_mask].X).sum(axis=1) if mt_mask.any() else np.zeros(adata.n_obs)
    with np.errstate(divide="ignore", invalid="ignore"):
        adata.obs["percent_mito"] = np.nan_to_num(mito_counts / adata.obs["n_counts"].values * 100)

    if plot_qc:
        sc.pl.violin(adata, ["n_genes", "n_counts", "percent_mito"], jitter=0.4, multi_panel=True)

    # Cell filters
    if min_genes:
        sc.pp.filter_cells(adata, min_genes=min_genes)
    if max_genes is not None:
        adata = adata[adata.obs["n_genes"] < max_genes, :].copy()
    if max_pct_mito is not None:
        adata = adata[adata.obs["percent_mito"] < max_pct_mito, :].copy()
    if verbose:
        print(f"Cells after filtering: {adata.n_obs}")

    if min_cells:
        sc.pp.filter_genes(adata, min_cells=min_cells)
    if verbose:
        print(f"Genes after filtering: {adata.n_vars}")

    if remove_doublets:
        try:
            import scrublet as scr
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Doublet removal needs scrublet; install with `pip install \"sccont[preprocess]\"` "
                "or pass remove_doublets=False"
            ) from exc
        scrub = scr.Scrublet(_dense(adata.X).copy())
        doublet_scores, predicted = scrub.scrub_doublets(verbose=verbose)
        if predicted is None:  # Scrublet could not find a threshold
            predicted = np.zeros(adata.n_obs, dtype=bool)
        adata.obs["doublet_score"] = doublet_scores
        adata.obs["predicted_doublet"] = predicted
        if plot_qc:
            import matplotlib.pyplot as plt

            scrub.plot_histogram()
            plt.show()
        adata = adata[~adata.obs["predicted_doublet"].astype(bool), :].copy()
        if verbose:
            print(f"Cells after doublet removal: {adata.n_obs}")

    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)

    if n_top_genes is not None and n_top_genes < adata.n_vars:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, subset=True, flavor="seurat")

    regress = []
    if regress_mito:
        regress.append("percent_mito")
    if regress_counts:
        regress.append("n_counts")
    if regress_cell_cycle:
        sc.tl.score_genes_cell_cycle(adata, s_genes=S_GENES, g2m_genes=G2M_GENES)
        regress += ["S_score", "G2M_score"]
    if regress:
        sc.pp.regress_out(adata, regress)

    sc.pp.scale(adata, max_value=max_value)
    if verbose:
        print(f"Final AnnData shape: {adata.shape} (cells x genes)")

    if return_anndata:
        return adata
    return pd.DataFrame(_dense(adata.X).T, index=adata.var_names, columns=adata.obs_names)

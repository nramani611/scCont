"""Loading and normalising scRNA-seq count matrices into ``AnnData``."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from .data import _dense

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
    Convert the result with :func:`to_anndata`.
    """
    df = pd.read_csv(path, sep=sep, **read_csv_kwargs)
    if gene_col in df.columns:
        df.index = df[gene_col]
        df = df.drop(gene_col, axis=1)
    df.index.name = None
    return df


def timepoints_from_columns(columns: Sequence[str], sep: str = "_", index: int = 1) -> list[str]:
    """Extract labels from cell names such as ``V12_T3`` -> ``'T3'``.

    Only a convenience for the naming convention of the datasets in this
    repository; any per-cell label can instead be placed in ``adata.obs``.
    """
    return [str(c).split(sep)[index] for c in columns]


def to_anndata(
    counts: pd.DataFrame,
    obs: pd.DataFrame | None = None,
    cells_by_genes: bool = False,
    **obs_columns,
) -> ad.AnnData:
    """Convert an expression table to an ``AnnData`` (cells x genes).

    Parameters
    ----------
    counts
        Genes x cells DataFrame (index = genes, columns = cells) as produced by
        :func:`load_expression_matrix`. Pass ``cells_by_genes=True`` if it is
        already cells x genes.
    obs
        Optional per-cell annotation table indexed by cell name; aligned to the
        cells in ``counts`` (extra rows dropped, missing cells raise).
    **obs_columns
        Extra per-cell labels given as ``name=sequence`` (length ``n_cells``),
        e.g. ``timepoint=[...]``, ``condition=[...]``.

    Returns
    -------
    AnnData
        Labels are in ``adata.obs``; scCont never uses them for training.
    """
    df = counts if cells_by_genes else counts.T
    adata = ad.AnnData(np.asarray(df.to_numpy(), dtype=np.float32))
    adata.obs_names = df.index.astype(str)
    adata.var_names = df.columns.astype(str)
    adata.var_names_make_unique()

    if obs is not None:
        missing = adata.obs_names.difference(obs.index.astype(str))
        if len(missing):
            raise ValueError(f"{len(missing)} cells missing from obs, e.g. {list(missing[:3])}")
        aligned = obs.copy()
        aligned.index = aligned.index.astype(str)
        for col in aligned.columns:
            adata.obs[col] = aligned.loc[adata.obs_names, col].to_numpy()
    for name, values in obs_columns.items():
        values = np.asarray(values)
        if len(values) != adata.n_obs:
            raise ValueError(f"obs column '{name}' has {len(values)} values for {adata.n_obs} cells")
        adata.obs[name] = values
    return adata


def preprocess(
    data: ad.AnnData | pd.DataFrame,
    *,
    min_genes: int = 200,
    max_genes: int | None = 6000,
    max_pct_mito: float | None = 10.0,
    min_cells: int = 3,
    remove_doublets: bool = True,
    normalize_total: bool = True,
    target_sum: float = 1e4,
    log1p: bool = True,
    n_top_genes: int | None = 3000,
    regress_mito: bool = True,
    regress_counts: bool = True,
    regress_cell_cycle: bool = True,
    scale: bool = True,
    max_value: float | None = 10.0,
    plot_qc: bool = False,
    verbose: bool = True,
    as_dataframe: bool = False,
) -> ad.AnnData | pd.DataFrame:
    """QC-filter, normalise, select HVGs, regress covariates and scale a count matrix.

    Every step has its own switch; the defaults reproduce the notebook's
    ``normalize = True`` block. Set a step's flag to ``False`` (or its threshold
    to ``None``/``0``) to skip it. For simulated data with no mitochondrial or
    cell-cycle signal use ``remove_doublets=False, regress_mito=False,
    regress_cell_cycle=False, max_pct_mito=None``.

    Parameters
    ----------
    data
        ``AnnData`` of raw counts (cells x genes; any ``obs`` labels are kept), or
        a genes x cells DataFrame (converted with :func:`to_anndata`).
    min_genes, max_genes
        Keep cells with ``min_genes <= n_genes < max_genes``. ``0``/``None`` disables the bound.
    max_pct_mito
        Drop cells with a higher percentage of ``MT-`` counts. ``None`` disables.
    min_cells
        Keep genes detected in at least this many cells. ``0`` disables.
    remove_doublets
        Run Scrublet and drop predicted doublets.
    normalize_total, target_sum
        Library-size normalisation with ``scanpy.pp.normalize_total``.
    log1p
        Apply ``scanpy.pp.log1p``.
    n_top_genes
        Number of highly variable genes to keep (``flavor='seurat'``). ``None`` keeps all.
    regress_mito, regress_counts, regress_cell_cycle
        Covariates for ``scanpy.pp.regress_out`` (``percent_mito``, ``n_counts``,
        ``S_score``/``G2M_score``). All ``False`` skips the regression.
    scale, max_value
        Z-score genes with ``scanpy.pp.scale``, clipping at ``max_value`` (``None`` = no clip).
    plot_qc
        Show the scanpy QC violin plot and the Scrublet histogram.
    verbose
        Print cell/gene counts after each filtering step.
    as_dataframe
        Return a genes x cells DataFrame instead of the ``AnnData`` (legacy).

    Returns
    -------
    AnnData (cells x genes) with QC metrics added to ``obs``
    (``n_genes``, ``n_counts``, ``percent_mito``, ``doublet_score``, ``S_score``, ...),
    or a genes x cells DataFrame if ``as_dataframe``.
    """
    if isinstance(data, ad.AnnData):
        adata = data.copy()
        adata.X = np.asarray(_dense(adata.X), dtype=np.float32)
    else:
        adata = to_anndata(data)
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
        import scrublet as scr

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

    if normalize_total:
        sc.pp.normalize_total(adata, target_sum=target_sum)
    if log1p:
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

    if scale:
        sc.pp.scale(adata, max_value=max_value)
    if verbose:
        print(f"Final AnnData shape: {adata.shape} (cells x genes)")

    if as_dataframe:
        return pd.DataFrame(_dense(adata.X).T, index=adata.var_names, columns=adata.obs_names)
    return adata

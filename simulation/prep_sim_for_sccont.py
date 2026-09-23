#!/usr/bin/env python3
"""
prep_sim_for_sccont.py — run the scCont preprocessing pipeline on simulate_sergio.py
output and emit a matrix in the same format as the real-data pipeline, with the
ground truth re-aligned to the surviving cells and genes.

The re-alignment is the point: QC filtering drops cells and genes, so the membership
and activity tables written by simulate_sergio.py no longer line up with the matrix
scCont sees. Scoring against unaligned ground truth silently reports garbage.

Steps kept from the real pipeline: gene filter, normalize_total(1e4), log1p, HVG,
regress_out(n_counts), scale(max_value=10).
Steps dropped (nothing in the simulation to act on): percent_mito filter and
regression, cell-cycle scoring and regression, Scrublet.

Usage:
  python prep_sim_for_sccont.py --indir runs/hvg3000 --outdir runs/hvg3000/prepped
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--indir", required=True, help="directory containing sim.h5ad")
    p.add_argument("--outdir", default=None, help="default: <indir>/prepped")
    p.add_argument("--min-genes", type=int, default=30,
                   help="scaled equivalent of min_genes=200 on a ~20k-gene real matrix; "
                        "set 0 to disable")
    p.add_argument("--max-genes", type=int, default=0, help="0 disables the upper filter")
    p.add_argument("--min-cells", type=int, default=3)
    p.add_argument("--n-top-genes", type=int, default=3000)
    p.add_argument("--target-sum", type=float, default=1e4)
    p.add_argument("--max-value", type=float, default=10.0)
    p.add_argument("--no-regress", action="store_true",
                   help="skip regress_out(n_counts)")
    args = p.parse_args()

    indir = Path(args.indir)
    outdir = Path(args.outdir or indir / "prepped")
    outdir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(indir / "sim.h5ad")
    adata.X = np.asarray(adata.X, dtype=np.float64)
    n0, g0 = adata.shape

    adata.obs["n_genes"] = (adata.X > 0).sum(1)
    adata.obs["n_counts"] = adata.X.sum(1)

    if args.min_genes > 0:
        sc.pp.filter_cells(adata, min_genes=args.min_genes)
    if args.max_genes > 0:
        adata = adata[adata.obs["n_genes"] < args.max_genes].copy()
    if args.min_cells > 0:
        sc.pp.filter_genes(adata, min_cells=args.min_cells)
    print(f"[qc]  {n0} -> {adata.n_obs} cells | {g0} -> {adata.n_vars} genes after filtering")

    sc.pp.normalize_total(adata, target_sum=args.target_sum)
    sc.pp.log1p(adata)
    if args.n_top_genes and args.n_top_genes < adata.n_vars:
        sc.pp.highly_variable_genes(adata, n_top_genes=args.n_top_genes,
                                    subset=True, flavor="seurat")
        print(f"[hvg] {adata.n_vars} genes retained "
              f"({int((~adata.var.is_background).sum())} program, "
              f"{int(adata.var.is_background.sum())} background)")
    else:
        print(f"[hvg] skipped: only {adata.n_vars} features present, "
              f"n_top_genes={args.n_top_genes} would be a no-op")

    if not args.no_regress:
        sc.pp.regress_out(adata, ["n_counts"])
    sc.pp.scale(adata, max_value=args.max_value)

    X = np.asarray(adata.X)
    # same orientation as the real pipeline: genes x cells
    pd.DataFrame(X.T, index=adata.var_names, columns=adata.obs_names).to_csv(
        outdir / "expression_matrix_normalized.csv")

    # ground truth, re-aligned to what survived
    mods = [f"prog_{m}" for m in range(adata.varm["gt_membership_signed"].shape[1])]
    pd.DataFrame(adata.varm["gt_membership_signed"],
                 index=adata.var_names, columns=mods).to_csv(
        outdir / "gt_membership_signed.csv")
    pd.DataFrame(adata.varm["gt_membership_binary"],
                 index=adata.var_names, columns=mods).to_csv(
        outdir / "gt_membership_binary.csv")
    pd.DataFrame(adata.obsm["gt_activity"], index=adata.obs_names, columns=mods).to_csv(
        outdir / "gt_activity.csv")
    keep = ["cell_type", "bin", "n_genes", "n_counts"]
    keep += [c for c in ("pseudotime", "branch") if c in adata.obs]
    adata.obs[keep].to_csv(outdir / "cell_metadata.csv")
    adata.var[["is_background", "is_master_regulator", "module_primary",
               "n_modules"]].to_csv(outdir / "gene_metadata.csv")
    adata.write_h5ad(outdir / "prepped.h5ad")

    print(f"[out] {outdir}/expression_matrix_normalized.csv  "
          f"({adata.n_vars} genes x {adata.n_obs} cells)")
    print(f"[out] ground truth re-aligned: gt_membership_*.csv, gt_activity.csv")


if __name__ == "__main__":
    main()

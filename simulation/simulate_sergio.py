#!/usr/bin/env python3
"""
simulate_sergio.py — SERGIO simulations with explicit ground-truth gene programs,
built for benchmarking scCont's latent -> program attribution.

Why this wrapper exists
-----------------------
Stock SERGIO gives you a GRN and a count matrix. What a scCont simulation study
needs is the *program-level* ground truth: for every latent dimension you claim
to interpret, a known set of genes (with direction) and a known per-cell activity
to score SHAP rankings against. This script constructs a modular GRN, runs SERGIO,
and emits that ground truth alongside the counts.

Emits (into --outdir):
  sim.h5ad                    X = UMI counts; layers['clean'] = noise-free
                              obs: cell_type, prog_*, (pseudotime, branch in dynamics mode)
                              var: is_master_regulator, module_primary, n_regulators
  gt_membership_signed.csv    genes x modules, signed weights in [-1, 1]
  gt_membership_binary.csv    genes x modules, {0,1}
  gt_activity.csv             cells x modules, ground-truth program activity
                              (in dynamics mode this is the *terminal* activity of the
                              cell's lineage state; obs['pseudotime'] carries where the
                              cell sits along the transition into it)
  gt_grn.csv                  regulator, target, K, coop_state, sign
  params.json                 every parameter + seed, for reproducibility
  sergio_inputs/              the targets/regs/bMat files handed to SERGIO

Usage
-----
  # steady-state, 8 programs, graded activity across 6 cell types
  python simulate_sergio.py --outdir runs/ss_seed0 --seed 0

  # bifurcating differentiation (continuous transitions; gives ground-truth pseudotime)
  python simulate_sergio.py --outdir runs/dyn_bif --mode dynamics --backbone bifurcating

  # ~3000 features to match a real HVG set: 800 simulated + 2200 distractors
  # (SERGIO integration is ~O(genes^1.7); padding costs seconds, simulating does not)
  python simulate_sergio.py --outdir runs/hvg3000 --n-modules 16 \
      --n-targets-per-module 49 --n-shared-targets 0 --n-background-genes 2200

  # negative control: no program structure at all
  python simulate_sergio.py --outdir runs/null_seed0 --null

  # noise ladder for the k-sensitivity sweep
  for p in 0 60 74 82 88; do
    python simulate_sergio.py --outdir runs/drop$p --dropout-percentile $p --seed 0
  done

SERGIO is fetched from github.com/PayamDiba/SERGIO on first use (no working PyPI
release; the package named `sergio` on PyPI is unrelated). Requires numpy, pandas,
scipy, networkx, anndata.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SERGIO_REPO = "https://github.com/PayamDiba/SERGIO.git"


# --------------------------------------------------------------------------- #
# SERGIO import
# --------------------------------------------------------------------------- #
def ensure_sergio(sergio_path: str | None = None, cache_dir: str | None = None):
    """Import SERGIO, cloning it if needed, and patch numpy>=1.24 incompatibilities.

    SERGIO's source still uses np.int / np.float, removed in numpy 1.24. We restore
    them as module attributes rather than editing the vendored source, so the clone
    stays a pristine checkout you can point a reviewer at.
    """
    for name, builtin in (("int", int), ("float", float), ("bool", bool),
                          ("object", object), ("str", str)):
        if name not in np.__dict__:      # __dict__, not hasattr: avoids numpy's warning shim
            setattr(np, name, builtin)

    candidates = []
    if sergio_path:
        candidates.append(Path(sergio_path))
    cache = Path(cache_dir or os.environ.get("SERGIO_HOME",
                                             Path.home() / ".cache" / "SERGIO"))
    candidates.append(cache)

    for cand in candidates:
        if (cand / "SERGIO" / "sergio.py").exists():
            sys.path.insert(0, str(cand))
            from SERGIO.sergio import sergio  # noqa: E402
            return sergio

    cache.parent.mkdir(parents=True, exist_ok=True)
    print(f"[setup] cloning SERGIO into {cache}", file=sys.stderr)
    subprocess.run(["git", "clone", "--depth", "1", DEFAULT_SERGIO_REPO, str(cache)],
                   check=True)
    sys.path.insert(0, str(cache))
    from SERGIO.sergio import sergio  # noqa: E402
    return sergio


# --------------------------------------------------------------------------- #
# Ground-truth GRN construction
# --------------------------------------------------------------------------- #
@dataclass
class GRN:
    n_genes: int
    n_modules: int
    mr_ids: list = field(default_factory=list)
    edges: list = field(default_factory=list)      # (reg, target, K, coop)
    membership: np.ndarray = None                  # n_genes x n_modules, signed
    gene_layer: np.ndarray = None                  # 0 = MR, 1 = direct, 2 = cascade


def build_modular_grn(rng: np.random.Generator,
                      n_modules: int,
                      n_targets_per_module: int,
                      n_shared_targets: int,
                      frac_repressive: float,
                      frac_cascade: float,
                      k_range=(1.0, 5.0),
                      coop_state: float = 2.0) -> GRN:
    """A modular DAG: one master regulator per module -> direct targets -> cascade targets.

    Design choices that matter for the benchmark:
      * Repressive edges (negative K in SERGIO) give *signed* program membership, so
        the signed-vs-absolute-SHAP question is testable rather than assumed.
      * Shared targets belong to two modules, so programs overlap the way real ones do
        and a method cannot win by assuming disjoint gene sets.
      * Cascade targets sit two hops from the MR, so program membership is not a
        one-layer, trivially-recoverable pattern.
    """
    mr_ids = list(range(n_modules))
    next_id = n_modules

    direct, cascade = {}, {}
    for m in range(n_modules):
        n_casc = int(round(frac_cascade * n_targets_per_module))
        n_dir = n_targets_per_module - n_casc
        if n_dir < 1:
            raise ValueError("frac_cascade leaves no direct targets in a module")
        direct[m] = list(range(next_id, next_id + n_dir))
        next_id += n_dir
        cascade[m] = list(range(next_id, next_id + n_casc))
        next_id += n_casc

    shared_ids = list(range(next_id, next_id + n_shared_targets))
    next_id += n_shared_targets
    n_genes = next_id

    membership = np.zeros((n_genes, n_modules), dtype=float)
    gene_layer = np.zeros(n_genes, dtype=int)
    edges = []

    def draw_k():
        mag = rng.uniform(*k_range)
        sign = -1.0 if rng.random() < frac_repressive else 1.0
        return sign * mag

    for m in range(n_modules):
        membership[mr_ids[m], m] = 1.0
        # MR -> direct targets
        for t in direct[m]:
            k = draw_k()
            edges.append((mr_ids[m], t, k, coop_state))
            membership[t, m] = np.sign(k) * abs(k) / k_range[1]
            gene_layer[t] = 1
        # direct target -> cascade target (two hops from the MR)
        for t in cascade[m]:
            parent = int(rng.choice(direct[m]))
            k = draw_k()
            edges.append((parent, t, k, coop_state))
            # effective direction relative to the module = product of signs along the path
            membership[t, m] = np.sign(k) * np.sign(membership[parent, m]) * abs(k) / k_range[1]
            gene_layer[t] = 2

    # shared targets: co-regulated by two distinct modules
    for t in shared_ids:
        mods = rng.choice(n_modules, size=2, replace=False)
        for m in mods:
            k = draw_k()
            edges.append((mr_ids[int(m)], t, k, coop_state))
            membership[t, int(m)] = np.sign(k) * abs(k) / k_range[1]
        gene_layer[t] = 1

    return GRN(n_genes=n_genes, n_modules=n_modules, mr_ids=mr_ids, edges=edges,
               membership=membership, gene_layer=gene_layer)


def make_activity(rng: np.random.Generator, n_bins: int, n_modules: int,
                  scheme: str, null: bool) -> np.ndarray:
    """Program activity per cell type (bins x modules), in [0, 1]."""
    if null:
        # every bin identical -> one homogeneous population, no program structure.
        return np.full((n_bins, n_modules), 0.5)
    if scheme == "binary":
        A = (rng.random((n_bins, n_modules)) < 0.45).astype(float)
        for m in range(n_modules):  # never all-on or all-off: that module carries no signal
            if A[:, m].sum() in (0, n_bins):
                A[rng.integers(n_bins), m] = 1.0 - A[0, m]
        return A
    if scheme == "graded":
        return rng.uniform(0.05, 1.0, size=(n_bins, n_modules))
    raise ValueError(f"unknown activity scheme: {scheme}")


#: how many children each non-leaf state gets, per backbone. "bifurcating" is a
#: binary tree and stays the default so every earlier run reproduces byte for byte.
BACKBONE_ARITY = {"linear": 1, "bifurcating": 2, "trifurcating": 3}


def parent_of(j: int, backbone: str) -> int:
    """Parent of lineage state j. Single definition, used by both the bifurcation
    matrix and the pseudotime depth calculation -- these were separate copies of
    the same expression before, which is exactly how a tree and its depths drift
    apart when a new backbone is added."""
    a = BACKBONE_ARITY[backbone]
    return j - 1 if a == 1 else (j - 1) // a


def make_bifurcation_matrix(n_bins: int, backbone: str, rate: float) -> np.ndarray:
    """bMat[i, j] > 0 means bin i differentiates into bin j. Each bin gets <= 1 parent."""
    b = np.zeros((n_bins, n_bins))
    for j in range(1, n_bins):
        b[parent_of(j, backbone), j] = rate
    return b


def lineage_depths(n_bins: int, backbone: str) -> np.ndarray:
    """Depth of every state in the lineage tree (root = 0)."""
    d = np.zeros(n_bins)
    for j in range(1, n_bins):
        d[j] = d[parent_of(j, backbone)] + 1
    return d


def write_sergio_inputs(grn: GRN, activity: np.ndarray, outdir: Path,
                        rate_low: float, rate_high: float) -> tuple[Path, Path]:
    """SERGIO input files.

    targets.txt: target, n_regs, reg1..regN, K1..KN, coop1..coopN
    regs.txt:    master_regulator, production_rate_bin1 .. production_rate_binB
    """
    outdir.mkdir(parents=True, exist_ok=True)
    by_target: dict[int, list] = {}
    for reg, tgt, k, coop in grn.edges:
        by_target.setdefault(tgt, []).append((reg, k, coop))

    tpath = outdir / "targets.txt"
    with open(tpath, "w") as fh:
        for tgt in sorted(by_target):
            regs = by_target[tgt]
            row = ([float(tgt), float(len(regs))]
                   + [float(r) for r, _, _ in regs]
                   + [float(k) for _, k, _ in regs]
                   + [float(c) for _, _, c in regs])
            fh.write(",".join(f"{v}" for v in row) + "\n")

    rates = rate_low + activity * (rate_high - rate_low)   # bins x modules
    rpath = outdir / "regs.txt"
    with open(rpath, "w") as fh:
        for m, mr in enumerate(grn.mr_ids):
            fh.write(",".join([f"{float(mr)}"] + [f"{r}" for r in rates[:, m]]) + "\n")
    return tpath, rpath


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #
def get_expressions_dynamics_with_steps(sim, rng: np.random.Generator):
    """Reimplementation of sim.getExpressions_dynamics() that also returns the
    simulation step each cell was sampled from -- i.e. ground-truth pseudotime,
    which stock SERGIO discards."""
    n_bins, n_genes, n_sc = sim.nBins_, sim.nGenes_, sim.nSC_
    U = np.zeros((n_bins, n_genes, n_sc))
    S = np.zeros((n_bins, n_genes, n_sc))
    steps = np.zeros((n_bins, n_sc), dtype=int)
    n_steps = np.zeros(n_bins, dtype=int)
    for bi in range(n_bins):
        conc = sim.binDict[bi][0].Conc
        n_sim_steps = len(conc[0]) * len(conc)
        if n_sc > n_sim_steps:
            raise RuntimeError(
                f"bin {bi}: asked for {n_sc} cells but only {n_sim_steps} simulated "
                f"steps exist. Lower --n-cells-per-type or raise --sampling-state.")
        # Conc is a list of INDEPENDENT trajectories, each n_traj_steps long, and
        # np.concatenate lays them out path-after-path -- so a flat index packs
        # (trajectory * n_traj_steps + timestep). Sampling from the whole pool is
        # correct; keeping the packed index as a time coordinate is not, because
        # the trajectory a cell came from is an arbitrary label, not a clock.
        n_traj_steps = len(conc[0])
        picked = rng.choice(n_sim_steps, size=n_sc, replace=False)
        for g in range(n_genes):
            U[bi, g, :] = np.take(np.concatenate(sim.binDict[bi][g].Conc, axis=0), picked)
            S[bi, g, :] = np.take(np.concatenate(sim.binDict[bi][g].Conc_S, axis=0), picked)
        steps[bi] = picked % n_traj_steps      # position WITHIN its trajectory
        n_steps[bi] = n_traj_steps
    return U, S, steps, n_steps


def add_technical_noise(sim, expr, args, dynamics=False):
    """SERGIO's technical-noise cascade: outliers -> library size -> dropout -> UMIs.
    Set --dropout-percentile 0 to skip dropout (clean end of the noise ladder)."""
    if dynamics:
        U, S = expr
        U, S = sim.outlier_effect_dynamics(U, S, outlier_prob=args.outlier_prob,
                                           mean=0.8, scale=1)
        _, U, S = sim.lib_size_effect_dynamics(U, S, mean=args.lib_mean, scale=args.lib_scale)
        if args.dropout_percentile > 0:
            bu, bs = sim.dropout_indicator_dynamics(U, S, shape=args.dropout_shape,
                                                    percentile=args.dropout_percentile)
            U, S = np.multiply(bu, U), np.multiply(bs, S)
        return sim.convert_to_UMIcounts_dynamics(U, S)

    e = sim.outlier_effect(expr, outlier_prob=args.outlier_prob, mean=0.8, scale=1)
    _, e = sim.lib_size_effect(e, mean=args.lib_mean, scale=args.lib_scale)
    if args.dropout_percentile > 0:
        binary = sim.dropout_indicator(e, shape=args.dropout_shape,
                                       percentile=args.dropout_percentile)
        e = np.multiply(binary, e)
    return sim.convert_to_UMIcounts(e)


def run_steady_state(sergio_cls, grn, tpath, rpath, args):
    sim = sergio_cls(number_genes=grn.n_genes, number_bins=args.n_bins,
                     number_sc=args.n_cells_per_type, noise_params=args.noise_params,
                     decays=args.decay, sampling_state=args.sampling_state,
                     noise_type=args.noise_type)
    sim.build_graph(input_file_taregts=str(tpath), input_file_regs=str(rpath),
                    shared_coop_state=args.coop_state)
    sim.simulate()
    clean = sim.getExpressions()                       # bins x genes x cells
    counts = add_technical_noise(sim, clean, args)
    return (np.concatenate(counts, axis=1).T,          # cells x genes
            np.concatenate(clean, axis=1).T,
            None, None)


def run_dynamics(sergio_cls, grn, tpath, rpath, args, rng):
    bmat = make_bifurcation_matrix(args.n_bins, args.backbone, args.bifurcation_rate)
    sim = sergio_cls(number_genes=grn.n_genes, number_bins=args.n_bins,
                     number_sc=args.n_cells_per_type, noise_params=args.noise_params,
                     decays=args.decay, sampling_state=args.sampling_state,
                     noise_type=args.noise_type, dynamics=True, bifurcation_matrix=bmat,
                     noise_params_splice=args.noise_params_splice, splice_ratio=args.splice_ratio)
    sim.build_graph(input_file_taregts=str(tpath), input_file_regs=str(rpath),
                    shared_coop_state=args.coop_state)
    sim.simulate_dynamics()
    U, S, steps, n_steps = get_expressions_dynamics_with_steps(sim, rng)
    cU, cS = add_technical_noise(sim, (U, S), args, dynamics=True)

    # pseudotime: depth of the bin in the lineage + fractional progress within it
    depth = lineage_depths(args.n_bins, args.backbone)
    pt = np.concatenate([depth[b] + steps[b] / max(n_steps[b] - 1, 1)
                         for b in range(args.n_bins)])
    pt = pt / pt.max() if pt.max() > 0 else pt
    return (np.concatenate(cS, axis=1).T, np.concatenate(S, axis=1).T,
            np.concatenate(cU, axis=1).T, pt), bmat


def add_background_genes(mats, n_bg, rng, n_strata=10):
    """Append genes with no program structure.

    Each background gene is a copy of a randomly chosen simulated gene whose values
    are permuted across cells *within library-size strata*: the gene's marginal
    distribution and its association with sequencing depth survive, all cell-type and
    program structure is destroyed. These are distractors -- a background gene showing
    up in a latent's top SHAP genes is a false positive, which gives you a per-latent
    FPR without needing a separate null run.
    """
    counts = mats[0]
    if n_bg <= 0:
        return mats, np.zeros(counts.shape[1], dtype=bool)

    depth = counts.sum(1)
    n_strata = min(n_strata, len(np.unique(depth)))
    strata = (pd.qcut(depth, n_strata, labels=False, duplicates="drop")
              if n_strata > 1 else np.zeros(len(depth), dtype=int))
    templates = rng.integers(0, counts.shape[1], size=n_bg)

    out = []
    perms = {}   # (gene j, stratum) -> permutation, shared across layers
    for mat in mats:
        if mat is None:
            out.append(None)
            continue
        bg = np.zeros((mat.shape[0], n_bg), dtype=mat.dtype)
        for j, t in enumerate(templates):
            for s in np.unique(strata):
                idx = np.where(strata == s)[0]
                key = (j, s)
                if key not in perms:
                    perms[key] = rng.permutation(idx)
                bg[idx, j] = mat[perms[key], t]
        out.append(np.hstack([mat, bg]))

    mask = np.concatenate([np.zeros(counts.shape[1], dtype=bool),
                           np.ones(n_bg, dtype=bool)])
    return tuple(out), mask


def to_anndata(counts, clean, unspliced, pseudotime, grn, activity, args, bg_mask=None):
    import anndata as ad
    n_cells = counts.shape[0]
    bins = np.repeat(np.arange(args.n_bins), args.n_cells_per_type)

    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_cells)])
    obs["cell_type"] = pd.Categorical([f"type_{b}" for b in bins])
    obs["bin"] = bins
    for m in range(grn.n_modules):
        obs[f"prog_{m}"] = activity[bins, m]           # ground-truth program activity
    if pseudotime is not None:
        obs["pseudotime"] = pseudotime
        obs["branch"] = pd.Categorical([f"branch_{b}" for b in bins])

    n_bg = 0 if bg_mask is None else int(bg_mask.sum())
    membership = grn.membership
    gene_layer = grn.gene_layer
    names = [f"g{i}" for i in range(grn.n_genes)]
    if n_bg:
        membership = np.vstack([membership, np.zeros((n_bg, grn.n_modules))])
        gene_layer = np.concatenate([gene_layer, np.full(n_bg, -1)])
        names += [f"bg{i}" for i in range(n_bg)]

    primary = np.where(np.abs(membership).max(axis=1) > 0,
                       np.abs(membership).argmax(axis=1), -1)
    n_regs = np.zeros(len(names), dtype=int)
    for _, tgt, _, _ in grn.edges:
        n_regs[tgt] += 1
    var = pd.DataFrame(index=names)
    var["gene_id"] = np.arange(len(names))
    var["is_master_regulator"] = np.isin(np.arange(len(names)), grn.mr_ids)
    var["is_background"] = np.arange(len(names)) >= grn.n_genes
    var["module_primary"] = primary
    var["n_modules"] = (np.abs(membership) > 0).sum(axis=1)
    var["n_regulators"] = n_regs
    var["layer"] = gene_layer

    adata = ad.AnnData(X=counts.astype(np.float32), obs=obs, var=var)
    adata.layers["clean"] = clean.astype(np.float32)
    if unspliced is not None:
        adata.layers["unspliced"] = unspliced.astype(np.float32)
    adata.varm["gt_membership_signed"] = membership
    adata.varm["gt_membership_binary"] = (np.abs(membership) > 0).astype(np.int8)
    adata.obsm["gt_activity"] = activity[bins]
    adata.uns["params"] = {k: (v if not isinstance(v, Path) else str(v))
                           for k, v in vars(args).items()}
    return adata


# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mode", choices=["steady", "dynamics"], default="steady")

    g = p.add_argument_group("GRN / programs")
    g.add_argument("--n-modules", type=int, default=8, help="ground-truth gene programs")
    g.add_argument("--n-targets-per-module", type=int, default=15)
    g.add_argument("--n-shared-targets", type=int, default=20,
                   help="genes belonging to two programs")
    g.add_argument("--frac-repressive", type=float, default=0.3,
                   help="fraction of edges that repress (gives signed ground truth)")
    g.add_argument("--frac-cascade", type=float, default=0.25,
                   help="fraction of a module's targets sitting two hops from the MR")
    g.add_argument("--coop-state", type=float, default=2.0)

    c = p.add_argument_group("cells")
    c.add_argument("--n-bins", type=int, default=6, help="cell types (steady) / lineage states (dynamics)")
    c.add_argument("--n-cells-per-type", type=int, default=300)
    c.add_argument("--activity", choices=["binary", "graded"], default="graded")
    c.add_argument("--null", action="store_true",
                   help="negative control: identical program activity in every bin")
    c.add_argument("--n-background-genes", type=int, default=0,
                   help="structureless distractor genes appended after simulation "
                        "(marginals and depth-association preserved, cell-type "
                        "structure destroyed). Use to reach a realistic feature count "
                        "without paying SERGIO's per-gene integration cost; any that "
                        "surface in a latent's top SHAP genes are false positives")
    c.add_argument("--rate-low", type=float, default=0.2)
    c.add_argument("--rate-high", type=float, default=3.5)

    d = p.add_argument_group("dynamics")
    d.add_argument("--backbone", choices=["linear", "bifurcating", "trifurcating"],
                   default="bifurcating",
                   help="linear = coherent transition; bifurcating = binary tree "
                        "(2 children per state); trifurcating = ternary tree "
                        "(3 children per state). Complete trees need "
                        "--n-bins = 1+a+a^2+... for arity a: 7 or 15 for binary, "
                        "13 or 40 for ternary. Other values give a ragged last "
                        "level, which is legal but makes the branches unequal.")
    d.add_argument("--bifurcation-rate", type=float, default=0.08)
    d.add_argument("--noise-params-splice", type=float, default=0.07)
    d.add_argument("--splice-ratio", type=float, default=4.0)

    s = p.add_argument_group("SERGIO kinetics / noise")
    s.add_argument("--noise-params", type=float, default=None,
                   help="default 1.0 (steady) / 0.2 (dynamics)")
    s.add_argument("--noise-type", default="dpd", choices=["sp", "spd", "dpd"])
    s.add_argument("--decay", type=float, default=0.8)
    s.add_argument("--sampling-state", type=int, default=None,
                   help="default 15 (steady) / 1 (dynamics)")
    s.add_argument("--outlier-prob", type=float, default=0.01)
    s.add_argument("--lib-mean", type=float, default=4.6,
                   help="log-scale library size; exp(mean) ~ counts/cell. SERGIO's "
                        "published default (4.6 ~ 100 counts) is shallow -- raise to "
                        "~7-8 for depths resembling 10x data")
    s.add_argument("--lib-scale", type=float, default=0.4)
    s.add_argument("--dropout-shape", type=float, default=6.5)
    s.add_argument("--dropout-percentile", type=float, default=82,
                   help="0 disables dropout; higher = sparser")

    p.add_argument("--sergio-path", default=None, help="existing SERGIO checkout")

    args = p.parse_args(argv)
    if args.noise_params is None:
        args.noise_params = 1.0 if args.mode == "steady" else 0.2
    if args.sampling_state is None:
        args.sampling_state = 15 if args.mode == "steady" else 1
    return args


def main(argv=None):
    args = parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)   # SERGIO uses the legacy global RNG internally

    sergio_cls = ensure_sergio(args.sergio_path)

    grn = build_modular_grn(rng, args.n_modules, args.n_targets_per_module,
                            args.n_shared_targets, args.frac_repressive,
                            args.frac_cascade, coop_state=args.coop_state)
    activity = make_activity(rng, args.n_bins, args.n_modules, args.activity, args.null)
    tpath, rpath = write_sergio_inputs(grn, activity, outdir / "sergio_inputs",
                                       args.rate_low, args.rate_high)
    print(f"[grn] {grn.n_genes} genes, {args.n_modules} programs, {len(grn.edges)} edges")

    unspliced = pseudotime = None
    if args.mode == "steady":
        counts, clean, unspliced, pseudotime = run_steady_state(
            sergio_cls, grn, tpath, rpath, args)
    else:
        (counts, clean, unspliced, pseudotime), bmat = run_dynamics(
            sergio_cls, grn, tpath, rpath, args, rng)
        np.savetxt(outdir / "sergio_inputs" / "bMat.tab", bmat, delimiter="\t")

    (counts, clean, unspliced), bg_mask = add_background_genes(
        (counts, clean, unspliced), args.n_background_genes, rng)

    adata = to_anndata(counts, clean, unspliced, pseudotime, grn, activity, args, bg_mask)

    genes = list(adata.var_names)
    mods = [f"prog_{m}" for m in range(args.n_modules)]
    pd.DataFrame(adata.varm["gt_membership_signed"], index=genes, columns=mods).to_csv(
        outdir / "gt_membership_signed.csv")
    pd.DataFrame(adata.varm["gt_membership_binary"], index=genes, columns=mods).to_csv(
        outdir / "gt_membership_binary.csv")
    pd.DataFrame(adata.obsm["gt_activity"], index=adata.obs_names, columns=mods).to_csv(
        outdir / "gt_activity.csv")
    pd.DataFrame([{"regulator": r, "target": t, "K": k, "coop_state": c,
                   "sign": int(np.sign(k))} for r, t, k, c in grn.edges]).to_csv(
        outdir / "gt_grn.csv", index=False)
    with open(outdir / "params.json", "w") as fh:
        json.dump(vars(args), fh, indent=2, default=str)

    adata.write_h5ad(outdir / "sim.h5ad")

    x = adata.X
    print(f"[out] {outdir/'sim.h5ad'}  {adata.n_obs} cells x {adata.n_vars} genes")
    print(f"[qc]  sparsity {(x == 0).mean():.3f} | "
          f"median counts/cell {np.median(x.sum(1)):.0f} | "
          f"median genes/cell {np.median((x > 0).sum(1)):.0f}")
    if args.n_background_genes:
        print(f"[qc]  {int(adata.var.is_background.sum())} background genes "
              f"({adata.var.is_background.mean():.0%} of features) carry no program signal.")
    if args.null:
        print("[qc]  NULL run: any coherent program scCont reports here is a false positive.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

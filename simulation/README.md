# SERGIO simulations for the scCont benchmark

This directory holds the scripts and parameter files that generated every simulated dataset
used in the scCont manuscript. Run everything from this directory.

| File | Purpose |
|---|---|
| `simulate_sergio.py` | Builds a modular gene regulatory network with known gene programs, runs [SERGIO](https://github.com/PayamDiba/SERGIO), and writes counts plus program-level ground truth. |
| `prep_sim_for_sccont.py` | Applies the real-data normalisation (gene filter, `normalize_total`, `log1p`, HVG, `regress_out(n_counts)`, `scale`) and re-aligns the ground truth to the surviving cells and genes. |
| `run_three_sims.sh` | Regenerates the three controlled variations of the reference dynamics run (three-way branching, diluted signal, doubled latent capacity). |
| `regen_reference_sim.sh` | Regenerates the reference dynamics run with the corrected pseudotime (see Provenance). |
| `configs/<run>/params.json` | Every parameter of each dataset exactly as written by the run that produced it. |

## Outputs

`simulate_sergio.py --outdir runs/<name>` writes:

| Path | Contents |
|---|---|
| `sim.h5ad` | `X` = UMI counts (cells x genes); `layers['clean']` = noise-free, `layers['unspliced']`; `obs`: `cell_type`, `bin`, `prog_*`, and in dynamics mode `pseudotime`, `branch`; `var`: `gene_id`, `is_master_regulator`, `is_background`, `module_primary`, `n_modules`, `n_regulators`, `layer`; `obsm['gt_activity']`, `varm['gt_membership_signed'|'gt_membership_binary']` |
| `gt_membership_signed.csv` | genes x programs, signed weights in [-1, 1] |
| `gt_membership_binary.csv` | genes x programs, {0, 1} |
| `gt_activity.csv` | cells x programs, ground-truth program activity |
| `gt_grn.csv` | regulator, target, K, coop_state, sign |
| `params.json` | every CLI parameter plus the seed |
| `sergio_inputs/` | the targets / regulators / `bMat.tab` files handed to SERGIO |

`prep_sim_for_sccont.py --indir runs/<name>` adds `runs/<name>/prepped/` with
`expression_matrix_normalized.csv` (genes x cells, the orientation scCont expects),
`gt_membership_*.csv` and `gt_activity.csv` re-aligned to the surviving genes and cells,
`cell_metadata.csv`, `gene_metadata.csv` and `prepped.h5ad`.

## Setup

```bash
pip install sccont               # numpy, pandas, scipy, anndata, scanpy are all pulled in
```

SERGIO has no working PyPI release (the `sergio` package on PyPI is unrelated). On first use
`simulate_sergio.py` clones `github.com/PayamDiba/SERGIO` into `$SERGIO_HOME` (default
`~/.cache/SERGIO`); pass `--sergio-path /path/to/checkout` to use an existing clone. SERGIO still
uses `np.int` / `np.float`, which numpy 1.24 removed; the script restores those aliases at import
time instead of editing the SERGIO source.

## Reproducing the datasets

All runs use `--seed 0`. The generic way to replay any shipped config is:

```bash
replay () {   # usage: replay configs/<run>/params.json --outdir runs/<run>
  python - "$1" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
for k, v in sorted(p.items()):
    if k in {"outdir", "sergio_path"} or v is None: continue
    flag = "--" + k.replace("_", "-")
    if isinstance(v, bool):
        if v: print(flag)
    else:
        print(flag); print(v)
PY
}
mapfile -t ARGS < <(replay configs/simulated_data_dyn_bifurc/params.json)
python simulate_sergio.py --outdir runs/simulated_data_dyn_bifurc "${ARGS[@]}"
python prep_sim_for_sccont.py --indir runs/simulated_data_dyn_bifurc --n-top-genes 3000
```

Shorthand commands for each dataset are below. Two flag blocks are shared:

```bash
COMMON="--seed 0 --activity graded --coop-state 2.0 --decay 0.8 --dropout-shape 6.5 \
        --frac-cascade 0.25 --frac-repressive 0.3 --lib-scale 0.4 --noise-type dpd \
        --outlier-prob 0.01 --rate-high 3.5 --rate-low 0.2 --bifurcation-rate 0.08 \
        --noise-params-splice 0.07 --splice-ratio 4.0"
STEADY="--mode steady   --noise-params 1.0 --sampling-state 15 --lib-mean 8.0 --dropout-percentile 60"
DYN="--mode dynamics    --noise-params 0.2 --sampling-state 1  --lib-mean 8.0 --dropout-percentile 60"
```

### Steady-state runs

| Run | Command (append `$COMMON`) | Used for |
|---|---|---|
| `simulated_data` | `--outdir runs/simulated_data --mode steady --noise-params 1.0 --sampling-state 15 --lib-mean 4.6 --dropout-percentile 82 --n-modules 8 --n-targets-per-module 15 --n-shared-targets 20 --n-bins 6 --n-cells-per-type 300` | first pilot (no background genes) |
| `hvg3000` | `--outdir runs/hvg3000 $STEADY --n-modules 16 --n-targets-per-module 49 --n-shared-targets 0 --n-background-genes 2200 --n-bins 6 --n-cells-per-type 300` | 3000-feature steady-state benchmark |
| `simulated_data_30bin` | `--outdir runs/simulated_data_30bin $STEADY --n-modules 16 --n-targets-per-module 49 --n-shared-targets 0 --n-background-genes 2200 --n-bins 30 --n-cells-per-type 60` | 30 cell types, same total cells |
| `simulated_data_null` | `--outdir runs/simulated_data_null $STEADY --null --n-modules 16 --n-targets-per-module 49 --n-shared-targets 0 --n-background-genes 2200 --n-bins 6 --n-cells-per-type 300` | negative control: no program structure |

### Dynamics (differentiation) runs

| Run | Command (append `$COMMON`) | Used for |
|---|---|---|
| `simulated_data_dyn_bifurc` | `--outdir runs/simulated_data_dyn_bifurc $DYN --backbone bifurcating --n-modules 16 --n-targets-per-module 49 --n-shared-targets 0 --n-background-genes 2200 --n-bins 8 --n-cells-per-type 225` | reference bifurcating run (original pseudotime) |
| `dyn_bifurc_ptfix` | same flags as above, `--outdir runs/dyn_bifurc_ptfix` | reference run regenerated after the pseudotime fix; identical counts |
| `dyn3way_m16_t49_bg2200` | `--outdir runs/dyn3way_m16_t49_bg2200 $DYN --backbone trifurcating --n-modules 16 --n-targets-per-module 49 --n-shared-targets 0 --n-background-genes 2200 --n-bins 13 --n-cells-per-type 138` | three-way branching, total cells matched |
| `dyn2way_m16_t24_bg2600` | `--outdir runs/dyn2way_m16_t24_bg2600 $DYN --backbone bifurcating --n-modules 16 --n-targets-per-module 24 --n-shared-targets 0 --n-background-genes 2600 --n-bins 8 --n-cells-per-type 225` | diluted signal (400 program genes) |
| `dyn2way_m32_t24_bg2200` | `--outdir runs/dyn2way_m32_t24_bg2200 $DYN --backbone bifurcating --n-modules 32 --n-targets-per-module 24 --n-shared-targets 0 --n-background-genes 2200 --n-bins 8 --n-cells-per-type 225` | latent capacity (32 programs) |

Every run was followed by

```bash
python prep_sim_for_sccont.py --indir runs/<run> --n-top-genes 3000
```

The 3000-feature runs keep all genes through the HVG step, so the signal/background ratio is
exactly what the flags specify.

`run_three_sims.sh` produces the last three dynamics runs by replaying
`configs/simulated_data_dyn_bifurc/params.json` with only the listed overrides, then prints a
table checking cell/gene counts and the number of root splits (3 for the three-way run, 2 otherwise).
`regen_reference_sim.sh` replays the same config into `runs/dyn_bifurc_ptfix`; when pointed at the
original run directory it also verifies that the counts are bit-identical and only pseudotime moved.

## Provenance: two fixes baked into `simulate_sergio.py`

1. **Trifurcating backbone.** `--backbone trifurcating` builds a ternary lineage tree. A single
   `parent_of()` drives both the SERGIO bifurcation matrix and the pseudotime depth calculation, so
   the tree and its depths cannot disagree. The binary backbone is unchanged, so earlier runs reproduce.
2. **Within-trajectory pseudotime.** SERGIO stores each lineage state as several independent
   trajectories. The original code kept the flat index `trajectory * T + timestep` as pseudotime,
   which correlated with the trajectory label rather than with time. The fix keeps
   `steps % n_traj_steps`. No random draw is added or reordered, so counts are identical before and
   after the fix and only `obs['pseudotime']` changes (`dyn_bifurc_ptfix` vs `simulated_data_dyn_bifurc`).

## Feeding a simulation into scCont

The prepped matrix is already normalised, so skip `sccont.preprocess`:

```python
import pandas as pd
import sccont

run = "runs/simulated_data_dyn_bifurc/prepped"
expr = pd.read_csv(f"{run}/expression_matrix_normalized.csv", index_col=0)   # genes x cells
meta = pd.read_csv(f"{run}/cell_metadata.csv", index_col=0)                 # cell_type, bin, pseudotime, branch
adata = sccont.to_anndata(expr, obs=meta)   # labels live in adata.obs; training never uses them

pairs = sccont.get_knn_pairs(adata, k=3)
encoder, projector, losses = sccont.train_contrastive(adata, pairs, latent_dim=32)
sccont.embed(encoder, adata)                 # -> adata.obsm["X_sccont"]
shap_values = sccont.compute_shap_values(encoder, adata)   # summary -> adata.varm

# Score attributions against the ground truth (aligned to adata.var_names)
gt = pd.read_csv(f"{run}/gt_membership_signed.csv", index_col=0).loc[adata.var_names]
```

Alternatively `prepped.h5ad` can be read directly with `anndata.read_h5ad`; it already carries the
same `obs` columns plus `varm["gt_membership_signed"]` and `obsm["gt_activity"]`.

#!/usr/bin/env bash
# ===========================================================================
# The three additional SERGIO simulations, with both patches applied.
#
#   sim1  three-way bifurcation, 800 driving / 16 programs / 2200 background
#   sim2  two-way   bifurcation, 400 driving / 16 programs / 2600 background
#   sim3  two-way   bifurcation, 800 driving / 32 programs / 2200 background
#
# All three total 3000 genes, so prep's --n-top-genes 3000 keeps everything and
# the HVG step cannot silently change the signal/background ratio between them.
# Gene count is  n_modules + n_modules * n_targets_per_module:
#   16 + 16*49 = 800    16 + 16*24 = 400    32 + 32*24 = 800
#
# EVERY OTHER PARAMETER IS READ FROM THE REFERENCE RUN'S params.json and
# replayed verbatim -- noise, decay, library size, dropout, sampling state, seed.
# An earlier version of this script hard-coded --lib-mean 4.6 and
# --dropout-percentile 82 from memory. If either guess had been wrong these
# would have been three different simulations rather than three controlled
# variations, and nothing in the output would have said so.
#
# --bifurcation-rate is inherited too, i.e. left at whatever the reference used.
# SERGIO's README recommends ~1 and the reference uses 0.08, but measurement
# showed cells from the same trajectory are no more similar than cells from
# different ones at matched time separation, so the low rate costs nothing that
# matters here and changing it would break comparability with the reference.
#
# Overwrites any previous output for these three directories.
# ===========================================================================
set -euo pipefail

SIM=./simulate_sergio.py
PREP=./prep_sim_for_sccont.py
REF=configs/simulated_data_dyn_bifurc    # shipped params.json of the reference run (edited for scCont/simulation)
OUT=runs
NTOP=3000
TOTAL_CELLS_MATCH=1                      # 1 = hold TOTAL cells constant for sim1

# --- 0. both patches must be applied --------------------------------------
grep -q 'trifurcating'  "$SIM" || { echo "ERROR: run  python patch_trifurcating.py"; exit 1; }
grep -q 'n_traj_steps'  "$SIM" || { echo "ERROR: run  python patch_pseudotime.py";   exit 1; }
[ -f "$REF/params.json" ] || { echo "ERROR: $REF/params.json not found; set REF="; exit 1; }

# --- 1. build one argument list, reference params + condition overrides ----
# Emits the full CLI for a condition. Overrides are key=value pairs.
args_for () {
  python - "$REF/params.json" "$@" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
ov = dict(a.split('=', 1) for a in sys.argv[2:])

# sim1 changes the number of states, which would change the CELL COUNT too if
# cells-per-state were held fixed. A ternary tree at the same depth has 13
# states where a binary tree has 7. Local neighbourhood density is exactly what
# the k analysis turns on, so total cells is held constant instead and
# cells-per-state is derived.
if ov.pop('_match_total', None) == '1' and 'n_bins' in ov:
    total = int(p['n_bins']) * int(p['n_cells_per_type'])
    ov['n_cells_per_type'] = str(max(1, round(total / int(ov['n_bins']))))

p.update({k: v for k, v in ov.items()})
SKIP = {'outdir', 'sergio_path'}
for k, v in sorted(p.items()):
    if k in SKIP or v is None:
        continue
    flag = '--' + k.replace('_', '-')
    if isinstance(v, bool):
        if v:
            print(flag)
    else:
        print(flag); print(v)
PY
}

run_one () {                       # $1 = outdir, rest = overrides
  local dir="$1"; shift
  echo
  echo "=================================================================="
  echo "  $dir"
  echo "=================================================================="
  mapfile -t A < <(args_for "$@")
  printf '  %s\n' "${A[*]}"
  rm -rf "$dir"
  python "$SIM" --outdir "$dir" "${A[@]}"
  python "$PREP" --indir "$dir" --n-top-genes "$NTOP"
}

# sim1: three-way. n_bins 13 (complete ternary tree, depth 2); cells rescaled.
run_one "$OUT/dyn3way_m16_t49_bg2200" \
  _match_total="$TOTAL_CELLS_MATCH" \
  backbone=trifurcating n_bins=13 \
  n_modules=16 n_targets_per_module=49 n_shared_targets=0 n_background_genes=2200

# sim2: diluted signal. Same tree and cell count as the reference.
run_one "$OUT/dyn2way_m16_t24_bg2600" \
  backbone=bifurcating \
  n_modules=16 n_targets_per_module=24 n_shared_targets=0 n_background_genes=2600

# sim3: latent capacity. Same tree and cell count as the reference.
run_one "$OUT/dyn2way_m32_t24_bg2200" \
  backbone=bifurcating \
  n_modules=32 n_targets_per_module=24 n_shared_targets=0 n_background_genes=2200

# --- 2. confirm each one is what was asked for -----------------------------
echo
echo "=== check ==="
printf '%-30s %6s %6s %8s %6s %9s %8s %s\n' \
       run cells genes signal bg 'bg%' programs 'root splits'
for d in "$OUT"/dyn3way_m16_t49_bg2200 "$OUT"/dyn2way_m16_t24_bg2600 "$OUT"/dyn2way_m32_t24_bg2200; do
  python - "$d" <<'PY'
import sys, json, numpy as np, anndata as ad
d = sys.argv[1]
p = json.load(open(f'{d}/params.json'))
a = ad.read_h5ad(f'{d}/sim.h5ad')
b = np.loadtxt(f'{d}/sergio_inputs/bMat.tab')
bg = int(a.var['is_background'].sum()) if 'is_background' in a.var else 0
print(f"{d.split('/')[-1]:<30} {a.n_obs:>6} {a.n_vars:>6} {a.n_vars-bg:>8} {bg:>6} "
      f"{bg/a.n_vars:>8.1%} {p['n_modules']:>8} {int((b[0] > 0).sum())}")
PY
done
echo
echo "root splits should read 3 for the three-way run and 2 for the other two."

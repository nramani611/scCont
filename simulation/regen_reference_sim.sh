#!/usr/bin/env bash
# ===========================================================================
# Regenerate the reference dynamics simulation with the pseudotime fix,
# into a NEW directory. The original is left untouched.
#
# The parameters are read back out of the ORIGINAL run's params.json and
# replayed verbatim -- not retyped from memory. That is the whole point: the
# only difference between the two runs must be the patch, and any parameter I
# guessed wrong (lib_mean, dropout, n_bins, cells per type) would silently make
# this a different simulation instead of a corrected one.
#
# Because the patch adds no RNG call, the counts matrix must come back
# bit-identical and only obs['pseudotime'] may move. Step 3 checks exactly that
# and fails loudly if the counts differ -- if they do, something other than the
# pseudotime fix changed and you should not use the output.
# ===========================================================================
set -euo pipefail

SIM=./simulate_sergio.py
PREP=./prep_sim_for_sccont.py
OLD=configs/simulated_data_dyn_bifurc  # shipped params.json of the reference run (edited for scCont/simulation);
                                       # point OLD= at the original run directory to enable the step-3 comparison
NEW=runs/dyn_bifurc_ptfix              # regenerated, pseudotime corrected
NTOP=3000                              # must match what prep used originally

# --- 0. refuse to run unpatched -------------------------------------------
if ! grep -q 'n_traj_steps' "$SIM"; then
  echo "ERROR: $SIM has not had the pseudotime fix applied."
  echo "       Run:  python patch_pseudotime.py"
  exit 1
fi
if [ ! -f "$OLD/params.json" ]; then
  echo "ERROR: $OLD/params.json not found. Point OLD= at the reference run."
  exit 1
fi

# --- 1. rebuild the exact CLI from the original run ------------------------
mapfile -t ARGS < <(python - "$OLD/params.json" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
SKIP = {'outdir', 'sergio_path'}          # per-run, not part of the simulation
for k, v in sorted(p.items()):
    if k in SKIP or v is None:
        continue
    flag = '--' + k.replace('_', '-')
    if isinstance(v, bool):               # store_true flags
        if v:
            print(flag)
    else:
        print(flag); print(v)
PY
)
echo "replaying ${#ARGS[@]} argument tokens from $OLD/params.json:"
printf '  %s\n' "${ARGS[*]}"
echo

# --- 2. simulate + prep ----------------------------------------------------
rm -rf "$NEW"
python "$SIM" --outdir "$NEW" "${ARGS[@]}"
python "$PREP" --indir "$NEW" --n-top-genes "$NTOP"

# --- 3. prove only pseudotime moved ----------------------------------------
# (edited for scCont/simulation) the repository ships only params.json, not the
# original sim.h5ad, so the bit-identity check runs only when OLD= is a full run.
if [ ! -f "$OLD/sim.h5ad" ]; then
  echo
  echo "NOTE: $OLD/sim.h5ad not found; skipping the counts-identity check."
  echo "      $NEW/prepped/ is the regenerated, pseudotime-corrected dataset."
  exit 0
fi
echo
echo "=== verification: counts must be identical, pseudotime must differ ==="
python - "$OLD" "$NEW" <<'PY'
import sys, numpy as np, anndata as ad
from pathlib import Path
old, new = Path(sys.argv[1]), Path(sys.argv[2])
A, B = ad.read_h5ad(old/'sim.h5ad'), ad.read_h5ad(new/'sim.h5ad')
den = lambda X: X.toarray() if hasattr(X, 'toarray') else np.asarray(X)
XA, XB = den(A.X), den(B.X)

ok = XA.shape == XB.shape and np.array_equal(XA, XB)
print(f'  shape        {XA.shape} vs {XB.shape}')
print(f'  counts       {"IDENTICAL" if ok else "*** DIFFER ***"}'
      + ('' if ok else f'   max|diff| {np.abs(XA-XB).max() if XA.shape==XB.shape else "n/a"}'))
print(f'  clean layer  {"IDENTICAL" if np.array_equal(A.layers["clean"], B.layers["clean"]) else "*** DIFFER ***"}')
print(f'  cell_type    {"IDENTICAL" if np.array_equal(A.obs.cell_type.to_numpy(), B.obs.cell_type.to_numpy()) else "*** DIFFER ***"}')

pa, pb = A.obs.pseudotime.to_numpy(), B.obs.pseudotime.to_numpy()
moved = not np.allclose(pa, pb)
print(f'  pseudotime   {"CHANGED (expected)" if moved else "*** UNCHANGED -- patch did not take effect ***"}'
      f'   corr(old,new) {np.corrcoef(pa,pb)[0,1]:+.3f}')

if not ok:
    sys.exit('\nFAILED: the counts changed, so something other than the pseudotime '
             'fix differs between these runs. Do not use this output.')
if not moved:
    sys.exit('\nFAILED: pseudotime is unchanged -- simulate_sergio.py was not patched.')
print(f'\nOK. {new}/prepped/ is the corrected dataset; {old}/ is untouched.')
PY

#!/bin/bash

#SBATCH --account=ucb744_asc1
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=01:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%j.txt
#SBATCH --job-name=arcsix-cre_preprocess_test
#SBATCH --partition=amilan
#SBATCH --qos=normal

# Regenerate a case's flt_cld_obs_info pickles with the current preprocess.py
# (which writes the `saa` key) and verify them, so the CRE array job no longer
# crashes with KeyError: 'saa' on a stale pickle.
#
# Usage: sbatch curc_shell_test.sh [CASE_ID]
#   CASE_ID : catalog case id to preprocess (default case_014)

module load anaconda intel/2022.1.2 hdf5/1.10.1 zlib/1.2.11 netcdf/4.8.1 swig/4.1.1 gsl/2.7
conda activate er3t

PROJECT_ROOT="/projects/yuch8913/arcsix_sfc/lrt_sim"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
cd "$PROJECT_ROOT"

CASE_ID="${1:-case_014}"

# Data dir holding flt_cld_obs_info (inferred from the CRE crash log's zpt path).
# Adjust if preprocessing writes elsewhere on CURC.
DATA_DIR="/pl/active/vikas-arcsix/yuch8913/arcsix/data"

# --- 1. Regenerate cloud-obs pickles (now including the saa key) ---
python -m ssfr_atm_corr.preprocess_runner "$CASE_ID" --overwrite --no-qc-plot

# --- 2. Verification report (download this for analysis) ---
REPORT="${PROJECT_ROOT}/case014_saa_check_${SLURM_JOB_ID:-local}.txt"
python - "$DATA_DIR" <<'PY' > "$REPORT" 2>&1
import pickle, glob, os, sys, numpy as np

d = os.path.join(sys.argv[1], 'flt_cld_obs_info')
fs = sorted(glob.glob(f'{d}/arcsix_cld_obs_info_p3b_20240607_cloudy_atm_corr_time_*_atm_corr.pkl'))
print(f'data dir   : {d}')
print(f'num pickles: {len(fs)}')

no_saa, sentinels, saa_means = [], [], []
for f in fs:
    o = pickle.load(open(f, 'rb'))
    b = os.path.basename(f)
    if not (isinstance(o, dict) and 'saa' in o):
        no_saa.append(b); continue
    if o.get('skip'):
        sentinels.append(b)
    saa_means.append(np.nanmean(o['saa']))

print(f'without saa  : {no_saa or "NONE"}')
print(f'skip-sentinel: {sentinels or "NONE"}')
if saa_means:
    sm = np.array(saa_means)
    print(f'saa range across legs (deg): {np.nanmin(sm):.2f} .. {np.nanmax(sm):.2f}  (mean {np.nanmean(sm):.2f})')

if fs:
    o0 = pickle.load(open(fs[0], 'rb'))
    print('\nfirst-leg keys:', sorted(o0.keys()))
    if 'saa' in o0:
        print('first-leg saa[:5]:', np.asarray(o0['saa'])[:5])
PY

echo "Wrote verification report: $REPORT"
cat "$REPORT"

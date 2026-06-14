#!/bin/bash

#SBATCH --account=ucb744_asc1
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%j.txt
#SBATCH --job-name=arcsix-cre_simulation
#SBATCH --partition=amem
#SBATCH --qos=mem-normal
# One array task per surface albedo in cre_cases.MANUAL_ALB_SWEEP (10 entries:
# indices 0-9). Keep this range in sync with len(MANUAL_ALB_SWEEP). %2 caps the
# job to 2 full amem nodes running concurrently.
#SBATCH --array=0-9%2

module load anaconda intel/2022.1.2 hdf5/1.10.1 zlib/1.2.11 netcdf/4.8.1 swig/4.1.1 gsl/2.7
conda activate er3t

PROJECT_ROOT="/projects/yuch8913/arcsix_sfc/lrt_sim"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
cd "$PROJECT_ROOT"

# Usage: sbatch curc_shell_alpine_high_mem_cre_runner.sh [CASE_ID] [MODE]
#   CASE_ID : catalog case id (default case_004)
#   MODE    : sw | lw | both (default both)
CASE_ID="${1:-case_004}"
MODE="${2:-both}"

# Reuse the prebuilt atmospheric profile (skips the MODIS-based rebuild); the
# matching ch4_profiles_* is derived automatically. Resolved under data/zpt/<date>/.
ATM_FILE="atm_profiles_20240613_cloudy_atm_corr_1_14.109_14.140_0.11km.dat"

# Each uvspec CRE run peaks near ~64 GB; amem gives ~15.5 GB/core. Size the pool
# by available RAM (not core count) so the flattened SZA x CWP sweep never OOMs.
MEM_PER_RUN_GB="${MEM_PER_RUN_GB:-64}"
CORE_MEM_GB=15                                   # ~15.5 GB/core on amem, rounded down
TOTAL_MEM_GB=$(( SLURM_NTASKS * CORE_MEM_GB ))
WORKERS=$(( TOTAL_MEM_GB / MEM_PER_RUN_GB ))
[ "$WORKERS" -lt 1 ] && WORKERS=1
echo "Alloc ${SLURM_NTASKS} cores (~${TOTAL_MEM_GB} GB); ${MEM_PER_RUN_GB} GB/run -> ${WORKERS} workers"

# This array task's surface albedo, pulled by index from the single source of
# truth (cre_cases.MANUAL_ALB_SWEEP) so the bash side never drifts from Python.
MANUAL_ALB="$(python -c "from cre.cre_cases import MANUAL_ALB_SWEEP as a; print(a[${SLURM_ARRAY_TASK_ID}])")" || {
    echo "ERROR: array index ${SLURM_ARRAY_TASK_ID} out of range for MANUAL_ALB_SWEEP" >&2
    exit 1
}
echo "Array task ${SLURM_ARRAY_TASK_ID}: albedo ${MANUAL_ALB}"

# Resumable by default: already-written (albedo, SZA) CSVs are skipped and any
# uvspec outputs already on disk are reused, so an interrupted/requeued job
# continues instead of restarting. Set OVERWRITE=1 to force a full recompute.
OVERWRITE_FLAG=""
[ "${OVERWRITE:-0}" = "1" ] && OVERWRITE_FLAG="--overwrite-lrt"

# python -m cre.cre_runner \
#     --case-id "$CASE_ID" \
#     --mode "$MODE" \
#     --atm-file "$ATM_FILE" \
#     --manual-alb "$MANUAL_ALB" \
#     --workers "$WORKERS" \
#     $OVERWRITE_FLAG


# python -m cre.cre_runner --case-id case_004 --mode 'lw' \
#     --manual-alb sfc_alb_20240603_14.735_14.752_0.34km_cre_alb.dat

# python -m cre.cre_runner --case-id case_004 --mode 'both' \
#     --manual-alb sfc_alb_20240613_15.834_15.883_0.12km_cre_alb_scale_0.987X.dat

python -m cre.cre_runner --case-id case_004 --mode 'both' \
    --manual-alb sfc_alb_20240611_14.968_15.347_0.12km_cre_alb.dat

# python -m cre.cre_runner --case-id case_019 --mode 'lw' \
#     --atm-file "$ATM_FILE" \
#     --manual-alb sfc_alb_20240613_14.109_14.140_0.11km_cre_alb.dat
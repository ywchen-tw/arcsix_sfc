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
# One array task per surface albedo in cre_cases.MANUAL_ALB_SWEEP (13 entries:
# indices 0-12). Keep this range in sync with len(MANUAL_ALB_SWEEP). %2 caps the
# job to 2 full amem nodes running concurrently.
#SBATCH --array=0-12%2

module load anaconda intel/2022.1.2 hdf5/1.10.1 zlib/1.2.11 netcdf/4.8.1 swig/4.1.1 gsl/2.7
conda activate er3t

PROJECT_ROOT="/projects/yuch8913/arcsix_sfc/lrt_sim"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
cd "$PROJECT_ROOT"

# Usage: sbatch curc_shell_alpine_high_mem_cre_runner.sh [CASE_ID] [MODE]
#   CASE_ID : catalog case id (default case_004)
#   MODE    : sw | lw | both (default both)
CASE_ID="${1:-case_019}"
MODE="${2:-both}"

# Reuse the prebuilt full-window atmospheric profile for the active case (skips the
# MODIS-based rebuild); the matching ch4_profiles_* is derived automatically.
# Resolved under data/zpt/<date>/. Uncomment the line matching CASE_ID above.
ATM_FILE="atm_profiles_20240613_cloudy_atm_corr_1_14.109_14.140_0.11km.dat"      # case_019 (2024-06-13)
# ATM_FILE="atm_profiles_20240607_cloudy_atm_corr_15.336_15.761_0.12km.dat"      # case_014 (2024-06-07)

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

# --- Shared atmospheric profile: guarded prebuild ------------------------------
# The array runs with --atm-file to REUSE a prebuilt full-window profile. The first
# time a case is run that profile does not exist yet, so exactly one task must build
# it from MODIS while the others wait -- otherwise all 13 array tasks would race on
# writing the same file. An atomic `mkdir` lock selects the single builder (works
# regardless of which task SLURM schedules first); the builder omits --atm-file so
# cre_sim builds and caches the profile as a side effect of its own albedo sweep.
# Once the profile exists (including on a resubmit) every task just reuses it. The
# matching ch4_profiles_* is derived automatically. Path resolved from settings so
# it never drifts from what cre_sim writes.
DATA_ROOT="$(python -c 'from ssfr_atm_corr.settings import _fdir_general_; print(_fdir_general_)')"
DATE_DIR="$(echo "$ATM_FILE" | sed -E 's/^atm_profiles_([0-9]{8})_.*/\1/')"
PROFILE_PATH="$DATA_ROOT/zpt/$DATE_DIR/$ATM_FILE"
LOCK_DIR="$DATA_ROOT/zpt/$DATE_DIR/.build_${ATM_FILE}.lock"

ATM_FLAG=(--atm-file "$ATM_FILE")
if [ ! -f "$PROFILE_PATH" ]; then
    if mkdir "$LOCK_DIR" 2>/dev/null; then
        # Won the lock -> this task builds the profile. The trap frees the lock on
        # exit (success or failure) so a resubmit is never blocked by a stale lock.
        echo "Task ${SLURM_ARRAY_TASK_ID}: won build lock -> building ${ATM_FILE} from MODIS."
        trap 'rmdir "$LOCK_DIR" 2>/dev/null' EXIT
        ATM_FLAG=()          # omit --atm-file so cre_sim builds (and caches) the profile
    else
        # Another task is building -> wait for the profile to appear (written early
        # in the run, before the long libRadtran sweep), then reuse it.
        echo "Task ${SLURM_ARRAY_TASK_ID}: another task is building ${ATM_FILE}; waiting ..."
        for _ in $(seq 1 720); do        # up to ~1 h
            [ -f "$PROFILE_PATH" ] && break
            sleep 5
        done
        if [ ! -f "$PROFILE_PATH" ]; then
            echo "ERROR: timed out waiting for ${PROFILE_PATH}; check the builder task." >&2
            exit 1
        fi
        sleep 5              # settle: let the small profile file finish writing
    fi
fi

python -m cre.cre_runner \
    --case-id "$CASE_ID" \
    --mode "$MODE" \
    "${ATM_FLAG[@]}" \
    --manual-alb "$MANUAL_ALB" \
    --workers "$WORKERS" \
    $OVERWRITE_FLAG


# python -m cre.cre_runner --case-id case_004 --mode 'lw' \
#     --manual-alb sfc_alb_20240603_14.735_14.752_0.34km_cre_alb.dat

# python -m cre.cre_runner --case-id case_004 --mode 'both' \
#     --manual-alb sfc_alb_20240613_15.834_15.883_0.12km_cre_alb_scale_0.987X.dat

# python -m cre.cre_runner --case-id case_004 --mode 'both' \
#     --manual-alb sfc_alb_20240603_14.716_14.749_0.34km_cre_alb.dat # peak 2-min broadband ~0.758

# python -m cre.cre_runner --case-id case_004 --mode 'both' \
#     --manual-alb sfc_alb_20240603_14.711_14.761_0.34km_cre_alb.dat # peak 3-min broadband ~0.751

# case_019 is now the array case above, so its own albedo (index 6 of
# MANUAL_ALB_SWEEP) is already covered by the 0-12 sweep. Single-albedo reference:
# python -m cre.cre_runner --case-id case_019 --mode 'both' \
#     --atm-file "$ATM_FILE" \
#     --manual-alb sfc_alb_20240613_14.109_14.140_0.11km_cre_alb.dat
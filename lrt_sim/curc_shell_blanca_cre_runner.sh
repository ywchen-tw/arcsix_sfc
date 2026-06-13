#!/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=32
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%j.txt
#SBATCH --job-name=arcsix-cre_simulation
#SBATCH --account=blanca-airs
#### #SBATCH --partition=blanca-airs
#SBATCH --qos=preemptable
# preemptable jobs can be killed mid-run; --requeue + the resumable run logic
# below let a requeued job pick up where it left off instead of restarting.
#SBATCH --requeue

module load anaconda intel/2022.1.2 hdf5/1.10.1 zlib/1.2.11 netcdf/4.8.1 swig/4.1.1 gsl/2.7
conda activate er3t

PROJECT_ROOT="/projects/yuch8913/arcsix_sfc/lrt_sim"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
cd "$PROJECT_ROOT"

# Usage: sbatch curc_shell_blanca_cre_runner.sh ALB_INDEX [CASE_ID] [MODE]
#   ALB_INDEX : 0-based index into cre_cases.MANUAL_ALB_SWEEP (one albedo per job)
#   CASE_ID   : catalog case id (default case_004)
#   MODE      : sw | lw | both (default both)
# Submit one job per albedo, e.g.:  for i in $(seq 0 9); do sbatch THIS.sh $i; done
ALB_INDEX="$1"
CASE_ID="${2:-case_004}"
MODE="${3:-both}"

if [ -z "$ALB_INDEX" ]; then
    echo "ERROR: ALB_INDEX (first arg) required, e.g. 'sbatch $0 0'." >&2
    exit 1
fi

# Reuse the prebuilt atmospheric profile (skips the MODIS-based rebuild); the
# matching ch4_profiles_* is derived automatically. Resolved under data/zpt/<date>/.
ATM_FILE="atm_profiles_20240603_cloudy_atm_corr_2_14.711_14.868_0.34km.dat"

# Size the worker pool by the RAM this allocation actually has, not by core
# count. Blanca regular nodes have ~2 GB/core, so a full 32-core node is only
# ~64 GB total -- much tighter than amem. Trust SLURM's reported memory when
# present; otherwise estimate from CORE_MEM_GB (default 2 for blanca regular).
#
# IMPORTANT: MEM_PER_RUN_GB defaults to a small value here because the real
# per-uvspec footprint is almost certainly far below the ~64 GB seen on amem
# (which was likely the allocation, not usage). MEASURE it once with
# `seff <jobid>` / `sstat -j <jobid> --format=MaxRSS` and set MEM_PER_RUN_GB.
MEM_PER_RUN_GB="${MEM_PER_RUN_GB:-2}"
PARENT_RESERVE_GB="${PARENT_RESERVE_GB:-4}"   # headroom for the Python parent + shared data
if [ -n "${SLURM_MEM_PER_NODE:-}" ]; then
    TOTAL_MEM_GB=$(( SLURM_MEM_PER_NODE / 1024 ))
elif [ -n "${SLURM_MEM_PER_CPU:-}" ]; then
    TOTAL_MEM_GB=$(( SLURM_MEM_PER_CPU * SLURM_NTASKS / 1024 ))
else
    CORE_MEM_GB="${CORE_MEM_GB:-2}"          # blanca regular node ~2 GB/core
    TOTAL_MEM_GB=$(( SLURM_NTASKS * CORE_MEM_GB ))
fi
USABLE_MEM_GB=$(( TOTAL_MEM_GB - PARENT_RESERVE_GB ))
if [ "$USABLE_MEM_GB" -lt "$MEM_PER_RUN_GB" ]; then
    echo "ERROR: node RAM ~${TOTAL_MEM_GB} GB (usable ${USABLE_MEM_GB}) < one run's ${MEM_PER_RUN_GB} GB." >&2
    echo "       This Blanca node is too small for a ${MEM_PER_RUN_GB} GB/run job. Verify the real" >&2
    echo "       per-run RAM (seff MaxRSS) and set MEM_PER_RUN_GB lower, or use the amem script." >&2
    exit 1
fi
WORKERS=$(( USABLE_MEM_GB / MEM_PER_RUN_GB ))
[ "$WORKERS" -lt 1 ] && WORKERS=1
# Never exceed the cores we hold.
[ "$WORKERS" -gt "$SLURM_NTASKS" ] && WORKERS="$SLURM_NTASKS"
echo "Alloc ${SLURM_NTASKS} cores (~${TOTAL_MEM_GB} GB, usable ${USABLE_MEM_GB}); ${MEM_PER_RUN_GB} GB/run -> ${WORKERS} workers"

# This job's surface albedo, pulled by index from the single source of truth
# (cre_cases.MANUAL_ALB_SWEEP) so the bash side never drifts from Python.
MANUAL_ALB="$(python -c "from cre.cre_cases import MANUAL_ALB_SWEEP as a; print(a[${ALB_INDEX}])")" || {
    echo "ERROR: ALB_INDEX ${ALB_INDEX} out of range for MANUAL_ALB_SWEEP" >&2
    exit 1
}
echo "ALB_INDEX ${ALB_INDEX}: albedo ${MANUAL_ALB}"

# Resumable by default: already-written (albedo, SZA) CSVs are skipped and any
# uvspec outputs already on disk are reused, so a preempted/requeued job
# continues instead of restarting. Set OVERWRITE=1 to force a full recompute.
OVERWRITE_FLAG=""
[ "${OVERWRITE:-0}" = "1" ] && OVERWRITE_FLAG="--overwrite-lrt"

python -m cre.cre_runner \
    --case-id "$CASE_ID" \
    --mode "$MODE" \
    --atm-file "$ATM_FILE" \
    --manual-alb "$MANUAL_ALB" \
    --workers "$WORKERS" \
    $OVERWRITE_FLAG

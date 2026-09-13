#!/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=64
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
# One array task per (albedo, SZA chunk) pair, flattened onto SLURM_ARRAY_TASK_ID:
# 15 albedos in cre_cases.MANUAL_ALB_SWEEP x 4 chunks in cre_cases.CRE_SZA_CHUNKS
# = 60 tasks (indices 0-59). Keep this range in sync with those two lists -- the
# guard below aborts a task whose index no longer maps onto a real pair. %2 caps
# the sweep to 2 concurrent nodes.
#SBATCH --array=0-59%2

module load anaconda intel/2022.1.2 hdf5/1.10.1 zlib/1.2.11 netcdf/4.8.1 swig/4.1.1 gsl/2.7
conda activate er3t

PROJECT_ROOT="/projects/yuch8913/arcsix_sfc/lrt_sim"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
cd "$PROJECT_ROOT"

# Usage: sbatch curc_shell_blanca_cre_runner.sh [CASE_ID] [MODE]
# The albedo and SZA-chunk indices are derived from SLURM_ARRAY_TASK_ID, not passed in.
#   CASE_ID : catalog case id (default case_004)
#   MODE    : sw | lw | both (default both)
#
# The whole (albedo x SZA chunk) matrix goes in with ONE submit:
#   sbatch curc_shell_blanca_cre_runner.sh                 # case_004, both modes
#   sbatch curc_shell_blanca_cre_runner.sh case_004 lw     # longwave only
#
# Re-run a subset by overriding the range on the command line, e.g. after a few
# tasks were preempted and did not requeue:
#   sbatch --array=12,17,40-43 curc_shell_blanca_cre_runner.sh
#
# Outside a job array (a quick interactive test) set the indices by hand:
#   ALB_INDEX=0 CHUNK_INDEX=0 bash curc_shell_blanca_cre_runner.sh
#
# The chunks together cover the full SZA grid PLUS the case-mean SZA, so a
# complete set is required before cre_plot can build its axis -- run every task.
CASE_ID="${1:-case_004}"
MODE="${2:-both}"

# Albedo and SZA-chunk counts come from Python so the bash side never drifts
# from the lists it is indexing.
N_ALB="$(python -c 'from cre.cre_cases import MANUAL_ALB_SWEEP as a; print(len(a))')" || exit 1
N_CHUNK="$(python -c 'from cre.cre_cases import CRE_SZA_CHUNKS as c; print(len(c))')" || exit 1
N_TASKS=$(( N_ALB * N_CHUNK ))

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    # Flatten (albedo, chunk) onto the array index: chunk varies fastest, so the
    # four chunks of one albedo are adjacent and finish at about the same time.
    ALB_INDEX=$((  SLURM_ARRAY_TASK_ID / N_CHUNK ))
    CHUNK_INDEX=$(( SLURM_ARRAY_TASK_ID % N_CHUNK ))

    if [ "$SLURM_ARRAY_TASK_ID" -ge "$N_TASKS" ]; then
        echo "ERROR: array task ${SLURM_ARRAY_TASK_ID} has no (albedo, chunk) pair:" >&2
        echo "       ${N_ALB} albedos x ${N_CHUNK} chunks = ${N_TASKS} tasks (0-$(( N_TASKS - 1 )))." >&2
        echo "       Update the '#SBATCH --array' range in $0." >&2
        exit 1
    fi
    # A deliberate subset (a re-run) is fine; a full submit that no longer spans
    # the matrix would silently skip pairs, so say so loudly.
    if [ -n "${SLURM_ARRAY_TASK_COUNT:-}" ] && [ "$SLURM_ARRAY_TASK_COUNT" -gt "$N_TASKS" ]; then
        echo "WARNING: array covers ${SLURM_ARRAY_TASK_COUNT} tasks but only ${N_TASKS} pairs exist." >&2
    fi
else
    # Not an array job: fall back to env overrides for a single manual run.
    ALB_INDEX="${ALB_INDEX:-}"
    CHUNK_INDEX="${CHUNK_INDEX:-}"
    if [ -z "$ALB_INDEX" ]; then
        echo "ERROR: not a job array and ALB_INDEX unset." >&2
        echo "       Submit with sbatch, or set ALB_INDEX (and optionally CHUNK_INDEX)." >&2
        exit 1
    fi
fi

# Empty CHUNK_INDEX -> no flag -> cre_runner uses the full SZA grid.
SZA_CHUNK_FLAG=()
if [ -n "$CHUNK_INDEX" ]; then
    SZA_CHUNK_FLAG=(--sza-chunk "$CHUNK_INDEX")
fi
echo "Task ${SLURM_ARRAY_TASK_ID:-manual}: albedo index ${ALB_INDEX}/$(( N_ALB - 1 )), SZA chunk ${CHUNK_INDEX:-<full grid>}"

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
# Blanca: hard cap on concurrent uvspec workers (override with MAX_WORKERS).
MAX_WORKERS="${MAX_WORKERS:-2}"
[ "$WORKERS" -gt "$MAX_WORKERS" ] && WORKERS="$MAX_WORKERS"
echo "Alloc ${SLURM_NTASKS} cores (~${TOTAL_MEM_GB} GB, usable ${USABLE_MEM_GB}); ${MEM_PER_RUN_GB} GB/run, cap ${MAX_WORKERS} -> ${WORKERS} workers"

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
    "${SZA_CHUNK_FLAG[@]}" \
    --workers "$WORKERS" \
    $OVERWRITE_FLAG

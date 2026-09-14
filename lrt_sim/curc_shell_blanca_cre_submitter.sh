#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:20:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=Yu-Wen.Chen@colorado.edu
#SBATCH --output=sbatch-output_%x_%j.txt
#SBATCH --job-name=arcsix-cre_submitter
#SBATCH --account=blanca-airs
#### #SBATCH --partition=blanca-airs
#SBATCH --qos=preemptable
# NO --requeue on purpose: a requeued submitter would re-submit albedos it had
# already submitted, doubling the sweep. If this job is preempted partway, look
# at the jobid log it writes and resume with ALB_START=<first albedo not listed>.

# This job does nothing but fan curc_shell_blanca_cre_runner.sh out over many
# sbatch submissions -- one per surface albedo -- instead of the single
# --array=0-63%2 job the runner declares on its own. That single job caps the
# WHOLE matrix at 2 concurrent tasks; N separate jobs each get their own cap, so
# the sweep actually spreads across the blanca preemptable pool.
#
# The runner is not modified: only --array and --job-name are overridden on the
# sbatch command line, which beats the #SBATCH lines inside the file.
#
# Usage (either works -- it is a one-core, few-minute job):
#   sbatch curc_shell_blanca_cre_submitter.sh                 # case_004, both
#   sbatch curc_shell_blanca_cre_submitter.sh case_004 lw     # longwave only
#   bash   curc_shell_blanca_cre_submitter.sh                 # straight from a login node
#
# Knobs (env vars, e.g. `DRY_RUN=1 bash curc_shell_blanca_cre_submitter.sh`):
#   DRY_RUN=1        print the sbatch lines, submit nothing
#   CONCURRENCY=N    per-job %N cap (default 2). Peak concurrent tasks across the
#                    sweep is (number of albedos submitted) x CONCURRENCY.
#   ALB_START/ALB_END  submit only albedos in this inclusive index range
#                    (default: the whole sweep) -- how to resume a partial run
#   SUBMIT_DELAY=S   seconds between submissions (default 1), to be gentle on slurmctld
#   EXTRA_SBATCH     extra flags passed to every sbatch, e.g. "--qos=normal"
#   FORCE=1          submit even if a job for that albedo is already queued/running
#   N_ALB/N_CHUNK    skip the Python lookup (only if the conda env is unavailable)
#
# Anything else in the environment is inherited by the submitted jobs (sbatch
# defaults to --export=ALL), so the runner's own knobs pass straight through:
#   OVERWRITE=1 MAX_WORKERS=4 sbatch curc_shell_blanca_cre_submitter.sh
# The one exception is the parent job's SLURM_* variables, which are scrubbed
# below -- see the comment on SLURM_SCRUB.

set -uo pipefail

module load anaconda intel/2022.1.2 hdf5/1.10.1 zlib/1.2.11 netcdf/4.8.1 swig/4.1.1 gsl/2.7 2>/dev/null
conda activate er3t 2>/dev/null

PROJECT_ROOT="/projects/yuch8913/arcsix_sfc/lrt_sim"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
cd "$PROJECT_ROOT" || exit 1

RUNNER="$PROJECT_ROOT/curc_shell_blanca_cre_runner.sh"
[ -f "$RUNNER" ] || { echo "ERROR: runner not found at $RUNNER" >&2; exit 1; }

if ! command -v sbatch >/dev/null 2>&1; then
    echo "ERROR: sbatch is not on PATH -- this node cannot submit jobs." >&2
    echo "       Run this script from a blanca login node instead: bash $0" >&2
    exit 1
fi

CASE_ID="${1:-case_004}"
MODE="${2:-both}"

# Same single source of truth the runner indexes into, so the slices here can
# never drift from the (albedo, chunk) flattening over there.
N_ALB="${N_ALB:-$(python -c 'from cre.cre_cases import MANUAL_ALB_SWEEP as a; print(len(a))')}" || {
    echo "ERROR: could not read MANUAL_ALB_SWEEP (conda env not active?)." >&2
    echo "       Activate er3t, or set N_ALB and N_CHUNK by hand." >&2
    exit 1
}
N_CHUNK="${N_CHUNK:-$(python -c 'from cre.cre_cases import CRE_SZA_CHUNKS as c; print(len(c))')}" || exit 1
N_TASKS=$(( N_ALB * N_CHUNK ))

ALB_START="${ALB_START:-0}"
ALB_END="${ALB_END:-$(( N_ALB - 1 ))}"
CONCURRENCY="${CONCURRENCY:-2}"
SUBMIT_DELAY="${SUBMIT_DELAY:-1}"
DRY_RUN="${DRY_RUN:-0}"
EXTRA_SBATCH="${EXTRA_SBATCH:-}"
FORCE="${FORCE:-0}"
ME="${USER:-$(whoami)}"

if [ "$ALB_START" -lt 0 ] || [ "$ALB_END" -ge "$N_ALB" ] || [ "$ALB_START" -gt "$ALB_END" ]; then
    echo "ERROR: albedo range ${ALB_START}-${ALB_END} is not inside 0-$(( N_ALB - 1 ))." >&2
    exit 1
fi

# When this script itself runs under sbatch, its allocation's SLURM_* variables
# are in the environment and --export=ALL would hand them to the jobs it submits.
# SLURM_MEM_PER_NODE / SLURM_MEM_PER_CPU / SLURM_NTASKS are exactly what the
# runner sizes its worker pool from, so a leaked "1 core, a few GB" from this
# submitter would silently starve -- or abort -- every job in the sweep. Drop
# all inherited SLURM_* so each child job sees only its own allocation.
SLURM_SCRUB=()
while IFS= read -r _v; do
    SLURM_SCRUB+=( -u "$_v" )
done < <(compgen -v | grep '^SLURM_' | sort -u)

JOBID_LOG="$PROJECT_ROOT/submitted_jobids_${CASE_ID}_$(date +%Y%m%d_%H%M%S).txt"

echo "Submitting ${CASE_ID} / ${MODE}: albedos ${ALB_START}-${ALB_END} of 0-$(( N_ALB - 1 ))"
echo "  one job per albedo, ${N_CHUNK} SZA chunks each, %${CONCURRENCY} concurrent per job"
echo "  peak concurrent tasks ~ $(( (ALB_END - ALB_START + 1) * CONCURRENCY ))"
echo "  matrix is ${N_ALB} albedos x ${N_CHUNK} chunks = ${N_TASKS} tasks"
[ "$DRY_RUN" = "1" ] && echo "  DRY_RUN=1 -- nothing will be submitted"
echo

N_OK=0
N_FAIL=0
N_SKIP=0
for (( ALB=ALB_START; ALB<=ALB_END; ALB++ )); do
    # Chunk varies fastest in the runner's flattening, so one albedo is the
    # contiguous block [ALB*N_CHUNK, ALB*N_CHUNK + N_CHUNK - 1].
    LO=$(( ALB * N_CHUNK ))
    HI=$(( LO + N_CHUNK - 1 ))
    JOB_NAME="arcsix-cre_${CASE_ID}_alb${ALB}"

    # Duplicate guard. Two jobs on the SAME (albedo, SZA) are only safe when the
    # first has finished: cre_sim skips an SZA whose CSV already exists, but that
    # CSV is written at the very END of the run, so an overlapping duplicate sees
    # it missing, runs the whole sweep again, and both write the same uvspec
    # input/output files under tmp -- which can be read back as finished while
    # half-written. So never submit an albedo that is already pending or running.
    # squeue filters by name server-side, so no truncation to worry about; a
    # finished job leaves no entry, and that case IS handled by the CSV skip.
    if [ "$FORCE" != "1" ]; then
        EXISTING="$( squeue -h -u "$ME" -n "$JOB_NAME" -o '%i' 2>/dev/null )"
        SQ_RC=$?
        if [ $SQ_RC -ne 0 ]; then
            echo "alb ${ALB}: WARNING -- squeue failed, cannot check for a duplicate; submitting anyway." >&2
        elif [ -n "$EXISTING" ]; then
            echo "alb ${ALB} (tasks ${LO}-${HI}): SKIPPED -- already queued/running as $(echo $EXISTING | tr '\n' ' ')"
            N_SKIP=$(( N_SKIP + 1 ))
            continue
        fi
    fi

    SBATCH_CMD=( sbatch
        --array="${LO}-${HI}%${CONCURRENCY}"
        --job-name="$JOB_NAME"
        ${EXTRA_SBATCH}
        "$RUNNER" "$CASE_ID" "$MODE" )

    if [ "$DRY_RUN" = "1" ]; then
        echo "alb ${ALB}: ${SBATCH_CMD[*]}"
        continue
    fi

    OUT="$( env "${SLURM_SCRUB[@]}" "${SBATCH_CMD[@]}" 2>&1 )"
    RC=$?
    if [ $RC -ne 0 ]; then
        # Usually a QOS/account limit (MaxSubmitJobs) -- keep going so a
        # transient rejection on one albedo does not drop the rest, but make the
        # failures impossible to miss in the summary below.
        echo "alb ${ALB} (tasks ${LO}-${HI}): SUBMIT FAILED -- ${OUT}" >&2
        N_FAIL=$(( N_FAIL + 1 ))
    else
        # Pull the id off sbatch's "Submitted batch job N" line specifically --
        # OUT also carries stderr, which may hold warnings around it.
        JOBID="$( printf '%s\n' "$OUT" | sed -n 's/^Submitted batch job \([0-9][0-9]*\).*/\1/p' | tail -1 )"
        if [ -z "$JOBID" ]; then
            echo "alb ${ALB} (tasks ${LO}-${HI}): submitted, but could not parse a job id from: ${OUT}" >&2
            N_OK=$(( N_OK + 1 ))
            sleep "$SUBMIT_DELAY"
            continue
        fi
        echo "alb ${ALB} (tasks ${LO}-${HI}): job ${JOBID}  ${JOB_NAME}"
        echo "$JOBID" >> "$JOBID_LOG"
        N_OK=$(( N_OK + 1 ))
    fi
    sleep "$SUBMIT_DELAY"
done

echo
if [ "$DRY_RUN" = "1" ]; then
    echo "DRY_RUN: $(( ALB_END - ALB_START + 1 - N_SKIP )) submission(s) printed, ${N_SKIP} skipped as already queued; none sent."
    exit 0
fi
echo "Submitted ${N_OK} job(s); ${N_SKIP} skipped (already in queue); ${N_FAIL} failed."
[ "$N_SKIP" -gt 0 ] && echo "  (re-submit a skipped albedo with FORCE=1 only if you are sure it is not running)"
if [ "$N_OK" -gt 0 ]; then
    echo "Job ids -> ${JOBID_LOG}"
    echo "Cancel the whole sweep with: scancel \$(cat ${JOBID_LOG})"
fi
[ "$N_FAIL" -gt 0 ] && exit 1
exit 0

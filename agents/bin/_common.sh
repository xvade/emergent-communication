# Shared settings for agents/bin launchers.
#
# Slurm placement, highest priority first:
#   1. launcher flags:  -A/--account ACCT  -p/--partition PART  -t/--minutes N
#   2. env vars:        MP_ACCOUNT, MP_PARTITION (e.g. in ~/.bashrc)
#   3. defaults below:  account stf, partition compute
# Run `hyakalloc` to see which account/partition pairs you can use.
MP_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
MP_SIF="${MP_SIF:-$MP_REPO/agents/apptainer/meltingpot.sif}"
MP_ACCOUNT="${MP_ACCOUNT:-stf}"
MP_PARTITION="${MP_PARTITION:-compute}"

if [[ ! -f "$MP_SIF" ]]; then
  echo "Container not found: $MP_SIF" >&2
  echo "Build it with: sbatch $MP_REPO/agents/apptainer/build.sbatch" >&2
  exit 1
fi

# Splits "$@" into Slurm flags (setting MP_ACCOUNT / MP_PARTITION /
# MP_MINUTES) and everything else (MP_ARGS array, passed to Python).
mp_parse_args() {
  MP_ARGS=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -A|--account)   MP_ACCOUNT="$2"; shift 2 ;;
      --account=*)    MP_ACCOUNT="${1#*=}"; shift ;;
      -p|--partition) MP_PARTITION="$2"; shift 2 ;;
      --partition=*)  MP_PARTITION="${1#*=}"; shift ;;
      -t|--minutes)   MP_MINUTES="$2"; shift 2 ;;
      --minutes=*)    MP_MINUTES="${1#*=}"; shift ;;
      *)              MP_ARGS+=("$1"); shift ;;
    esac
  done
}

# Run python inside the container with the repo bound and agents/ importable.
# (MP_EXEC=exec replaces the shell, so signals reach Python directly.)
mp_python() {
  ${MP_EXEC:-} apptainer exec --cleanenv \
    --bind "$MP_REPO" --pwd "$MP_REPO" \
    --env "PYTHONPATH=$MP_REPO/agents,USER=$USER" \
    --env "OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}" \
    "$MP_SIF" python "$@"
}

# mp_run JOBNAME MEM python-args...
# Runs `python python-args...` in the container: directly if already inside a
# Slurm allocation, otherwise in a new 1-CPU allocation via srun.
mp_run() {
  local name="$1" mem="$2"; shift 2
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    mp_python "$@"
    return
  fi
  echo "Requesting 1 CPU / $mem for ${MP_MINUTES} min:" \
       "partition=$MP_PARTITION account=$MP_ACCOUNT" >&2
  # Re-enter this script's helpers inside the allocation. Arguments are
  # passed as real argv, so none are added or lost.
  #
  # srun ignores a single Ctrl-C (it just prints job status), and on SIGTERM
  # it SIGKILLs the step outright. So srun runs in the background (shielded
  # from the terminal's SIGINT) and Ctrl-C becomes SIGUSR1, which srun passes
  # through untouched; the play server saves and exits on SIGUSR1.
  # A second Ctrl-C force-stops the job.
  srun --account="$MP_ACCOUNT" --partition="$MP_PARTITION" \
    --cpus-per-task=1 --mem="$mem" --time="$MP_MINUTES" \
    --job-name="$name" --unbuffered \
    bash -c 'source "$0"; MP_EXEC=exec mp_python "$@"' \
    "$MP_REPO/agents/bin/_common.sh" "$@" &
  local pid=$! rc=0
  local stops=0
  _mp_stop() {
    stops=$((stops + 1))
    if (( stops == 1 )); then
      echo " stopping (saving any recording; Ctrl-C again to force)..." >&2
      kill -USR1 "$pid" 2>/dev/null
    else
      echo " force-stopping..." >&2
      kill -TERM "$pid" 2>/dev/null
    fi
  }
  trap _mp_stop INT TERM HUP
  while true; do
    wait "$pid" && rc=0 || rc=$?
    kill -0 "$pid" 2>/dev/null || break   # wait was interrupted by the trap
  done
  trap - INT TERM HUP
  return "$rc"
}

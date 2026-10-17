#!/usr/bin/env bash
#SBATCH --job-name=alpamayo-carla
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --time=01:00:00
set -euo pipefail
mode=${1:-async}
if [[ $# -gt 0 ]]; then shift; fi
: "${CARLA_OUTPUT_DIR:?Export asset variables before sbatch}"
repo_dir="$SLURM_SUBMIT_DIR"
[[ -f "$repo_dir/examples/embodiment/carla/train_alpamayo.sh" ]] || {
  echo "Submit this script from the repository root." >&2; exit 2;
}
cd "$repo_dir"
export RAY_TMPDIR="/tmp/rlinf-carla-${SLURM_JOB_ID}"
export TMPDIR="$CARLA_OUTPUT_DIR/.tmp/${SLURM_JOB_ID}"
mkdir -p "$CARLA_OUTPUT_DIR" "$RAY_TMPDIR" "$TMPDIR"
ulimit -n 65535
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cleanup() { python examples/embodiment/carla/cleanup_slurm_children.py; }
trap cleanup EXIT
trap 'exit 143' TERM
nvidia-smi --query-gpu=index,name,memory.total --format=csv
ray start --head --port=$((27000+SLURM_JOB_ID%1000)) \
  --num-cpus="$SLURM_CPUS_PER_TASK" --num-gpus=8 \
  --temp-dir="$RAY_TMPDIR" --disable-usage-stats
bash examples/embodiment/carla/train_alpamayo.sh "$mode" "$@"

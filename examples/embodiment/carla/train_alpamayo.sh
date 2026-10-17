#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == --help || $# -lt 1 ]]; then
  echo 'Usage: bash examples/embodiment/carla/train_alpamayo.sh {async|colocate} [Hydra overrides...]'
  exit 0
fi
mode=$1
shift
case "$mode" in
  async) entry=examples.embodiment.carla.async_grpo.train_async_carla ;;
  colocate) entry=examples.embodiment.carla.train_alpamayo_colocate ;;
  *) echo "Unknown mode: $mode" >&2; exit 2 ;;
esac
: "${CARLA_SERVER_DIR:?Set CARLA_SERVER_DIR}"
: "${CARLA_ROUTE_FILE:?Set CARLA_ROUTE_FILE}"
: "${CARLA_SFT_CHECKPOINT:?Set CARLA_SFT_CHECKPOINT}"
: "${CARLA_OUTPUT_DIR:?Set CARLA_OUTPUT_DIR}"
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_dir"
[[ -d "$CARLA_SERVER_DIR" && -e "$CARLA_ROUTE_FILE" && -d "$CARLA_SFT_CHECKPOINT" ]] || {
  echo 'Missing CARLA server, route or checkpoint.' >&2; exit 2;
}
mkdir -p "$CARLA_OUTPUT_DIR"
export EMBODIED_PATH="$repo_dir/examples/embodiment"
export REPO_PATH="$repo_dir"
export PYTHONPATH="$repo_dir:$repo_dir/third_party/alpamayo:$repo_dir/examples/embodiment/carla/runtime${ALPAMAYO_RUNTIME_DIR:+:$ALPAMAYO_RUNTIME_DIR}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export WANDB_MODE=${WANDB_MODE:-offline}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
if [[ "$mode" == async ]]; then
  export RLINF_PACE_TO_STALENESS=${RLINF_PACE_TO_STALENESS:-1}
  export RLINF_ACTOR_CONSUME_ONCE=${RLINF_ACTOR_CONSUME_ONCE:-1}
fi
python examples/embodiment/carla/check_alpamayo.py
exec python -m "$entry" --config-path "$EMBODIED_PATH/config" \
  --config-name "carla_grpo_alpamayo_full_$mode" "$@"

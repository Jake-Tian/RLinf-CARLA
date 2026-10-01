#!/usr/bin/env bash
# Run one of the published CARLA GRPO configurations from any working directory.
set -euo pipefail

usage() {
  echo "Usage: bash examples/embodiment/carla/train_grpo.sh {sync|async|async-split}"
  echo "Requires CARLA_SERVER_DIR, CARLA_ROUTE_FILE, CARLA_SFT_CHECKPOINT, CARLA_OUTPUT_DIR."
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi
if [[ $# -ne 1 ]]; then
  usage >&2
  exit 2
fi

case "$1" in
  sync) config=carla_grpo_starvla ;;
  async) config=carla_grpo_starvla_async ;;
  async-split) config=carla_grpo_starvla_async_split ;;
  *) usage >&2; exit 2 ;;
esac

: "${CARLA_SERVER_DIR:?Set CARLA_SERVER_DIR to the CARLA server directory}"
: "${CARLA_ROUTE_FILE:?Set CARLA_ROUTE_FILE to a route file}"
: "${CARLA_SFT_CHECKPOINT:?Set CARLA_SFT_CHECKPOINT to a compatible checkpoint}"
: "${CARLA_OUTPUT_DIR:?Set CARLA_OUTPUT_DIR to a results directory}"

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_dir"
[[ -d "$CARLA_SERVER_DIR" ]] || { echo "Missing CARLA server: $CARLA_SERVER_DIR" >&2; exit 2; }
[[ -f "$CARLA_ROUTE_FILE" ]] || { echo "Missing route: $CARLA_ROUTE_FILE" >&2; exit 2; }
[[ -e "$CARLA_SFT_CHECKPOINT" ]] || { echo "Missing checkpoint: $CARLA_SFT_CHECKPOINT" >&2; exit 2; }
mkdir -p "$CARLA_OUTPUT_DIR"
export EMBODIED_PATH="$repo_dir/examples/embodiment"

python examples/embodiment/carla/verify_env_config.py "$config"

if [[ "$config" == "carla_grpo_starvla" ]]; then
  exec python examples/embodiment/train_embodied_agent.py --config-name "$config"
fi
exec python -m examples.embodiment.carla.async_grpo.train_async_carla \
  --config-path "$EMBODIED_PATH/config" --config-name "$config"

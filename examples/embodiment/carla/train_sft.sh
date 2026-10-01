#!/usr/bin/env bash
# Train StarVLA on a checked CARLA demonstration dataset.
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: bash examples/embodiment/carla/train_sft.sh"
  echo "Optional: STARVLA_DIR, CARLA_SFT_CONFIG, CARLA_OUTPUT_DIR, SFT_NUM_PROCESSES, SFT_RUN_ID."
  exit 0
fi
if [[ $# -ne 0 ]]; then
  echo "Unexpected arguments. Use --help." >&2
  exit 2
fi

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
starvla_dir="${STARVLA_DIR:-$repo_dir/.venv/starVLA}"
config="${CARLA_SFT_CONFIG:-$starvla_dir/examples/RLinfCARLA/CARLA/train_files/starvla_carla.yaml}"
run_root="${CARLA_OUTPUT_DIR:-$repo_dir/results}/sft"
num_processes="${SFT_NUM_PROCESSES:-8}"
run_id="${SFT_RUN_ID:-carla_sft_$(date +%Y%m%d_%H%M%S)}"

[[ -d "$starvla_dir" ]] || { echo "Missing StarVLA checkout: $starvla_dir" >&2; exit 2; }
[[ -f "$config" ]] || { echo "Missing SFT config: $config" >&2; exit 2; }
[[ "$num_processes" =~ ^[1-9][0-9]*$ ]] || { echo "SFT_NUM_PROCESSES must be positive" >&2; exit 2; }
command -v accelerate >/dev/null || { echo "Activate the StarVLA training environment first" >&2; exit 2; }

python "$repo_dir/examples/embodiment/carla/check_sft_dataset.py" \
  --starvla "$starvla_dir" --config "$config"
mkdir -p "$run_root"
export WANDB_MODE="${WANDB_MODE:-disabled}"

cd "$starvla_dir"
exec accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "$num_processes" \
  starVLA/training/train_starvla.py \
  --config_yaml "$config" \
  --run_root_dir "$run_root" \
  --run_id "$run_id"

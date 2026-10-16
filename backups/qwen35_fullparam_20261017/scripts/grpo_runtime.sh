#!/usr/bin/env bash
# Shared runtime for the CARLA GRPO Slurm jobs. Source after setting ROOT, REPO,
# LINE, VENV and RESULT_ROOT; install the EXIT trap after preflight succeeds.

setup_grpo_runtime() {
  cd "$REPO" || exit 1
  source "$VENV/bin/activate"

  export HF_HOME="$ROOT/.cache/huggingface"
  export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
  export EMBODIED_PATH="$REPO/examples/embodiment"
  export REPO_PATH="$REPO"
  if [[ "$1" == async ]]; then
    # Ray workers must be able to import the custom actor and rollout classes.
    export PYTHONPATH="$LINE:$REPO:${PYTHONPATH:-}"
  else
    export PYTHONPATH="$REPO:${PYTHONPATH:-}"
  fi
  export OMP_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export NUMEXPR_NUM_THREADS=1
  # Ray's Unix socket paths must stay below 108 bytes.
  export RAY_TMPDIR="/tmp/rlinf-ray-${SLURM_JOB_ID}"
  export TMPDIR="$REPO/.tmp/${SLURM_JOB_ID}"
  export WANDB_MODE=offline
  export WANDB_SILENT=true
  mkdir -p "$RESULT_ROOT" "$RAY_TMPDIR" "$TMPDIR" "$LINE/logs"
}

start_local_ray() {
  ray stop --force >/dev/null 2>&1 || true
  # Keep the dashboard enabled so failed workers can be shut down cleanly.
  ray start --head \
    --port="$((24000 + SLURM_JOB_ID % 1000))" \
    --num-cpus=48 \
    --num-gpus=8 \
    --disable-usage-stats
}

stop_local_ray() { ray stop --force >/dev/null 2>&1 || true; }

reap() {
  # Ray shutdown alone leaves CARLA servers holding the next phase's ports.
  pkill -9 -f CarlaUE4 2>/dev/null || true
  stop_local_ray
}

clear_carla_logs() {
  for _ in 1 2 3; do
    rm -rf "$REPO/carla_env_logs" 2>/dev/null && return 0
    sleep 2
  done
  echo "note: could not clear $REPO/carla_env_logs; logs may mix phases"
}

start_gpu_sampler() {
  local out="$1"
  ( while true; do
      nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
        --format=csv,noheader,nounits >> "$out" 2>/dev/null
      sleep 10
    done ) &
  GPU_SAMPLER_PID=$!
}

stop_gpu_sampler() {
  [[ -n "$GPU_SAMPLER_PID" ]] && kill "$GPU_SAMPLER_PID" 2>/dev/null
  GPU_SAMPLER_PID=""
}

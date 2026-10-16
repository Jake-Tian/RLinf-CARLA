#!/usr/bin/env bash
#SBATCH --job-name=qwen35-full-colocate
#SBATCH --partition=gpu_24h
#SBATCH --reservation=yztian25_4
#SBATCH --nodelist=projgpu13
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint=highcpucount
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:rtx3090:8
#SBATCH --time=01:00:00
#SBATCH --output=/research/d7/gds/yztian25/RLinf-CARLA/logs/qwen35_colocate_g8_%j.out
#SBATCH --error=/research/d7/gds/yztian25/RLinf-CARLA/logs/qwen35_colocate_g8_%j.err
# 8-GPU colocate arm for the qwen35_efficiency_20261015 study (sync runner).
# Layout: env keeps 6 dedicated cards 0-5 (12 envs), actor+rollout share 6-7.
# Usage: run_colocate_g8.sbatch <smoke|validate|steady> <study_dir> [overrides] [candidate_path] [config_name]
# The candidate path is prepended to PYTHONPATH so the sync runner picks up the
# same frozen policy package as the async arm (alpamayo_carla.policy is shared).
set -uo pipefail
# Refuse late starts past the final verification reserve.
if [[ $(date +%s) -ge 1792162200 ]]; then echo SKIPPED_BUDGET; exit 0; fi
ulimit -u 65536 || true
ulimit -n 65535 || true
ROOT=/research/d7/gds/yztian25
REPO="$ROOT/RLinf"
LINE="$ROOT/RLinf-CARLA"
VENV="$REPO/.venv-starvla-libero"
PHASE=${1:-smoke}
STUDY=${2:?study directory required}
OVERRIDES=${3:-}
EXTRA_PATH=${4:?frozen full candidate required}
CONFIG_NAME=${5:-carla_grpo_qwen35_full_colocate_v6}
EXTRA_ARGS=()
[[ -n "$OVERRIDES" ]] && read -ra EXTRA_ARGS <<< "$OVERRIDES"
RESULT_ROOT="$STUDY/colocate_${PHASE}_${SLURM_JOB_ID}"
mkdir -p "$RESULT_ROOT"
[[ -n "$OVERRIDES" ]] && printf '%s\n' "$OVERRIDES" > "$RESULT_ROOT/extra_overrides.txt"
[[ -n "$EXTRA_PATH" ]] && printf '%s\n' "$EXTRA_PATH" > "$RESULT_ROOT/extra_pythonpath.txt"
cd "$REPO" || exit 1
source "$VENV/bin/activate"
PP="$LINE/.runtime-alpamayo:$LINE:$REPO:$ROOT/Autopilot-VeRL-prefetch-video"
[[ -n "$EXTRA_PATH" ]] && PP="$EXTRA_PATH:$PP"
export PYTHONPATH="$PP:${PYTHONPATH:-}"
cd "$EXTRA_PATH" || exit 2
export HF_HUB_OFFLINE=1
export EMBODIED_PATH="$REPO/examples/embodiment"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export WANDB_MODE=offline
export WANDB_SILENT=true
export XDG_CACHE_HOME="$LINE/.cache"
export TRITON_CACHE_DIR="$LINE/.cache/triton"
export TILELANG_CACHE_DIR="$LINE/.cache/tilelang"
export RAY_TMPDIR="/tmp/rlinf-ray-${SLURM_JOB_ID}"
export TMPDIR="$REPO/.tmp/${SLURM_JOB_ID}"
export CARLA_SERVER_DIR="$ROOT/shared_assets/CARLA_0916"
export CARLA_CACHE_DIR="$LINE/.cache/carla"
export CARLA_ROUTE_FILE="$LINE/sft_data/routes/route_0076_from41_to10.txt"
export ALPAMAYO_CHECKPOINT="$ROOT/Autopilot-VeRL-prefetch-video/models/alpamayo2-2B-sft"
export CARLA_SFT_CHECKPOINT="$ALPAMAYO_CHECKPOINT"
export CARLA_OUTPUT_DIR="$RESULT_ROOT"
unset RLINF_POLICY_INITIAL_LORA RLINF_ACTOR_INITIAL_LORA RLINF_ACTOR_CAPTURE_DIR
export RLINF_FULL_AUDIT=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RLINF_AUDIT_FULL_STATE=1
unset RLINF_SYNC_STATE_AUDIT
unset RLINF_CAPTURE_FIRST_UPDATE
mkdir -p "$RAY_TMPDIR" "$TMPDIR" "$CARLA_CACHE_DIR"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
nvidia-smi --query-gpu=index,power.limit,enforced.power.limit --format=csv > "$RESULT_ROOT/gpu_power_limits.csv"
nvidia-smi topo -m > "$RESULT_ROOT/gpu_topology.txt"
nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv > "$RESULT_ROOT/nvidia_mapping.csv"
python -c 'import os,torch,json; print(json.dumps({"cuda_visible_devices":os.getenv("CUDA_VISIBLE_DEVICES"),"slurm_job_gpus":os.getenv("SLURM_JOB_GPUS"),"devices":[{"logical_index":i,"uuid":str(getattr(torch.cuda.get_device_properties(i),"uuid","unknown"))} for i in range(torch.cuda.device_count())]}))' > "$RESULT_ROOT/gpu_mapping.json"
python -c 'import torch, transformers; from transformers import AutoConfig; import os; c=AutoConfig.from_pretrained(os.environ["ALPAMAYO_CHECKPOINT"]); assert c.model_type=="qwen3_5"; print("native_model",c.model_type,"cuda_devices",torch.cuda.device_count(),"transformers",transformers.__version__)' || exit 1
GPU_COUNT=$(python -c 'import torch; print(torch.cuda.device_count())')
[[ "$GPU_COUNT" == "8" ]] || exit 2
if [[ "$PHASE" == bootstrap ]]; then
  RUN_ARGS=(runner.max_epochs=1 runner.max_steps=1 runner.save_interval=-1 env.train.max_episode_steps=16 env.train.max_steps_per_rollout_epoch=16)
  LIMIT=700
elif [[ "$PHASE" == quick ]]; then
  RUN_ARGS=(runner.max_epochs=3 runner.max_steps=3 runner.save_interval=-1)
  LIMIT=2400
elif [[ "$PHASE" == smoke ]]; then
  RUN_ARGS=(runner.max_epochs=1 runner.max_steps=1 runner.save_interval=-1 env.train.max_episode_steps=16 env.train.max_steps_per_rollout_epoch=16)
  LIMIT=600
elif [[ "$PHASE" == validate ]]; then
  RUN_ARGS=(runner.max_epochs=20 runner.max_steps=20 runner.save_interval=20)
  LIMIT=3400
else
  RUN_ARGS=(runner.max_epochs=7 runner.max_steps=7 runner.save_interval=-1)
  LIMIT=2200
fi
REMAINING=$((1792162800-$(date +%s)-600))
(( LIMIT > REMAINING )) && LIMIT=$REMAINING
(( LIMIT > 0 )) || exit 3
GPU_SAMPLER_PID=""
CPU_SAMPLER_PID=""
cleanup() {
  [[ -n "$CPU_SAMPLER_PID" ]] && kill "$CPU_SAMPLER_PID" 2>/dev/null || true
  [[ -n "$GPU_SAMPLER_PID" ]] && kill "$GPU_SAMPLER_PID" 2>/dev/null || true
  python "$LINE/scripts/cleanup_slurm_children.py" || true
}
trap cleanup EXIT
ray start --head --port=$((26000+SLURM_JOB_ID%1000)) --num-cpus="$SLURM_CPUS_PER_TASK" --num-gpus="$GPU_COUNT" --temp-dir="$RAY_TMPDIR" --disable-usage-stats || exit 1
(while true; do
  nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw --format=csv,noheader,nounits >> "$RESULT_ROOT/gpu_util.csv"
  sleep 1
done) &
GPU_SAMPLER_PID=$!
python "$LINE/probes/qwen35_efficiency_20261014/sample_affinity_cpu.py" "$RESULT_ROOT/cpu_util.jsonl" &
CPU_SAMPLER_PID=$!
cp "$STUDY/colocate_configs/$CONFIG_NAME.yaml" "$RESULT_ROOT/base_config.yaml"
sha256sum "$STUDY/colocate_configs/$CONFIG_NAME.yaml" \
  "$EXTRA_PATH/alpamayo_carla/policy.py" \
  "$EXTRA_PATH/rlinf/workers/actor/embodied_fsdp_actor_worker.py" \
  "$EXTRA_PATH/rlinf/workers/rollout/hf/huggingface_worker.py" \
  "$EXTRA_PATH/rlinf/config.py" "$EXTRA_PATH/full_colocate_workers.py" "$EXTRA_PATH/train_full_colocate.py" "$0" > "$RESULT_ROOT/code_hashes.txt"
[[ -n "$EXTRA_PATH" ]] && sha256sum "$EXTRA_PATH/fla/__init__.py" 2>/dev/null >> "$RESULT_ROOT/code_hashes.txt"
sha256sum "$LINE/probes/qwen35_efficiency_20261014/sample_affinity_cpu.py" >> "$RESULT_ROOT/code_hashes.txt"
python -c 'import importlib.util,json,hashlib,pathlib,sys; sys.path.insert(0,sys.argv[1]); names=["rlinf.workers.actor.embodied_fsdp_actor_worker","rlinf.workers.rollout.hf.huggingface_worker","alpamayo_carla.policy"]; paths={n:pathlib.Path(importlib.util.find_spec(n).origin).resolve() for n in names}; assert all(str(p).startswith(str(pathlib.Path(sys.argv[1]).resolve())+"/") for p in paths.values()); print(json.dumps({n:{"path":str(p),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for n,p in paths.items()},indent=2))' "$EXTRA_PATH" > "$RESULT_ROOT/actual_imports.json" || exit 2
REMAINING=$((1792162800-$(date +%s)-600))
(( LIMIT > REMAINING )) && LIMIT=$REMAINING
(( LIMIT > 0 )) || exit 3
STARTED=$(date +%s)
timeout --kill-after=15s "$LIMIT" python "$EXTRA_PATH/train_full_colocate.py" \
  --config-path "$STUDY/colocate_configs" --config-name "$CONFIG_NAME" \
  "${RUN_ARGS[@]}" "${EXTRA_ARGS[@]}" \
  "runner.logger.logger_backends=[tensorboard,wandb]" \
  "runner.logger.experiment_name=qwen35-token-grpo-colocate-g8-j${SLURM_JOB_ID}" \
  > "$RESULT_ROOT/run.log" 2>&1
rc=$?
ENDED=$(date +%s)
printf 'job_id=%s\nphase=%s\nmodel=qwen3_5\naction=discrete_accel_curvature_tokens\ngpus=%s\nplacement=see_base_config.yaml\nmicro_batch=1\nglobal_batch=6\ntrain_envs=12\nrc=%s\nwall_s=%s\nresult_dir=%s\n' \
  "$SLURM_JOB_ID" "$PHASE" "$GPU_COUNT" "$rc" "$((ENDED-STARTED))" "$RESULT_ROOT" > "$RESULT_ROOT/summary.txt"
[[ -n "$OVERRIDES" ]] && printf 'overrides=%s\n' "$OVERRIDES" >> "$RESULT_ROOT/summary.txt"
[[ -n "$EXTRA_PATH" ]] && printf 'extra_pythonpath=%s\n' "$EXTRA_PATH" >> "$RESULT_ROOT/summary.txt"
printf 'config_name=%s\n' "$CONFIG_NAME" >> "$RESULT_ROOT/summary.txt"
tail -65 "$RESULT_ROOT/run.log"
exit "$rc"

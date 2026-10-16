#!/usr/bin/env bash
#SBATCH --job-name=qwen35-full-actor
#SBATCH --partition=gpu_24h
#SBATCH --reservation=yztian25_4
#SBATCH --nodelist=projgpu13
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:rtx3090:8
#SBATCH --time=01:00:00
#SBATCH --output=/research/d7/gds/yztian25/RLinf-CARLA/logs/qwen35_full_%j.out
#SBATCH --error=/research/d7/gds/yztian25/RLinf-CARLA/logs/qwen35_full_%j.err
set -uo pipefail
[[ $(date +%s) -lt 1792162800 ]] || exit 3
ulimit -u 65536 || true
ulimit -n 65535 || true
ROOT=/research/d7/gds/yztian25
REPO="$ROOT/RLinf"
LINE="$ROOT/RLinf-CARLA"
VENV="$REPO/.venv-starvla-libero"
PHASE=${1:-smoke}
ARM=${2:-A}
STUDY=${3:?study directory required}
OVERRIDES=${4:-}
EXTRA_PATH=${5:-}
CONFIG_DIR=${SIX_CONFIG_DIR:-$STUDY/configs}
CONFIG_NAME=${SIX_CONFIG_NAME:-carla_grpo_qwen35_full_async_v3}
EXTRA_ARGS=()
[[ -n "$OVERRIDES" ]] && read -ra EXTRA_ARGS <<< "$OVERRIDES"
if [[ "$ARM" == L ]]; then ACTOR=2-7; ROLLOUT=0-1; MICRO=1; STORE=1; elif [[ "$ARM" == K ]]; then ACTOR=2-7; ROLLOUT=0-5; MICRO=1; STORE=1; elif [[ "$ARM" == J ]]; then ACTOR=5-7; ROLLOUT=0-5; MICRO=1; STORE=2; elif [[ "$ARM" == H ]]; then ACTOR=5-7; ROLLOUT=3-4; MICRO=1; STORE=1; elif [[ "$ARM" == G ]]; then ACTOR=6-7; ROLLOUT=0-5; MICRO=1; STORE=3; elif [[ "$ARM" == F ]]; then ACTOR=3; ROLLOUT=0-2; MICRO=2; STORE=3; elif [[ "$ARM" == E ]]; then ACTOR=3-5; ROLLOUT=0-2; MICRO=2; STORE=1; elif [[ "$ARM" == A ]]; then ACTOR=3-5; ROLLOUT=6-7; MICRO=1; STORE=1; elif [[ "$ARM" == D ]]; then ACTOR=3-4; ROLLOUT=5-7; MICRO=3; STORE=1; elif [[ "$ARM" == I ]]; then ACTOR=6-7; ROLLOUT=3-5; MICRO=1; STORE=3; else ACTOR=3; ROLLOUT=4-5; MICRO=3; STORE=3; fi
[[ "$ARM" == C ]] && MICRO=6
ENV_GPUS=0-2
[[ "$ARM" == K ]] && ENV_GPUS=0-5
[[ "$ARM" == L ]] && ENV_GPUS=0-5
[[ "$ARM" == J ]] && ENV_GPUS=0-5
[[ "$ARM" == G ]] && ENV_GPUS=0-5
RESULT_ROOT="$STUDY/${PHASE}_${ARM}_${SLURM_JOB_ID}"
mkdir -p "$RESULT_ROOT"
[[ -n "$OVERRIDES" ]] && printf '%s\n' "$OVERRIDES" > "$RESULT_ROOT/extra_overrides.txt"
[[ -n "$EXTRA_PATH" ]] && printf '%s\n' "$EXTRA_PATH" > "$RESULT_ROOT/extra_pythonpath.txt"
printf 'extra_prefix=%s\n' "${SIX_EXTRA_PREFIX:-<none>}" > "$RESULT_ROOT/extra_prefix.txt"
source "$LINE/scripts/grpo_runtime.sh"
setup_grpo_runtime async
# SIX_EXTRA_PREFIX：可选的前置 PYTHONPATH 层（02:29 为 fla A/B 增加；空=不启用）。
# A/B 用法：SIX_EXTRA_PREFIX=$LINE/.runtime-fla（0.5.2）叠加在 candidate 之前；其余环境不变。
PRE=""
[[ -n "${SIX_EXTRA_PREFIX:-}" ]] && PRE="$SIX_EXTRA_PREFIX:"
if [[ -n "$EXTRA_PATH" ]]; then
  export PYTHONPATH="$PRE$EXTRA_PATH:$LINE/.runtime-alpamayo:$PYTHONPATH:$ROOT/Autopilot-VeRL-prefetch-video"
else
  export PYTHONPATH="$PRE$LINE/.runtime-alpamayo:$PYTHONPATH:$ROOT/Autopilot-VeRL-prefetch-video"
fi
cd "$EXTRA_PATH" || exit 2
export HF_HUB_OFFLINE=1
export XDG_CACHE_HOME="$LINE/.cache"
export TRITON_CACHE_DIR="$LINE/.cache/triton"
export TILELANG_CACHE_DIR="$LINE/.cache/tilelang"
export CARLA_SERVER_DIR="$ROOT/shared_assets/CARLA_0916"
export CARLA_CACHE_DIR="$LINE/.cache/carla"
export CARLA_ROUTE_FILE="$LINE/sft_data/routes/route_0076_from41_to10.txt"
export ALPAMAYO_CHECKPOINT="$ROOT/Autopilot-VeRL-prefetch-video/models/alpamayo2-2B-sft"
export CARLA_SFT_CHECKPOINT="$ALPAMAYO_CHECKPOINT"
export CARLA_OUTPUT_DIR="$RESULT_ROOT"
unset RLINF_POLICY_INITIAL_LORA RLINF_ACTOR_INITIAL_LORA RLINF_ACTOR_CAPTURE_DIR
export RLINF_FULL_AUDIT=1
export RLINF_AUDIT_FULL_STATE=1
export RLINF_ALL_RANK_AGE_AUDIT=1
export RLINF_PACE_TO_STALENESS=1
export RLINF_ACTOR_CONSUME_ONCE=1
# v3 修复（2026-10-16 01:46）：共享卡 OOM 缺口为碎片级（14-38MiB 请求失败、405MiB reserved-fragmented，
# 两次 OOM 消息均自荐 expandable_segments）。环境变量、零模型语义，全进程继承。相对 v2 的唯一变更。
# 02:08 收窄为仅 ARM K 生效：G+GC 对照必须与 18591 的 allocator 基线严格一致。
[[ "$ARM" == K ]] && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PROFILE_OUTPUT="$RESULT_ROOT/policy_profile.json"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
nvidia-smi --query-gpu=index,power.limit,enforced.power.limit --format=csv > "$RESULT_ROOT/gpu_power_limits.csv"
nvidia-smi topo -m > "$RESULT_ROOT/gpu_topology.txt"
nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv > "$RESULT_ROOT/nvidia_mapping.csv"
python -c 'import os,torch,json; print(json.dumps({"cuda_visible_devices":os.getenv("CUDA_VISIBLE_DEVICES"),"slurm_job_gpus":os.getenv("SLURM_JOB_GPUS"),"devices":[{"logical_index":i,"uuid":str(getattr(torch.cuda.get_device_properties(i),"uuid","unknown"))} for i in range(torch.cuda.device_count())]}))' > "$RESULT_ROOT/gpu_mapping.json"
python -c 'import torch, transformers; from transformers import AutoConfig; import os; c=AutoConfig.from_pretrained(os.environ["ALPAMAYO_CHECKPOINT"]); assert c.model_type=="qwen3_5"; print("native_model",c.model_type,"cuda_devices",torch.cuda.device_count(),"transformers",transformers.__version__)' || exit 1
if [[ "$PHASE" == replay ]]; then
  timeout 280s python "$LINE/scripts/replay_actor_batch.py" "$STUDY/capture" > "$RESULT_ROOT/run.log" 2>&1
  rc=$?
  tail -60 "$RESULT_ROOT/run.log"
  exit "$rc"
fi
if [[ "$PHASE" == smoke ]]; then
  RUN_ARGS=(runner.max_epochs=1 runner.max_steps=1 runner.save_interval=-1 env.train.max_episode_steps=16 env.train.max_steps_per_rollout_epoch=16)
  # Canonical initialization is immutable; smoke must not capture over it.
  LIMIT=280
elif [[ "$PHASE" == bootstrap ]]; then
  # 18852 loaded actors at minute four, so 280s did not cover initialization.
  # Keep the same four-update workload; extend only startup/exit validation.
  RUN_ARGS=(runner.max_epochs=1 runner.max_steps=1 runner.save_interval=-1 env.train.max_episode_steps=16 env.train.max_steps_per_rollout_epoch=16)
  LIMIT=610
elif [[ "$PHASE" == diagnostic ]]; then
  RUN_ARGS=(runner.max_epochs=1 runner.max_steps=1 runner.save_interval=-1)
  LIMIT=1000
elif [[ "$PHASE" == quick ]]; then
  RUN_ARGS=(runner.max_epochs=5 runner.max_steps=5 runner.save_interval=-1)
  LIMIT=2700
elif [[ "$PHASE" == validate ]]; then
  RUN_ARGS=(runner.max_epochs=20 runner.max_steps=20 runner.save_interval=20)
  LIMIT=5400
else
  RUN_ARGS=(runner.max_epochs=7 runner.max_steps=7 runner.save_interval=-1)
  LIMIT=1700
fi
python -c 'import importlib.util; print("frozen_policy",importlib.util.find_spec("alpamayo_carla.policy").origin)'
GPU_COUNT=$(python -c 'import torch; print(torch.cuda.device_count())')
EXPECTED=6
[[ "$ARM" == H ]] && EXPECTED=8
[[ "$ARM" == G ]] && EXPECTED=8
[[ "$ARM" == I ]] && EXPECTED=8
[[ "$ARM" == K ]] && EXPECTED=8
[[ "$ARM" == L ]] && EXPECTED=8
[[ "$ARM" == J ]] && EXPECTED=8
[[ "$ARM" == F ]] && EXPECTED=4
[[ "$ARM" == A ]] && EXPECTED=8
[[ "$ARM" == D ]] && EXPECTED=8
[[ "$PHASE" == replay ]] || [[ "$GPU_COUNT" == "$EXPECTED" ]] || exit 2
GPU_SAMPLER_PID=""
CPU_SAMPLER_PID=""
HANG_WATCH_PID=""
cleanup() {
  if [[ -d "$RAY_TMPDIR" ]]; then
    mkdir -p "$RESULT_ROOT/ray_worker_logs"
    find "$RAY_TMPDIR" -path '*/logs/*' -type f -name '*worker*' -exec cp -n {} "$RESULT_ROOT/ray_worker_logs/" \; 2>/dev/null || true
  fi
  [[ -n "$HANG_WATCH_PID" ]] && kill "$HANG_WATCH_PID" 2>/dev/null || true
  [[ -n "$CPU_SAMPLER_PID" ]] && kill "$CPU_SAMPLER_PID" 2>/dev/null || true
  [[ -n "$GPU_SAMPLER_PID" ]] && kill "$GPU_SAMPLER_PID" 2>/dev/null || true
  python "$LINE/scripts/cleanup_slurm_children.py" || true
}
trap 'exit 143' TERM
trap cleanup EXIT
ray start --head --port=$((26000+SLURM_JOB_ID%1000)) --num-cpus="$SLURM_CPUS_PER_TASK" --num-gpus="$GPU_COUNT" --temp-dir="$RAY_TMPDIR" --disable-usage-stats || exit 1
(while true; do
  nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw --format=csv,noheader,nounits >> "$RESULT_ROOT/gpu_util.csv"
  sleep 1
done) &
GPU_SAMPLER_PID=$!
python "$LINE/probes/qwen35_efficiency_20261014/sample_affinity_cpu.py" "$RESULT_ROOT/cpu_util.jsonl" &
CPU_SAMPLER_PID=$!
python "$LINE/probes/qwen35_six_actor_20261016/watch_hang_injob.py" "$RESULT_ROOT" >> "$RESULT_ROOT/hang_watch.log" 2>&1 &
HANG_WATCH_PID=$!
cp "$CONFIG_DIR/$CONFIG_NAME.yaml" "$RESULT_ROOT/base_config.yaml"
sha256sum "$LINE/async_grpo/actor.py" "$LINE/async_grpo/rollout.py" "$EXTRA_PATH/alpamayo_carla/policy.py" "$0" > "$RESULT_ROOT/code_hashes.txt"
[[ -n "$EXTRA_PATH" ]] && sha256sum "$EXTRA_PATH/fla/__init__.py" >> "$RESULT_ROOT/code_hashes.txt"
sha256sum "$LINE/probes/qwen35_efficiency_20261014/sample_affinity_cpu.py" >> "$RESULT_ROOT/code_hashes.txt"
sha256sum "$EXTRA_PATH/rlinf/workers/actor/async_ppo_fsdp_worker.py" "$EXTRA_PATH/async_grpo/actor.py" "$EXTRA_PATH/async_grpo/rollout.py" "$EXTRA_PATH/rlinf/hybrid_engines/fsdp/utils.py" "$EXTRA_PATH/rlinf/hybrid_engines/fsdp/strategy/fsdp2.py" "$EXTRA_PATH/full_audit.py" >> "$RESULT_ROOT/code_hashes.txt"
python -c 'import importlib.util,json,hashlib,pathlib,sys; sys.path.insert(0,sys.argv[1]); names=["rlinf.workers.actor.async_ppo_fsdp_worker","alpamayo_carla.policy","rlinf.hybrid_engines.fsdp.utils","rlinf.hybrid_engines.fsdp.strategy.fsdp2"]; paths={n:pathlib.Path(importlib.util.find_spec(n).origin).resolve() for n in names}; assert all(str(p).startswith(str(pathlib.Path(sys.argv[1]).resolve())+"/") for p in paths.values()); print(json.dumps({n:{"path":str(p),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for n,p in paths.items()},indent=2))' "$EXTRA_PATH" > "$RESULT_ROOT/actual_imports.json" || exit 2
REMAINING=$((1792162800-$(date +%s)-600))
[[ "$REMAINING" -gt 0 ]] || exit 3
[[ "$LIMIT" -le "$REMAINING" ]] || LIMIT="$REMAINING"
STARTED=$(date +%s)
timeout --kill-after=15s "$LIMIT" python "$LINE/async_grpo/train_async_carla.py" \
  --config-path "$CONFIG_DIR" --config-name "$CONFIG_NAME" \
  cluster.component_placement.env="$ENV_GPUS" cluster.component_placement.actor="$ACTOR" cluster.component_placement.rollout="$ROLLOUT" \
  actor.micro_batch_size="$MICRO" algorithm.rollout_store_size_per_rank="$STORE" "${RUN_ARGS[@]}" "${EXTRA_ARGS[@]}" \
  "runner.logger.logger_backends=[tensorboard,wandb]" \
  "runner.logger.experiment_name=qwen35-token-grpo-${ARM}-j${SLURM_JOB_ID}" \
  > "$RESULT_ROOT/run.log" 2>&1
rc=$?
ENDED=$(date +%s)
printf 'job_id=%s\nphase=%s\nmodel=qwen3_5\naction=discrete_accel_curvature_tokens\ngpus=%s\nplacement=env:%s actor:%s rollout:%s\nmicro_batch=%s\nrc=%s\nwall_s=%s\nresult_dir=%s\n' \
  "$SLURM_JOB_ID" "$PHASE" "$GPU_COUNT" "$ENV_GPUS" "$ACTOR" "$ROLLOUT" "$MICRO" "$rc" "$((ENDED-STARTED))" "$RESULT_ROOT" > "$RESULT_ROOT/summary.txt"
[[ -n "$OVERRIDES" ]] && printf 'overrides=%s\n' "$OVERRIDES" >> "$RESULT_ROOT/summary.txt"
[[ -n "$EXTRA_PATH" ]] && printf 'extra_pythonpath=%s\n' "$EXTRA_PATH" >> "$RESULT_ROOT/summary.txt"
printf 'config=%s/%s.yaml\n' "$CONFIG_DIR" "$CONFIG_NAME" >> "$RESULT_ROOT/summary.txt"
tail -65 "$RESULT_ROOT/run.log"
if [[ "$PHASE" == bootstrap ]] && [[ -f "$RESULT_ROOT/first_microbatch_rank0.pt" ]]; then
  cleanup
  GPU_SAMPLER_PID=""; CPU_SAMPLER_PID=""; HANG_WATCH_PID=""
  export RLINF_FULL_AUDIT=0
  REF_LIMIT=$((1792162800-$(date +%s)-600))
  [[ "$REF_LIMIT" -le 280 ]] || REF_LIMIT=280
  if [[ "$REF_LIMIT" -gt 0 ]]; then
    timeout "$REF_LIMIT" python "$STUDY/replay_native_logprobs.py" "$RESULT_ROOT" "$CONFIG_DIR/$CONFIG_NAME.yaml" "$RESULT_ROOT/native_reference" > "$RESULT_ROOT/native_reference.log" 2>&1
    printf 'native_reference_rc=%s\n' "$?" >> "$RESULT_ROOT/summary.txt"
    tail -12 "$RESULT_ROOT/native_reference.log"
  fi
fi
exit "$rc"

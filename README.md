# RLinf + CARLA: Qwen3.5 Full-Parameter GRPO

English | [简体中文](README.zh-CN.md)

This repository integrates CARLA 0.9.16 with RLinf for full-parameter online GRPO using `alpamayo2-2B-sft`, native Qwen3.5 and discrete acceleration/curvature action tokens. Both async and colocate execution are available. The required model and preprocessing source is bundled: **these Alpamayo entry points need no external Autopilot-VeRL checkout, VeRL installation or StarVLA installation**. Model weights, the CARLA server and maps, and route files remain separate assets.

The [experiment report, in Chinese](docs/carla/Qwen35-CARLA训练效率调优.md) explains accepted optimizations, settings inherited from LoRA experiments, rejected candidates and measurement limits. The previous StarVLA continuous-action workflow is preserved in the [legacy guide](docs/carla/legacy_starvla.md). Its checkpoint is not interchangeable with this model.

## Install

From the repository root on a Linux NVIDIA GPU machine, create a separate Python 3.11 environment. Do not overwrite an existing StarVLA environment with Transformers 5. Do not install the old `.[embodied]` extra for this model: it restricts Transformers to version 4.

```bash
git clone https://github.com/Jake-Tian/RLinf-CARLA.git
cd RLinf-CARLA
python3.11 -m venv .venv-alpamayo
source .venv-alpamayo/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements/embodied/alpamayo.txt
python -m pip install flash-attn==2.8.3 --no-build-isolation
python -m pip install -e . -e third_party/alpamayo
```

Flash Attention requires a wheel or build toolchain compatible with PyTorch and CUDA. The CARLA Python package must match the 0.9.16 server. See the [historical runtime versions](backups/qwen35_fullparam_20261017/dependencies/versions.json); the additional FLA dependency is pinned to `fla-core==0.5.0`. Release acceptance reuses existing pinned libraries while isolating external source imports. A fresh-system installation has not been tested. Acceptance evidence is under [validation](docs/carla/validation/).

## Prepare assets

The model directory must contain its original weights, vocabulary, `config.json`, `reasoning_vla_config.json` and processor configuration. Use the compatible native Qwen3.5 discrete-action checkpoint. A general Qwen model or the previous StarVLA action head is not a direct replacement. Preserve normalization and the action-token ABI.

```bash
export CARLA_SERVER_DIR=/absolute/path/to/CARLA_0916
export CARLA_ROUTE_FILE=/absolute/path/to/route.txt
export CARLA_SFT_CHECKPOINT=/absolute/path/to/alpamayo2-2B-sft
export CARLA_OUTPUT_DIR=/absolute/path/to/results/async
export WANDB_MODE=offline
```

A route is an `x y` text file with at least two distinct points. The checkpoint may live in any asset directory. It need not be inside an Autopilot checkout. The launcher replaces inherited `PYTHONPATH` with this repository and explicitly supplied runtime libraries. Set `ALPAMAYO_RUNTIME_DIR` only when reusing isolated dependency directories; a normal venv installation does not need it.

## Run training

Default configurations target one machine with eight RTX3090 GPUs. Run CARLA, training and profiling on compute nodes. With Slurm, obtain an allocation and execute the commands inside the job, not on the login node. Each mode uses all eight GPUs when run separately.

| Mode | Env | Actor | Rollout | Weight transfer |
| --- | --- | --- | --- | --- |
| async | 0–5 | 2–7 | 0–1 | GPU buckets, no model offload |
| colocate | 0–5 | 2–7 | 2–7 | CPU buckets, alternating Actor/Rollout offload |

Start with a short training check. Use a separate output directory for each run. Export asset variables before starting Ray so workers inherit them.

```bash
# Run on a compute node with allocated GPUs.
ray start --head --num-gpus=8 --num-cpus=64
export RLINF_FULL_AUDIT=1
export RLINF_ALL_RANK_AGE_AUDIT=1
bash examples/embodiment/carla/train_alpamayo.sh async \
  runner.max_epochs=1 runner.max_steps=1 \
  env.train.max_episode_steps=16 env.train.max_steps_per_rollout_epoch=16
```

For colocate, use a new output directory and replace `async` with `colocate`. A [Slurm example](examples/embodiment/carla/run_alpamayo.sbatch) is provided; adapt its partition, reservation and resource arguments to your cluster. Outside Slurm, manage this task's Ray and CARLA lifecycle without stopping other tasks' processes.

The full workload has 12 environments, group size 2, horizon 128, global batch 6 and microbatch 1, normally producing 32 actual global optimizer updates per round. This example runs three rounds and saves a checkpoint at the end:

```bash
bash examples/embodiment/carla/train_alpamayo.sh async \
  runner.max_epochs=3 runner.max_steps=3 runner.save_interval=3
```

Keep the supplied full-parameter configuration, BF16 computation with FP32 state, gradient checkpointing, resharding, B1 generation without KV cache, within-generation vision reuse and full-weight synchronization. `staleness_threshold=1` counts publication versions, not individual optimizer updates. Retain the original behavior logprob instead of replacing it with rescoring under current weights.

The optional async scheduling experiment is disabled by default. Append `rollout.yield_before_prediction=true` to enable it. It completed an eight-round historical run but has no demonstrated end-to-end throughput gain.

## Results and validation

The launch log prints the resolved configuration. The output directory contains model/checkpoint preflight evidence, TensorBoard and W&B offline data. Setting `RLINF_FULL_AUDIT=1` also records first-update logprobs, gradients, parameter changes and optimizer state, adding initialization and first-round overhead.

The recommended six-Actor layout achieved about 0.1367 actual updates/s, 17.3% above the three-Actor layout, in short steady measurements on eight RTX3090s. Full-parameter colocate lacks a matched full-horizon steady comparison, so async is not proven universally faster. The final yield run completed 256 updates and checkpoint saving, without optimizer-restore or convergence validation.

```bash
python -m pytest tests/unit_tests/test_carla.py \
  tests/unit_tests/test_utils.py -k 'carla or Alpamayo or clip_grad_updates_original_storage'
```

See [source and licensing](third_party/alpamayo/README.md) for the bundled compatibility code's upstream commit. Retained optional SFT helpers still contain lazy VeRL imports; the CARLA online path does not call them. The [historical source backup](backups/qwen35_fullparam_20261017/) is unchanged. Its scripts contain old absolute paths and deadlines and are provenance records, not current launch entry points.

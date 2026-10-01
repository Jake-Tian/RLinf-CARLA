# RLinf + CARLA

English | [简体中文](README.zh-CN.md)

This repository integrates CARLA 0.9.16 with [RLinf](https://github.com/RLinf/RLinf) for synchronous and asynchronous GRPO training of a StarVLA policy with three-dimensional continuous controls. It is based on RLinf commit `db66ac56d1aa4a9c8441c4026e4212b21811970d` and preserves the default behavior of existing environments and models.

## What changed

| Location | Changes |
| --- | --- |
| [CARLA environment](rlinf/envs/sim/carla/) | Adds server management, parallel environments, routes, observations, rewards, episode termination, and RLinf environment registration. |
| [StarVLA integration](rlinf/models/) | Handles three-dimensional action unnormalization, configurable LoRA target modules, and configurations without a value head. |
| [GRPO configurations](examples/embodiment/config/) | Adds synchronous, asynchronous colocated, and asynchronous split-GPU CARLA configurations. |
| [Asynchronous entry point](examples/embodiment/carla/async_grpo/) | Extends RLinf's runner with GRPO group-integrity checks and optional rollout pacing. |
| [CARLA utilities](examples/embodiment/carla/) | Adds an environment smoke test, configuration preflight, SFT data collection, and StarVLA data adapters. |

The training configurations use Qwen3-VL-2B with a StarVLA MLP action head. The control order is `steer, throttle, brake`, and the action horizon is 6. The default placement targets one machine with eight RTX 3090 GPUs. This repository does not include the CARLA server, model weights, datasets, or route files.

## 1. Install

Run these commands from the repository root on a Linux machine with NVIDIA GPUs. RLinf's installation script does not currently offer `--env carla`, so install the StarVLA + Libero dependency combination, then install a CARLA Python wheel that matches the **0.9.16** server version.

```bash
git clone https://github.com/Jake-Tian/RLinf-CARLA.git
cd RLinf-CARLA
bash requirements/install.sh embodied --model starvla --env libero --install-rlinf
source .venv/bin/activate
python -m pip install /absolute/path/to/carla-0.9.16-wheel.whl
```

The install script defaults to Python 3.11.14 and places the StarVLA checkout in `.venv/starVLA/`. Obtain the CARLA server, matching wheel, and maps using the [official CARLA installation guide](https://carla.readthedocs.io/en/latest/start_quickstart/).

## 2. Configure assets

Each line of `route.txt` contains an `x y` coordinate; the file needs at least two distinct points. The checkpoint must match Qwen3-VL-2B and the three-dimensional action head. An unadapted general-purpose VLM checkpoint cannot be used directly. Run the following from the repository root:

```bash
export EMBODIED_PATH="$PWD/examples/embodiment"
export CARLA_SERVER_DIR=/absolute/path/to/CARLA_0916
export CARLA_ROUTE_FILE=/absolute/path/to/route.txt
export CARLA_SFT_CHECKPOINT=/absolute/path/to/steps_8152_pytorch_model.pt
export CARLA_OUTPUT_DIR=/absolute/path/to/results
export WANDB_MODE=offline
mkdir -p "$CARLA_OUTPUT_DIR"
```

If you do not yet have a compatible checkpoint, follow [SFT data and model](#sft-data-and-model) first. Export these variables before starting Ray so its workers can read the paths.

## 3. Run preflight and an environment smoke test

The configuration preflight requires the RLinf/StarVLA environment and a real checkpoint path. The environment smoke test starts CARLA and uses fixed actions to check images and basic interaction. It uses a generated test route, so it does not measure driving performance on `CARLA_ROUTE_FILE`.

```bash
python examples/embodiment/carla/verify_env_config.py carla_grpo_starvla
python examples/embodiment/carla/smoke_carla_env.py \
  --server-dir "$CARLA_SERVER_DIR" --steps 12
```

Before a full run, use a short training run on the target route to check the environment, model, Ray setup, and output paths. CARLA servers consume substantial GPU memory. If you reduce the number of environments, adjust the GRPO group and batch settings accordingly.

## 4. Run GRPO

The [GRPO launch script](examples/embodiment/carla/train_grpo.sh) checks the asset paths, runs the configuration preflight, and starts the selected RLinf training entry point. Run it from the activated environment after exporting the variables in step 2. RLinf connects to an existing Ray cluster when available and otherwise initializes Ray locally.

```bash
# Synchronous GRPO
bash examples/embodiment/carla/train_grpo.sh sync

# Asynchronous GRPO with separate actor and rollout GPUs and pacing enabled
RLINF_PACE_TO_STALENESS=1 bash examples/embodiment/carla/train_grpo.sh async-split
```

The mode selects a configuration; pacing is a separate switch that affects only asynchronous rollout:

| Mode | Config | GPU placement (env / actor / rollout) | Train envs |
| --- | --- | --- | --- |
| `sync` | `carla_grpo_starvla` | `0-3 / 4-7 / 4-7` | 16 |
| `async` | `carla_grpo_starvla_async` | `0-3 / 4-7 / 4-7` | 16 |
| `async-split` | `carla_grpo_starvla_async_split` | `0-2 / 3-5 / 6-7` | 12 |

Use `bash examples/embodiment/carla/train_grpo.sh async` for the colocated asynchronous comparison. Set `RLINF_PACE_TO_STALENESS=1` for paced asynchronous rollout, or leave it unset for unpaced rollout. All three configurations assume eight GPUs. The configured `runner.max_steps: 5` is for a short integration run. Set `runner.max_steps` and `runner.max_epochs` for a longer run. Report the differing environment counts when comparing throughput. On Slurm or another scheduler, request resources appropriate for the selected configuration and invoke the same script inside the job.

## SFT data and model

[`collect_sft_data.py`](examples/embodiment/carla/collect_sft_data.py) uses CARLA's BehaviorAgent to collect demonstrations and routes. Its default is 80 episodes, compatible with the YAML's eight held-out episodes:

```bash
python examples/embodiment/carla/collect_sft_data.py \
  --server-dir "$CARLA_SERVER_DIR" \
  --cache-dir /absolute/path/to/carla-cache \
  --out /absolute/path/to/carla-data \
  --routes /absolute/path/to/carla-data/routes
```

Copy [`data_config.py`](examples/embodiment/carla/starVLA_carla/data_config.py) into `examples/RLinfCARLA/CARLA/train_files/data_registry/` in the StarVLA checkout, and [`starvla_carla.yaml`](examples/embodiment/carla/starVLA_carla/starvla_carla.yaml) into its parent `train_files/` directory:

```bash
SFT_DIR="$PWD/.venv/starVLA/examples/RLinfCARLA/CARLA/train_files"
mkdir -p "$SFT_DIR/data_registry"
cp examples/embodiment/carla/starVLA_carla/data_config.py "$SFT_DIR/data_registry/"
cp examples/embodiment/carla/starVLA_carla/starvla_carla.yaml "$SFT_DIR/"
```

Set `base_vlm` and `data_root_dir` in the copied YAML to local paths, then check the dataset. If you collect fewer episodes, reduce `holdout_episodes` in the YAML accordingly:

```bash
python examples/embodiment/carla/check_sft_dataset.py \
  --starvla "$PWD/.venv/starVLA" \
  --config "$PWD/.venv/starVLA/examples/RLinfCARLA/CARLA/train_files/starvla_carla.yaml"
```

Start SFT from the activated environment with the [SFT launch script](examples/embodiment/carla/train_sft.sh). It runs the dataset check before calling StarVLA's `accelerate` trainer:

```bash
bash examples/embodiment/carla/train_sft.sh
```

The script defaults to eight processes and writes under `CARLA_OUTPUT_DIR/sft`. Set `SFT_NUM_PROCESSES` and adjust the YAML batch settings for another GPU count. Set `STARVLA_DIR` and `CARLA_SFT_CONFIG` when using a different StarVLA checkout or YAML. If the SFT checkpoint lacks action-window fields, inspect it first with `--dry-run` in the [repair script](examples/embodiment/carla/repair_starvla_ckpt_config.py), apply the suggested repair, and set `CARLA_SFT_CHECKPOINT` to the checkpoint path. See the [CARLA subdirectory guide](examples/embodiment/carla/README.md) for more detail.

## Validation scope

The original experiments ran synchronous and asynchronous training on eight RTX 3090 GPUs on linux7. Synchronous Job 14412 took about 345.2 seconds per steady-state step. Split-GPU asynchronous Job 14491 with pacing took about 246.0 seconds per step over steps 10–17. The runs used different environment counts, so these are efficiency observations and do not establish improved driving success. The published code passed local static and unit checks but has not undergone a complete GPU rerun from a fresh public checkout.

## Origin and licenses

This repository retains upstream RLinf's [Apache-2.0 license](LICENSE). Obtain the CARLA server, assets, and StarVLA under their respective licenses. Their code and model weights are not redistributed as repository files. The original upstream RLinf Chinese README is preserved as [README.upstream.zh-CN.md](README.upstream.zh-CN.md).

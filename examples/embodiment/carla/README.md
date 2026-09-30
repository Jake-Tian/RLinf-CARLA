# RLinf + CARLA

这个目录说明如何在 RLinf 中运行 CARLA 0.9.16 的同步与异步 GRPO。仓库基于 RLinf 提交 `db66ac56d1aa4a9c8441c4026e4212b21811970d`。CARLA 适配器位于 `rlinf/envs/sim/carla/`，GRPO 配置位于 `examples/embodiment/config/`，异步入口位于本目录的 `async_grpo/`。原有环境和模型配置保持默认行为。

## 准备资产

需要 Linux、NVIDIA GPU、CARLA 0.9.16 server 与匹配的 Python 3.11 wheel、RLinf StarVLA 依赖，以及一个已训练的 Qwen3-VL-2B + 3 维连续动作头 checkpoint。模型权重、CARLA 安装包、示范数据和 route 文件不随仓库分发。

在仓库根目录设置路径：

```bash
export EMBODIED_PATH="$PWD/examples/embodiment"
export CARLA_SERVER_DIR=/absolute/path/to/CARLA_0916
export CARLA_ROUTE_FILE=/absolute/path/to/route.txt
export CARLA_SFT_CHECKPOINT=/absolute/path/to/steps_8152_pytorch_model.pt
export CARLA_OUTPUT_DIR=/absolute/path/to/results
export WANDB_MODE=offline
```

先按 RLinf 的 `requirements/install.sh` 安装 StarVLA 训练环境，再在同一环境安装与 server 匹配的 CARLA 0.9.16 Python wheel。配置中的 `actor.model.lora_target_modules` 会避开 Qwen3-VL 视频 patch embedding。此仓库在 RLinf 的动作反归一化函数中处理 3 维动作，不需要修改 StarVLA checkout。

## 检查与运行

下面的命令从仓库根目录执行。`verify_env_config.py` 只校验配置，`smoke_carla_env.py` 需要实际 CARLA server 和 GPU。正式训练前先用一条代表性 route 做 smoke。

```bash
python examples/embodiment/carla/verify_env_config.py carla_grpo_starvla
python examples/embodiment/carla/smoke_carla_env.py \
  --server-dir "$CARLA_SERVER_DIR" --steps 12
```

RLinf 训练入口需要先连接到可用的 Ray 集群。按本机资源调整 `cluster.component_placement`、`total_num_envs` 和 batch size。仓库内的默认 placement 是原实验的 8 张 RTX 3090 布局。

```bash
python examples/embodiment/train_embodied_agent.py \
  --config-name carla_grpo_starvla

RLINF_PACE_TO_STALENESS=1 python -m examples.embodiment.carla.async_grpo.train_async_carla \
  --config-path "$PWD/examples/embodiment/config" \
  --config-name carla_grpo_starvla_async_split
```

另外有 `carla_grpo_starvla_async.yaml`，用于 actor 与 rollout 共卡的对照。异步入口沿用 RLinf 的 runner，只增加 GRPO group 完整性检查和可选的 rollout 配速。`RLINF_PACE_TO_STALENESS=1` 对应已测的配速版本，未设置时可复现未配速路径。

## SFT 数据适配

`collect_sft_data.py` 可采集 CARLA 示范数据。`starVLA_carla/data_config.py` 是 StarVLA 的数据注册插件，`starvla_carla.yaml` 是对应训练配置。使用前，把二者分别放进 StarVLA checkout 的 `examples/RLinfCARLA/CARLA/train_files/data_registry/` 与 `examples/RLinfCARLA/CARLA/train_files/`，并将 YAML 中的模型和数据路径改为本机绝对路径。运行 `check_sft_dataset.py --starvla <checkout>` 验证 episode 切分、归一化和 action chunk，再启动 StarVLA SFT。SFT checkpoint 若缺少 action-window 字段，可用 `repair_starvla_ckpt_config.py --run-dir <run_dir> --dry-run` 检查后修复。

## 已验证范围

原实验在 linux7 的 8 张 RTX 3090 上完成同步与异步训练链路。同步 Job 14412 的稳态约 345.2 秒/步，异步配速且分卡的 Job 14491 在第 10–17 步约 246.0 秒/步。这是效率观察，不能据此声称驾驶成功率提升。14491 同时把环境数从 16 改为 12，跨臂成功率指标仍需审计。公开仓库的路径配置与入口调整经过本地静态和单元检查，尚未在 GPU 集群重新运行完整训练。

## 来源与许可

RLinf 原始代码遵循仓库根目录 `LICENSE` 中的 Apache-2.0。`async_grpo/train_async_carla.py` 从上游 `examples/embodiment/train_async.py` 调整而来，保留了原文件版权声明。StarVLA 代码不打包进此仓库，运行时依赖其独立安装。

# RLinf + CARLA

[English](README.md) | 简体中文

在 [RLinf](https://github.com/RLinf/RLinf) 上接入 CARLA 0.9.16，使 StarVLA 的三维连续控制策略可以进行同步和异步 GRPO 训练。本仓库以 RLinf 提交 `db66ac56d1aa4a9c8441c4026e4212b21811970d` 为基线，保持原有环境和模型的默认行为。

## 做了哪些更新

| 位置 | 更新 |
| --- | --- |
| [CARLA 环境](rlinf/envs/sim/carla/) | 增加 server 管理、并行环境、route、观测、奖励和 episode 结束条件，并接入 RLinf 环境注册。 |
| [StarVLA 适配](rlinf/models/) | 处理三维动作反归一化，支持可配置的 LoRA 目标模块，并兼容没有 value head 的配置。 |
| [GRPO 配置](examples/embodiment/config/) | 新增同步、异步共卡、异步分卡三套 CARLA 配置。 |
| [异步入口](examples/embodiment/carla/async_grpo/) | 基于 RLinf runner 增加 GRPO group 完整性检查和可选的 rollout 配速。 |
| [CARLA 工具](examples/embodiment/carla/) | 增加环境 smoke、配置预检、SFT 数据采集及 StarVLA 数据适配。 |

训练配置使用 Qwen3-VL-2B + StarVLA MLP 动作头，输出 `steer, throttle, brake`，action horizon 为 6。默认 placement 面向单机 8 张 RTX 3090。仓库不包含 CARLA server、模型权重、数据集或 route 文件。

## 1. 安装

在 Linux NVIDIA GPU 机器上，从仓库根目录执行。RLinf 安装脚本目前没有 `--env carla`，这里用 StarVLA + Libero 依赖组合，再安装与 CARLA server 同为 **0.9.16** 的 Python wheel。

```bash
git clone https://github.com/Jake-Tian/RLinf-CARLA.git
cd RLinf-CARLA
bash requirements/install.sh embodied --model starvla --env libero --install-rlinf
source .venv/bin/activate
python -m pip install /absolute/path/to/carla-0.9.16-wheel.whl
```

安装脚本默认使用 Python 3.11.14，并将 StarVLA checkout 放在 `.venv/starVLA/`。CARLA server、匹配的 wheel 和地图请按 [CARLA 官方安装说明](https://carla.readthedocs.io/en/latest/start_quickstart/)准备。

## 2. 配置资产

`route.txt` 每行包含一个 `x y` 坐标，至少需要两个不同的点。checkpoint 必须匹配 Qwen3-VL-2B 和三维动作头，不能直接使用未适配的通用 VLM 权重。以下命令在仓库根目录执行：

```bash
export EMBODIED_PATH="$PWD/examples/embodiment"
export CARLA_SERVER_DIR=/absolute/path/to/CARLA_0916
export CARLA_ROUTE_FILE=/absolute/path/to/route.txt
export CARLA_SFT_CHECKPOINT=/absolute/path/to/steps_8152_pytorch_model.pt
export CARLA_OUTPUT_DIR=/absolute/path/to/results
export WANDB_MODE=offline
mkdir -p "$CARLA_OUTPUT_DIR"
```

如果尚无适配 checkpoint，可先完成下方的 [SFT 数据与模型](#sft-数据与模型)流程。启动 Ray 前先导出环境变量，让 worker 能读取这些路径。

## 3. 预检和环境 smoke

配置预检需要已安装的 RLinf/StarVLA 环境和真实 checkpoint 路径。环境 smoke 会启动 CARLA，用固定动作检查图像及基础交互。smoke 使用脚本生成的测试 route，不能代表给定 `CARLA_ROUTE_FILE` 的驾驶成绩。

```bash
python examples/embodiment/carla/verify_env_config.py carla_grpo_starvla
python examples/embodiment/carla/smoke_carla_env.py \
  --server-dir "$CARLA_SERVER_DIR" --steps 12
```

正式训练前，建议在目标 route 上做短训练链，确认环境、模型、Ray 和结果路径都正常。CARLA server 占用较多显存；缩小环境数量时，也需同步调整 GRPO group 和 batch 配置。

## 4. 运行 GRPO

[GRPO 启动脚本](examples/embodiment/carla/train_grpo.sh)会检查资产路径、运行配置预检，再调用相应的 RLinf 训练入口。激活训练环境并设置第 2 步的变量后运行。RLinf 会优先连接已有 Ray 集群，没有时在本机初始化。

```bash
# 同步 GRPO
bash examples/embodiment/carla/train_grpo.sh sync

# 异步 GRPO，actor 与 rollout 分卡，启用配速
RLINF_PACE_TO_STALENESS=1 bash examples/embodiment/carla/train_grpo.sh async-split
```

用 `bash examples/embodiment/carla/train_grpo.sh async` 运行 actor 与 rollout 共卡的异步对照。不设置 `RLINF_PACE_TO_STALENESS` 即运行未配速路径。三个配置默认均为 8 GPU 布局，运行前核对各自的 `cluster.component_placement`。配置中的 `runner.max_steps: 5` 是短链检查值，正式训练前按预算调整 `runner.max_steps` 和 `runner.max_epochs`。同步和共卡异步配置默认 16 个环境，分卡异步默认 12 个环境，比较效率时应同时报告该差异。在 Slurm 等调度器上，申请与配置匹配的资源，并在作业中调用同一脚本。

## SFT 数据与模型

[`collect_sft_data.py`](examples/embodiment/carla/collect_sft_data.py) 使用 CARLA BehaviorAgent 采集示范数据和 route。默认采集 80 条 episode，满足 YAML 中保留 8 条 episode 的设置：

```bash
python examples/embodiment/carla/collect_sft_data.py \
  --server-dir "$CARLA_SERVER_DIR" \
  --cache-dir /absolute/path/to/carla-cache \
  --out /absolute/path/to/carla-data \
  --routes /absolute/path/to/carla-data/routes
```

将 [`data_config.py`](examples/embodiment/carla/starVLA_carla/data_config.py) 放到 StarVLA checkout 的 `examples/RLinfCARLA/CARLA/train_files/data_registry/`，将 [`starvla_carla.yaml`](examples/embodiment/carla/starVLA_carla/starvla_carla.yaml) 放到上一级 `train_files/`。可按下列命令复制：

```bash
SFT_DIR="$PWD/.venv/starVLA/examples/RLinfCARLA/CARLA/train_files"
mkdir -p "$SFT_DIR/data_registry"
cp examples/embodiment/carla/starVLA_carla/data_config.py "$SFT_DIR/data_registry/"
cp examples/embodiment/carla/starVLA_carla/starvla_carla.yaml "$SFT_DIR/"
```

把 YAML 中的 `base_vlm` 和 `data_root_dir` 改为本机路径，然后检查数据。若采集较少的 episode，也要相应降低 YAML 中的 `holdout_episodes`：

```bash
python examples/embodiment/carla/check_sft_dataset.py \
  --starvla "$PWD/.venv/starVLA" \
  --config "$PWD/.venv/starVLA/examples/RLinfCARLA/CARLA/train_files/starvla_carla.yaml"
```

在已激活的训练环境中调用 [SFT 启动脚本](examples/embodiment/carla/train_sft.sh)。它会先检查数据集，再调用 StarVLA 的 `accelerate` 训练入口：

```bash
bash examples/embodiment/carla/train_sft.sh
```

脚本默认使用 8 个进程，结果写入 `CARLA_OUTPUT_DIR/sft`。使用其他 GPU 数量时，设置 `SFT_NUM_PROCESSES` 并调整 YAML 中的 batch 配置。使用其他 StarVLA checkout 或 YAML 时，设置 `STARVLA_DIR` 和 `CARLA_SFT_CONFIG`。SFT checkpoint 如果缺少 action-window 字段，先用 [修复脚本](examples/embodiment/carla/repair_starvla_ckpt_config.py) 的 `--dry-run` 检查，再按提示修复，最后将 checkpoint 设为 `CARLA_SFT_CHECKPOINT`。细节见 [CARLA 子目录说明](examples/embodiment/carla/README.md)。

## 验证范围

原实验在 linux7 的 8 张 RTX 3090 上运行过同步与异步训练链路。同步 Job 14412 稳态约 345.2 秒/步；分卡异步且配速的 Job 14491 在第 10 至 17 步约 246.0 秒/步。两者环境数不同，这些数字仅是效率观察，不能证明驾驶成功率提升。公开代码经过本地静态和单元检查，尚未在新的公开 checkout 上重跑完整 GPU 训练。

## 来源与许可

本仓库保留上游 RLinf 的 [Apache-2.0 许可](LICENSE)。CARLA server、资产和 StarVLA 分别按各自项目的许可获取，其代码和权重没有作为仓库文件再分发。上游 RLinf 的原版中文介绍保存在 [README.upstream.zh-CN.md](README.upstream.zh-CN.md)。

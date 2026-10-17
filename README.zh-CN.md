# RLinf + CARLA：Qwen3.5 全参数 GRPO

[English](README.md) | 简体中文

本仓库在RLinf中接入CARLA 0.9.16，使用 `alpamayo2-2B-sft` 的原生Qwen3.5与离散加速度/曲率action token进行全参数online GRPO，支持async和colocate。模型与预处理必需源码已经内置，**无需另外克隆Autopilot-VeRL或安装VeRL、StarVLA来运行这两个Alpamayo入口**。模型权重、CARLA服务端与地图、路线文件需另行准备。

[实验报告](docs/carla/Qwen35-CARLA训练效率调优.md)说明有效优化、从LoRA继承的配置、未采用方案和比较边界。旧版StarVLA连续动作训练保持可用，见[旧版说明](docs/carla/legacy_starvla.zh-CN.md)，不要混用两种checkpoint。

## 安装

在Linux NVIDIA GPU机器上，从仓库根目录创建独立Python 3.11环境。不要把Qwen3.5的Transformers 5依赖直接覆盖到旧StarVLA环境，也不要为此模型安装旧的`.[embodied]` extra，它限制Transformers 4。

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

Flash Attention需要与PyTorch/CUDA匹配的wheel或编译工具链。CARLA Python包必须与0.9.16服务端匹配。测试环境的完整版本见[历史依赖记录](backups/qwen35_fullparam_20261017/dependencies/versions.json)，新增FLA依赖固定为`fla-core==0.5.0`。发布验收复用已有版本运行库，并隔离外部源码导入，尚未完成全新系统安装测试。验收证据见[validation](docs/carla/validation/)。

## 准备资产

模型目录应包含原始权重、词表、`config.json`、`reasoning_vla_config.json`和processor配置。必须使用兼容的原生Qwen3.5离散动作checkpoint，不能直接用通用Qwen模型或旧StarVLA动作头替换。保持原始normalization及动作token ABI。

```bash
export CARLA_SERVER_DIR=/absolute/path/to/CARLA_0916
export CARLA_ROUTE_FILE=/absolute/path/to/route.txt
export CARLA_SFT_CHECKPOINT=/absolute/path/to/alpamayo2-2B-sft
export CARLA_OUTPUT_DIR=/absolute/path/to/results/async
export WANDB_MODE=offline
```

路线为至少两个不同点的`x y`文本坐标。checkpoint可放在任意资产目录，无需位于Autopilot仓库中。启动脚本会替换继承的`PYTHONPATH`，只加入本仓库的实现和明确指定的依赖运行库。若复用隔离的第三方依赖目录，可设置`ALPAMAYO_RUNTIME_DIR`，正常venv安装无需设置。

## 启动训练

默认配置面向单机8张RTX3090。CARLA、训练、profiling都应在计算节点运行。使用Slurm时先申请资源，再在作业中执行命令，不要在登录节点直接启动训练。两种模式在同一时间分别使用全部8卡。

| 模式 | Env | Actor | Rollout | 权重传输 |
| --- | --- | --- | --- | --- |
| async | 0–5 | 2–7 | 0–1 | GPU bucket，无模型offload |
| colocate | 0–5 | 2–7 | 2–7 | CPU bucket，Actor/Rollout分阶段offload |

先验证短训练。每次使用独立输出目录，设置资产变量后启动Ray，使worker继承环境。

```bash
# 在已分配GPU的计算节点执行。
ray start --head --num-gpus=8 --num-cpus=64
export RLINF_FULL_AUDIT=1
export RLINF_ALL_RANK_AGE_AUDIT=1
bash examples/embodiment/carla/train_alpamayo.sh async \
  runner.max_epochs=1 runner.max_steps=1 \
  env.train.max_episode_steps=16 env.train.max_steps_per_rollout_epoch=16
```

测试colocate时使用新的输出目录，并把`async`换成`colocate`。Slurm示例见[run_alpamayo.sbatch](examples/embodiment/carla/run_alpamayo.sbatch)，按集群调整分区、reservation和资源参数。非Slurm运行应自行管理本任务的Ray和CARLA生命周期，避免停止其他任务的进程。

完整horizon配置为12环境、group2、horizon128、global batch6、microbatch1，一轮通常32次实际更新。以下运行三轮，并在末轮保存checkpoint：

```bash
bash examples/embodiment/carla/train_alpamayo.sh async \
  runner.max_epochs=3 runner.max_steps=3 runner.save_interval=3
```

当前推荐保持完整配置中的全参数训练、BF16计算/FP32状态、gradient checkpointing、reshard、B1无KV cache生成、生成内视觉复用与完整权重同步。`staleness_threshold=1`按发布版本计数，不代表最多落后一次optimizer更新。不要用当前权重重算并替换旧行为logprob。

yield是可选调度实验，默认关闭，可在async命令末尾加`rollout.yield_before_prediction=true`。它已完成八轮历史验证，但尚无明确的整体吞吐收益。

## 查看结果与验证

启动日志打印已解析配置。输出目录包含模型来源及checkpoint预检记录、TensorBoard和W&B offline。开启`RLINF_FULL_AUDIT=1`后还保存首更新的logprob、梯度、参数变化和optimizer状态证据，会增加初始化与首轮开销。

历史推荐六Actor布局稳态约0.1367实际updates/s，相比三Actor布局高17.3%，仅为八张3090上的短稳态结果。全参colocate尚无匹配完整horizon的公平稳态对照，不能宣称async一定更优。最后yield长测完成256次更新及存盘，但未验证optimizer恢复或训练收敛。

```bash
python -m pytest tests/unit_tests/test_carla.py \
  tests/unit_tests/test_utils.py -k 'carla or Alpamayo or clip_grad_updates_original_storage'
```

[代码来源和许可](third_party/alpamayo/README.md)记录内置兼容源码的上游commit。保留的可选SFT辅助接口仍有延迟VeRL导入，当前CARLA在线训练不调用这些接口。历史[源码备份](backups/qwen35_fullparam_20261017/)保持不变，其中脚本有旧绝对路径和截止时间，仅供追溯，不作为运行入口。

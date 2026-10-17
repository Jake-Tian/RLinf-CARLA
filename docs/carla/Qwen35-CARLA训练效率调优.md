# Qwen3.5 + CARLA 全参数 GRPO 实验报告

这份报告说明 `alpamayo2-2B-sft` 在 RLinf + CARLA 上的全参数 online GRPO 适配、效率优化和验证边界。截至 2026-10-17，推荐使用六 Actor、双 Rollout 的八卡 async 配置。一次生成内复用视觉特征和重新分配 GPU 资源有明确证据，最后一轮 yield 改动仅缩短了同步阶段，没有证明整体吞吐提升。全参数 colocate 已完成短训练正确性检查，尚不能判定 async 必然优于 colocate。

## 1. 当前可以使用的结果

| 项目 | 结论 | 证据范围 |
| --- | --- | --- |
| 全参数训练 | 约 22.21 亿参数参与训练，无 LoRA、无隐藏视觉冻结 | 首更新梯度、权重变化、optimizer 状态及完整在线轮次 |
| 推荐 async 布局 | Env 0–5，Actor 2–7，Rollout 0–1 | 同一机器八张 RTX3090，三轮布局比较 |
| 六 Actor 相对三 Actor | 吞吐提高约 17.3% | 两边各排除首轮，仅两轮稳态，不是多 seed 长测 |
| 生成内视觉复用 | 一次生成中视觉执行 8 次降为 1 次，局部耗时下降 42.31% | 真实观测局部诊断与集成 logprob 检查，不是 pipeline 提速 42.31% |
| 最后一轮 yield | 同步平均耗时下降约 20%，整条 pipeline 吞吐无明确提升 | 八轮、256 次更新，正常退出并存盘，默认关闭 |
| colocate | 已跑通全参数短训练 | 历史 Job18943 为 horizon16、4 次更新，无完整 horizon 稳态结论 |

正式入口见仓库 README。历史 LoRA 数据和逐次实验日志保留，但不再作为默认推荐或全参数性能依据。

## 2. 模型与训练工作量

模型起点是原生 Qwen3.5 的 `alpamayo2-2B-sft`。模型输出离散 acceleration/curvature action token，经 checkpoint 中的分桶和 normalization 还原为物理控制量，再通过反馈控制器作用到 CARLA。它与旧版 StarVLA 的 `steer, throttle, brake` 连续动作头不同，两个 checkpoint 和控制接口不能互换。

观测来自 CARLA 中实际采集的四路相机、四帧图像和 ego 历史位姿。每张图像为 640×360，CARLA 运行在 20 Hz。一次生成输出四个 10 Hz 原生控制，重复到八个仿真 tick。重复 tick 不被当作新的独立动作采样。没有使用数据集中的未来轨迹作为在线观测。

| 配置 | 全参数主线 |
| --- | --- |
| 参数 | 2,220,993,344 个参数，617 个参数张量全部 trainable |
| 环境与 GRPO group | 12 个环境，group size 2，完整 group 进入优势计算 |
| 训练量 | horizon128，global batch6，microbatch1，update epoch1 |
| 实际更新 | 通常每轮 32 次全局 optimizer 更新，不跨 DP rank 相加 |
| 精度 | BF16 计算，FP32 主权重、梯度归约及 Adam 一二阶状态 |
| 优化器 | AdamW，lr 1e-6，全局梯度裁剪 2 |
| 显存策略 | 非 reentrant gradient checkpointing，reshard_after_forward=true |
| 生成 | 每次模型推理 B1，无 KV cache，单次生成内视觉复用 |
| 同步 | 完整权重，128 MiB bucket，保留原始 dtype，阻塞发布 |
| async 数据 | consume once、接收限流、背压、每 Actor store1，staleness threshold1 |

历史参测 YAML 显式指定 `route_0039_from146_to14.txt`。旧启动脚本另设的 `CARLA_ROUTE_FILE` 不覆盖 YAML 中的显式路径。新发布入口改为从环境变量读取路线，复测历史数据时必须选择同一路线。路线、模型或图像规格变化后的吞吐不能直接并入旧表。

### 怎样衡量效率

主指标是实际 optimizer updates/s，辅以固定更新工作量的墙钟。令稳态区间实际完成更新数为 $N$，对应 runner 轮次耗时之和为 $T$，则

$$
\text{updates/s}=\frac{N}{T}.
$$

主表排除第一轮启动及预热，另外保留 Slurm 总耗时。存盘、启动和退出不能悄悄混入某一组的稳态。async 同时记录实际接收、使用的数据及其版本，不用采样量替代有效训练量。

GPU utilization、功耗和零利用率采样占比用于解释空闲与争用。零利用率采样占比是 `nvidia-smi` 采样中 util=0 的时间占比，不等于严格的 pipeline idle。共享卡的指标包含 CARLA 与训练或推理，不能直接归因给某个角色。阶段存在重叠，不能把 Actor、Rollout 和同步时间简单相加。

## 3. 哪些改动有效

### 3.1 一次生成内复用视觉特征

原先无 KV cache 的自回归生成每输出一个 token，都会再次处理相同图像。现在只在同一次生成调用内复用视觉结果，生成结束后恢复，不跨观测或权重更新缓存。训练仍读取原始图像并计算视觉梯度。

Job18891 用两份真实观测、固定原始动作测量：视觉执行次数 8→1，平均生成耗时 3.106624→1.792252 秒，局部加速约 1.733 倍。Job18908 进一步在集成路径中检查原始与受控修改后的视觉权重，生成 logprob 对相应权重的原生 teacher forcing 最大差为 0，权重修改确实改变输出。该实现已进入全参正式测量。

这解决重复视觉计算，不能据此宣称整条训练快了 42.31%。

### 3.2 用六 Actor 共享布局替代三 Actor 独占布局

两种方案都使用八张卡。六 Actor 方案的 GPU0–1 共享 Env/Rollout，GPU2–5 共享 Env/Actor，GPU6–7 为 Actor 独占。三 Actor 方案的 Env 为 GPU0–2，Actor 为 GPU3–5，Rollout 为 GPU6–7。

| 指标 | 三 Actor，18915 | 六 Actor，18911 |
| --- | ---: | ---: |
| 完成更新 | 96 | 96 |
| 稳态更新吞吐 | 0.116505 | 0.136661 |
| 平均秒/轮 | 274.67 | 234.16 |
| Actor 训练秒/轮 | 227.79 | 121.53 |
| Rollout 预测秒/轮 | 193.71 | 199.54 |
| 全八卡平均 util | 48.33% | 61.89% |
| 零 util 采样占比 | 43.96% | 31.28% |
| 平均功耗/卡 | 177.97 W | 197.92 W |

在固定训练协议下，六 Actor 布局的短稳态吞吐提高约 17.3%。这是资源分配组合的收益，不能单独归因为 DP 数翻倍。两组都只有两轮稳态，尚无多 seed 稳健性确认。

### 3.3 接收限流和背压

快的 Actor 过去可能从共享 channel 抢取过多轨迹，store 满时继续 drain 还可能挤掉尚未训练的数据，造成 rank 供数失衡和等待。consume-once 路径现在限制未消费轨迹额度，store 满时暂停 drain，在丢弃过期数据或消费后释放额度。

该机制先在历史 LoRA 路径定位并验证，后被全参路径继承。它保护数据消费和活性，默认未开启该路径时保持旧行为。没有可独立归因的全参数吞吐提升百分比。

### 3.4 支持稳定运行的必要改动

- **完整权重同步与预测互斥**：预测不能在 bucket apply 的中途读取一半新、一半旧的权重。CPU 并发回归和 GPU 在线发布均已检查。
- **全局梯度裁剪修复**：FSDP2 从梯度的实际 mesh 获取归约组，裁剪时把转换后的结果写回原梯度。全参首更新核对了真实全局范数，不能只看日志中的裁剪数值。
- **位置编码精度保护**：关闭 `cast_forward_inputs`，避免把 FP32 rotary 输入隐式转成 BF16，造成行为与训练 logprob 差异。
- **显存碎片处理**：六 Actor 完整 horizon 曾在 reduce-scatter 申请缓冲时 OOM，启用 expandable segments 后，后续完整实验成功。未冻结模块或缩减训练参数。
- **日志完整退出**：先处理完日志队列再结束 logger，避免最后一轮指标缺失或退出等待。

这些是正确性或可运行性修复，不应全部包装成纯性能收益。

## 4. 从 LoRA 阶段继承了什么

全参数主线沿用了此前测量得到的工程经验，但重新核验了模型、梯度、optimizer 状态和完整权重同步。下面的“继承”不代表 LoRA 的提速百分比可以外推。

| 项目 | 继承方式 | 全参验证与限制 |
| --- | --- | --- |
| 12 环境、group2、horizon128、global6/micro1 | 保留固定工作量，便于迭代 | 已完成全参完整轮次，未穷举 batch 最优值 |
| staleness1、rollout pacing、consume once | 继承调度与消费协议 | 保留旧行为 logprob 和版本，有全参在线审计 |
| 接收额度与背压 | 继承修复实现，容量按新布局选择 | 当前每 rank store1，不是照搬历史 store3 |
| GPU 图像预处理 | 继承已有实现 | 已在全参组合中运行，没有独立全参收益消融 |
| 避免每次更新清 CUDA cache | 继承可配置开关 | async 与 colocate 均采用，未声称独立全参提速比例 |
| BF16 计算与测量/审计 | 继承工具和思路 | FP32 master、梯度归约、Adam 状态在全参重新检查 |
| GPU 分配、reshard、checkpointing | 根据全参显存重新选择 | 不把历史 LoRA 的 no-reshard 结果当作全参结论 |

**未继承为全参优化**：冻结主模型常驻、冻结视觉特征缓存、仅同步 LoRA key、LoRA 初始化权重加载。全参数模式开启所有参与动作预测的可学习参数。降学习率属于训练超参数调整，也不计入系统加速。

## 5. 哪些方向没有带来明确提升

### 三 Rollout

Job18959 将 Rollout 增加到 GPU0–1、7，保留 Actor2–7 和 Env0–5。它完成三轮 96 次更新，稳态吞吐为 0.129996，比双 Rollout 的 18911 低 4.88%。

Rollout 预测时间从 199.54 降到 164.96 秒，但 Actor 训练由 121.53 升到 177.48 秒。全八卡 util 虽升到 78.81%，共享 GPU7 的争用抵消了供数收益，因此未替换推荐配置。

### 最后一轮 yield

改动是在每个完整 prediction batch 前可选地执行一次 `await asyncio.sleep(0)`，让排队的权重同步有机会被调度。不拆 batch，不改变锁、行为 logprob 或版本语义。

| 指标 | 同日基线19029 | yield长测19042 |
| --- | ---: | ---: |
| 完成轮数 / 实际更新 | 3 / 96 | 8 / 256 |
| 排除首轮后的 updates/s | 0.138170 | 0.137629 |
| 平均秒/轮 | 231.60 | 232.51 |
| 全八卡 util | 63.87% | 60.18% |
| 零 util 采样占比 | 29.54% | 32.64% |
| 平均功耗/卡 | 201.31 W | 196.28 W |
| 全部热同步平均耗时 | 26.46 秒 | 21.11 秒 |
| Rollout 进入同步函数前平均延迟 | 8.50 秒 | 4.90 秒 |

局部同步下降约 20.2%，整体吞吐差约 −0.39%。两边稳态窗口分别为两轮和七轮，不能据此作精确因果估计。只挑候选第2、3轮又会得到约 +4.3%，说明短窗口容易误导。因此保留为默认关闭的可选开关，不宣称新最佳。

```yaml
rollout:
  yield_before_prediction: false
```

19042 的六 Actor 首 LP 差均为 0，无缺失梯度，最终 version8 在实际 counter256 稳定发布，正常退出并生成六份 DCP shard 和 metadata，约 28.7 GB。存盘成功尚不等于 optimizer 恢复已验证。末轮 approx KL 约 1.118，没有学习质量或收敛结论。

本轮执行还发生过启动门槛失误：19036 因要求 LP 差严格等于 0，拒绝了短测中一个 rank 的 2.38e-7 差异，未进入训练，且未及时自动重试，造成约一个半小时空档。修正为明确的有限绝对误差阈值 1e-6 后才运行19042。失败证据保留，不把空档算作训练或补跑预算。

### 其他未采用方向

显式 `logits_to_keep=1` 的局部耗时约 1.778→1.773 秒，差异不足以接受。部分更大 batch 或 KV cache 生成路径出现过生成/训练 LP 不一致，因此当前仍使用已验证的 B1、无 KV cache 路径，不能为速度掩盖数值差异。

## 6. async 与 colocate 能比较到什么程度

18943 验证了 colocate 的 horizon16、4 次实际更新，包括首 LP、完整梯度、FP32 Adam 和裁剪，但没有完整 horizon128 稳态结果。

18944 的六 Actor async 将 staleness threshold 改为 0，三轮完成 96 次更新，稳态 0.085669 updates/s，373.53 秒/轮。它明显慢于 staleness1，但新鲜度约束不同，不能把差值当作同协议系统收益。

threshold 按发布版本计数，不按 optimizer step 计数。一轮内部有32次更新，所以 threshold1 不能解释为数据最多落后一次 optimizer 更新。比较必须结合实际 behavior version 和 optimizer-age 审计，保留采样时的旧 logprob，不能用当前权重重算后替换。

当前没有足够证据判定全参数 async 一定优于 colocate。历史 LoRA 的完整比较不能代替这项缺失实验。

## 7. 发布代码与复现

发布代码以原 RLinf-CARLA 为基底，仅修改模型接入、CARLA观测、必要的FSDP/worker修复。Alpamayo模型适配位于 `rlinf/models/embodiment/alpamayo2/`，上游必需源码位于 `third_party/alpamayo/`，async扩展和colocate入口位于 `examples/embodiment/carla/`。

不需要外部Autopilot-VeRL checkout。保留 `autopilot` 包名是为了兼容checkpoint的Hydra target，并不代表运行依赖外部仓库。部分保留的上游SFT辅助函数有延迟VeRL导入，CARLA在线训练不调用它们，不承诺整个兼容包所有历史接口都无VeRL依赖。

模型权重、CARLA服务端与路线仍需另行准备，模型权重路径不再绑定到外部源码目录。README给出安装、配置和两种模式的启动命令。旧StarVLA入口保留，勿将其连续动作checkpoint与当前离散动作模型混用。

本次发布验收已在8张3090上分别完成async与colocate的horizon16、4次实际更新。两边六Actor首LP差均为0，完整参数、梯度、FP32状态、裁剪和源码来源检查通过。VeRL与StarVLA导入被主动禁止，模型和processor来自本仓库。19048训练rc0后因过严的逐shard抽样变化断言使外层作业失败，修正验收条件后原结果复核通过，19050的colocate正常COMPLETED0:0。详细原始证据与限制见[发布验收](validation/README.md)。

### 复现结果时的检查顺序

1. 固定checkpoint、词表、normalization、路线、图像、horizon、group、global batch和复用协议。
2. 核对实际模型类型、trainable参数、首更新梯度和optimizer状态。
3. 先做horizon16短测，确认旧行为LP、完整group和同步发布。
4. 再测完整horizon，分别统计启动、首轮、稳态、同步、存盘和退出。
5. 按实际全局更新计数，不跨DP rank累加，不以高util替代吞吐。

## 8. 尚未完成的验证

- 全参async与colocate在完整horizon和匹配新鲜度下的稳态对照。
- 原推荐配置的多seed长测和更多路线上的学习质量评估。
- 完整checkpoint与optimizer恢复后的训练一致性。
- 从全新操作系统和全新依赖环境进行安装验收。此次发布验收复用隔离版本的已有运行库，不能替代干净安装测试。

接下来若继续优化，主要关注生成/供数关键路径与共享GPU争用。当前六Actor约122秒训练、约197至200秒Rollout预测，单独减少几秒同步未必缩短轮次。实验已停止，不自动恢复调优。

## 附录：证据索引

历史备份 `backups/qwen35_fullparam_20261017/` 保持不变。其配置和启动脚本是历史记录，不能直接当作新的运行入口。主要Job及用途如下：

| Job | 用途 |
| --- | --- |
| 18891 / 18908 | 视觉复用局部测量 / 集成数值验证 |
| 18894 / 18895 | GPU / CPU完整权重同步读回 |
| 18911 / 18915 | 六Actor / 三Actor完整horizon比较 |
| 18943 | 全参colocate短训练正确性 |
| 18944 | async staleness0对照 |
| 18959 | 三Rollout对照，未采用 |
| 19029 / 19035 | 同日基线 / yield短测 |
| 19036 | 启动门槛失败，未训练 |
| 19042 | yield八轮、256更新及存盘 |
| 19048 / 19050 | 独立源码async / colocate短训练验收 |

本次发布的简要验收结果单独保存于 `docs/carla/validation/`，GPU验收不用于声称新的效率提升。原始GPU/CPU采样、W&B offline、trace及大checkpoint保留在实验存储，不上传权重或原始大日志。表格中的小型原始指标JSON随 `docs/carla/results/` 发布，便于复核。

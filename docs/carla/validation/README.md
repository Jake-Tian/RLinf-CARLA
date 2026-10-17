# 独立源码运行验收

2026-10-17在projgpu13的8张RTX3090、64CPU上验证当前发布代码。async（19048）与colocate（19050）分别完成horizon16、4次实际全局更新，训练命令均rc0。六Actor初始logprob最大差均为0，无缺失梯度，全局裁剪约2，1234份Adam状态FP32，视觉和语言参数总量2,220,993,344全部trainable。完整结果见[summary.json](summary.json)。

验收使用本仓库的模型和processor源码，并在driver及Ray worker中禁止VeRL和StarVLA导入。逐rank来源文件已转换为仓库相对路径，方便对照。模型权重仍由外部资产目录提供，第三方运行库使用已有隔离版本和此次独立安装的fla-core0.5.0。兼容包wheel构建和7项相关CPU测试通过。

19048的训练正常结束，但外层Slurm显示FAILED1:0：第一版验收脚本要求每个rank抽查的权重片段都变化，rank5的语言片段变化为0。第二版保留全部原始记录，改为要求全局视觉、语言均有实际参数变化，逐rank梯度、缺失项、裁剪、FP32状态和LP要求不变，原结果复核通过。19050的colocate也有一个rank的语言抽查片段变化为0，其余rank有变化，正常通过并COMPLETED0:0。没有把作业失败状态改写为成功，也未修改训练数据或logprob。

[source_hashes.json](source_hashes.json)记录GPU验收的源码和配置，确认运行期间未改变。验收后仅补充版权、整理import/空白格式和未使用的诊断变量名，模型计算和import绑定未变，见[post_acceptance_formatting.json](post_acceptance_formatting.json)。整理后重新通过全部7项相关CPU测试和两种模式的结果复核，当前文件hash见[release_source_hashes.json](release_source_hashes.json)。[model_relocation.json](model_relocation.json)证明三个模型适配模块重定位和格式化前后的AST一致，以及23份上游源码hash不变。[audit_acceptance.py](audit_acceptance.py)可从当前clone运行：`python docs/carla/validation/audit_acceptance.py /absolute/path/to/result async`，或将末尾模式换成colocate。

这是现有依赖环境下的源码隔离与短训练验收，不是全新系统安装、完整horizon稳态比较、optimizer恢复或收敛验证。历史19042的八轮长测仍对应yield候选，不能把本次短测记成新代码完成长测。

每种模式还保留了启动日志中的已解析配置，资产绝对路径替换为`<checkpoint_asset>`等占位符，不能直接作为Hydra启动配置。实际运行请使用正式YAML和环境变量。预检记录保存了checkpoint元数据hash。

上游23份源码保持原样，格式工具跳过该目录，`.gitattributes`仅对这部分原样源码关闭空白差异检查，不影响本仓库新增代码的检查。兼容wheel也已在独立目标目录安装，并在禁止VeRL/StarVLA导入的条件下成功加载。

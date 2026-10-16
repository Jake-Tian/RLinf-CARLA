# Qwen3.5 CARLA 全参数 GRPO 源码备份

本目录备份截至2026-10-17已验证的代码和配置，用于恢复实验代码。它是原仓库main上的源码快照，不是新的默认安装入口。依赖拆分、可移植安装和对外发布整理尚未进行。

## 内容与恢复顺序

1. 以manifest.json中public_base_commit对应的完整RLinf-CARLA仓库为基底。
2. 将source/async/覆盖到基底的一个独立恢复目录，获得async适配、必要框架修改、完整梯度审计与FLA命名空间shim。patches/rlinf_async.patch是框架差异的另一种恢复方式，两者不重复应用。
3. 需要colocate时再覆盖source/colocate/。该层只包含相对async的同步路径差异及同步入口。
4. dependencies/autopilot.json记录外部commit，dependencies/autopilot_source/保存模型加载、动作/历史tokenizer和processor的必要源码闭包。可覆盖到对应外部checkout进行恢复；当前导入关系未改变。autopilot_runtime.patch只保存改变AST的必需运行时差异，空补丁表示没有此类差异。只删除许可证注释的本地清理不纳入补丁，快照保留原许可证声明。
5. dependencies/versions.json记录实际依赖层的版本和checkpoint元数据hash。保留原checkpoint、词表、normalization和动作语义。权重、数据、第三方运行库、CARLA二进制、缓存和原始日志不在备份中。FLA的shim不等于完整安装包。
6. configs/是历史参测base config，最终覆盖参数、GPU布局及环境变量需结合scripts/、manifest.json和results/读取。原脚本保留旧路径和截止时间，仅作为启动记录，恢复时必须设置新的资源与时间预算。历史grpo_runtime.sh仅setup_grpo_runtime用于参测，其他清理函数不是推荐入口。

运行python verify_backup.py可验证文件hash、Python语法及框架补丁是否能应用到public基底，不加载模型、不提交GPU作业。

## 已验证范围

- 模型alpamayo2-2B-sft，原生Qwen3.5，离散acceleration/curvature token，语言与视觉全参数训练。
- async：18911/18915/18944均3轮96次实际全局更新并正常退出。推荐六Actor/双Rollout，staleness1，global6/micro1，12环境/group2/horizon128，稳态0.136661 updates/s。
- colocate：18943仅horizon16、4更新及数值/梯度检查通过，未完成完整horizon稳态比较。
- 三Rollout对照18959：正常完成96更新，稳态0.129996 updates/s，比18911低4.88%。全8卡util78.81%，但Actor训练177.48秒高于双Rollout的121.53秒。它是已测对照，不替换推荐配置。GPU利用率高不等于更新吞吐高。
- 生成batch1、use_cache=false，单次生成内视觉复用。旧行为logprob和版本保留，训练使用原始图像。原子同步保护、完整GRPO group、限流/背压、FSDP精度与全局裁剪修复均随源码备份。
- 缺失：全参数长期训练与完整optimizer checkpoint恢复、colocate稳态公平对照、干净环境安装复现。不能宣称async必然优于colocate。

## 来源与许可

框架沿用仓库Apache-2.0。外部Alpamayo兼容模块来自Autopilot-VeRL，commit和URL见dependencies/autopilot.json，许可见dependencies/Autopilot-LICENSE；保留各文件原有版权声明。本备份不上传模型权重，也不声明额外权重分发权。

# Track B 独立执行阶段

版本：2026-10-17。状态：开发与运行计划，实际 GPU 进展以回传日志为准。

本文件配合[主说明](TRACK_B.md)与[Codex Cloud 开发指令](CODEX_CLOUD_PROMPT.md)使用。现有 HANDOFF 记录原 Phase 1–2 已完成、Phase 3 待开始；以下编号按最新独立方案重新组织，已有有效代码和环境直接复用。

## 执行方式与默认配置

Codex Cloud 负责仓库代码、配置、CPU 检查和运行说明。CONCH/MUSK 缓存、VLM 训练与评估由组员在 Windows RTX 4080 Laptop 上执行，显存按设备实测值配置，当前目标设备为 12 GB。用户本人电脑承担项目沟通与材料查看。

组员通过 git 或代码归档取得版本，按 PowerShell 运行说明启动 Python 命令，回传日志和小型结果文件。Cloud 根据这些结果修正代码、配置和下一轮任务。GPU 可见性决定实测状态，Cloud 持续完成各 GPU 阶段的实现和命令准备。

启动模型为 Qwen/Qwen2-VL-2B-Instruct，BF16、语言端 LoRA、视觉主干初始冻结。配置优先匹配组员设备；显存需要时采用四位量化基座加 LoRA，批大小和梯度累积根据实测调整。merger、视觉适配或模型升级服务明确的训练问题，并适配同一 GPU。资源选择依据任务能力与实际运行条件，成本从取舍条件中移除。

每阶段汇报执行位置、完成内容、产物、当前结果和下一项任务。运行时长估算服务交付排期。新增工作直接改善数据、训练、推理或应用流程。

## Phase 0：接续现有工程并写入独立范围

**位置：Codex Cloud。**

以工作仓库固定提交为起点，读取 HANDOFF、现有说明、data.py、encoders.py 和基础脚本。复用已完成的环境与数据功能，将 B 自有数据、困难挖掘、描述符、评估和演示写入执行配置。A 的交接状态从运行依赖中移出。

原先围绕小模型成本、额外原理验证与答辩消融的安排由本版任务范围接替。模型更新围绕 R1 鉴别、R2 区域解释、迭代 R3 与类别反馈推进。凭据通过组员环境管理，仓库保存代码和配置。

**产物与完成条件：**最新工作范围、现有功能清单、待实现模块清单及版本记录完成，开发任务从现有代码继续。

## Phase 1：B 自主数据、角色与类别反馈

**位置：Codex Cloud；真实文件映射在组员机器完成。**

将现有 manifest_standin.csv 的记录提升为 B 正式 data/manifest.csv，保留 tile_id、标签和既有分区，来源元数据标为 B。运行固定读取该清单；后续 A 产物通过显式导入形成新的运行版本。

默认 train 为每类 2,000 张，共 18,000 张。使用固定种子 seed=0，在每类 train 中划出 200 张开发样本：train_dev 共 1,800 张，train_fit 共 16,200 张。角色保存为独立 sidecar，保持原 manifest 字段。CRC-VAL-HE-7K 的 pool/eval 各 3,590 张沿用现有划分并冻结。

R1/R2/R3 接受 train_fit；阈值、checkpoint 和停止条件使用 train_dev；反馈接受登记的 pool ID 与 train_fit 回放；eval 服务阶段报告。其余 NCT 真标签图块按实际错误类型补入 train_fit，并登记新增 ID。

反馈记录写入 data/feedback/feedback_records.jsonl，包含 tile_id、corrected_label、feedback_source、查询时预测、时间和模型/目标版本。来源使用 simulated_dataset_label 或 expert_label。维护同图块当前有效标签，真实专家纠正覆盖较早标签，并刷新相关目标与回放索引。

**产物与完成条件：**正式清单、角色 sidecar、来源元数据和类别反馈 schema 可被各构建器读取；数据来源切换为显式操作。

## Phase 2：教师数据与训练骨架并行实现

**位置：Codex Cloud。**

**教师数据线：**复用 encoders.py，实现带 tile_id 索引的缓存与恢复。困难对启动使用 STR/MUS、DEB/MUC、NORM/TUM；在 B 的 train_fit 上用 CONCH embedding 进行五折逻辑回归，生成逐图块交叉验证分数和困难采样权重。初始困难图块权重为普通图块三倍。

建立 B 自管的 configs/descriptors_crc.yaml。官方 prompts_nct.json 提供九类类别提示，区域形态描述由 B 单独维护。实现三行三列、112 像素、步长 56 的裁剪，横纵起点为 0、56、112，保存原图坐标。CONCH 提供区域描述符匹配，MUSK 辅助筛选，接受条件在 train_dev 上调整。

**训练骨架线：**实现 R1 builder、LoRA 训练、九类评分、解释、评估，以及 R2/R3/feedback 构建器。R1 先使用真实类别推进；区域目标接入后复用同一训练入口。增加 checkpoint 保存恢复、目标版本、有效标签覆盖及运行日志。

九类评分固定候选答案、提示与计分规则；解释接收同图像及评分确定的预测类别。R2 训练目标围绕真实类别构建，推理解释围绕预测类别展开。

**产物与完成条件：**各阶段代码与配置接口贯通，教师目标缓存和分类训练可以分别运行。

## Phase 3：Cloud CPU 检查与 Windows 运行包

**位置：Codex Cloud。**

使用小型 fixtures 检查 manifest 路由、类别反馈覆盖、模板字段、裁剪坐标和评分边界。fixture 只服务工程检查。预备真实 GPU 的全部阶段命令与配置，处理 Windows 路径和 Python 子进程入口。

组员使用 PowerShell，路径通过环境变量或 YAML 指定。以下是 coding agent 需要实现的目标接口：

```powershell
python scripts/run_pipeline.py --stage prepare --config configs/laptop_4080.yaml
python scripts/run_pipeline.py --stage cache --config configs/laptop_4080.yaml
python scripts/run_pipeline.py --stage zeroshot --config configs/laptop_4080.yaml
python scripts/run_pipeline.py --stage r1 --config configs/laptop_4080.yaml
python scripts/run_pipeline.py --stage r2 --config configs/laptop_4080.yaml
python scripts/run_pipeline.py --stage r3 --config configs/laptop_4080.yaml
python scripts/run_pipeline.py --stage feedback --config configs/laptop_4080.yaml
python scripts/run_pipeline.py --stage report --config configs/laptop_4080.yaml
```

运行说明覆盖数据与模型位置、阶段前置产物、恢复方式，以及日志、指标、代表预测和错误摘要的回传位置。零样本评估加载原始模型，冒烟适配器单独保存。

**产物与完成条件：**全阶段代码、配置和运行说明完成；Cloud 检查结果与待 GPU 实测项目分别登记，组员可按顺序执行。

## Phase 4：组员 GPU 冒烟、零样本与正式 R1

**位置：组员 Windows GPU；Cloud 根据回传修正。**

执行 prepare 和 cache 的训练特征、困难清单部分，核对真实数据映射；区域缓存安排在 R2 前完成。使用约 50 条样本、20 步检查完整训练，重点确认监督标签 mask、可训练参数、checkpoint 恢复和九类评分。记录实际显存、步速、精度与视觉 token 数。

重新加载原始模型输出 zeroshot。R1 训练九分类和困难对任务，使用真标签及困难采样；根据 train_dev 的持续错误补充对应 NCT 图块或调整适配模块。选定模型后报告固定 eval 结果。

**产物与完成条件：**真实 smoke 日志、zeroshot、r1_sft checkpoint 与指标回传。后续运行以实测配置继续。

## Phase 5：R2 区域证据训练

**位置：组员 GPU。**

围绕每张 train_fit 图块的真实类别，利用区域分数构建 Diagnosis、Supporting、Against、Conclusion。自动教师提供逐图块弱监督，目标说明相对支持强弱，并保留区域 ID 和坐标。模板或语言模型改写沿用已有证据含义。

继续训练后，依据 train_dev 的分类表现和结构化输出质量选取 checkpoint。推理类别与解释使用同一个预测。教师匹配通过率用于调整自动目标构建。

**产物与完成条件：**r2_grounded 模型、有效训练目标和可直接显示的区域解释保存，固定 eval 输出完成。

## Phase 6：完整迭代 R3 自训练

**位置：组员 GPU。**

当前模型对困难 train_fit 图块生成候选，初始每图四至八条。真实训练标签用于事后筛选类别，引用格式和教师规则筛选自动证据。处理重复答案，记录构建与接受数量。

有效候选与高质量 R2 数据混合继续训练，初始约一半使用回放数据。默认规划两轮生成、筛选、更新，根据 train_dev 选择下一轮模型并记录实际完成轮数；平台期优先调整样本、监督和可训练模块。每轮保存 checkpoint，r3_selftrain 映射到选定版本。

自监督在具体视觉表征或域适配问题出现时启用：将同图语义保持视图的一致性与区分目标接到学生可训练视觉模块，服务后续分类训练。

**产物与完成条件：**迭代循环可恢复，实际轮次、样本和模型版本完整记录，完成轮次有相应指标。

## Phase 7：类别纠正驱动更新

**位置：组员 GPU；类别纠正在 B 演示中录入。**

从 train_dev 当前最佳可用模型开始，保存 anchor_checkpoint 和 model_version。每批先按当前模型在剩余 pool 中的较低 margin 选定图块 ID，再收取这些 ID 的类别反馈；模拟入口仅在选定后读取对应 manifest.label。margin 为前两候选归一化分数之差。

累计 0、50、100、200 指获得类别反馈的独立 pool 图块数 n_feedback。另记 n_changed，即反馈类别与查询时预测不同的图块数；确认原类别也计入有效监督。真实反馈按实际到达数量推进，fb_{n} 中的 n 使用实际 n_feedback。

反馈训练使用 corrected_label 的分类损失，混合既往有效类别反馈与 train_fit 的 R2 回放。同图块类别修订替换旧目标与旧反馈回放，相关缓存和索引同步更新。模拟标签和专家标签分别记录来源及实际数量。类别反馈直接参与分类学习，解释保持自动教师来源。

更新后记录新模型和固定 eval 结果，并继续接收下一批纠正。从反馈 checkpoint 再次启动 R3 时，R3 仍使用 train_fit。

**产物与完成条件：**有效类别反馈表、累计节点与实际数量、fb_{n} 模型、指标和版本映射齐备。

## Phase 8：B 自有演示与提交

**位置：Cloud 实现；组员机器运行和导出。**

提供轻量本地演示：打开图块，查看组织类别、竞争候选及引用区域，录入类别纠正，加入模型更新队列，查看更新后的版本和后续图块结果。批量更新由组员机器执行命令完成。

B 自行生成困难对误判率与错误数量、九类平衡准确率、逐类召回率、混淆矩阵和反馈曲线。保留 preds_vlm_{stage}.csv 的 tile_id、pred、九列 p_<CLASS>、top2、margin，以及 rationales 的区域字段；stage 沿用 zeroshot、r1_sft、r2_grounded、r3_selftrain、fb_{n}。结果字段采用主说明第 10 节的 B 自有合同，运行版本映射和状态保存在 run_manifest。

团队结果冻结为 2026-10-18 15:30，America/Chicago。官方提交为 2026-10-21，摘要及可独立阅读 PPT/PDF，主体上限 20 页。演示和提交材料使用实际运行结果，交代三个目标的完成进展。

**产物与完成条件：**B 自有结果图、演示、模型更新记录和提交材料可直接交接。

## 固定依据

- [工作仓库 HANDOFF](https://github.com/alin4638/hardpair-trackb-handoff/blob/18f5f0a3266f91cb0d40dff80a9ab16ffe8949de/HANDOFF.md)：原阶段状态与现有工程。
- [工作仓库 CLAUDE](https://github.com/alin4638/hardpair-trackb-handoff/blob/18f5f0a3266f91cb0d40dff80a9ab16ffe8949de/CLAUDE.md)：原数据、训练和输出合同；本版采用最新独立范围。
- [官方参考仓库](https://github.com/ncsa/ashby-hackathon-2026/tree/78576d3824c70679053db34458d79cc0df70d259)：编码器、数据和运行示例参考。
- 本次附件 2026_Ashby_Hackathon_Opening_Session.pptx 第44页：提交日期、格式与主体页数。

自主清单、类别反馈范围和 Cloud/GPU 分工遵循本轮用户确定的执行方式；完整范围参见 [TRACK_B.md](TRACK_B.md)。

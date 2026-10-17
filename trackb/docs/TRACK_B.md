# Track B：困难组织鉴别、区域证据与类别反馈学习

版本：2026-10-17，独立执行版。配套文件：[阶段规划](TRACK_B_PHASES.md)、[Codex Cloud 任务提示词](CODEX_CLOUD_PROMPT.md)。

本文定义接下来的实现目标。现有实现状态以仓库代码和真实运行记录为准；下文列出的新增模块、命令和结果属于待实现交付。

## 1. 已确定的任务与执行条件

做一个结直肠组织图块复核助手：接收图像，区分容易混淆的组织类别，显示对应区域的鉴别依据，接收类别纠正，再通过模型更新改善后续图块的表现。

本版吸收 Track B2 的三项目标：困难鉴别、基于区域证据的解释与迭代自训练、反馈驱动的持续学习。比赛取舍依据为任务效果、实际功能进展和交付完成度。计算成本从取舍条件中移除。每个新增模块应明确它将改善哪一段训练或应用流程。

| 项目 | 当前有效决定 |
| --- | --- |
| B 与 A 的关系 | B 自主管理数据、困难类别对、描述符、训练、评估和演示。A 后续产物可作为显式导入的协作输入。 |
| 专家反馈 | 默认仅提供类别纠正；类别确认同样可形成监督。 |
| 开发位置 | Codex Cloud：代码、配置、可在 Cloud 执行的必要验证、组员运行手册。 |
| GPU 执行位置 | 组员的 RTX 4080 Laptop 机器：编码器缓存、模型训练、推理与实际评估。仓库记录为 Windows、Python 3.12、12 GB 显存，启动时探测实际环境。 |
| 用户本人电脑 | 文档与代码交接、结果查看。 |
| 默认交接方式 | Git 提交或代码归档交接；组员执行 GPU 命令，回传小型结果和诊断日志。 |
| 原方案的验证性扩展 | 图像置换、遮挡对照、答辩专用消融和独立原理验证移出本次主清单。 |

目标与开发交接信息已齐备。训练目录、CUDA、可用磁盘、模型访问状态和每步速度由组员运行启动脚本取得，作为实际配置输入。

## 2. 仓库与已完成起点

- **工作仓库：** [alin4638/hardpair-trackb-handoff](https://github.com/alin4638/hardpair-trackb-handoff)，本次核对提交 `18f5f0a3266f91cb0d40dff80a9ab16ffe8949de`。
- **官方参考仓库：** [ncsa/ashby-hackathon-2026](https://github.com/ncsa/ashby-hackathon-2026/tree/main)，本次核对提交 `78576d3824c70679053db34458d79cc0df70d259`。用途为模型接口、数据类别和赛事工具参考。
- 工作范围集中在工作仓库的 `trackb/`。先读已有项目指导与交接文档，再把本次用户确定的范围写入 `trackb/AGENTS.md`，并同步 `trackb/CLAUDE.md` 中对应的执行约定。保留其余有效项目规则。

`HANDOFF.md` 记录 Phase 1–2 完成：数据读取、路径映射、训练样本入口和本地 CONCH/MUSK 编码器已搭建，相关基础检查已通过。Phase 3 处于待开始状态。复用 `tb/data.py`、`tb/encoders.py` 和已有 smoke 脚本，从当前进度继续。

官方 `prompts_nct.json` 提供九个类别的简短文本，适合初始化类别映射。B 的区域形态描述符由本项目单独维护。官方服务运行于 Delta 的安排作为参考背景；本项目 GPU 执行采用组员机器。[S1–S4]

## 3. B 自有数据合同

### 3.1 正式 manifest

将仓库现有 `data/manifest_standin.csv` 的内容提升为 B 的正式 `data/manifest.csv`：保留图块 ID、相对路径、类别与当前 train/pool/eval 划分，更新来源说明并记录 `manifest_version`。这份清单从本版起成为 B 的正式实验依据。

`manifest_source()` 改为读取 B 的显式配置。未来协作清单通过明确的导入步骤建立新运行版本；已经冻结的 B 运行继续使用其原版本。

基础列为 `tile_id,path,label,split`。本地实际路径由可配置的 `data_root` 解析。路径默认指向组员机器的 `trackb/data/raw/`，同时支持配置到已有数据目录。缓存记录数据集与相对图像路径、tile_id、manifest 版本和模型/处理器版本，供跨机器和断点恢复复用。

### 3.2 数量、用途与开发集

| 数据角色 | 来源与默认数量 | 用途 |
| --- | --- | --- |
| 基础 train | NCT-CRC-HE-100K，现有每类 2,000 张，共 18,000 张 | B 的初始训练样本集合 |
| train_fit | 上述 train 中每类 1,800 张，共 16,200 张 | 困难挖掘、自动目标生成、R1/R2/R3 参数训练 |
| train_dev | 上述 train 中每类 200 张，共 1,800 张 | checkpoint、阈值和训练停止条件选择 |
| pool | 现有 CRC-VAL-HE-7K 划分中的 3,590 张 | 查询与取得类别反馈 |
| eval | 现有 CRC-VAL-HE-7K 划分中的 3,590 张 | 固定阶段报告与最终结果 |
| extra_train | NCT 中基础清单之外的有标签图块 | 按持续发生的错误补充训练 |

train 内按类别、固定 `seed=0` 划出开发集，保存到 `data/roles.csv`，本版按图块划分并记录这一实际粒度。已有开发划分持续复用。九类顺序固定为 `ADI,BACK,DEB,LYM,MUC,MUS,NORM,STR,TUM`。[S2、S4、S5]

普通训练构建器从 train_fit 和登记的 extra_train 取数。反馈构建器接入登记的 pool 反馈与训练数据回放。train_dev 和 eval 分别承担表中的固定用途。

额外 NCT 图块直接使用已有真实类别。R1 先采用基础清单中的 16,200 张 train_fit 图块训练，再根据开发集错误补入有效样本。新数据角色、数量与来源写入运行配置。

## 4. 困难样本与本地描述符

启动困难类别对为 `STR/MUS`、`DEB/MUC`、`NORM/TUM`。B 在 train_fit 的 CONCH 图像特征上进行五折交叉验证，获得每张训练图块由其他折训练出的分类器预测。该预测用于选择困难图块，选取过程写入 `data/hard_train.csv`。

困难排序同时考虑错误预测和真实类别相对竞争类别的得分。`true_rival_margin` 表示真实类别得分减去配对类别得分；数值越小，图块越优先进入困难样本队列，负值表示配对类别得分更高。后续根据模型持续出现的错误更新队列。

自管描述符文件为 `configs/descriptors_crc.yaml`，包含 `descriptor_id`、类别、英文形态描述和来源。每类先整理少量可以在图块中观察的形态特征，优先覆盖三个困难类别对。官方类别短句作为类级提示；区域形态文本通过实际图块和教师匹配结果逐步调整，记录为人工编写的自动监督配置。

CONCH 用于主要图文匹配，MUSK 提供辅助分数。图像和文本采用各自编码器匹配的投影空间。两套教师分阶段加载、输出缓存，随后释放显存供 VLM 使用。缓存按消费阶段分别构建与恢复：R1 先使用训练特征和困难清单，R2 前完成所需区域缓存。现有 `encoders.py` 可继续承担编码职责。[S2–S4]

## 5. 学生模型与训练配置

默认主力沿用 `Qwen/Qwen2-VL-2B-Instruct`，在语言注意力与 MLP 投影上训练 LoRA 参数，视觉主干初始冻结。LoRA 使用附加的低秩参数学习任务行为，便于保存训练阶段和切换版本。

该选择利用当前已经搭建的接口，并匹配指定的 4080 Laptop 执行条件。训练配置记录精度、实际可训练模块、视觉 token 数和显存峰值。支持 bf16 时优先采用 bf16；单卡批大小从 1 起，通过梯度累积形成有效批次。实际显存需要时采用四位量化基座加 LoRA、梯度检查点和合适的序列长度，按组员机器的实测结果确定。

视觉与语言连接需要适配时启用 merger/projector；视觉表征出现明确瓶颈时再扩大视觉可训练范围。基座升级同样服务明确的任务问题，并在指定 GPU 上完成运行检查。训练步骤和模型选择依据开发表现与比赛时间窗口安排。

训练实现采用模型对应的聊天模板和图像处理器。SFT 损失作用于 assistant 目标词元；用户提示、图像占位符与填充位置采用相应掩码。保存 adapter、可训练视觉参数、优化器、学习率调度器、随机状态、训练步数与基础模型版本，支持跨阶段加载和中断续训。

## 6. R1：困难组织鉴别

R1 以正确类别为核心监督，包含九分类任务和困难类别对任务。困难图块的初始采样权重为普通图块的三倍。先跑通类别监督，再接入困难清单；类级短说明可作为辅助文本，逐图块区域理由由 R2 提供。

目标构建记录图像、用户提示、assistant 答案、tile_id、监督来源和目标版本。每轮根据 train_dev 的相互误判和逐类召回率调整取样与有效监督。R1 完成后保存选择的 checkpoint，并输出固定 eval 结果。

B2 中的自监督用于明确的视觉表征或域适配需求：以同一训练图像的语义保持视图构建学习目标，接入可训练的学生视觉模块。图像自身产生监督信号时记录为自监督；教师伪标签、自生成答案分别记录其实际来源。当前主执行路径先完整落实 R1、R2、R3 和类别反馈，表征适配按观察到的问题接入。

## 7. R2：区域证据与鉴别解释

原始图块采用三行三列重叠区域，每个裁剪为 112 像素；224 像素图像的横纵起点均为 0、56、112。保存区域 ID、原图坐标和变换后的坐标映射。

用教师编码器比较裁剪与本地描述符。CONCH 主分数决定自动目标是否被接受，MUSK 辅助分数用于定位教师分歧、调整描述符与训练材料。阈值在 train_dev 上选择，并记录接受数量与输出质量。

训练目标围绕真实类别组织支持证据，围绕困难类别对或当前竞争候选组织鉴别比较。模板首先输出：

```text
Diagnosis: <class>
Supporting: <visible descriptor> [region row,col]
Against <rival>: <relative support comparison> [region row,col]
Conclusion: <brief conclusion>
```

`Against` 表达该区域对竞争类别的相对支持强弱。该信息由图文匹配提供，记录为自动区域证据。推理时，解释使用同一图像、已经确定的模型预测与竞争候选，界面类别和解释标题共用该预测。

R2 目标接入同一训练引擎。教师匹配与引用格式用于选择可用自动目标，典型区域图直接用于调整训练材料和应用显示。

## 8. R3：迭代自训练

从当前模型尚需改善的 train_fit 图块中取样，为每张图像生成多个候选答案，起始数量为四至八条。生成输入使用图像和任务提示，训练真值用于生成之后的类别筛选；自动证据规则用于筛选候选引用。

去重后，将有效候选与高质量 R2 数据混合训练，起始回放比例为一半。默认规划两轮“生成—筛选—更新”，每轮根据 train_dev 的结果选择后续模型，并记录实际完成轮数。出现平台期时先调整错误样本、监督内容或可训练模块，再决定后续更新。

pool 的类别反馈通过反馈构建器管理。后续从反馈更新的 checkpoint 继续自训练时，R3 图块来源仍遵循其训练角色。

## 9. 类别反馈学习

### 9.1 反馈内容与查询

本版默认专家提供 `corrected_label`。最小有效记录为：

```json
{
  "tile_id": "<selected_pool_tile_id>",
  "corrected_label": "STR",
  "feedback_source": "expert_label",
  "query_model_version": "<checkpoint_version>",
  "original_prediction": "MUS",
  "timestamp": "<ISO-8601-time>"
}
```

上述记录为格式示例。实际类别来自提供反馈的人或明确启用的模拟数据入口。

两种来源使用固定值：`expert_label` 表示专家提供的类别；`simulated_dataset_label` 表示查询后揭示的数据集原标签。每次运行记录实际来源。初次流程跑通可采用模拟入口，专家类别到达后进入相同训练接口并记录对应来源。

每批先根据当前模型在剩余 pool 上的低置信度选择图块 ID，再取得这些图块的反馈类别。预测第一候选与第二候选得分之差越小，代表当前模型越难区分。模拟入口也按已选 ID 读取标签。

### 9.2 更新与累计节点

从 train_dev 上当前最佳可用 checkpoint 开始，记录 `anchor_checkpoint`。以累计获得有效类别反馈的独立图块数 `n_feedback` 记录 0、50、100、200 等节点；另记 `n_changed`，表示反馈类别与查询时预测不同的图块数。确认原预测的类别也可形成训练监督。实际反馈按到达数量推进，重复修订同一图块更新其有效版本。

训练使用当前有效类别目标、既往有效类别反馈和 train_fit 中的 R2 回放。类别反馈始终进入分类损失；区域与理由继续采用自动监督来源。专家的区域绘制、描述符修订和理由编辑留在本版范围之外。

新的类别修订成为该图块的当前有效训练标签，替换旧的类别目标和反馈回放记录。与旧类别冲突的自动训练目标退役，目标缓存与索引同步刷新。历史反馈保留在 `data/feedback/feedback_records.jsonl`，训练时解析当前有效版本。

更新后在固定 eval 上记录该版本表现，并继续接受下一批类别反馈。真实反馈与模拟反馈在输出中保留清晰的来源字段，变化只按实际运行结果陈述。

## 10. 推理、指标与输出合同

分类使用固定九类候选答案评分：将一个候选答案内各词元的条件概率取自然对数后相加，得到该类别的序列分数；再对九个分数取指数，并分别除以九个指数值之和，得到归一化类别得分。条件概率指模型在图像、提示和已经给出的答案词元条件下生成下一词元的概率。类别答案、词元长度和计分规则跨阶段保持一致。

主要指标用于选择训练方向和展示实际进展：困难类别对相互误判率、九类平衡准确率、逐类召回率、结构化输出完成率和教师区域匹配通过率。平衡准确率是九个类别召回率的平均；一个类别的召回率是该类被正确识别的图块占该类全部图块的比例。某困难对的相互误判率，是两类互相被误判的图块数之和除以这两类全部评估图块数。

train_dev 用于模型选择；固定 eval 用于阶段报告。B 自有零样本结果及各训练阶段形成主要进展图。困难挖掘已有的 CONCH 分类器可作为同一 B 划分上的参考，标明训练样本与版本。

| 输出 | 本版合同 |
| --- | --- |
| `results/preds_vlm_{stage}.csv` | `tile_id,pred,p_ADI,p_BACK,p_DEB,p_LYM,p_MUC,p_MUS,p_NORM,p_STR,p_TUM,top2,margin`；top2 为第二候选类别，margin 为前两候选归一化得分之差 |
| `results/vlm_rationales_{stage}.jsonl` | `tile_id,pred,rationale,cited_regions,grounding_pass`，并记录模型版本与自动证据来源 |
| `results/vlm_summary.csv` | `run_id,stage,model_version,manifest_version,n_eval,balanced_accuracy,pair_error_STR_MUS,pair_error_DEB_MUC,pair_error_NORM_TUM,format_rate,grounding_pass_rate` |
| `results/vlm_feedback_curve.csv` | `run_id,stage,feedback_source,anchor_checkpoint,model_version,n_feedback,n_changed,n_eval,balanced_accuracy,pair_error_STR_MUS,pair_error_DEB_MUC,pair_error_NORM_TUM` |
| `outputs/run_manifest.json` | 代码/数据/描述符版本、设备、配置、实际完成阶段和 checkpoint 映射 |
| `outputs/target_build_log.jsonl` | 每轮生成、接受、去重与训练数量 |

阶段名沿用 `zeroshot,r1_sft,r2_grounded,r3_selftrain,fb_{n}`，其中 n 为实际累计反馈数量。多轮自训练在运行记录内保留各轮模型，`r3_selftrain` 指向选定版本。R1 纯类别输出的解释指标可标为 `not_applicable`，同时记录指标样本数；各指标据其实际生成的产物填写。

## 11. Cloud 开发与组员 GPU 运行

Codex Cloud 完成全流程实现：数据角色、困难挖掘、裁剪、目标构建、训练、评分、解释、R3、类别反馈、结果导出和轻量本地应用。Cloud 使用适配其环境的依赖和小型数据样例，验证关键数据流、目标格式、区域坐标、反馈覆盖及命令入口。

GPU 侧加载、梯度更新、显存、吞吐和模型效果由组员机器实际运行。现有工作环境优先复用；CONCH、MUSK 和 VLM 按阶段使用单块 GPU。运行状态分别记录为 `code_ready`、`cpu_checked`、`gpu_pending`、`gpu_verified`，每种状态附对应证据。

coding agent 需要交付以下新增入口；这些名称是本次目标接口：

```text
python scripts/run_pipeline.py --stage prepare --config configs/laptop_4080.yaml
python scripts/run_pipeline.py --stage cache --config configs/laptop_4080.yaml
python scripts/run_pipeline.py --stage zeroshot --config configs/laptop_4080.yaml
python scripts/run_pipeline.py --stage r1 --config configs/laptop_4080.yaml
python scripts/run_pipeline.py --stage r2 --config configs/laptop_4080.yaml
python scripts/run_pipeline.py --stage r3 --config configs/laptop_4080.yaml
python scripts/run_pipeline.py --stage feedback --config configs/laptop_4080.yaml
python scripts/run_pipeline.py --stage report --config configs/laptop_4080.yaml
```

从 `trackb/` 执行。入口支持配置检查、明确的阶段前置产物、断点恢复和日志目录。组员手册提供可直接使用的 PowerShell 命令；路径由 Python `pathlib` 与配置管理。Cloud 的安装依赖、组员的训练依赖分别保存，已有编码器兼容处理随版本记录一起交接。

第一次 GPU 运行先以约 50 条样本、20 步核实训练与恢复流程，再实测安排正式轮次。零样本阶段加载原始基座，冒烟 adapter 使用独立目录。较长任务采用断点保存并给出实测完成时间估计。

GPU 结果回传包包含设备诊断、有效运行配置、训练日志、上述小型结果文件、代表性区域案例与运行 manifest。权重、完整图像数据和大型缓存保存在组员的执行环境；模型由执行者使用相应授权来源取得。任何实际远程执行通过已经建立并获授权的连接开展；当前默认使用代码与结果交接。

## 12. 独立演示与比赛交付

实现 B 自有轻量本地 Web 演示，默认采用单一轻量框架。流程为：打开图块，查看类别与竞争候选，查看自动区域证据，选择正确类别并提交反馈，进入更新队列，切换实际生成的新模型版本后查看后续图块表现。

批量训练由运行入口执行，界面显示队列和真实模型版本。开发样例用于界面开发；比赛案例使用真实保存的模型输出与反馈记录。

结果以困难组织鉴别、区域证据、类别纠正后的模型变化三部分组织。团队结果冻结时间沿用 **2026-10-18 15:30，America/Chicago**。所提供开场材料规定 **2026-10-21** 提交摘要和可独立阅读的 PPT/PDF，主体最多 20 页。[S6]

## 13. 新 agent 的完整交接清单

必需的开发上下文由两个仓库 URL 和本次三份 Markdown 组成：`TRACK_B.md`、`TRACK_B_PHASES.md`、`CODEX_CLOUD_PROMPT.md`。把文档放入工作仓库 `trackb/docs/`，并在 Codex Cloud 中选择该工作仓库；提示词正文用于启动任务。Cloud 的仓库 checkout、setup script 和变更 diff 工作方式见官方说明。[S7、S8]

现有 `HANDOFF.md`、`trackb/README.md`、基础代码和初始清单已经随工作仓库提供。B2、旧 curr、旧 phases 的当前有效要求已吸收到本版。组员机器已有环境、数据与模型可直接复用；可用路径和首轮诊断日志作为运行时交接信息。

## 14. 资料依据

- **[S1]** [工作仓库 HANDOFF.md，固定提交](https://github.com/alin4638/hardpair-trackb-handoff/blob/18f5f0a3266f91cb0d40dff80a9ab16ffe8949de/HANDOFF.md)：完成状态、硬件、交接范围。
- **[S2]** [data.py](https://github.com/alin4638/hardpair-trackb-handoff/blob/18f5f0a3266f91cb0d40dff80a9ab16ffe8949de/trackb/tb/data.py)、[encoders.py](https://github.com/alin4638/hardpair-trackb-handoff/blob/18f5f0a3266f91cb0d40dff80a9ab16ffe8949de/trackb/tb/encoders.py)、[README.md](https://github.com/alin4638/hardpair-trackb-handoff/blob/18f5f0a3266f91cb0d40dff80a9ab16ffe8949de/trackb/README.md)：已存在的数据/编码器实现与环境说明。
- **[S3]** [官方 README](https://github.com/ncsa/ashby-hackathon-2026/blob/78576d3824c70679053db34458d79cc0df70d259/README.md)、[STUDENT_GUIDE.md](https://github.com/ncsa/ashby-hackathon-2026/blob/78576d3824c70679053db34458d79cc0df70d259/STUDENT_GUIDE.md)：赛事模型工具与服务使用方式。
- **[S4]** [官方 prompts_nct.json](https://github.com/ncsa/ashby-hackathon-2026/blob/78576d3824c70679053db34458d79cc0df70d259/prompts_nct.json)：九个 CRC 组织类别和类别短句。
- **[S5]** [Zenodo 1214456](https://zenodo.org/records/1214456)：NCT-CRC-HE-100K 与 CRC-VAL-HE-7K 数据说明。
- **[S6]** 用户提供的 `2026_Ashby_Hackathon_Opening_Session.pptx` 第 44 页：提交时间与主体页数；第 45 页：评审维度。团队冻结时间来自当前工作仓库交接说明。
- **[S7]** [OpenAI 官方 Codex Cloud 环境说明](https://learn.chatgpt.com/docs/environments/cloud-environment)：工作仓库、setup script 与代码 diff。
- **[S8]** [OpenAI 官方 AGENTS.md 说明](https://learn.chatgpt.com/docs/agent-configuration/agents-md)：项目指导文件的使用。
- **[S9]** 用户提供的 `Track-B2-(TBD).txt`：困难鉴别、区域证据与迭代自训练、专家反馈学习三项目标。

独立 B、类别反馈范围、Cloud/GPU 分工与成本取舍来自 2026-10-17 本次用户明确要求；数据角色、文件合同和实现顺序为据此确定的执行设计。

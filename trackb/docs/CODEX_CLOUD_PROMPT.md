# 给 Codex Cloud coding agent 的执行提示词

请把下述 Track B 项目推进到完整代码交接状态。配套 `trackb/docs/TRACK_B.md`、`trackb/docs/TRACK_B_PHASES.md` 是本次目标与执行说明；本提示词明确工作环境、权限和交付边界。

## 1. 接手与工作范围

工作仓库：https://github.com/alin4638/hardpair-trackb-handoff

已审阅基准提交：`18f5f0a3266f91cb0d40dff80a9ab16ffe8949de`。交接记录为 Phase 1–2 完成、Phase 3 待开始。

官方只读参考：https://github.com/ncsa/ashby-hackathon-2026/tree/main

参考提交：`78576d3824c70679053db34458d79cc0df70d259`。

先读取实际仓库中的 `AGENTS.md`、`CLAUDE.md`、`HANDOFF` 及目录内说明，核对当前 HEAD、工作区改动和基准后的新增成果，复用有效实现。最新用户范围按本文落实，上层有效且相容的规则继续执行。根目录旧 Track B 规划作为历史参考。

在 `trackb/` 内完成代码、配置和文档，创建或更新该目录的 `AGENTS.md`，同步 `trackb/CLAUDE.md` 至当前范围。官方仓库用于查阅，引用必要来源即可。工作围绕 B 独立运行展开，自有数据、描述符、困难样本、评估和演示全部落地。

阶段汇报用于同步进展。持续完成所有可完成的代码工作，常规实现选择自主处理。阶段末说明完成内容、实际检查与下一步，继续推进。

## 2. 开发环境与执行边界

Codex Cloud 负责开发和 CPU 可行的必要检查。在环境 setup 脚本准备 CPU 开发依赖，并配置依赖源与官方仓库读取所需的网络访问。所有 GPU 编码、推理和训练由组员在 Windows、RTX 4080 Laptop 12 GB 上执行；用户本人电脑用于协作。交接默认采用 git 或 ZIP、跨平台 Python 和 PowerShell 命令。

先复用 `tb/data.py`、`tb/encoders.py` 和现有 smoke。以任务效果配置资源，费用从取舍因素中移出。复用组员已成功的 Windows Python 3.12 路线；Qwen2-VL-2B、bf16、语言端 LoRA、初始冻结视觉主干为启动配置，4-bit、梯度累积及可训练视觉模块按实测调整。CONCH、MUSK 分阶段缓存并释放，再加载 VLM。保留已跑通的 tokenizer 适配，明确 Cloud 与 Windows 依赖约束。

新增工作直接改善模型、数据或应用；独立原理验证、答辩专用消融和额外专家验证从清单中移出。实际 GPU 执行由组员操作；Cloud 完成可运行入口、配置、runbook 和结果接收流程。

## 3. 固定 B 自有数据与困难样本

把现有 `data/manifest_standin.csv` 的内容升级为 B 正式 `data/manifest.csv`，保留图块 ID、类别、划分和图像映射。加载器固定读取 B 配置中的清单路径，取代运行时自动选择 A 目录的逻辑。

初始 train 为每类 2,000 张、共 18,000 张；按类固定抽取 200 张作为 train_dev，共 1,800 张，其余 train_fit 共 16,200 张。将本地角色另存并固定 seed=0。pool、eval 各 3,590 张，保持现有划分。参数训练与自动目标生成使用 train_fit，开发选择使用 train_dev，阶段报告使用 eval，反馈按登记的 pool 图块进入训练。

额外 NCT 图块按持续出现的错误补充，直接使用已有真实标签。缓存关联实际图像身份、清单版本及当前 tile_id。原始数据由组员从 Zenodo 1214456 等授权来源恢复；已有数据、模型缓存直接复用。

在 train_fit 上实现 CONCH 特征的五折逻辑回归，生成 B 自有困难样本与类别对。启动类别对为 STR/MUS、DEB/MUC、NORM/TUM，后续按训练结果更新。每类编写三至五条形态描述符并保存来源与版本；官方 `prompts_nct` 提供的是类别短名，形态描述需要在 B 内补充。

## 4. 完成 B2 三条学习主线

**R1：困难鉴别。** 实现九分类、困难类别对训练及困难图块加权，类别真值作为主要监督。额外有标签数据用于解决具体混淆。

**R2：逐图块区域证据。** 在 224 像素原图上建立三行三列、112 像素裁剪、56 像素步长的区域，保存 ID 和坐标，增强时同步映射。CONCH 匹配形态描述，MUSK 辅助筛选；训练目标围绕真实类别与竞争类别，包含 Diagnosis、Supporting、Against、Conclusion 和区域引用。目标来自当前图块的裁剪证据，自动理由标明 teacher 来源。

**R3：迭代自训练。** 从待改善的 train_fit 图块生成多个答案，按真实类别、引用格式和教师证据规则筛选，去重并控制单图贡献，与有效 R2 数据回放混合继续训练。实现重复生成、筛选、更新与 train_dev 选模。该流程称为自训练。自监督扩展在具体视觉表征需求出现时，将明确定义的图像自身监督目标接到学生可训练视觉模块；现有冻结教师编码 API 继续服务目标构建。

**反馈：默认类别确认与纠正。** 将 corrected_label、查询时预测、来源、时间及模型版本写入 `data/feedback/feedback_records.jsonl`。来源固定为 `expert_label` 或 `simulated_dataset_label`。有效专家标签覆盖同图块旧类别目标及旧反馈回放；冲突自动理由退出回放。专家区域、理由编辑和反馈理由生成留在本轮范围之外。

从 train_dev 最佳可用 checkpoint 开始顺序更新。每批先按当前模型剩余 pool 的低 margin 选 ID，再获取这些 ID 的标签；margin 为前两候选归一化分数之差，模拟入口才读取选中行的 manifest.label。每个节点使用截至当前全部有效类别反馈，同图块取当前版本，以类别损失训练，并混合 train_fit 的有效 R2 回放。累计 0、50、100、200 个独立图块为展示节点；`n_feedback` 统计实际获得反馈的图块，`n_changed` 统计反馈类别与查询时预测有差异的图块，确认原预测也计入有效反馈。按实际数量推进，记录 anchor、ID、步数与版本。

## 5. 完整工程入口与交付

实现下列真实接口，命令列出的是待完成目标：

```powershell
python scripts/run_pipeline.py --stage prepare --config configs/laptop_4080.yaml
```

`--stage` 支持 `prepare/cache/zeroshot/r1/r2/r3/feedback/report`，并提供 `--help`、`--dry-run`。完成训练、九类答案似然评分、解释生成、反馈、独立演示和报告导出。类别评分归一化用于固定候选排序；解释使用同图像、该预测与竞争候选。报告困难类别相互误判率、平衡准确率、逐类表现和输出质量。

保留 `preds_vlm_{stage}.csv`、`vlm_rationales_{stage}.jsonl`、`vlm_feedback_curve.csv`、`vlm_summary.csv`。演示支持选图、显示预测与区域、录入类别纠正、写入更新队列和展示已有新版本结果；更新操作对应真实保存的 checkpoint。

训练保存 adapter、可训练视觉参数、optimizer、scheduler、RNG、step 及配置，支持恢复。交付 `requirements-cloud.txt`、Windows 训练环境约束和 PowerShell runbook，覆盖 setup、数据恢复、各阶段运行、续跑与结果导出。必要 CPU 检查围绕命令、数据角色、目标构建、坐标与报告接口，展示实际执行输出。

完成后给出可审阅 diff、实现清单与状态：`code_ready` 表示代码齐备，`cpu_checked` 表示已完成对应 CPU 检查，`gpu_pending` 表示等待组员执行，`gpu_verified` 依据实际回传确认。附组员执行命令及首轮回传清单：运行配置、环境版本、样本数、峰值显存、步耗时、损失、错误日志、checkpoint 标识及小型结果文件。训练权重保留在组员的合法运行环境。

## 使用方法与交接清单

1. 将三份新 Markdown 放进工作仓库的 `trackb/docs/`，在 Codex Cloud 选择该仓库，粘贴本提示词正文并保留两个 URL，即可开始。
2. 三份文档与两个 URL 已包含启动所需上下文。原 B2、curr、phases 内容已吸收；原活动 slides 可作额外背景参考。
3. 组员依据 Windows runbook 恢复数据、模型缓存与所需凭证，再执行 GPU 阶段。模型权重通过组员机器 setup 获取，交接以代码、说明和小型成果为主。
4. 将首轮诊断与结果文件回传给 coding agent，继续完成实际运行问题修复与模型迭代。

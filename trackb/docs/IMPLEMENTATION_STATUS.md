# Track B 代码交付状态

日期：2026-10-17。基准 HEAD：`18f5f0a3266f91cb0d40dff80a9ab16ffe8949de`，当前分支 `main`。
未新建/切换分支。用户已授权将交付提交并推送至 `alin4638/hardpair-trackb-handoff` 的现有 `main`；
发布提交以 `git log -1` 与交付消息中的提交号为准。所有项目修改限定在工作仓库 `trackb/`。
本次执行位置实际为本地 Codex 桌面环境，完成了 Cloud 可复用的 CPU 开发与验证，
不将本地执行称为已在 Codex Cloud 运行。没有访问 Lop/Delta，没有下载预训练权重或数据集。

## 交付模块与证据

| 模块 | 状态 | 证据/产物 |
|---|---|---|
| B 正式 manifest、固定角色、hash 与路径解析 | code_ready / cpu_checked | `data/manifest*.json,csv`、`data/roles*.json,csv`；与 stand-in 逐记录一致 |
| train_fit / train_dev / pool / eval 隔离、额外 NCT 真标签登记 | code_ready / cpu_checked | `tb/data.py`、`scripts/register_extra.py`；16200/1800/3590/3590 |
| CONCH 五折 OOF 困难挖掘、权重与 pair margin | code_ready / cpu_checked / gpu_pending | `tb/hard.py`；CPU 合成特征检查；真实教师特征未生成 |
| 27 个 B 自有形态描述符、9 个区域与坐标映射 | code_ready / cpu_checked | `configs/descriptors_crc.yaml`、`tb/crops.py` |
| 两教师逐图块缓存、图像 hash、模型 revision、开发阈值 | code_ready / cpu_checked / gpu_pending | `tb/cache.py`、`tb/grounding.py`；真实编码器运行待组员 |
| R1/R2 builder、assistant mask、九候选序列评分 | code_ready / cpu_checked / gpu_pending | `tb/build_sft.py`、`tb/scoring.py`、`tb/model.py` |
| LoRA / 可选视觉参数 / optimizer / scheduler / RNG 恢复 | code_ready / cpu_checked / gpu_pending | `tb/train_lora.py`；CPU 随机小模型真实训练与恢复参数完全一致 |
| 两轮 R3 生成—筛选—去重—回放—开发选模 | code_ready / cpu_checked / gpu_pending | `tb/pipeline.py`；模拟后端完整两轮与重复恢复测试 |
| 先查询后取标签、累计反馈、专家覆盖、同数量修订 | code_ready / cpu_checked / gpu_pending | `tb/feedback.py`；2→4 条累计训练与再次修订测试 |
| 阶段评估、recall、混淆矩阵、反馈曲线与回传包 | code_ready / cpu_checked / gpu_pending | `tb/evaluate.py`、`report`；分母/恢复测试；真实模型指标未取得 |
| Streamlit 演示、版本、区域、反馈与队列 | code_ready / gpu_pending | `app.py`；语法检查，真实图像及 GPU 结果浏览待组员 |
| 命令、前置检查、PowerShell 手册与数据恢复 | code_ready / cpu_checked / gpu_pending | `scripts/run_pipeline.py`、`docs/WINDOWS_RUNBOOK.md`；八阶段 dry-run |

当前没有任何真实 GPU 阶段可登记为 `gpu_verified`。运行时只有相应 GPU 阶段完成，才会写入该状态。
CPU 模拟后端在临时目录生成的流程记录仅作为测试，不能混入正式 results。

## 已验证边界

详见 `docs/validation/cpu-evidence.json` 与 `cpu-tests.xml`：记录实际执行命令、退出码、
Python/平台、输出及源文件 SHA-256。主要测试覆盖：

- stand-in 记录完全保留，按类 seed=0 固定 200 张 train_dev；冻结角色被改动即报错。
- train_dev/eval/pool 不能通过改 ID、伪造 extra 或绕过反馈标签进入训练。
- 专家标签覆盖旧模拟版本，后续模拟标签不能覆盖专家；确认也计入当前有效反馈。
- 224/112/56 裁剪与非 224 原图坐标映射、错误类别/区域/描述符拒绝。
- 九个有限候选分数稳定归一化；assistant mask 覆盖全部提示和 padding。
- 预测恢复拒绝不同模型/图像；每类召回与困难对错误具有实际分母。
- 真实 CPU autograd、AdamW、scheduler 和 Python/NumPy/torch RNG：连续 6 步与 3+3 续训结果一致；
  缺失 latest 指针仍恢复；损坏 checkpoint 组件被拒绝。
- 模拟推理/训练后端验证两轮 R3、50% 回放、去重、反馈选取顺序、累计 2/4 条与同数修订。

## 运行合同与限制

正式结果保存在 `outputs/<run_id>/results/`；按模型完整历史在 `results/by_model/`，
根目录同名 CSV/JSONL 是最近完整版本。`hard_train.csv`、目标构建日志、缓存和 checkpoint
同样按 run_dir 隔离。正式类别反馈按主文档位于 `data/feedback/feedback_records.jsonl`；
smoke 使用独立反馈目录。以上目录配置和日志路径均在命令计划中可见。

本实现的解释使用固定模板和精确描述符引用。推理预填评分决定的 Diagnosis，
类别与标题共用同一预测；不合格式/证据不通过的输出保留失败标记，不当作有效证据。
区域匹配属于自动弱监督，开发集阈值用类别匹配代理选择，不代表病理专家验证。
R1 纯分类解释指标为空并标明 n_rationale=0。

CPU 检查没有执行真实 HF tokenizer/processor、CUDA 前后向、bf16/fp16/4-bit、
CONCH/MUSK 模型加载或 Streamlit 浏览器交互。这些都在 GPU/操作员验收清单内，
不是已验证兼容性。Windows 固定依赖是待冒烟验证的交付配置。
完整数据、模型访问凭据、实际吞吐与显存、R2/R3 接受率和全部模型效果均待回传。

同运行 code/config/manifest 改动会被拒绝，以避免续训混合版本。调参或修正代码后使用新的
run_dir/feedback_dir；不要为了绕过检查手改 hash。额外 NCT 通过显式新清单启用。
当前提供随机中断续训与跨阶段模型接续；不承诺不同硬件 CUDA 算子位级确定性。

## 参考核对

已只读核对官方固定提交 `78576d3824c70679053db34458d79cc0df70d259` 的
[README](https://github.com/ncsa/ashby-hackathon-2026/blob/78576d3824c70679053db34458d79cc0df70d259/README.md)、
[STUDENT_GUIDE](https://github.com/ncsa/ashby-hackathon-2026/blob/78576d3824c70679053db34458d79cc0df70d259/STUDENT_GUIDE.md) 和
[prompts_nct.json](https://github.com/ncsa/ashby-hackathon-2026/blob/78576d3824c70679053db34458d79cc0df70d259/prompts_nct.json)。
官方类别短句用于代码含义映射；27 个形态描述符由 B 维护。
Qwen 模型/处理器入口参照 [Transformers Qwen2-VL 文档](https://huggingface.co/docs/transformers/main/en/model_doc/qwen2_vl)，
具体交付锁定 Transformers 4.57.1，兼容性仍以组员实际 smoke 为准。

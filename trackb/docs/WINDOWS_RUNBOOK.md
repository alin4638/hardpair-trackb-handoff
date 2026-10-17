# Windows RTX 4080 Laptop 操作手册

本手册从 `trackb/` 执行。GPU 机器由组员操作；当前交付只完成 CPU 检查。
默认 Python 3.12、CUDA PyTorch 2.8.0、12GB 显存是假定启动配置，先做诊断。
正式结果冻结时间：2026-10-18 15:30 America/Chicago。耗时按冒烟日志实测排期。

## 1. 取得代码

交接基准是 `18f5f0a3266f91cb0d40dff80a9ab16ffe8949de`。本次交付按用户授权
提交到工作仓库的现有 `main`，组员直接取得最新 `main`：

```powershell
git clone --branch main https://github.com/alin4638/hardpair-trackb-handoff.git
Set-Location hardpair-trackb-handoff
git rev-parse HEAD
git log -1 --oneline
Set-Location trackb
```

已有 checkout 时，先确认当前是 `main` 且工作区没有未保存的修改，再从上面工作仓库
执行 fast-forward pull。某些 checkout 的 `origin` 是个人 fork，应先用 `git remote -v`
确认 URL，避免拉到旧基准或把 fork 当成工作仓库。

也可继续使用先前提供的代码 ZIP/patch 离线交接，但不要把旧 patch 再应用到已经更新的 main。
源码包的文件清单和 SHA-256 位于该包配套的 `receipt.json`；发布提交以远端 main 和本次消息给出的提交号为准。

## 2. 独立环境与模型访问

优先保留组员已成功的旧环境，用新 `.venv` 验证本次固定依赖。
无需更改 PowerShell 执行策略，直接调用环境内 Python：

```powershell
py -3.12 -m venv .venv
$py = (Resolve-Path .venv\Scripts\python.exe).Path
& $py -m pip install --upgrade pip
& $py -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
& $py -m pip install -r requirements-windows.txt
New-Item -ItemType Directory -Force third_party | Out-Null
git clone https://github.com/mahmoodlab/CONCH.git third_party/CONCH
git -C third_party/CONCH checkout 141cc09c7d4ff33d8eda562bd75169b457f71a62
git clone https://github.com/lilab-stanford/MUSK.git third_party/MUSK
git -C third_party/MUSK checkout 714b666969c1911e5efe70d991140a21030f4ef3
& $py -m pip install --no-deps -e third_party/CONCH -e third_party/MUSK
```

CONCH/MUSK 固定到旧交接已记录的实现提交。`--no-deps` 避免 MUSK 的旧
Torch 2.0.1 声明降级整个环境。原有 CONCH tokenize、MUSK SentencePiece 兼容代码保留。
本次 Windows 依赖组合仍须 GPU 冒烟验证。`pip check` 的第三方旧固定依赖冲突
需要保留诊断，不能用一次依赖检查通过替代实际编码器验证。

由组员在自己的 Hugging Face 账户接受 CONCH/MUSK 访问条款。凭据保存在用户环境；
不要把 token 写入配置、脚本或回传包。可复用已有 HF 缓存，或设定磁盘目录：

```powershell
$env:HF_HOME = 'D:\Ashby\hf-cache'
& .venv\Scripts\hf.exe auth login
& .venv\Scripts\hf.exe download MahmoodLab/CONCH
& .venv\Scripts\hf.exe download xiangjx/musk
& .venv\Scripts\hf.exe download Qwen/Qwen2-VL-2B-Instruct
& $py scripts/diagnose.py --config configs/smoke_4080.yaml
& $py -m pip freeze | Out-File -Encoding utf8 outputs/smoke_4080_v1/packages.txt
nvidia-smi | Out-File -Encoding utf8 outputs/smoke_4080_v1/nvidia-smi.txt
```

查看 `environment.json` 的实际设备、显存、bf16、CUDA 与第三方提交。
准备至少原始数据、教师权重、学生权重、多份 adapter/优化器和缓存的可用空间；
程序记录剩余磁盘，不在未实测前声称足够。所有 GPU 阶段要求 CUDA；不会默默回落到用户 CPU。

## 3. 恢复数据与冻结清单

已有目录可直接复用。编辑两份 YAML 中的 `data_root` 为 Windows 绝对路径，
推荐用正斜杠或单引号。必须包含：

```text
D:/datasets/CRC/
  NCT-CRC-HE-100K/ADI/*.tif ... TUM/*.tif
  CRC-VAL-HE-7K/ADI/*.tif ... TUM/*.tif
```

没有数据时，在组员机器下载；脚本支持部分文件续传，并核对交接已记录的 ZIP MD5：

```powershell
& $py scripts/download_data.py --root data/raw
& $py scripts/run_pipeline.py --stage prepare --config configs/smoke_4080.yaml
& $py scripts/check_data.py --config configs/smoke_4080.yaml
& $py -m pytest -q tests
```

正式 manifest/roles 随代码提供，不能重新生成任意新划分。`.gitattributes` 固定文本 LF，避免 Windows autocrlf 改变哈希。`prepare` 会核对既有角色与
seed=0 分层规则完全一致。`--skip-images` 只用于无数据的开发环境，不能作为 GPU 启动数据通过证据。

## 4. 第一轮 GPU 冒烟与恢复

`smoke_4080.yaml` 使用每类 6 个 train_fit（54 个图块）、18 个 train_dev、20 个更新步；
验证 eval 也只取 54 个图块，**所有 smoke 结果均非正式指标**。目录与正式 adapter 隔离。
先构建训练特征，CONCH/MUSK 逐个加载：

```powershell
& $py scripts/run_pipeline.py --stage cache --cache-kind features --config configs/smoke_4080.yaml
& $py scripts/run_pipeline.py --stage zeroshot --config configs/smoke_4080.yaml
& $py scripts/run_pipeline.py --stage r1 --config configs/smoke_4080.yaml --stop-after 10
& $py scripts/run_pipeline.py --stage r1 --config configs/smoke_4080.yaml --resume
& $py scripts/run_pipeline.py --stage cache --cache-kind regions --config configs/smoke_4080.yaml
& $py scripts/run_pipeline.py --stage r2 --config configs/smoke_4080.yaml
& $py scripts/run_pipeline.py --stage r3 --config configs/smoke_4080.yaml
```

在第 10 步检查 checkpoint，然后续到 20；不改 YAML 的 steps，避免改变调度器合同。
`logs/*_train.jsonl` 包含 loss、实际 visual_tokens、峰值显存、每优化器步秒数和 LR；
`checkpoints/<stage>/trainable.json` 包含可训练参数名/形状、精度和基座 revision。
验收：有限损失和梯度、assistant mask 检查成功、完成 checkpoint 恢复、九类得分归一化、
训练与 eval 输出 ID 符合对应样本子集。

R3 零条候选通过时会明确停止并保留全部生成记录；不能把空数据训练或纯回放称作完成 R3。
先检查 `r3/*/*.json`、阈值与 R2 目标；对模板/描述符或容量做针对性修正后开新运行。

反馈默认是真实类别来源，首次命令生成队列并返回 `awaiting_feedback`：

```powershell
& $py scripts/run_pipeline.py --stage feedback --config configs/smoke_4080.yaml
& $py -m streamlit run app.py -- --config configs/smoke_4080.yaml
# 在界面提交已查询 pool 图块的类别后：
& $py scripts/run_pipeline.py --stage feedback --config configs/smoke_4080.yaml --resume
& $py scripts/run_pipeline.py --stage report --config configs/smoke_4080.yaml
```

若首轮要自动模拟反馈，**第一次 prepare 前**在 smoke YAML 中将
`feedback.source` 改为 `simulated_dataset_label`。程序先保存选定 ID，再读取这些 ID 的数据集类别。
模拟与真实反馈分别记来源。不要在已有 run_dir 内变更配置；重新开目录和独立 feedback_dir。

## 5. 正式完整运行

依据 smoke 的显存/速度调整正式 YAML，随后冻结配置。默认 1,000 步、梯度累积 8、每 100 步保存、
每 250 步在完整 1,800 train_dev 上选择，正式 eval 固定 3,590 图块。默认步数是运行配置，
不是已经验证的收敛结论。九候选逐一评分较慢，应将开发评估和阶段评估时间一并纳入排期。

```powershell
& $py scripts/run_pipeline.py --stage prepare --config configs/laptop_4080.yaml
& $py scripts/run_pipeline.py --stage cache --cache-kind features --config configs/laptop_4080.yaml
& $py scripts/run_pipeline.py --stage zeroshot --config configs/laptop_4080.yaml
& $py scripts/run_pipeline.py --stage r1 --config configs/laptop_4080.yaml
& $py scripts/run_pipeline.py --stage cache --cache-kind regions --config configs/laptop_4080.yaml
& $py scripts/run_pipeline.py --stage r2 --config configs/laptop_4080.yaml
& $py scripts/run_pipeline.py --stage r3 --config configs/laptop_4080.yaml
& $py scripts/run_pipeline.py --stage feedback --config configs/laptop_4080.yaml
& $py -m streamlit run app.py -- --config configs/laptop_4080.yaml
# 提交反馈后，重复：
& $py scripts/run_pipeline.py --stage feedback --config configs/laptop_4080.yaml --resume
& $py scripts/run_pipeline.py --stage report --config configs/laptop_4080.yaml
```

`cache --cache-kind features` 生成训练特征和 `outputs/<run>/hard_train.csv`；困难清单按运行隔离，
包含真实类相对竞争类 margin、OOF 预测、rank、3 倍困难权重。
`cache --cache-kind regions` 生成逐图块区域缓存、开发集阈值和 3 个代表区域示例。
阈值是 train_dev 类别匹配代理指标，不能作为专家区域标注质量的证明。
CONCH 主判定，MUSK 分歧分数保留供检查。

显存不足时，结束进程，在**新 run_dir** 中启用 `model.load_in_4bit: true`，并安装：

```powershell
& $py -m pip install bitsandbytes==0.48.1
```

四位路径与可选 `model.train_visual: [visual.merger]` 都需要重新冒烟。
默认视觉主干冻结；不要把可配置视觉适配说成已经跑过。保持 batch=1、梯度检查点，
调整序列长度时必须保证完整目标，程序不会静默截断答案。

新增有真实类别的 NCT 图块：

```powershell
& $py scripts/register_extra.py --config configs/laptop_4080.yaml --classes STR MUS --per-class 500 --out data/extra_str_mus_v1.csv --reason 'train_dev 当前 STR/MUS 错误较多，见本轮开发结果'
```

把新路径写入新运行配置的 `extra_manifest`，重新执行 prepare/cache/R1；清单之外的 NCT
按目录真实标签加入，train_dev 和原始 manifest 图块不会再次作为 extra 进入训练。

## 6. Checkpoint 与失败恢复

每个阶段目录：

```text
outputs/<run>/checkpoints/<stage>/
  latest.json, best.json, trainable.json
  step_0000100/
    adapter/                 LoRA 参数与 PEFT 配置
    processor/               图像处理器与聊天模板
    visual.pt                显式可训练视觉参数（默认空字典）
    training.pt              optimizer、scheduler、GradScaler、Python/NumPy/torch/CUDA RNG、step
    checkpoint.json          基座名称/解析 revision、目标/配置合同、模型版本
    checksums.json           组件 SHA-256
    dev_metrics.json         仅在开发评估完成后存在
```

保存发生在梯度累积完成的优化器边界，先写 `.incomplete` 再原子重命名。
恢复扫描最新完整目录，即使 latest 指针写入前宕机仍可恢复。组件校验失败则报错。
中途梯度累积不保存，最多重做最近完整 checkpoint 之后的步骤。
跨阶段只加载模型参数并新建优化器；同阶段 `--resume` 恢复全部训练状态。
选择 best 只看 train_dev 平衡准确率。R3 每轮记录候选数量和被选模型；反馈 anchor
从 R1/R2/R3 当前开发最佳版本确定后固定，之后按批次顺序继续更新。

```powershell
& $py scripts/run_pipeline.py --stage r2 --config configs/laptop_4080.yaml --resume
```

缓存与预测也逐图块落盘；重跑跳过合同匹配的已完成项。版本、图像哈希、目标或有效配置改变会报错，
避免混用结果。建议保留相同目录结构；完整 checkpoint 可以整体备份，但迁移到另一绝对路径后
需要同步 manifest 中的 checkpoint 路径，不能只复制 adapter 冒充训练恢复。

强制结束进程可能留下 `outputs/<run>/pipeline.lock`；只有确认该运行没有 Python 工作进程后，
才删除该锁并恢复。正常异常与 Ctrl+C 会释放锁。不要同时运行同一 run_dir 的两个流水线命令。
训练读取当前反馈快照；训练期间新来的修订会在下一轮循环被纳入。

类别反馈当前版本保存在 `data/feedback/feedback_records.jsonl`（正式配置），
smoke 在独立 `data/feedback_smoke/`。历史保留，专家版本优先于后来模拟版本；
同数量的新修订也会创建新模型版本，结果历史在 `results/by_model/`。
回放只接收 train_fit R2，pool 的旧自动理由不会进入反馈回放。

## 7. 回传与运行状态

`report` 生成 `outputs/<run>/return_bundle.zip`，包含环境诊断、有效配置、run_manifest、
构建日志、全部阶段日志、小型 CSV/JSONL、各类 recall、混淆矩阵、反馈历史、
checkpoint 标识、代表区域图和进展图。**不含权重、原始数据或大型教师缓存。**

另外回传 `packages.txt`、`nvidia-smi.txt`（可手动放入 return_bundle 后再压缩），
以及失败时的完整 stderr/log。第一轮至少需要：

- 实际设备/驱动/Python/CUDA、依赖列表和配置。
- 54 图块 / 20 步实际日志、mask/可训练参数记录、每步 loss、峰值 VRAM、步速。
- 第 10 步与第 20 步 checkpoint.json/checksums.json、恢复命令及成功日志。
- 九类预测小文件、开发与 eval 样本数、R2 通过/拒绝数量、R3 实际候选与轮数。
- 反馈 selected IDs、来源、n_feedback、n_changed、anchor 与更新模型版本。

`code_ready`：实现与入口齐备；`cpu_checked`：有实际 CPU 检查记录；
`gpu_pending`：本次尚未在组员机器执行；`gpu_verified`：组员实际完成相应 GPU 阶段且产物存在。
任何阶段运行成功都不等于准确率提升；改进结论只能来自回传的真实固定 eval 结果。

同一更新中同时存在真实与模拟标签时，反馈曲线为每种实际来源各写一行，
`n_feedback` 是该模型使用的总独立图块数，`n_source_feedback` 是该来源当前数量；
两行共享同一模型的总体 eval 指标，不代表两次独立实验或某来源的因果贡献。


反馈模型之后再继续 R3，可明确选择实际反馈 checkpoint：

```powershell
& $py scripts/run_pipeline.py --stage r3 --r3-from feedback --config configs/laptop_4080.yaml --resume
```

该命令为当前反馈模型创建独立两轮循环，生成与回放仍只来自 train_fit；不会把 pool
图块用于 R3。重复同一命令恢复该循环，更换反馈模型后则有新的循环标识。

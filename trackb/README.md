# Track B — independent pathology VLM training and feedback

**Final experiment:** after the completed 200-label GPU follow-up, use
[FINAL_GPU.md](docs/FINAL_GPU.md) and `configs/final_4080.yaml`. This adds pathology-feature
classification and fast cumulative feedback while reusing the existing Qwen explanations.

Implementation lives entirely in `trackb/`. Read [AGENTS.md](AGENTS.md),
[scope](docs/TRACK_B.md), [phases](docs/TRACK_B_PHASES.md) and
[delivery status](docs/IMPLEMENTATION_STATUS.md).

**GPU operator: start with [Windows PowerShell runbook](docs/WINDOWS_RUNBOOK.md).**
If you already completed the October 17 GPU run, use the
[regional explanation and feedback follow-up](docs/FOLLOWUP_GPU.md): it reuses your
R2 weights/caches and runs true-label feedback from the isolated pool.
The default model is Qwen2-VL-2B with language LoRA. CONCH/MUSK caches are built
sequentially and released before student training. No remote GPU service is required.

```powershell
cd hardpair-trackb-handoff/trackb
python scripts/run_pipeline.py --help
python scripts/run_pipeline.py --stage prepare --config configs/laptop_4080.yaml --dry-run
```

Stages: `prepare`, `cache` (features or regions), `evidence` (CPU calibration from cached scores),
`probe` (CPU pathology classifier), `explain` (guided inference), `zeroshot`, `r1`, `r2`, `r3`,
`feedback`, `report`. Formal and smoke runs have separate configurations, targets,
checkpoints, results and feedback histories. `--resume` validates run/target identity.

CPU development: Python 3.11 or 3.12; `bash scripts/setup_cloud.sh` installs a CPU-only
PyTorch and development packages, then runs `scripts/validate_cpu.py`. No datasets,
pretrained models or GPU execution are involved. Without torch, the basic tests run
with `python -m pip install -r requirements-cloud.txt; python -m pytest -q tests`,
but the real optimizer/RNG recovery test is skipped and must not be reported as passed.

Frozen data: `data/manifest.csv` retains every original stand-in record;
`data/roles.csv` fixes 16,200 train_fit / 1,800 train_dev / 3,590 pool / 3,590 eval.
Both have SHA-256 metadata. Data images belong under configurable `data_root`.
A artifacts are never discovered automatically. Optional extra NCT training rows
must be explicitly registered with real directory labels using `scripts/register_extra.py`.

The demo consumes actual saved results and queued pool queries:

```powershell
python -m streamlit run app.py -- --config configs/laptop_4080.yaml
```

No weights or raw data belong in the handoff. `python scripts/make_handoff.py` exports
all nonignored Track B source (including untracked additions), a patch against HEAD,
and file hashes to `outputs/handoff/`. For the published handoff, pull `main` from
`https://github.com/alin4638/hardpair-trackb-handoff.git`; verify the remote URL because
`origin` in an existing checkout may point to a personal fork.

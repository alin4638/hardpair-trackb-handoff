# Teammate agent: start Track B GPU execution

Take over the Ashby Hackathon Track B project on the teammate's Windows RTX 4080 Laptop machine.
Get the latest existing `main` from https://github.com/alin4638/hardpair-trackb-handoff.
Check the remote URL: some checkouts use `origin` for a personal fork. Preserve local changes;
do not create a branch, force-push, or replace an existing environment without instruction.
Work only inside `trackb/`.

First read `trackb/AGENTS.md`, `trackb/CLAUDE.md`, `trackb/docs/TRACK_B.md`,
`trackb/docs/TRACK_B_PHASES.md`, `trackb/docs/CODEX_CLOUD_PROMPT.md`,
`trackb/docs/WINDOWS_RUNBOOK.md`, and `trackb/docs/IMPLEMENTATION_STATUS.md`.
The newer Track B scope supersedes incompatible historical root-level plans.

The implementation has passed 15 CPU tests, including real optimizer/RNG checkpoint recovery,
and all eight stage dry-runs. Real model loading, CUDA execution, VRAM, throughput and accuracy
are still `gpu_pending`. Verify the actual GPU, driver, Python, CUDA, disk space and model access.
Reuse existing datasets and authorized Hugging Face caches where possible. Follow the PowerShell
runbook for the pinned environment and encoder source commits. Never commit credentials, weights,
raw images, or large caches.

Start with `configs/smoke_4080.yaml`: 54 training tiles and 20 optimizer steps in an isolated run.
Set paths before starting the run. From `trackb/`, with the intended environment activated:

```powershell
python scripts/diagnose.py --config configs/smoke_4080.yaml
python scripts/run_pipeline.py --stage prepare --config configs/smoke_4080.yaml
python scripts/check_data.py --config configs/smoke_4080.yaml
python -m pytest -q tests
python scripts/run_pipeline.py --stage cache --cache-kind features --config configs/smoke_4080.yaml
python scripts/run_pipeline.py --stage zeroshot --config configs/smoke_4080.yaml
python scripts/run_pipeline.py --stage r1 --config configs/smoke_4080.yaml --stop-after 10
python scripts/run_pipeline.py --stage r1 --config configs/smoke_4080.yaml --resume
```

Verify the saved step-10 checkpoint and successful continuation to step 20. Then follow the runbook
through region caching, R2, R3, feedback and report. Fix concrete compatibility/runtime problems
and retain failure logs. Do not bypass identity/hash checks or report empty R3 acceptance as success.
Use measured smoke results to configure the formal run, then proceed through all stages using
`configs/laptop_4080.yaml`. Use separate run and feedback directories when changing a frozen configuration.

Preserve the fixed 16,200 train_fit / 1,800 train_dev / 3,590 pool / 3,590 eval roles.
Use train_dev for checkpoint/threshold selection and eval only for stage reporting. Select pool IDs
before acquiring labels. Feedback defaults to `expert_label`; wait for actual class feedback when
requested. If simulation is explicitly enabled, configure `simulated_dataset_label` before the run
and retain that provenance. Every update uses all current valid feedback plus train_fit R2 replay.

Return environment diagnostics, effective configuration, package/driver versions, training and error
logs, loss, peak VRAM, step timing, checkpoint identifiers and recovery evidence, actual sample counts,
small prediction/metric files, R3 acceptance counts and feedback provenance. Run the `report` stage to
produce `outputs/<run>/return_bundle.zip`; keep weights and large caches on the GPU machine.
Update `code_ready`, `cpu_checked`, `gpu_pending`, and `gpu_verified` only from actual evidence.

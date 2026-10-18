# GPU follow-up: regional explanations and cumulative class feedback

Source: the teammate's `5e93f91` run (`docs/GPU_RUN_2026-10-17.md`). This change continues
that run from its saved weights/cache. Keep `outputs/laptop_4080_v1` on the GPU machine.
No model or dataset download and no teacher re-encoding is needed when those artifacts remain.
Use the same Windows Python 3.12 / torch 2.8+cu128 environment that completed the original run.

## What changes

- `tb/region_guide.py`: normalize each descriptor against train_fit scores, retain useful
  descriptors by train_fit discrimination, choose the crop by relative support against a
  competing class, and calibrate each ordered class pair on train_dev. This reads cached
  CONCH/MUSK scores on CPU. It does not use pool/eval labels.
- R2 now receives the selected descriptors and coordinates in its prompt. The new config
  refines the old R2 adapter for 300 steps at learning rate 2e-5, retaining 50% class replay.
- Inference uses the predicted top-two classes to retrieve evidence, then asks the student
  for an explanation conditioned on that evidence. A failed student response falls back to
  an explicitly tagged teacher template. Insufficient support yields no region citation.
  Raw student output, evidence scores, selected descriptor IDs and guide version are saved.
- Feedback starts from the best available train_dev checkpoint, including the original R2.
  It selects pool IDs before reading their dataset labels, then trains on all current labels
  at 50/100/200 unique tiles. Updates take 100/200/400 optimizer steps at 2e-5 with accumulation 4.
  Half of training draws are feedback; replay combines valid train_fit R2 targets and class
  targets. Revisions replace prior labels. Expert labels take precedence over simulated labels.
- The Streamlit app shows evidence provenance, query/model versions and pending updates;
  it accepts individual corrections or a CSV and can launch/resume feedback on the GPU host.

## Data for feedback and acceptance

The existing 3,590 pool tiles have not been used for initial R1/R2/R3 training. Use their true
labels as `simulated_dataset_label`; this exercises the class-feedback mechanism without
waiting for a human expert. Keep the other 3,590 eval tiles out of updates. Pool tiles become
feedback training examples when selected; they are not an independent test set afterward.
Train_dev remains the checkpoint-selection set. This is separation from this project's
training, not a claim about the foundation models' pretraining data.

MHIST is an HP-versus-SSA polyp classification task, whereas this model scores nine CRC
**tissue** classes. Do not remap HP/SSA to NORM/TUM or import them into the nine-class feedback
queue. A future MHIST extension needs its own candidates, descriptors and feedback/test split.
Official dataset: https://bmirds.github.io/MHIST/

## PowerShell: update and inspect prerequisites

Run from the existing teammate checkout. Keep local untracked weights and caches in place.

```powershell
git pull --ff-only
cd trackb
$py = '.\.venv\Scripts\python.exe'
& $py scripts/run_pipeline.py --help
& $py scripts/run_pipeline.py --stage prepare --config configs/followup_4080.yaml --dry-run
```

If this checkout's remote is a fork, pull the handoff repository's `main` explicitly instead:
`git pull --ff-only https://github.com/alin4638/hardpair-trackb-handoff.git main`.

Before `prepare`, edit `data_root`, `inherit_run` and `cache_dir` in **both follow-up configs**
only if local paths differ. They default to `data/raw`, `outputs/laptop_4080_v1`, and that run's
`cache` directory. `prepare` imports checkpoint identities, pins the existing base revision,
and rebases old absolute checkpoint paths if the checkout moved. It does not copy weights.
New runs use separate feedback directories and select new queries with their chosen checkpoint.
Leave the original run/config unchanged. All config edits must precede the first prepare.

## First GPU smoke

```powershell
$cfg = 'configs/followup_smoke.yaml'
& $py scripts/run_pipeline.py --stage prepare --config $cfg
& $py scripts/run_pipeline.py --stage evidence --config $cfg
& $py scripts/run_pipeline.py --stage r2 --config $cfg --stop-after 10
& $py scripts/run_pipeline.py --stage r2 --config $cfg --resume
& $py scripts/run_pipeline.py --stage feedback --config $cfg --resume
& $py scripts/run_pipeline.py --stage report --config $cfg
```

Run commands one at a time; inspect a nonzero exit before continuing. The smoke uses 54 fit
and eval tiles, 18 dev tiles, 20 R2 steps and 2/4-label feedback updates (20 steps each).
The stop at step 10 exercises checkpoint resume with unchanged configuration. Smoke metrics
are for this small smoke run only. Successful feedback creates two entries in
`data/feedback_followup_smoke_v1/state.json`, with `n_feedback` 2 and 4 and actual new model IDs.

## Formal continuation

```powershell
$cfg = 'configs/followup_4080.yaml'
& $py scripts/run_pipeline.py --stage prepare --config $cfg
& $py scripts/run_pipeline.py --stage evidence --config $cfg
& $py scripts/run_pipeline.py --stage r2 --config $cfg --resume
& $py scripts/run_pipeline.py --stage feedback --config $cfg --resume
& $py scripts/run_pipeline.py --stage report --config $cfg
& $py -m streamlit run app.py -- --config $cfg
```

R2 refinement is recommended for the evidence-conditioned generation prompt. If the immediate
priority is feedback, `evidence -> feedback` also works from the inherited best checkpoint.
`--stage explain` optionally evaluates the best available checkpoint with guided explanations;
R2 and feedback already export explanations. Do not rerun cache, zero-shot, R1, or the old R3
rounds for this handoff. Further R3 remains available with guided evidence, but is not needed
before completing the requested feedback updates.

The app's button starts `feedback --resume` as a subprocess on the machine running Streamlit;
run it on the GPU laptop. Refresh to see its logs/version changes. A pipeline lock prevents
concurrent training. The class CSV schema is `tile_id,corrected_label`; download the selected
queue template from the app. Imports accept only already-selected pool IDs and an explicit
source (`expert_label` for human review; `simulated_dataset_label` for database truth).

## Resume and outputs

Use the same command with `--resume`. Do not change configuration during a run. Checkpoints
remain under `outputs/<run_id>/checkpoints/<stage>/step_*`: adapter, processor, trainable visual
parameters (`visual.pt`, empty when vision is frozen), optimizer, scheduler, scaler, Python/
NumPy/CPU/CUDA random states and step (`training.pt`), metadata and checksums. Incomplete
checkpoint directories are skipped. `latest.json` resumes optimizer progress; `best.json`
selects by train_dev. If a process was killed and left `pipeline.lock`, first confirm that its
PID is no longer running, then remove that stale lock and resume.

Logs are under `outputs/<run_id>/logs/`, including app-launched subprocess logs. Feedback records,
current labels, queued queries and update history are under `data/feedback_followup_v1/`.
`results/vlm_feedback_curve.csv` records actual `n_feedback`, `n_changed`, source and model ID.
`results/vlm_summary.csv` includes classification, regional support and raw student acceptance.

`grounding_pass_rate` in this mode measures the delivered guided explanation (including
explicit template fallback); `student_grounding_rate` measures raw student compliance with
the retrieved evidence across all requested tiles. `retrieved_support_rate` measures evidence
availability. Per-result `metrics.json` also counts each `explanation_source`. These are teacher
score proxies, not expert region annotations. Do not present a template fallback as an improved
student-generated explanation or directly compare its pass rate to the old free-generation rate.
Classification scoring is unchanged.

After smoke and formal runs, inspect a few result and feedback tiles in the app: the cited
rectangle, descriptor, explanation source, saved query version and actual updated model version.
Export `--stage report` again after any further updates. Push only the small return artifacts
into `trackb/docs/validation/gpu_followup_<date>/` plus a concise run note, using the existing
handoff repository main. Include environment/effective config, guide/inheritance files, logs
(VRAM/step time), summary/feedback curves, rationales/predictions and checkpoint identifiers.
The `return_bundle.zip` gathers these. Keep weights, images, teacher caches and credentials out
of Git. Report actual failures or partial completion and which command can resume them.

## Delivery status

- `code_ready`: guided evidence, parent-run continuation, R2 refinement, cumulative simulated
  feedback, class replay, app update/CSV flow, stage entrypoints and resume paths implemented.
- `cpu_checked`: see `docs/validation/followup_2026-10-18/cpu-evidence.json` and `cpu-tests.xml`.
  CPU fixtures check 2/4-label updates, revisions, source/role handling, guide selection,
  generated-output fallback and portable continuation; the CLI help and dry-runs are exercised.
- `gpu_pending`: all new guide quality, fine-tuning, GPU feedback and interactive app behavior.
- `gpu_verified`: applies to the original teammate run only, documented in
  `docs/GPU_RUN_2026-10-17.md`; no new GPU result is claimed here.

# GPU follow-up run (2026-10-18): guided regional evidence and cumulative pool-label feedback

Executed on the same RTX 4080 Laptop / pinned environment as [GPU_RUN_2026-10-17.md](GPU_RUN_2026-10-17.md)
(torch 2.8.0+cu128, transformers 4.57.1, peft 0.17.1), code `dce6bf0`, following
[FOLLOWUP_GPU.md](FOLLOWUP_GPU.md) with no code or config changes. Both follow-up configs used their default
paths (`data/raw`, `outputs/laptop_4080_v1`, its `cache`), which match this checkout. No teacher re-encoding and
no model download. Small evidence: [`validation/gpu_followup_2026-10-18/`](validation/gpu_followup_2026-10-18/).
Weights, caches, images and `return_bundle.zip` (smoke 2.1 MB, formal 12.4 MB) remain on the GPU machine.

## Commands and exit status

| Run | Command sequence | Result |
|---|---|---|
| `followup_smoke_v1` | prepare → evidence → r2 `--stop-after 10` → r2 `--resume` → feedback `--resume` → report | all exit 0 |
| `followup_4080_v1` | prepare → evidence → r2 `--resume` → feedback `--resume` → report | all exit 0 |

No failures. 20/20 CPU tests passed before the GPU work. Inheritance imported the parent identities
(R1 `step_0001000-9dbc7783bac4`, R2/R3 `step_0001000-a7a3c2c550c5`) without copying weights, and selected 0 old queries.

## Smoke (54 fit/eval tiles, 18 dev; not formal metrics)

- Evidence: guide `f49cbfb312df992b`, 32 guided R2 targets.
- R2 refine from `r2_parent`: paused at step 10, `--resume` continued at step 11 (LR 1.0e-5 → 9.0e-6) to step 20.
  0.65 s/step (grad_accum 2), peak 4.97 GiB. Step-10 checksums: adapter `5aa377f5…`, `training.pt` `eb7415d6…`.
- Feedback: 2 updates as specified; `state.json` holds `fb_2` (`step_0000020-c58bf678fdcc`, n_changed 1) and
  `fb_4` (`step_0000020-63e105ca941b`, n_changed 3). 20 steps each, ~1.3 s/step, peak 4.98 GiB.

| Smoke stage | BA | Student explanation (raw) | Teacher template fallback | Insufficient evidence |
|---|---|---|---|---|
| r2_grounded / fb_0 | 0.778 | 25/54 (0.463) | 11 | 18 |
| fb_2 | 0.778 | 24/54 (0.444) | 9 | 21 |
| fb_4 | 0.796 | 31/54 (0.574) | 2 | 21 |

## Formal

### Evidence guide (`evidence`, CPU, 318 s)

Guide `bca2de537a0f5080` fitted on 16,200 train_fit tiles (normalization, descriptor AUC) with pair thresholds
on 1,800 train_dev tiles. 24/27 descriptors kept (AUC ≥ 0.55). 6,571 guided R2 targets. Train_dev balanced
pair accuracy for the hard pairs: DEB/MUC 0.848, MUS/STR 0.695, NORM/TUM 0.673, MUC/DEB 0.655, TUM/NORM 0.610,
STR/MUS 0.565. These are teacher-score proxies, not expert region annotations.

### R2 refinement (300 steps, lr 2e-5, grad_accum 8, 50% class replay; 4,391 s)

2.74 s/step, peak VRAM 4.99 GiB, loss 0.058 → 0.027 (first/last 50). Dev BA at 100/200/300: 0.911 / 0.907 / 0.911.
Selected `step_0000100-565954366c0d` (dev 0.911; parent R2 dev 0.908). This became the feedback anchor.

### Cumulative simulated feedback (15,178 s)

`feedback.source = simulated_dataset_label`, configured before prepare. At each node the current model scored the
remaining pool, the lowest-margin IDs were written to `queries.jsonl`, and only then were the dataset labels for
those IDs read. Every update trained on all current labels plus train_fit replay (R2 regional and class targets).
Verified from the target files: the feedback training sets contain only `pool` and `train_fit` tiles (no eval,
no train_dev). All 200 records are selected pool tiles with source `simulated_dataset_label`.

| Stage | Model | n_feedback | n_changed | Steps | s/step | Peak VRAM | Dev BA (selected) |
|---|---|---|---|---|---|---|---|
| fb_0 (anchor) | `step_0000100-565954366c0d` | 0 | 0 | – | – | – | 0.911 |
| fb_50 | `step_0000100-adedac735126` | 50 | 26 | 100 | 1.38 | 4.99 GiB | 0.906 |
| fb_100 | `step_0000200-145e0e179623` | 100 | 57 | 200 | 1.32 | 4.99 GiB | 0.900 |
| fb_200 | `step_0000200-a755eb3b2def` | 200 | 111 | 400 | 1.30 | 4.99 GiB | 0.911 |

### Fixed eval (n = 3,590)

| Stage | BA | STR/MUS | DEB/MUC | NORM/TUM | Raw student explanation | Template fallback | Insufficient evidence |
|---|---|---|---|---|---|---|---|
| parent r2_grounded (10-17) | 0.835 | 0.316 | 0.001 | 0.074 | free generation, not comparable | – | – |
| r2_grounded (refined) = fb_0 | 0.842 | 0.296 | 0.000 | 0.063 | 2,012 (0.560) | 0 | 1,578 |
| fb_50 | 0.864 | 0.266 | 0.000 | 0.067 | 2,076 (0.578) | 0 | 1,514 |
| fb_100 | 0.865 | 0.245 | 0.000 | 0.078 | 2,239 (0.624) | 0 | 1,351 |
| fb_200 | **0.885** | **0.229** | 0.000 | 0.059 | 2,315 (0.645) | 0 | 1,275 |

**Raw student vs. template.** In the formal run, `student_grounding_rate` = `retrieved_support_rate` = delivered
`grounding_pass_rate` at every stage. Whenever the guide retrieved sufficient evidence, the student reproduced the
evidence-conditioned explanation verbatim, and **no teacher-template fallback was delivered**. The remaining
explanations are honest "insufficient evidence" outputs with no region cited. The rising rate across feedback
stages reflects changed predictions/rivals for which evidence was retrievable, not better free-form writing.
In the smoke run the template fallback did occur (11, 9 and 2 tiles), so the fallback path is exercised.

**Interpretation.** 200 pool labels raised eval BA from 0.842 to 0.885 and cut STR/MUS confusion from 0.296 to 0.229,
while dev BA (NCT train_dev) stayed about 0.90–0.91. Pool and eval are both halves of CRC-VAL-HE-7K, so much of this
gain is likely adaptation to that cohort rather than better NCT discrimination. Pool tiles used as feedback are
training data now, and eval stays the untouched report set.

## Streamlit check (headless `streamlit.testing.AppTest`, `app_check.py`)

Run against a throwaway copy of the smoke run's state and feedback directory, so no real records were changed and
no GPU job was launched:

- Shows the actual feedback model version (`step_0000020-63e105ca941b`) and result stages r2_grounded/fb_0/fb_2/fb_4.
- The results tab shows the saved rationale verbatim for each `explanation_source` (`evidence_conditioned_student`,
  `retrieved_teacher_template`, `insufficient_evidence`) with cited regions, model version and evidence JSON.
  Two image elements render. The red rectangle is drawn from the saved `cited_regions`; pixels were not inspected.
- The feedback tab shows `query_model_version` and the original prediction for a queried tile, plus the retrieved evidence.
- Form submission appended an `expert_label` revision, and the pending-update count went 0 → 1.
- CSV import (`import_labels`, the function behind the app's import button; AppTest cannot drive file
  uploads) accepted 2 selected pool IDs, and the pending count went to 3. It rejected an eval ID and an unselected pool ID.
- The launch button is disabled while `pipeline.lock` exists. Not tested: an interactive browser session and launching
  `feedback --resume` from the button.

## Resume

Every stage can be resumed with the same command plus `--resume` and an unchanged config. Further feedback nodes need
a new config value for `feedback.nodes`, and so a new run/feedback directory.

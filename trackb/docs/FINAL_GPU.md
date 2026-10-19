# Final Track B experiment: pathology classification with reusable Qwen explanations

Continue the completed `followup_4080_v1` run from commit `096fa64`. The final recipe is
`configs/final_4080.yaml`. It reuses all 200 real saved simulated-feedback records, their query
provenance, the trained Qwen feedback checkpoint, the calibrated region guide and teacher caches.
It trains a new **CONCH + MUSK whole-tile classifier**, then collects only 200 additional labels
at cumulative counts 300 and 400. There is no new R1/R2/R3 training or zero-shot run.

## Design and reuse

The classifier follows Track A's successful basic recipe: StandardScaler on train_fit, multinomial
logistic regression with C=0.01 and feedback weight 20. It trains on all 16,200 train_fit labels and
all currently effective pool corrections. The regularization and weight are fixed in the config;
there is no parameter sweep or new validation split. Source train_dev is reported, not trained on.
Separate CONCH and MUSK heads provide disagreement for querying; the concatenated head classifies.
These heads are small CPU models. They warm-start from the previous update and converge to the
weighted classification objective; there is no LoRA optimizer-step schedule in this final recipe.

Half of each new query batch targets the predicted MUS/STR boundary, splitting the quota between
both predicted directions where available. The remaining slots prioritize encoder disagreement
and low margin. Feature diversity within the shortlist avoids near-duplicate queries. Querying
receives predictions/features only; selected IDs are saved before dataset labels are looked up.
The existing 200 labels remain `simulated_dataset_label`; human revisions remain `expert_label`
and take precedence. All current labels are used on every refit.

Only missing **whole-tile feature** caches are encoded. A normal completed parent run already has
both encoders' train_fit features; the missing 1,800 train_dev + 3,590 pool + 3,590 eval tiles are
encoded once per encoder. Existing region scores are reused unchanged. No model weights need to
be downloaded again when the existing local Hugging Face cache is intact.

The final classifier's prediction and runner-up drive region retrieval and the existing Qwen
`fb_200` checkpoint writes the corresponding explanation. Parent explanations are reused when
Qwen version, guide version, prediction and rival match. Only remaining supported tiles require
Qwen generation. The system records **classifier_version** and **explanation_model_version**
separately: this improves the system's classifier; it is not additional Qwen fine-tuning.
Insufficient evidence remains explicit; generated student text and teacher fallback stay distinct.

## Execute once on the teammate Windows laptop

Use the same working environment that completed the follow-up. Keep both historical output
directories and `data/feedback_followup_v1`. Pull the existing handoff main:

```powershell
git pull --ff-only https://github.com/alin4638/hardpair-trackb-handoff.git main
cd trackb
$py = '.\.venv\Scripts\python.exe'
$cfg = 'configs/final_4080.yaml'
& $py scripts/run_pipeline.py --stage prepare --config $cfg --dry-run
```

Before prepare, adjust only local paths in the config if the checkout moved. In particular:

- `inherit_run`: `outputs/followup_4080_v1` (the completed run with 200 labels)
- `inherit_feedback_dir`: `data/feedback_followup_v1` (actual records and queries)
- `cache_dir`: `outputs/laptop_4080_v1/cache` (existing teacher feature/region cache)
- `data_root`: `data/raw`

Keep the new `run_dir` and `feedback_dir` distinct. Run these commands in order, stopping if a
command fails so its actual error can be fixed and the same stage resumed:

```powershell
& $py scripts/run_pipeline.py --stage prepare --config $cfg
& $py scripts/run_pipeline.py --stage cache --cache-kind features --config $cfg --resume
& $py scripts/run_pipeline.py --stage probe --config $cfg --resume
& $py scripts/run_pipeline.py --stage feedback --config $cfg --resume
& $py scripts/run_pipeline.py --stage explain --config $cfg --resume
& $py scripts/run_pipeline.py --stage report --config $cfg
& $py -m streamlit run app.py -- --config $cfg
```

`prepare` copies the small history/guide/feedback records and references existing Qwen weights.
`probe` fits with the inherited 200 labels. `feedback` performs cumulative 300/400-label updates.
Both classifier stages run on CPU, score the fixed eval set in batches, and record elapsed fitting
time and convergence. `explain` does one final generation pass using the saved Qwen checkpoint.
There is no separate GPU smoke experiment: the existing encoder/Qwen paths have already run, and
new classifier fitting, continuation, query selection and resume are covered by CPU checks.

## What to inspect and return

Use the reported classifier results, not the Qwen-only baseline, for the final system:

- `results/vlm_summary.csv`: accuracy, balanced accuracy, pair errors, explanation coverage.
- `results/vlm_feedback_curve.csv`: cumulative feedback counts and actual changed predictions.
- `results/by_model/probe-*/per_class_recall.csv`: especially MUS and STR, plus all other classes.
- `history/qwen_metrics.json`: copied previous fb_200 result (BA 0.8848); do not rerun it.
- `history/qwen_rationales.jsonl`: reused original explanations, unchanged.
- `results/by_model/probe-*/explanation_summary.json`: how many final explanations were reused.
- `classifier_identifier.json` / `classifier.json` in the report bundle: actual model identity,
  feedback/training counts, solver iterations, convergence, fit time and fixed settings.

In Streamlit, the newest result stage is selected by default. The header shows the classifier
and Qwen versions separately. The feedback tab displays the original query prediction and the
current classifier prediction. Submitting a correction queues a real classifier refit; the update
button runs the CPU feedback stage. After a new refit, immediate retrieved evidence is available;
run `explain --resume` when you want saved Qwen explanations for that new classifier version.

Export `report` after the final run. Push a concise run note and the small return artifacts to
`trackb/docs/validation/gpu_final_<date>/` on the same repository's existing main. Include environment,
effective config, run manifest, historical baseline reference, classifier metadata, feedback
records/queries/state, metrics, predictions, final rationales, explanation reuse counts, logs,
feature-cache throughput/VRAM and Qwen checkpoint IDs. Keep image/embedding caches, Qwen weights,
`weights.npz` classifier parameters and credentials out of Git. The source/classifier metadata
and results are sufficient for the handoff; model files stay on the GPU machine.

## Recovery and scope

Rerun the interrupted command with `--resume`, keeping the same config. Completed per-image
features, fitted classifiers and per-tile explanations are reused. An interrupted classifier fit
restarts that small CPU fit; completed models are stored under `classifiers/probe-*/weights.npz`
and `classifier.json`. Classifier artifacts contain the scalers, nine-class coefficients and
intercepts for all three heads; loading them needs no pickle or sklearn estimator object.
Completed feedback updates persist their effective records and current model in the new feedback
directory. Re-running prepare after successful inheritance does not erase new corrections.

If killed while running, remove a leftover `pipeline.lock` only after confirming its process has
ended. There is no need to rerun historical LoRA resume tests or old baselines.

Status at code handoff: `code_ready` and `cpu_checked` (see `validation/final_2026-10-18/`). New
whole-tile encoding, final hybrid model metrics and GPU explanation generation are `gpu_pending`.
The old `gpu_verified` results remain at `validation/gpu_followup_2026-10-18/`. The operator should
record the actual final outcome, including a partial run or a regression, rather than starting
extra experiments to force a preferred result.

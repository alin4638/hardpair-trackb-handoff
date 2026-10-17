# Track B current execution contract (2026-10-17)

Scope: edit only this trackb directory. The user's current request and docs/TRACK_B.md,
docs/TRACK_B_PHASES.md, docs/CODEX_CLOUD_PROMPT.md supersede incompatible older plans.
Do not create or switch branches without the user's instruction. Respond mainly in Mandarin.

B owns its explicit manifest, roles, hard mining, descriptors, training, evaluation and demo.
Never automatically consume Track A artifacts. Imports must be explicit and versioned.
Frozen seed=0 roles: 16,200 train_fit, 1,800 train_dev, original 3,590 pool and 3,590 eval.
Train/dev/eval separation is mandatory; pool enters training only through selected, recorded
class feedback. Extra NCT uses real labels, never pseudo-labels. No training on train_dev/eval.
Feedback sources: expert_label or simulated_dataset_label. Select IDs before retrieving labels.
Current corrections replace old targets; every update uses all current feedback and valid R2 replay.

Development and CPU checks are Cloud-compatible; GPU execution belongs to the teammate's
Windows RTX 4080 Laptop. Do not run GPU tasks on the user's computer, Lop, Delta or Kaggle.
Do not download model weights or datasets in the development environment. Keep weights,
caches, credentials and .env out of Git. Existing encoder tokenizer compatibility stays intact.
Default Qwen2-VL-2B, language LoRA, frozen visual tower; optional quantization/visual adaptation
must be configured and logged. First GPU smoke: ~50 samples, 20 optimizer steps, separate run.
Cost is not a selection criterion. No independent principle studies or presentation-only ablations.

Continue implementation across phases; routine choices require no new approval. Distinguish
code_ready, cpu_checked, gpu_pending and gpu_verified with actual evidence. Never infer GPU
success, quality or efficacy from fixtures. Save resumable optimizer/scheduler/RNG/step state,
adapter and trainable visual tensors. Checkpoint selection uses train_dev only.
Team freeze: 2026-10-18 15:30 America/Chicago; submission: 2026-10-21.

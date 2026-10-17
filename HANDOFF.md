# Handoff: Track B, Phase 3 (local encoders, hard cases, crops)

Written 2026-10-17. Read `CLAUDE.md` (team-level) and `trackb/CLAUDE.md` first: they hold the rules,
data contracts and method. Results freeze **Sun Oct 18, 3:30pm**; go/no-go Sun noon.

## What's in this folder

```
CLAUDE.md                 project instructions (lives at hardpair/CLAUDE.md in the real repo)
trackb/                   Track B code, i.e. hardpair/trackb/ (the only place Track B may edit)
  CLAUDE.md README.md requirements.txt .env.example .gitignore
  tb/data.py              manifest join, local paths, leakage guard, training_rows()  [done, Phase 2]
  tb/encoders.py          local CONCH + MUSK, batched image/text embeddings            [done, Phase 1]
  scripts/smoke_encoders.py smoke_vlm.py check_data.py download_data.sh
  data/manifest_standin.csv  stand-in for Track A's manifest (deterministic, rebuilt by tb/data.py)
```

**Left out on purpose** (restore them on the target machine, see `trackb/README.md`):
- `.venv/`: recreate with the `uv` commands in the README.
- `data/raw/` (NCT-CRC-HE-100K + CRC-VAL-HE-7K, ~11 GB): `bash scripts/download_data.sh`, or Zenodo 1214456.
- `third_party/CONCH`, `third_party/MUSK`: `git clone` at the commits named in the README, then `pip install --no-deps -e`.
- CONCH/MUSK/Qwen weights: never copy them (CC-BY-NC-ND). Download them with your own HF access.
- No `.env` existed, and no secrets are included.

## State

- Phases 1–2 are done: encoders pass the smoke test, and `check_data.py` passes (counts, stand-in manifest, leakage guard).
- **Track A's artifacts are not present locally**: there is no `../data/manifest.csv`, `../prompts.json`,
  `../results/hard_pairs.json`, or Track A train-CV preds. Code must fall back gracefully:
  - manifest: `tb/data.py` already falls back to `data/manifest_standin.csv`;
  - hard pairs: STR/MUS, DEB/MUC, NORM/TUM;
  - hard cases: run a local 5-fold LR probe (see below);
  - descriptors: no `prompts.json` yet. Write a stand-in (`trackb/configs/prompts_standin.json`, a few
    morphology descriptors per class) and read `../prompts.json` when it appears. Ask the team about its schema.
- Phase 3 was not started: no Phase 3 files exist yet.
- Hardware: RTX 4080 Laptop 12 GB, bf16. Windows, Git Bash/PowerShell, Python 3.12 in `trackb/.venv`.

## Next task: Phase 3 (the user's spec, verbatim)

> **tb/encoders.py**: batched CONCH/MUSK image and text embeddings (local, so text works, unlike Track A's
> servers). Cache embeddings for the train sample and the 7K to data/emb/.
>
> **Hard pairs**: read ../results/hard_pairs.json if it exists, else use STR/MUS, DEB/MUC, NORM/TUM.
>
> **Hard cases**: if Track A's train CV preds exist, use their margins. Otherwise fit a quick local
> logistic-regression probe with 5-fold CV on train and take the lowest-margin tiles within hard pairs.
> Write data/hard_train.csv.
>
> **tb/crops.py + tb/grounding.py**: 3×3 overlapping 112 px crops. Score each crop against the descriptors
> from ../prompts.json with CONCH, cross-check with MUSK. grounding_pass(tile, descriptor, region,
> rival_descriptor) returns true if the cited descriptor beats the rival on that crop. Set the margin
> threshold on a train holdout. Show me 3 example tiles with the top crop per descriptor saved as an image.

Suggested plan:
1. Add a `cache_embeddings(enc, manifest, splits)` helper (in `tb/encoders.py` or a script) that writes
   `data/emb/{conch,musk}_{train,7k}.npy` plus a tile_id index. Use DataLoader workers for .tif reading.
   Estimate time first on 256 tiles (rule 6). Embedding the 7K is fine because it's inference, not training,
   but the probe must fit on train only.
2. `tb/hard.py`: hard pairs loader + 5-fold LR on CONCH train embeddings (optionally concatenated with MUSK) →
   per-tile CV margin → `data/hard_train.csv` (tile_id, label, pair, rival, margin, rank).
3. `tb/crops.py`: 3×3 grid, stride 56, 112 px crops, regions named `r,c`.
4. `tb/grounding.py`: descriptor text embeddings + crop embeddings, so the score is a cosine;
   `grounding_pass` = score(desc) − score(rival_desc) > τ under CONCH, with the MUSK agreement reported.
   Pick τ on a held-out slice of train (e.g. the τ that maximises the pass-rate gap between the true-class
   descriptor and the rival descriptor). Never pick it on pool/eval.
5. Figure: 3 tiles from hard pairs, each with the top crop per descriptor outlined →
   `outputs/grounding_examples.png`. Show it to the user.

Hard rules to keep in mind: never train or tune on `eval`; don't edit outside `trackb/`; no weights in the repo/Kaggle;
use local compute only (no Delta; Lumen only with OK).

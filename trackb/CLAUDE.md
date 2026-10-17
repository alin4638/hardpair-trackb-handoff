# Track B: post-training a pathology VLM on hard tissue pairs

Lives at `hardpair/trackb/` inside the team repo. Owner: Alex (laptop + Kaggle).
Track A (`hardpair/00_…` to `08_…`, `app.py`) belongs to a teammate; see `../pipeline_plan.md`.
**Don't edit anything outside `trackb/`.** Contract changes are agreed in team chat first.

Ashby Hackathon 2026. Results freeze **Sun Oct 18, 3:30pm** (team rule). Delta servers stop
Sun 6pm. Deck + abstract due **Wed Oct 21**.

## The question Track B answers

Track A's frozen-encoder probe hits ~96%. Off-the-shelf VLMs scored 2–6 of 18 tiles in the
organizers' test, so Track A's LLM never sees pixels. **Can post-training make a small VLM
see the tissue: classify the hard pairs, cite visual evidence, and improve from corrections?**
Track B reports on Track A's exact eval set and metrics, so its results become extra
bars and lines on Track A's figures.

## Data (same as Track A)

- **Train pool:** NCT-CRC-HE-100K, 9 classes (ADI BACK DEB LYM MUC MUS NORM STR TUM), 224 px.
- **Test:** CRC-VAL-HE-7K, split 50/50 stratified into `pool` (corrections come from here) and `eval` (every reported number).
- Both are public (Zenodo record 1214456, also mirrored as public Kaggle datasets). Download them
  directly to the laptop and Kaggle, since only the teammate's account has Delta compute.
- **Track A's `data/manifest.csv` is the source of truth** for `tile_id, path, label, split`.
  Track B maps its local file paths onto those `tile_id`s and uses the manifest's 2,000/class
  train sample, plus more NCT tiles if useful. Never anything from `pool`/`eval`, except
  corrections in Aim 3.
- **Hard pairs** come from Track A's 5-fold CV on train (expected STR/MUS, DEB/MUC, NORM/TUM).
  Read `results/hard_pairs.json` when it exists; until then use those three.
- Optional Round 0 data: PathGen captions for TCGA-COAD/READ tiles (domain captioning).

## Hard rules

1. **Never train on `eval`.** Train only on `pool` in the correction-loop experiment, and log
   exactly which pool tiles were used. Hard cases are mined on train only.
2. **Never upload or commit CONCH/MUSK weights**, including to Kaggle datasets (CC-BY-NC-ND).
   Alex runs them locally with his own HF access. Only derived JSONL/CSV goes to Kaggle.
3. **No secrets in code or git.** HF token, Lumen key and server keys go in `.env` (gitignored).
4. **Leave Track A's Delta GPU alone.** Track B compute = laptop (RTX 4080 Laptop 12 GB, bf16)
   for main runs, and Kaggle T4/P100 (fp16 only, ~12 h sessions, ~30 GPU-h/week per account) for
   ablations and sampling.
5. **Lumen coins are shared and scarce** (10 coins, hourly refill), so Track A has first claim.
   Track B writes training targets with a **local** instruct LLM (e.g. Qwen2.5-7B-Instruct
   in 4-bit on the laptop) or with templates. Lumen only with a teammate's OK.
6. Small verified steps: 50 examples / 20 steps before any long run. Estimate time and VRAM
   before anything longer than 10 minutes.
7. Same code on laptop and Kaggle: precision auto-detected (bf16, else fp16 with fp32 LoRA),
   checkpoint every N steps, auto-resume.

## Outputs (match Track A's contracts)

- `results/preds_vlm_{stage}.csv` in Track A's preds format:
  `tile_id, pred, p_<CLASS> x9, top2, margin`. Get the probabilities by **scoring each
  class-name answer's log-likelihood** under the model (restricted choice), not by parsing
  free text, so the VLM slots into F2 (bars per stage) and F4 (accuracy vs coverage).
- `results/vlm_rationales_{stage}.jsonl`: `tile_id, pred, rationale, cited_regions, grounding_pass`.
- `results/vlm_feedback_curve.csv`: same columns as Track A's `feedback_curve.csv`, so it can be
  a third line on F1.
- Stages: `zeroshot`, `r1_sft`, `r2_grounded`, `r3_selftrain`, `fb_{n}`.

## Metrics (from Track A's plan)

- **Pair confusion rate** for each hard pair (A, B): (A→B + B→A) / (n_A + n_B), on `eval`.
- 9-class balanced accuracy on `eval`.
- Track B extras:
  - format compliance;
  - **grounding pass rate**: does a cited descriptor actually score higher than the rival
    class's descriptor on the cited crop, under local CONCH, with MUSK as a cross-check;
  - optional faithfulness check: masking the cited crop changes the answer.

## Method

Model: `Qwen/Qwen2-VL-2B-Instruct` with LoRA on the LLM projections and the vision tower frozen. Try
unfreezing the merger/projector as an ablation. 224 px tiles make about 64 visual tokens, so
training is cheap.

- **Round 0 (optional):** caption SFT on PathGen COAD/READ tiles, for domain vocabulary.
- **Round 1, discrimination (Aim 1):**
  - prompts: a 9-way prompt ("Which tissue type is this? Options: …") and pair prompts ("Stroma or smooth muscle?");
  - targets: label + 1-line reason drawn from that class's descriptors in Track A's `prompts.json`;
  - weighting: oversample hard tiles 3× (lowest CV margin within a hard pair);
  - semi-supervised add-on: extra NCT tiles pseudo-labeled where local CONCH and MUSK probes agree with p ≥ 0.9.
- **Round 2, grounded differential (Aim 2):** fixed output format
  ```
  Diagnosis: <A>
  Supporting: <descriptor> [region r,c]; ...
  Against <B>: <descriptor of B absent or weaker> [region r,c]; ...
  Conclusion: <one sentence>
  ```
  - regions: a 3×3 grid of overlapping 112 px crops, scored locally against the descriptors with CONCH;
  - target writing: a local LLM, or a template, fills the format from label + top crops;
  - filter: keep a target only if every citation passes grounding.
- **Round 3, self-training (Aim 2):** sample 4–8 answers for each hard train tile; keep the ones that are
  correct and pass grounding; continue LoRA on them plus 50% replay of the Round 2 data.
- **Aim 3, corrections:** take pool tiles in Track A's lowest-margin order. At 0/50/100/200 corrections,
  continue LoRA training on the corrections + replay, then evaluate on `eval`. Use fewer, larger rounds than
  Track A, because each refit is a training run, not a sub-second refit.

Go/no-go **Sun noon**: if the VLM is clearly behind the probe *and* not improving with rounds,
stop training and write up the gap and its causes. That's a real finding for the deck.

## Layout

```
trackb/
  CLAUDE.md  README.md  requirements.txt  .env.example
  configs/        yaml per run; descriptors read from ../prompts.json
  tb/             data.py (manifest join, local paths), crops.py, encoders.py (local CONCH/MUSK),
                  build_sft.py (r0–r3, fb), train_lora.py, score.py (class log-lik → preds csv),
                  generate.py, grounding.py, evaluate.py
  kaggle/         train.ipynb, make_dataset.sh
  scripts/        smoke tests
  data/ outputs/ third_party/   (gitignored)
```

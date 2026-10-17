# Track B: post-training a pathology VLM on hard tissue pairs

See `CLAUDE.md` for rules, data contracts and method.

## Setup (Windows native, also Linux/Kaggle)

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe torch torchvision --torch-backend cu128
uv pip install --python .venv/Scripts/python.exe -r requirements.txt "timm>=0.9.8" einops fairscale sentencepiece ftfy regex h5py safetensors tokenizers
git clone https://github.com/mahmoodlab/CONCH.git third_party/CONCH        # tested at 141cc09
git clone https://github.com/lilab-stanford/MUSK.git third_party/MUSK      # tested at 714b666
uv pip install --python .venv/Scripts/python.exe --no-deps -e third_party/CONCH -e third_party/MUSK
hf auth login        # your own approved access to MahmoodLab/CONCH and xiangjx/musk
hf download MahmoodLab/CONCH && hf download xiangjx/musk && hf download Qwen/Qwen2-VL-2B-Instruct
```

`--no-deps` for the encoders is deliberate: MUSK pins `torch==2.0.1`/`timm==0.9.8`. `tb/encoders.py`
works around two transformers-5 breakages (CONCH's `batch_encode_plus`, MUSK's `.spm` XLM-R tokenizer)
without patching `third_party/`.

## Smoke tests

```bash
python scripts/smoke_encoders.py   # CONCH + MUSK: shapes, norms, known-answer similarities
python scripts/smoke_vlm.py        # Qwen2-VL-2B: 1 generation, visual tokens, peak VRAM
```

"""Smoke test: load a Qwen-VL model, 1 greedy generation on a pathology image, report visual tokens and peak VRAM.

    python scripts/smoke_vlm.py [--model Qwen/Qwen2-VL-2B-Instruct] [--size 224]
"""
import argparse
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")
ap.add_argument("--image", default=str(ROOT / "third_party/MUSK/assets/lungaca1014.jpeg"))
ap.add_argument("--size", type=int, default=224, help="resize to size x size, like the 224 px CRC tiles")
args = ap.parse_args()

dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
t0 = time.time()
model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=dtype, device_map="cuda", attn_implementation="sdpa").eval()
proc = AutoProcessor.from_pretrained(args.model)
print(f"loaded {args.model} ({dtype}) in {time.time() - t0:.1f}s, weights {torch.cuda.memory_allocated() / 2**30:.2f} GiB")

img = Image.open(args.image).convert("RGB").resize((args.size, args.size))
options = "adipose, background, debris, lymphocytes, mucus, smooth muscle, normal colon mucosa, cancer-associated stroma, colorectal adenocarcinoma epithelium"
msgs = [{"role": "user", "content": [
    {"type": "image"},
    {"type": "text", "text": f"Which tissue type is this H&E tile? Options: {options}. Answer with one option, then one sentence of visual evidence."},
]}]
text = proc.apply_chat_template(msgs, add_generation_prompt=True)
inputs = proc(text=[text], images=[img], return_tensors="pt").to("cuda")
n_vis = int((inputs["input_ids"] == model.config.image_token_id).sum())

torch.cuda.reset_peak_memory_stats()
t0 = time.time()
with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
dt = time.time() - t0
new = out[0, inputs["input_ids"].shape[1]:]
print(f"visual tokens: {n_vis}  prompt tokens: {inputs['input_ids'].shape[1]}  new tokens: {len(new)}  gen time {dt:.2f}s")
print(f"peak VRAM during generate: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
print("answer:", proc.decode(new, skip_special_tokens=True).strip())
sys.exit(0 if len(new) > 0 else 1)

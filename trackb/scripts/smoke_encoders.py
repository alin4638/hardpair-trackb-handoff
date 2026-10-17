"""Smoke test: 2 images + 4 texts per encoder -> embedding shapes, norms and image-text similarities.

    python scripts/smoke_encoders.py
Known answers: roi1.jpg is CONCH's zero-shot example (invasive ductal > lobular carcinoma),
lungaca1014.jpeg is MUSK's demo image (lung adenocarcinoma > squamous cell carcinoma).
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tb import encoders  # noqa: E402

IMAGES = [ROOT / "third_party/CONCH/docs/roi1.jpg", ROOT / "third_party/MUSK/assets/lungaca1014.jpeg"]
TEXTS = ["an H&E image of invasive ductal carcinoma", "an H&E image of invasive lobular carcinoma",
         "an H&E image of lung adenocarcinoma", "an H&E image of lung squamous cell carcinoma"]
EXPECT = [(0, 1), (2, 3)]  # per image: (index that should win, index that should lose)

ok = True
imgs = [Image.open(p).convert("RGB") for p in IMAGES]
for name in ["conch", "musk"]:
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    enc = encoders.load(name)
    t_load = time.time() - t0
    zi, zt = enc.embed_images(imgs), enc.embed_texts(TEXTS)
    sim = zi @ zt.T
    print(f"\n[{name}] load {t_load:.1f}s  peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    print(f"  image emb {zi.shape} norms {np.linalg.norm(zi, axis=1).round(3)}")
    print(f"  text  emb {zt.shape} norms {np.linalg.norm(zt, axis=1).round(3)}")
    print("  cosine sim (rows = images, cols = texts):")
    for p, row in zip(IMAGES, sim):
        print(f"    {p.name:17s} " + "  ".join(f"{t[15:]}={s:+.3f}" for t, s in zip(TEXTS, row)))
    known = [bool(sim[i, w] > sim[i, l]) for i, (w, l) in enumerate(EXPECT)]
    check = zi.shape == (2, enc.dim) and zt.shape == (4, enc.dim) and all(known)
    print(f"  CHECK shapes=({enc.dim}) and known answers {known}: {'PASS' if check else 'FAIL'}")
    ok &= check
    del enc
    torch.cuda.empty_cache()

sys.exit(0 if ok else 1)

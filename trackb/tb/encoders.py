"""Local CONCH / MUSK encoders: batched image and text embeddings in the shared image-text space.

Weights stay in the HF cache (CC-BY-NC-ND): never copy them into the repo or a Kaggle dataset.
Both encoders return L2-normalised float32 numpy arrays, so cosine similarity = dot product.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

CONCH_DIM, MUSK_DIM = 512, 1024


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


class Conch:
    name, dim = "conch", CONCH_DIM

    def __init__(self, device=None, dtype=torch.float16):
        from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer
        from huggingface_hub import hf_hub_download

        self.device = device or _device()
        self.dtype = dtype if self.device == "cuda" else torch.float32
        ckpt = hf_hub_download("MahmoodLab/CONCH", "pytorch_model.bin")
        self.model, self.preprocess = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path=ckpt, device=self.device)
        self.model = self.model.to(self.dtype).eval()
        self.tokenizer = get_tokenizer()

    @torch.inference_mode()
    def embed_images(self, images: list[Image.Image], batch_size: int = 64) -> np.ndarray:
        out = []
        for i in range(0, len(images), batch_size):
            x = torch.stack([self.preprocess(im.convert("RGB")) for im in images[i:i + batch_size]])
            z = self.model.encode_image(x.to(self.device, self.dtype), proj_contrast=True, normalize=True)
            out.append(z.float().cpu().numpy())
        return np.concatenate(out)

    def tokenize(self, texts: list[str]) -> torch.Tensor:
        # Same as conch.open_clip_custom.tokenize, which calls batch_encode_plus (removed in transformers 5):
        # pad to 127, then append <pad> as the placeholder for the <cls> slot (context length 128).
        ids = self.tokenizer(texts, max_length=127, add_special_tokens=True, return_token_type_ids=False,
                             truncation=True, padding="max_length", return_tensors="pt")["input_ids"]
        return torch.nn.functional.pad(ids, (0, 1), value=self.tokenizer.pad_token_id)

    @torch.inference_mode()
    def embed_texts(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            tok = self.tokenize(texts[i:i + batch_size]).to(self.device)
            out.append(self.model.encode_text(tok, normalize=True).float().cpu().numpy())
        return np.concatenate(out)


class _XlmrSpm:
    """Slow XLMRobertaTokenizer ids from a sentencepiece file (transformers 5 can no longer build one from .spm).

    fairseq layout: <s>=0 <pad>=1 </s>=2 <unk>=3, every other piece id is spm id + 1. Only what
    musk.utils.xlm_tokenizer needs: encode() with <s> ... </s>, plus bos/eos/pad ids.
    """
    bos_token_id, pad_token_id, eos_token_id, unk_token_id = 0, 1, 2, 3

    def __init__(self, path):
        import sentencepiece as sp

        self.sp = sp.SentencePieceProcessor(model_file=path)

    def encode(self, text: str) -> list[int]:
        ids = [i + 1 if i != self.sp.unk_id() else self.unk_token_id for i in self.sp.encode(text)]
        return [self.bos_token_id] + ids + [self.eos_token_id]


class Musk:
    """MUSK with its retrieval head + out_norm (zero-shot space), no multiscale aug, 384 px input."""
    name, dim = "musk", MUSK_DIM

    def __init__(self, device=None, dtype=torch.float16):
        import os

        import torchvision.transforms as T
        from huggingface_hub import hf_hub_download
        from musk import modeling, utils  # noqa: F401  (registers musk_large_patch16_384 with timm)
        from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
        from timm.models import create_model

        self.device = device or _device()
        self.dtype = dtype if self.device == "cuda" else torch.float32
        self.utils = utils
        # Pass the cached file path directly; the default "hf_hub:" route re-downloads 3 GB into ~/.cache/.
        ckpt = hf_hub_download("xiangjx/musk", "model.safetensors")
        self.model = create_model("musk_large_patch16_384")
        utils.load_model_and_may_interpolate(ckpt, self.model, "model|module", "")
        self.model = self.model.to(self.device, self.dtype).eval()
        self.preprocess = T.Compose([
            T.Resize(384, interpolation=3, antialias=True),
            T.CenterCrop((384, 384)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
        ])
        spm = os.path.join(os.path.dirname(modeling.__file__), "models", "tokenizer.spm")
        self.tokenizer = _XlmrSpm(spm)

    @torch.inference_mode()
    def embed_images(self, images: list[Image.Image], batch_size: int = 32) -> np.ndarray:
        out = []
        for i in range(0, len(images), batch_size):
            x = torch.stack([self.preprocess(im.convert("RGB")) for im in images[i:i + batch_size]])
            z = self.model(image=x.to(self.device, self.dtype), with_head=True, out_norm=True,
                           ms_aug=False, return_global=True)[0]
            out.append(z.float().cpu().numpy())
        return np.concatenate(out)

    @torch.inference_mode()
    def embed_texts(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            rows = [self.utils.xlm_tokenizer(t, self.tokenizer, max_len=100) for t in texts[i:i + batch_size]]
            ids = torch.tensor([r[0] for r in rows], device=self.device)
            pad = torch.tensor([r[1] for r in rows], device=self.device)
            z = self.model(text_description=ids, padding_mask=pad, with_head=True, out_norm=True,
                           ms_aug=False, return_global=True)[1]
            out.append(z.float().cpu().numpy())
        return np.concatenate(out)


ENCODERS = {"conch": Conch, "musk": Musk}


def load(name: str, **kw):
    return ENCODERS[name](**kw)

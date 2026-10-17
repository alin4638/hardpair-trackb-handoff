"""Manifest join, local tile paths, and the train/pool/eval leakage guard.

Source of truth is Track A's `../data/manifest.csv` (tile_id, path, label, split). Until it exists we
use `data/manifest_standin.csv`, built with the rule from pipeline_plan.md (00_manifest.py):
2,000 tiles/class from NCT-CRC-HE-100K with seed 0, plus all of CRC-VAL-HE-7K split 50/50
stratified into pool/eval. Track A's exact sampling code is unknown, so the stand-in's train
sample and pool/eval assignment will NOT match tile-for-tile; `compare_manifests` reports by how much.

Every SFT builder must get its rows from `training_rows()`; nothing else in Track B reads tiles for training.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]          # trackb/
RAW = ROOT / "data" / "raw"
TRACK_A_MANIFEST = ROOT.parent / "data" / "manifest.csv"
STANDIN_MANIFEST = ROOT / "data" / "manifest_standin.csv"

CLASSES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
DATASETS = {"100k": "NCT-CRC-HE-100K", "7k": "CRC-VAL-HE-7K"}
# Kather et al. 2019 (Zenodo 1214456) per-class counts, used to verify the local copy.
EXPECTED_COUNTS = {
    "100k": {"ADI": 10407, "BACK": 10566, "DEB": 11512, "LYM": 11557, "MUC": 8896,
             "MUS": 13536, "NORM": 8763, "STR": 10446, "TUM": 14317},
    "7k": {"ADI": 1338, "BACK": 847, "DEB": 339, "LYM": 634, "MUC": 1035,
           "MUS": 592, "NORM": 741, "STR": 421, "TUM": 1233},
}
SPLITS = ("train", "pool", "eval")
N_PER_CLASS, SEED = 2000, 0


# ---------------------------------------------------------------- local files

def list_tiles(ds: str) -> pd.DataFrame:
    """All local tiles of a dataset ('100k' or '7k'): columns rel_path, label, fname. Sorted, so deterministic."""
    base = RAW / DATASETS[ds]
    rows = [(p.relative_to(RAW).as_posix(), c, p.name)
            for c in CLASSES for p in sorted((base / c).glob("*.tif"))]
    if not rows:
        raise FileNotFoundError(f"no tiles under {base}; run scripts/download_data.sh")
    return pd.DataFrame(rows, columns=["rel_path", "label", "fname"])


def check_counts(ds: str, tiles: pd.DataFrame | None = None) -> pd.DataFrame:
    tiles = list_tiles(ds) if tiles is None else tiles
    got = tiles.label.value_counts().reindex(CLASSES, fill_value=0)
    exp = pd.Series(EXPECTED_COUNTS[ds])
    return pd.DataFrame({"local": got, "expected": exp, "ok": got == exp})


def _dataset_of(path: str) -> str:
    """'7k' or '100k' from the folder name (Delta: .../CRC-VAL-HE-7K/..., .../NCT-CRC-HE-100K/...)."""
    s = path.upper()
    if "HE-7K" in s:
        return "7k"
    if "HE-100K" in s:
        return "100k"
    raise ValueError(f"cannot tell dataset from path {path!r}")


def local_path(manifest_path: str) -> Path:
    """Map a manifest path (Track A's Delta absolute path or our relative one) to the local file.

    Matching is on (dataset, class folder, filename), which is identical across copies of Zenodo 1214456.
    """
    parts = re.split(r"[\\/]", manifest_path)
    fname, cls = parts[-1], parts[-2]
    p = RAW / DATASETS[_dataset_of(manifest_path)] / cls / fname
    if not p.exists():
        raise FileNotFoundError(f"{manifest_path!r} -> {p} (missing locally)")
    return p


# ---------------------------------------------------------------- manifest

def build_standin(out: Path = STANDIN_MANIFEST) -> pd.DataFrame:
    """Stand-in for Track A's manifest, same columns; tile_ids follow pipeline_plan.md (train_STR_0412, 7k_STR_0123)."""
    from sklearn.model_selection import train_test_split

    big, val = list_tiles("100k"), list_tiles("7k")
    rng = np.random.default_rng(SEED)
    train = []
    for c in CLASSES:
        sub = big[big.label == c]
        pick = np.sort(rng.choice(len(sub), N_PER_CLASS, replace=False))
        s = sub.iloc[pick].reset_index(drop=True)
        s["tile_id"] = [f"train_{c}_{i:04d}" for i in range(len(s))]
        train.append(s)
    train = pd.concat(train)
    train["split"] = "train"

    val = val.copy()
    val["tile_id"] = val.groupby("label").cumcount().map("{:04d}".format).radd("7k_" + val.label + "_")
    pool_idx, eval_idx = train_test_split(np.arange(len(val)), test_size=0.5, stratify=val.label, random_state=SEED)
    val["split"] = ""
    val.iloc[pool_idx, val.columns.get_loc("split")] = "pool"
    val.iloc[eval_idx, val.columns.get_loc("split")] = "eval"

    m = pd.concat([train, val])[["tile_id", "rel_path", "label", "split"]].rename(columns={"rel_path": "path"})
    m = m.reset_index(drop=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        f.write("# STAND-IN for Track A data/manifest.csv (rule from pipeline_plan.md; not tile-identical)\n")
        m.to_csv(f, index=False)
    return m


def manifest_source() -> Path:
    return TRACK_A_MANIFEST if TRACK_A_MANIFEST.exists() else STANDIN_MANIFEST


def load_manifest(with_local: bool = True) -> pd.DataFrame:
    """Track A's manifest if present, else the stand-in (built on first use). Adds `local_path`, `fname`."""
    src = manifest_source()
    if src == STANDIN_MANIFEST and not src.exists():
        build_standin()
    m = pd.read_csv(src, comment="#")
    assert list(m.columns[:4]) == ["tile_id", "path", "label", "split"], f"unexpected columns {list(m.columns)}"
    assert m.tile_id.is_unique, "duplicate tile_id in manifest"
    assert set(m.split) <= set(SPLITS), f"unexpected splits {set(m.split)}"
    assert set(m.label) <= set(CLASSES), f"unexpected labels {set(m.label)}"
    m.attrs["source"] = src.name
    m["fname"] = m.path.str.split(r"[\\/]").str[-1]
    if with_local:
        m["local_path"] = [str(local_path(p)) for p in m.path]
    return m


def compare_manifests(real: pd.DataFrame, standin: pd.DataFrame) -> dict:
    """What changes when Track A's manifest replaces the stand-in (matching tiles by filename)."""
    rep = {"columns_equal": list(real.columns[:4]) == list(standin.columns[:4])}
    for s in SPLITS:
        a, b = set(real.fname[real.split == s]), set(standin.fname[standin.split == s])
        rep[s] = {"real": len(a), "standin": len(b), "overlap": len(a & b)}
    rep["counts_real"] = real.groupby(["split", "label"]).size().unstack(fill_value=0).to_dict("index")
    return rep


# ---------------------------------------------------------------- leakage guard

class LeakError(AssertionError):
    pass


def assert_no_leak(rows: pd.DataFrame, manifest: pd.DataFrame, allow_pool_ids=()) -> None:
    """Raise if any row is an eval tile, or a pool tile not explicitly allowed (Aim 3 corrections).

    Checks tile_id AND filename, so a tile can't slip in under another id (e.g. an 'extra NCT' id).
    """
    allow = set(allow_pool_ids)
    ev = manifest[manifest.split == "eval"]
    po = manifest[(manifest.split == "pool") & ~manifest.tile_id.isin(allow)]
    bad = pd.concat([
        rows[rows.tile_id.isin(ev.tile_id) | rows.fname.isin(ev.fname)].assign(why="eval"),
        rows[rows.tile_id.isin(po.tile_id) | rows.fname.isin(po.fname)].assign(why="pool (not allowed)"),
    ])
    if allow - set(manifest.tile_id[manifest.split == "pool"]):
        raise LeakError(f"allow_pool_ids contains non-pool ids: {sorted(allow - set(manifest.tile_id[manifest.split == 'pool']))[:5]}")
    if len(bad):
        raise LeakError(f"{len(bad)} held-out tiles reached a training builder, e.g.\n{bad[['tile_id', 'fname', 'why']].head()}")


def extra_nct(manifest: pd.DataFrame) -> pd.DataFrame:
    """100K tiles outside the manifest's train sample (for pseudo-labelling). ids: nct_<filename stem>."""
    big = list_tiles("100k")
    big = big[~big.fname.isin(manifest.fname)].copy()
    big["tile_id"] = "nct_" + big.fname.str.replace(".tif", "", regex=False)
    big["split"] = "train_extra"
    big["local_path"] = [str(RAW / p) for p in big.rel_path]
    return big.rename(columns={"rel_path": "path"})[["tile_id", "path", "label", "split", "fname", "local_path"]]


def training_rows(manifest: pd.DataFrame | None = None, *, include_extra: bool = False,
                  pool_ids=(), log_path: Path | None = None) -> pd.DataFrame:
    """The ONLY entry point for training tiles: train split (+ extra NCT) (+ logged pool corrections).

    pool_ids: Aim 3 corrections; they are written to `log_path` (required when pool_ids is non-empty).
    """
    m = load_manifest() if manifest is None else manifest
    parts = [m[m.split == "train"]]
    if include_extra:
        parts.append(extra_nct(m))
    if len(pool_ids):
        if log_path is None:
            raise ValueError("pool corrections must be logged: pass log_path")
        corr = m[m.tile_id.isin(set(pool_ids)) & (m.split == "pool")]
        if len(corr) != len(set(pool_ids)):
            raise LeakError("some pool_ids are not pool tiles")
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        corr[["tile_id", "label"]].to_csv(log_path, index=False)
        parts.append(corr)
    rows = pd.concat(parts, ignore_index=True)
    assert_no_leak(rows, m, allow_pool_ids=pool_ids)
    return rows

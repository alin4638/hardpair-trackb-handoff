"""B-owned frozen manifest, role assignment, path mapping and leakage guards."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]          # trackb/
RAW = ROOT / "data" / "raw"
MANIFEST = ROOT / "data" / "manifest.csv"
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


def manifest_source(config=None) -> Path:
    return Path(config['manifest']) if config else MANIFEST


def freeze_roles(m, path, n_dev=200):
    from .common import file_hash, atomic_json, read_json
    path = Path(path)
    rng = np.random.default_rng(0)
    expected = m[['tile_id', 'split']].rename(columns={'split': 'role'}).copy()
    expected.loc[expected.role == 'train', 'role'] = 'train_fit'
    for c in CLASSES:
        ids = sorted(m.loc[(m.split == 'train') & (m.label == c), 'tile_id'])
        if len(ids) <= n_dev: raise ValueError(f'Insufficient train rows for {c}')
        dev = rng.choice(ids, n_dev, replace=False)
        expected.loc[expected.tile_id.isin(dev), 'role'] = 'train_dev'
    if path.exists():
        old = pd.read_csv(path)
        if not old.equals(expected): raise ValueError('Frozen roles differ; create a new explicit data version')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        expected.to_csv(path, index=False)
    atomic_json(path.with_suffix('.meta.json'), {'seed': 0, 'split_unit': 'tile',
                'sha256': file_hash(path), 'counts': expected.role.value_counts().to_dict()})
    return expected


def load_manifest(with_local=True, config=None):
    from .common import file_hash, read_json
    src = manifest_source(config)
    m = pd.read_csv(src, comment='#')
    if list(m.columns[:4]) != ['tile_id', 'path', 'label', 'split']: raise ValueError('Invalid manifest columns')
    if not m.tile_id.is_unique or not m.path.is_unique: raise ValueError('Duplicate image identity')
    if not set(m.split) <= set(SPLITS) or not set(m.label) <= set(CLASSES): raise ValueError('Invalid manifest roles/classes')
    meta = read_json(src.with_suffix('.meta.json'))
    if not meta or meta['sha256'] != file_hash(src): raise ValueError('Manifest hash mismatch or missing metadata')
    roles = Path(config['roles']) if config else ROOT / 'data/roles.csv'
    if not roles.exists(): raise FileNotFoundError('Run prepare to freeze roles first')
    r = pd.read_csv(roles)
    if not r.tile_id.is_unique or set(r.tile_id) != set(m.tile_id): raise ValueError('Roles identity mismatch')
    rm = read_json(roles.with_suffix('.meta.json'))
    if not rm or rm['sha256'] != file_hash(roles): raise ValueError('Frozen roles hash mismatch')
    m = m.merge(r, on='tile_id', validate='one_to_one', sort=False)
    if not ((m.split.eq('train') & m.role.isin(['train_fit','train_dev'])) | m.split.eq(m.role)).all():
        raise ValueError('Role/split mismatch')
    m['fname'] = m.path.str.split(r'[\\/]').str[-1]
    raw = Path(config['data_root']) if config else RAW
    m['local_path'] = [str(raw / Path(p)) for p in m.path]
    if any(Path(p).is_absolute() or '..' in Path(p).parts for p in m.path): raise ValueError('Manifest paths must be relative')
    if with_local:
        missing = [p for p in m.local_path if not Path(p).is_file()]
        if missing: raise FileNotFoundError(f'{len(missing)} images missing; first: {missing[0]}')
    m.attrs.update(source=str(src), manifest_version=meta['manifest_version'])
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
    ev = manifest[(manifest.split == "eval") | (manifest.get("role", manifest.split) == "train_dev")]
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


def training_rows(manifest=None, *, include_extra=False, pool_ids=(), log_path=None,
                  feedback=None, extra=None):
    """Only supervised fit rows and explicit current corrected pool labels enter training."""
    m = load_manifest() if manifest is None else manifest
    if 'role' not in m: raise LeakError('Frozen roles are required')
    parts = [m[m.role == 'train_fit'].copy()]
    if include_extra and extra is None: raise ValueError('Register extra NCT in an explicit extra manifest first')
    if extra is not None:
        if not extra.path.str.startswith('NCT-CRC-HE-100K/').all(): raise LeakError('Extra data must be NCT')
        if not set(extra.label) <= set(CLASSES): raise ValueError('Invalid extra labels')
        if set(extra.tile_id) & set(m.tile_id) or set(extra.fname) & set(m.fname): raise LeakError('Extra images overlap frozen manifest')
        parts.append(extra)
    ids = set(pool_ids)
    if ids:
        if feedback is None or set(feedback) != ids: raise ValueError('Current feedback labels required')
        if log_path is None: raise ValueError('Feedback training must be logged')
        corr = m[m.tile_id.isin(ids) & (m.role == 'pool')].copy()
        if len(corr) != len(ids): raise LeakError('Feedback contains non-pool IDs')
        corr['label'] = corr.tile_id.map(lambda x: feedback[x]['corrected_label'])
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        corr[['tile_id','label']].to_csv(log_path,index=False)
        parts.append(corr)
    rows = pd.concat(parts,ignore_index=True)
    assert_no_leak(rows,m,allow_pool_ids=ids)
    return rows

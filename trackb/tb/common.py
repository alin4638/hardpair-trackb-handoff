"""Small portable I/O helpers. No torch import in CPU planning paths."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import tempfile
import yaml

ROOT = Path(__file__).resolve().parents[1]

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()

def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(value, f, indent=2, default=str, allow_nan=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def read_json(path, default=None):
    return json.loads(Path(path).read_text(encoding='utf-8')) if Path(path).exists() else default

def read_jsonl(path):
    if not Path(path).exists(): return []
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]

def write_jsonl(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(''.join(json.dumps(x, default=str, allow_nan=False) + '\n' for x in rows), encoding='utf-8')
    os.replace(tmp, path)

def append_jsonl(path, row):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, default=str, allow_nan=False) + '\n')
        f.flush(); os.fsync(f.fileno())

def load_config(path):
    c = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    for key in ('manifest', 'roles', 'data_root', 'descriptors', 'run_dir', 'feedback_dir'):
        p = Path(os.path.expandvars(c[key])).expanduser()
        c[key] = str(p if p.is_absolute() else ROOT / p)
    for key in ('extra_manifest',):
        if c.get(key):
            p = Path(os.path.expandvars(c[key])).expanduser()
            c[key] = str(p if p.is_absolute() else ROOT / p)
    if c['model']['name'] != 'Qwen/Qwen2-VL-2B-Instruct':
        raise ValueError('This backend implements Qwen2-VL; validate another model in a separate backend')
    if c['train']['steps'] <= 0 or c['train']['grad_accum'] <= 0: raise ValueError('Invalid training steps')
    if not 4 <= c['r3']['candidates'] <= 8: raise ValueError('R3 requires 4–8 candidates')
    if c['feedback']['source'] not in ('expert_label', 'simulated_dataset_label'): raise ValueError('Invalid feedback source')
    return c

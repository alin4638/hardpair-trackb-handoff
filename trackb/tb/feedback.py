"""Append-only category feedback; persistent selection precedes label acquisition."""
from datetime import datetime,timezone
from pathlib import Path
import os
from contextlib import contextmanager
from .common import append_jsonl,read_jsonl,digest,write_jsonl
from .data import CLASSES
SOURCES={'expert_label','simulated_dataset_label'}

@contextmanager
def lock(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    try: fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError: raise RuntimeError(f'Writer active (or stale lock after crash): {path}')
    try:
        os.write(fd,str(os.getpid()).encode());os.close(fd);yield
    finally: path.unlink(missing_ok=True)

def current(records):
    result={}
    for r in records:
        if r['corrected_label'] not in CLASSES or r['feedback_source'] not in SOURCES: raise ValueError('Invalid feedback record')
        old=result.get(r['tile_id'])
        if old and old['feedback_source']=='expert_label' and r['feedback_source']=='simulated_dataset_label': continue
        result[r['tile_id']]=r
    return result

def submit(path,query,label,source):
    if label not in CLASSES or source not in SOURCES: raise ValueError('Invalid class/source')
    required={'tile_id','query_model_version','original_prediction','query_id'}
    if not required<=set(query): raise ValueError('Feedback must reference an existing selection')
    row={k:query[k] for k in required};row.update(corrected_label=label,feedback_source=source,timestamp=datetime.now(timezone.utc).isoformat())
    with lock(Path(path).with_suffix('.lock')):
        row['revision']=len(read_jsonl(path))+1
        append_jsonl(path,row)
    return row

def validate_records(records,queries,m):
    pool=set(m.loc[m.role=='pool','tile_id']);q={(x['query_id'],x['tile_id']):x for x in queries}
    for r in records:
        origin=q.get((r.get('query_id'),r['tile_id']))
        if r['tile_id'] not in pool or origin is None: raise ValueError('Feedback was not selected from pool')
        if any(r[k]!=origin[k] for k in ('query_model_version','original_prediction')): raise ValueError('Feedback query provenance mismatch')
    return current(records)

def select(preds,used,n,version):
    rows=preds[~preds.tile_id.isin(used)].sort_values(['margin','tile_id']).head(n)
    query_id=digest({'model':version,'ids':rows.tile_id.tolist()})[:20]
    return [{'tile_id':r.tile_id,'query_model_version':version,'original_prediction':r.pred,
             'top2':getattr(r,'top2',None), 'margin':float(r.margin),'query_id':query_id,'selected_at':datetime.now(timezone.utc).isoformat()} for r in rows.itertuples()]

def import_labels(frame,queries,manifest,path,source):
    """Import only already-selected pool IDs; CSV labels never create selections."""
    if source not in SOURCES: raise ValueError('Choose expert_label or simulated_dataset_label')
    if not {'tile_id','corrected_label'}<=set(frame.columns): raise ValueError('CSV requires tile_id,corrected_label')
    if not frame.tile_id.is_unique: raise ValueError('One current label per tile is required')
    lookup={q['tile_id']:q for q in queries};pool=set(manifest.loc[manifest.role=='pool','tile_id'])
    for row in frame.itertuples():
        if row.tile_id not in pool or row.tile_id not in lookup: raise ValueError(f'Not a selected pool tile: {row.tile_id}')
        if row.corrected_label not in CLASSES: raise ValueError(f'Unknown class: {row.corrected_label}')
    return [submit(path,lookup[r.tile_id],r.corrected_label,source) for r in frame.itertuples()]

"""Supervision provenance and unified target builders."""
import random
from .data import CLASSES, training_rows
from .grounding import evidence, render, generation_prompt
from .common import digest,file_hash
from pathlib import Path
CLASS_NAMES=dict(zip(CLASSES, ['adipose tissue','background with no tissue','debris and necrosis',
    'lymphocytes','mucus','smooth muscle','normal colon mucosa','cancer-associated stroma',
    'colorectal adenocarcinoma epithelium']))
CLASS_PROMPT='Which tissue type is this H&E image? Choose exactly one code: '+ '; '.join(f'{c} = {CLASS_NAMES[c]}' for c in CLASSES)+'. Answer with the code only.'

def target(row,prompt,answer,source,weight=1):
    obj={'tile_id':row.tile_id,'path':row.path,'local_path':row.local_path,'label':row.label,
         'prompt':prompt,'answer':answer,'supervision_source':source,'weight':weight,
         'image_sha256':file_hash(row.local_path) if Path(row.local_path).is_file() else None}
    obj['target_version']=digest({k:v for k,v in obj.items() if k!='local_path'})
    return obj

def r1(rows,hard=None):
    hard={} if hard is None else hard.set_index('tile_id').to_dict('index')
    out=[]
    for row in rows.itertuples():
        h=hard.get(row.tile_id,{})
        out.append(target(row,CLASS_PROMPT,row.label,'dataset_label',h.get('weight',1)))
        if h.get('weight',1)>1:
            # Alphabetical order avoids leaking which of the two options is the target.
            pair=sorted([row.label,h['rival']])
            out.append(target(row,'Which tissue type is this H&E image? Choose '+ ' or '.join(f'{cl} ({CLASS_NAMES[cl]})' for cl in pair)+'. Answer with the code only.',row.label,'dataset_label',3))
    return out

def r2(rows,cache,threshold):
    if not set(rows.role)<= {'train_fit','extra_train'}: raise ValueError('R2 cannot train held-out roles')
    out=[]
    for row in rows.itertuples():
        scores=cache.read('conch',row,'regions');e=evidence(scores,cache.descriptors,row.label)
        if e['margin']<=threshold: continue
        musk=cache.read('musk',row,'regions')
        t=target(row,generation_prompt(cache.descriptors),render(e),'automatic_CONCH_region_evidence')
        t['evidence']={**e,'musk_margin':float(musk[e['region_index'],e['support_index']]-musk[e['region_index'],e['against_index']])}
        out.append(t)
    return out

def mix(primary,replay,fraction=0.5,seed=0):
    if not primary: raise ValueError('No accepted primary targets; inspect target_build_log before proceeding')
    if not 0 <= fraction < 1: raise ValueError('Invalid replay fraction')
    if fraction and not replay: raise ValueError('No valid R2 replay')
    n=round(len(primary)*fraction/(1-fraction))
    rng=random.Random(seed)
    return primary+[dict(x) for x in rng.choices(replay,k=n)] if n else primary

def feedback_targets(m,current,replay,log_path,fraction=0.5):
    rows=training_rows(m,pool_ids=list(current),feedback=current,log_path=log_path)
    rows=rows[rows.tile_id.isin(current)]
    targets=[target(row,CLASS_PROMPT,row.label,current[row.tile_id]['feedback_source']) for row in rows.itertuples()]
    fit=set(m.loc[m.role=='train_fit','tile_id'])
    valid=[t for t in replay if t['tile_id'] in fit and (t['tile_id'] not in current or t['label']==current[t['tile_id']]['corrected_label'])]
    return mix(targets,valid,fraction)

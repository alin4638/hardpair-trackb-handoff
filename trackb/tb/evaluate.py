"""Stage reports with explicit denominators and no invented absent metrics."""
from pathlib import Path
import numpy as np
import pandas as pd
from .data import CLASSES
from .hard import PAIRS
from .common import atomic_json,read_jsonl,write_jsonl,read_json,file_hash
from .grounding import generation_prompt,validate

def metrics(rows,preds,rationales=None):
    if not preds.tile_id.is_unique or set(rows.tile_id)!=set(preds.tile_id): raise ValueError('Evaluation IDs must match exactly')
    joined=rows[['tile_id','label']].merge(preds,on='tile_id',validate='one_to_one')
    recalls={c:float((joined.loc[joined.label==c,'pred']==c).mean()) if (joined.label==c).any() else None for c in CLASSES}
    result={'n_eval':len(joined),'balanced_accuracy':float(np.mean(list(recalls.values()))) if all(x is not None for x in recalls.values()) else None,
            'recall':recalls,'class_counts':joined.label.value_counts().to_dict()}
    for a,b in PAIRS:
        n=int(joined.label.isin([a,b]).sum()); errors=int(((joined.label.eq(a)&joined.pred.eq(b))|(joined.label.eq(b)&joined.pred.eq(a))).sum())
        result[f'pair_error_{a}_{b}']=errors/n if n else None
        result[f'pair_counts_{a}_{b}']={'errors':errors,'denominator':n}
    result['confusion_matrix']=pd.crosstab(joined.label,joined.pred).reindex(index=CLASSES,columns=CLASSES,fill_value=0).to_dict('index')
    rr=rationales or []
    result.update(n_rationale=len(rr),format_rate=sum(bool(x['format_pass']) for x in rr)/len(rr) if rr else None,
                  grounding_pass_rate=sum(bool(x['grounding_pass']) for x in rr)/len(rr) if rr else None)
    guided=[x for x in rr if 'evidence_status' in x]
    result.update(explanation_sources={s:sum(x.get('explanation_source','free_student')==s for x in rr)
                                      for s in {x.get('explanation_source','free_student') for x in rr}},
                  retrieved_support_rate=sum(x['evidence_status']=='supported' for x in guided)/len(guided) if guided else None,
                  student_grounding_rate=sum(bool(x.get('student_grounding_pass')) for x in guided)/len(guided) if guided else None)
    return result

def score_rows(student,rows,path,version,rationales_path=None,cache=None,threshold=0):
    """Flush after each tile; restart skips only matching model and image identities."""
    path=Path(path); partial=path.with_suffix('.partial.jsonl')
    contract={'model_version':version,'ids':rows.tile_id.tolist(),'paths':rows.path.tolist(),'image_hashes':[file_hash(p) for p in rows.local_path],
              'scoring':'sum log p(class code tokens); no EOS; fixed prompt v1; nine candidates batched right-padded'}
    old=read_json(path.with_suffix('.contract.json'))
    if old and old!=contract: raise ValueError('Prediction contract changed; use a new output filename')
    atomic_json(path.with_suffix('.contract.json'),contract)
    done={r['tile_id']:r for r in read_jsonl(partial)}
    rr={r['tile_id']:r for r in read_jsonl(rationales_path)} if rationales_path else {}
    guide=None
    if rationales_path and getattr(student,'c',{}).get('grounding',{}).get('mode')=='guided':
        from .region_guide import RegionGuide
        guide=RegionGuide.load(cache)
        rp=Path(rationales_path).with_suffix('.contract.json')
        rc={'model_version':version,'guide_version':guide.version,'mode':'guided'}
        old=read_json(rp)
        if old and old!=rc: raise ValueError('Rationale guide changed; use a new output filename')
        atomic_json(rp,rc)
    from .common import append_jsonl
    for row in rows.itertuples():
        if row.tile_id not in done:
            pred={'tile_id':row.tile_id,**student.scores(row.local_path)}
            append_jsonl(partial,pred);done[row.tile_id]=pred
    if rationales_path:
        # Greedy rationales in small left-padded batches; each tile is still flushed as soon as it is checked.
        pending=[row for row in rows.itertuples() if row.tile_id not in rr]
        size=max(1,int(getattr(student,'c',{}).get('model',{}).get('generate_batch',8)))
        for offset in range(0,len(pending),size):
            batch=pending[offset:offset+size];ps=[done[row.tile_id] for row in batch]
            if guide is not None:
                evidence=[guide.retrieve(row,p['pred'],p['top2']) for row,p in zip(batch,ps)]
                indices=[i for i,e in enumerate(evidence) if e['accepted']]
                texts=student.generate_many([batch[i].local_path for i in indices],
                        [guide.prompt(evidence[i]) for i in indices],
                        predictions=[ps[i]['pred'] for i in indices]) if indices else []
                generated=dict(zip(indices,texts))
                for i,(row,p) in enumerate(zip(batch,ps)):
                    rr[row.tile_id]={**guide.finish(row,p['pred'],p['top2'],generated.get(i)),'model_version':version}
                    append_jsonl(rationales_path,rr[row.tile_id])
                continue
            texts=student.generate_many([row.local_path for row in batch],
                                        [generation_prompt(cache.descriptors,p['pred'],p['top2']) for p in ps],
                                        predictions=[p['pred'] for p in ps])
            for row,p,text in zip(batch,ps,texts):
                check=validate(text,p['pred'],cache.read('conch',row,'regions'),cache.descriptors,threshold,cache.read('musk',row,'regions'),expected_rival=p['top2'])
                # Do not repair a contradictory generated answer into a claimed successful output.
                rr[row.tile_id]={'tile_id':row.tile_id,'pred':p['pred'],'rationale':text,'model_version':version,**check}
                append_jsonl(rationales_path,rr[row.tile_id])
    frame=pd.DataFrame([done[x] for x in rows.tile_id])
    frame.drop(columns=['candidate_token_lengths']).to_csv(path,index=False)
    return frame,list(rr.values())

def upsert_csv(path,row,keys):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    old=pd.read_csv(path) if path.exists() else pd.DataFrame()
    if len(old):
        mask=np.ones(len(old),dtype=bool)
        for key in keys: mask &= old[key].astype(str).eq(str(row[key])).to_numpy()
        old=old.loc[~mask]
    tmp=path.with_suffix('.tmp');pd.concat([old,pd.DataFrame([row])],ignore_index=True).to_csv(tmp,index=False);tmp.replace(path)

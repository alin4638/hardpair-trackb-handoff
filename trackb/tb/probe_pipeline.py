"""Final experiment: cached pathology features, CPU feedback, one final Qwen explanation pass."""
from pathlib import Path
import gc
import logging
import shutil
import numpy as np
import pandas as pd
from .common import atomic_json,read_json,read_jsonl,write_jsonl,append_jsonl,digest
from .data import CLASSES
from .feedback import validate_records,submit
from .probe import feature_bank,fit_probe,PathologyProbe,subset,diverse_queries
from .evaluate import metrics,upsert_csv
from .region_guide import RegionGuide


def save(c,state):atomic_json(Path(c['run_dir'])/'run_manifest.json',state)


def publish(c,m,bank,probe,state,active):
    root=Path(c['run_dir']);name=f'probe_fb_{len(active)}'
    folder=root/'results/by_model'/probe.version;folder.mkdir(parents=True,exist_ok=True)
    rows=m[m.role=='eval'];idx=np.flatnonzero(m.role.eq('eval'))
    preds=probe.predict(rows.tile_id,subset(bank,idx));predpath=folder/f'preds_vlm_{name}.csv'
    preds.to_csv(predpath,index=False)
    rrpath=folder/f'vlm_rationales_{name}.jsonl'
    if not rrpath.exists():write_jsonl(rrpath,[])
    result=metrics(rows,preds,read_jsonl(rrpath))
    result.update(run_id=c['run_id'],stage=name,model_version=probe.version,manifest_version=m.attrs['manifest_version'],
                  classification_backend='pathology_probe',n_feedback=len(active))
    atomic_json(folder/'metrics.json',result)
    pd.DataFrame([{'class':cl,'recall':result['recall'][cl],'n':result['class_counts'].get(cl,0)} for cl in CLASSES]).to_csv(folder/'per_class_recall.csv',index=False)
    pd.DataFrame.from_dict(result['confusion_matrix'],orient='index').to_csv(folder/'confusion_matrix.csv')
    keys=('run_id','stage','model_version','manifest_version','classification_backend','n_feedback','n_eval','accuracy','balanced_accuracy',
          'pair_error_STR_MUS','pair_error_DEB_MUC','pair_error_NORM_TUM','format_rate','grounding_pass_rate',
          'n_rationale','retrieved_support_rate','student_grounding_rate')
    upsert_csv(root/'results/vlm_summary.csv',{k:result[k] for k in keys},['run_id','stage','model_version'])
    n_changed=sum(r['corrected_label']!=r['original_prediction'] for r in active.values())
    sources=sorted({r['feedback_source'] for r in active.values()}) or [c['feedback']['source']]
    for source in sources:
        curve={k:result[k] for k in ('run_id','stage','model_version','n_eval','balanced_accuracy','pair_error_STR_MUS','pair_error_DEB_MUC','pair_error_NORM_TUM')}
        curve.update(n_feedback=len(active),n_changed=n_changed,feedback_source=source,
                     n_source_feedback=sum(r['feedback_source']==source for r in active.values()),classification_backend='pathology_probe')
        upsert_csv(root/'results/vlm_feedback_curve.csv',curve,['run_id','stage','model_version','feedback_source'])
    for path in (predpath,rrpath):shutil.copy2(path,root/'results'/path.name)
    state.setdefault('results',{})[name]={'path':str(folder),'model_version':probe.version,'classification_backend':'pathology_probe'}
    state['classifier']={k:v for k,v in probe.metadata.items() if k!='contract'}
    state['classifier']['result_stage']=name
    save(c,state)
    logging.info('Classifier %s: n_feedback=%d eval_BA=%.4f',probe.version,len(active),result['balanced_accuracy'])
    return result


def run_probe(c,m,cache,state,advance=False):
    root=Path(c['run_dir']);folder=Path(c['feedback_dir']);folder.mkdir(parents=True,exist_ok=True)
    records=folder/'feedback_records.jsonl';queries=folder/'queries.jsonl'
    bank=feature_bank(c,m,cache)
    fb=read_json(folder/'state.json',{})
    if fb and fb.get('backend')!='pathology_probe':raise ValueError('Use a separate feedback directory for the classifier')
    while True:
        qs=read_jsonl(queries);active=validate_records(read_jsonl(records),qs,m)
        if len(active)<c['probe'].get('minimum_inherited_feedback',0):
            raise ValueError('Expected completed parent feedback; check inherit_feedback_dir before continuing')
        if fb.get('trained_feedback_hash')!=digest(active):
            previous=PathologyProbe.load(fb['current']['path']) if fb else None
            probe=fit_probe(c,m,bank,active,cache.identity,previous)
            dev=m.role.eq('train_dev').to_numpy()
            dev_result=metrics(m[dev],probe.predict(m.loc[dev,'tile_id'],subset(bank,np.flatnonzero(dev))))
            atomic_json(Path(probe.metadata['path'])/'dev_metrics.json',dev_result)
            publish(c,m,bank,probe,state,active)
            current={k:v for k,v in probe.metadata.items() if k!='contract'}
            if not fb:fb={'backend':'pathology_probe','anchor':current,'updates':[]}
            fb.update(current=current,trained_feedback_hash=digest(active))
            fb['updates'].append({'model_version':probe.version,'n_feedback':len(active),
                                  'seconds':probe.metadata['seconds'],'dev_balanced_accuracy':dev_result['balanced_accuracy']})
            atomic_json(folder/'effective_feedback.json',active)
            atomic_json(folder/'state.json',fb)
        else:
            probe=PathologyProbe.load(fb['current']['path'])
        if not advance:return {'status':'cpu_checked','model_version':probe.version,'n_feedback':len(active)}
        node=next((n for n in c['feedback']['nodes'] if n>len(active)),None)
        if node is None:return {'status':'cpu_checked','model_version':probe.version,'n_feedback':len(active),'queue':'complete'}
        outstanding=[q for q in qs if q['tile_id'] not in active]
        if not outstanding:
            mask=m.role.eq('pool')&~m.tile_id.isin(active);idx=np.flatnonzero(mask)
            pred=probe.predict(m.loc[mask,'tile_id'],subset(bank,idx))
            outstanding=diverse_queries(pred,subset(bank,idx),node-len(active),probe.version,
                c['probe']['hard_pair_query_fraction'],c['probe']['diversity_weight'])
            if not outstanding:return {'status':'pool_exhausted','n_feedback':len(active)}
            write_jsonl(queries,qs+outstanding) # selection persisted before any label lookup
        if c['feedback']['source']=='expert_label':
            return {'status':'awaiting_feedback','outstanding':len(outstanding),'model_version':probe.version}
        labels=m.set_index('tile_id').label
        for q in outstanding:submit(records,q,labels[q['tile_id']],'simulated_dataset_label')


def explain_probe(c,m,cache,state):
    """Only the final classifier gets explanations. Reuse parent generations when evidence is unchanged."""
    from .model import Student
    cp=state['checkpoints'][c['probe'].get('explanation_checkpoint','feedback')]
    guide=RegionGuide.load(cache);classifier=state['classifier'];name=classifier['result_stage']
    folder=Path(state['results'][name]['path']);rrpath=folder/f'vlm_rationales_{name}.jsonl'
    contract={'classifier':classifier['model_version'],'qwen':cp['model_version'],'guide':guide.version}
    old=read_json(folder/'explanation_contract.json')
    if old and old!=contract:raise ValueError('Explanation components changed; choose a new run')
    atomic_json(folder/'explanation_contract.json',contract)
    rows=m[m.role=='eval'];preds=pd.read_csv(folder/f'preds_vlm_{name}.csv').set_index('tile_id')
    done={r['tile_id']:r for r in read_jsonl(rrpath)}
    history={r['tile_id']:r for r in read_jsonl(Path(c['run_dir'])/'history/qwen_rationales.jsonl')}
    history_meta=read_json(Path(c['run_dir'])/'history/reference.json',{})
    student=None;pending=[];size=c['model'].get('generate_batch',8)
    def record(row,p,rr):
        rr.update(model_version=classifier['model_version'],classification_backend='pathology_probe',
                  classifier_version=classifier['model_version'],explanation_model_version=cp['model_version'])
        append_jsonl(rrpath,rr);done[row.tile_id]=rr
    def flush():
        nonlocal student
        if not pending:return
        if student is None:student=Student(c,cp['path'])
        texts=student.generate_many([r.local_path for r,p,e in pending],[guide.prompt(e) for r,p,e in pending],
                                    predictions=[p['pred'] for r,p,e in pending])
        for (row,p,e),text in zip(pending,texts):
            record(row,p,{**guide.finish(row,p['pred'],p['top2'],text),'reused_parent_explanation':False})
        pending.clear()
    try:
        for row in rows.itertuples():
            if row.tile_id in done:continue
            p=preds.loc[row.tile_id];e=guide.retrieve(row,p['pred'],p['top2']);prior=history.get(row.tile_id)
            if (prior and history_meta.get('model_version')==cp['model_version'] and prior['pred']==p['pred']
                and prior.get('evidence',{}).get('rival')==p['top2'] and prior.get('evidence',{}).get('guide_version')==guide.version):
                record(row,p,{**prior,'reused_parent_explanation':True})
            elif not e['accepted']:
                record(row,p,{**guide.finish(row,p['pred'],p['top2']),'reused_parent_explanation':False})
            else:
                pending.append((row,p,e))
                if len(pending)>=size:flush()
        flush()
    finally:
        if student is not None:
            del student;gc.collect();__import__('torch').cuda.empty_cache()
    active=validate_records(read_jsonl(Path(c['feedback_dir'])/'feedback_records.jsonl'),
                            read_jsonl(Path(c['feedback_dir'])/'queries.jsonl'),m)
    probe=PathologyProbe.load(classifier['path']);bank=feature_bank(c,m,cache)
    publish(c,m,bank,probe,state,active)
    summary={'status':'gpu_verified' if any(r.get('reused_parent_explanation') is False and r.get('student_rationale') is not None for r in done.values()) else 'cpu_checked',
             'n_explanations':len(done),'n_reused':sum(r.get('reused_parent_explanation',False) for r in done.values()),
             'classifier_version':classifier['model_version'],'qwen_version':cp['model_version']}
    atomic_json(folder/'explanation_summary.json',summary)
    return summary

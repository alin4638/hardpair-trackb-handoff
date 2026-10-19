"""Continue an existing GPU run without copying weights or recomputing teacher caches."""
from pathlib import Path
import shutil
from .common import read_json,atomic_json,read_jsonl,write_jsonl,file_hash

def restored_path(stored,source):
    path=Path(stored)
    if path.exists():return path
    parts=str(stored).replace('\\','/').split('/')
    if 'outputs' in parts:
        candidate=source.parent.joinpath(*parts[parts.index('outputs')+1:])
        if candidate.exists():return candidate
    return source.joinpath(*parts[parts.index('checkpoints'):]) if 'checkpoints' in parts else path


def inherit(c,state):
    if not c.get('inherit_run') or state.get('inherited_from'): return
    source=Path(c['inherit_run']);root=Path(c['run_dir'])
    if source.resolve()==root.resolve(): raise ValueError('Continuation requires a new run_dir')
    parent=read_json(source/'run_manifest.json')
    if not parent: raise FileNotFoundError(f'Missing parent run manifest: {source}')
    if parent.get('manifest_version') and parent['manifest_version']!=read_json(Path(c['manifest']).with_suffix('.meta.json'))['manifest_version']:
        raise ValueError('Parent and continuation manifests differ')
    if parent['descriptor_sha256']!=file_hash(c['descriptors']): raise ValueError('Descriptors differ from parent cache')
    if parent['effective_config']['model']['name']!=c['model']['name']: raise ValueError('Base model mismatch')
    for key,cp in parent['checkpoints'].items():
        # Old round keys must not mark a new R3 cycle as already trained.
        if key not in ('r1','r2','r3','feedback'): continue
        path=restored_path(cp['path'],source)
        if not (path/'checkpoint.json').exists(): raise FileNotFoundError(f'Checkpoint missing: {path}')
        state['checkpoints'][key]={**cp,'path':str(path.resolve())}
    if 'r2' in state['checkpoints']:
        state['checkpoints']['r2_parent']=dict(state['checkpoints']['r2'])
    for name in ('threshold.json','region_guide.json','hard_train.csv','model_revision.json'):
        if (source/name).exists(): shutil.copy2(source/name,root/name)
    targets=read_jsonl(source/'targets/r2.jsonl')
    for t in targets: t['local_path']=str(Path(c['data_root'])/t['path'])
    write_jsonl(root/'targets/r2.jsonl',targets)
    queue=Path(c['feedback_dir']);queue.mkdir(parents=True,exist_ok=True)
    old_queue=Path(c.get('inherit_feedback_dir') or parent['effective_config']['feedback_dir'])
    if old_queue.resolve()==queue.resolve(): raise ValueError('Continuation requires a separate feedback_dir')
    # Optional final continuation carries actual labels, preserving selection and source provenance.
    queries=read_jsonl(old_queue/'queries.jsonl') if c.get('inherit_queries',True) else []
    if not queries and c.get('inherit_queries',True):
        queries=read_jsonl(source/'return_bundle/feedback/queries.jsonl')
    write_jsonl(queue/'queries.jsonl',queries)
    if c.get('inherit_feedback',False):
        from .feedback import validate_records
        from .data import load_manifest
        records=read_jsonl(old_queue/'feedback_records.jsonl')
        active=validate_records(records,queries,load_manifest(False,c))
        if len(active)<c.get('probe',{}).get('minimum_inherited_feedback',0):
            raise ValueError('Parent feedback records missing; check inherit_feedback_dir')
        write_jsonl(queue/'feedback_records.jsonl',records)
        state['inherited_feedback_count']=len(active)
    if c.get('classification_backend')=='pathology_probe':
        cp=state['checkpoints'][c['probe'].get('explanation_checkpoint','feedback')]
        entries=[(name,v) for name,v in parent.get('results',{}).items() if v['model_version']==cp['model_version']]
        if entries:
            name,entry=entries[-1];old=restored_path(entry['path'],source)
            history=root/'history';history.mkdir(parents=True,exist_ok=True)
            for oldname,newname in ((f'vlm_rationales_{name}.jsonl','qwen_rationales.jsonl'),
                                     (f'preds_vlm_{name}.csv','qwen_predictions.csv'),('metrics.json','qwen_metrics.json')):
                if (old/oldname).exists():shutil.copy2(old/oldname,history/newname)
            atomic_json(history/'reference.json',{'model_version':cp['model_version'],'stage':name,'source_run':str(source)})
    state['inherited_from']={'run_dir':str(source),'code_head':parent.get('code_head'),
                             'selected_queries':len(queries),'cache_dir':c.get('cache_dir')}
    atomic_json(root/'inheritance.json',state['inherited_from'])

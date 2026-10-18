"""Continue an existing GPU run without copying weights or recomputing teacher caches."""
from pathlib import Path
import shutil
from .common import read_json,atomic_json,read_jsonl,write_jsonl,file_hash


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
        path=Path(cp['path'])
        if not path.exists():
            parts=cp['path'].replace('\\','/').split('/')
            path=source.joinpath(*parts[parts.index('checkpoints'):])
        if not (path/'checkpoint.json').exists(): raise FileNotFoundError(f'Checkpoint missing: {path}')
        state['checkpoints'][key]={**cp,'path':str(path.resolve())}
    if 'r2' in state['checkpoints']:
        state['checkpoints']['r2_parent']=dict(state['checkpoints']['r2'])
    for name in ('threshold.json','hard_train.csv','model_revision.json'):
        if (source/name).exists(): shutil.copy2(source/name,root/name)
    targets=read_jsonl(source/'targets/r2.jsonl')
    for t in targets: t['local_path']=str(Path(c['data_root'])/t['path'])
    write_jsonl(root/'targets/r2.jsonl',targets)
    queue=Path(c['feedback_dir']);queue.mkdir(parents=True,exist_ok=True)
    old_queue=Path(parent['effective_config']['feedback_dir'])
    if old_queue.resolve()==queue.resolve(): raise ValueError('Continuation requires a separate feedback_dir')
    # Carry the already-selected expert queries, but no labels or old training state.
    queries=read_jsonl(old_queue/'queries.jsonl') if c.get('inherit_queries',True) else []
    if not queries and c.get('inherit_queries',True):
        queries=read_jsonl(source/'return_bundle/feedback/queries.jsonl')
    write_jsonl(queue/'queries.jsonl',queries)
    state['inherited_from']={'run_dir':str(source),'code_head':parent.get('code_head'),
                             'selected_queries':len(queries),'cache_dir':c.get('cache_dir')}
    atomic_json(root/'inheritance.json',state['inherited_from'])

"""Exercise real CPU classifier fitting, cumulative feedback, and reuse with tiny fixtures."""
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pandas as pd
import pytest
from tb.common import ROOT,load_config,read_json,read_jsonl,atomic_json,write_jsonl,file_hash
from tb.data import CLASSES,load_manifest
from tb.feedback import select,submit
from tb.probe import fit_probe,PathologyProbe,diverse_queries,subset
from tb.probe_pipeline import run_probe,explain_probe
from tb.continuation import inherit
from tb.region_guide import fit_guide
from tb.grounding import render


def fixture(tmp_path):
    c=load_config(ROOT/'configs/final_4080.yaml');c.update(run_dir=str(tmp_path/'run'),feedback_dir=str(tmp_path/'feedback'))
    c['probe'].update(minimum_inherited_feedback=0,max_iter=500)
    c['feedback']['nodes']=[2,4]
    Path(c['run_dir']).mkdir();rng=np.random.default_rng(2);rows=[];x=[]
    for role,n in [('train_fit',6),('train_dev',1),('pool',3),('eval',1)]:
        for j,cl in enumerate(CLASSES):
            for i in range(n):
                tid=f'{role}_{cl}_{i}';rows.append({'tile_id':tid,'role':role,'label':cl,'path':tid+'.png','local_path':tid+'.png'})
                x.append(np.eye(9)[j]+rng.normal(0,.04,9))
    m=pd.DataFrame(rows);m.attrs['manifest_version']='cpu_fixture'
    bank={'conch':np.asarray(x,dtype=np.float32),'musk':np.asarray(x,dtype=np.float32)*.9}
    import yaml
    class Cache:
        identity={'fixture':True}
        descriptors=yaml.safe_load((ROOT/'configs/descriptors_crc.yaml').read_text())['descriptors']
        def read(self,name,row,kind):
            if kind=='features':return bank[name][m.index[m.tile_id==row.tile_id][0]]
            s=np.zeros((9,len(self.descriptors)))
            s[4,[i for i,d in enumerate(self.descriptors) if d['label']==row.label]]=1
            return s
    cache=Cache();cache.c=c
    return c,m,bank,cache,{'checkpoints':{},'stages':{}}


def test_real_fit_roundtrip_and_feedback_revision(tmp_path):
    c,m,bank,cache,state=fixture(tmp_path)
    baseline=fit_probe(c,m,bank,{},cache.identity)
    predicted=baseline.predict(m.tile_id,bank)
    assert (predicted.pred==m.label).mean()>.95
    loaded=PathologyProbe.load(baseline.metadata['path'])
    np.testing.assert_allclose(loaded.probabilities(bank),baseline.probabilities(bank))
    tid=m[m.role=='pool'].iloc[0].tile_id
    active={tid:{'corrected_label':'STR','feedback_source':'expert_label'}}
    updated=fit_probe(c,m,bank,active,cache.identity,baseline)
    i=m.index[m.tile_id==tid][0]
    assert updated.probabilities(bank)[i,CLASSES.index('STR')]>baseline.probabilities(bank)[i,CLASSES.index('STR')]
    assert updated.version!=baseline.version
    assert updated.metadata['contract']['feedback'][tid]['corrected_label']=='STR'
    assert not any(t.startswith(('train_dev','eval')) for t in updated.metadata['contract']['train_fit_ids'])


def test_queries_are_unique_label_blind_and_focus_both_directions():
    n=40;preds=pd.DataFrame({'tile_id':[f't{i}' for i in range(n)],'pred':['MUS','STR']*20,
        'top2':['STR','MUS']*20,'margin':np.linspace(.01,.8,n),'encoder_disagreement':[True]*n})
    bank={'conch':np.eye(n),'musk':np.eye(n)}
    qs=diverse_queries(preds,bank,8,'probe-version',hard_fraction=1)
    assert len(qs)==len({q['tile_id'] for q in qs})==8
    assert sum(q['original_prediction']=='STR' for q in qs)==4
    with pytest.raises(ValueError):diverse_queries(preds.assign(label='STR'),bank,8,'bad')


def test_cpu_feedback_runs_and_resumes_with_latest_labels(tmp_path):
    c,m,bank,cache,state=fixture(tmp_path)
    run_probe(c,m,cache,state)
    result=run_probe(c,m,cache,state,advance=True)
    assert result['n_feedback']==4
    folder=Path(c['feedback_dir']);fb=read_json(folder/'state.json')
    assert [x['n_feedback'] for x in fb['updates']]==[0,2,4]
    before=list(fb['updates']);run_probe(c,m,cache,state,advance=True)
    assert read_json(folder/'state.json')['updates']==before
    qs=read_jsonl(folder/'queries.jsonl');records=read_jsonl(folder/'feedback_records.jsonl')
    assert len(qs)==len(records)==4
    assert all(r['timestamp']>=next(q['selected_at'] for q in qs if q['tile_id']==r['tile_id']) for r in records)
    submit(folder/'feedback_records.jsonl',qs[0],'STR','expert_label')
    run_probe(c,m,cache,state,advance=True)
    trained=read_json(folder/'effective_feedback.json')
    assert trained[qs[0]['tile_id']]['corrected_label']=='STR'
    assert len(read_json(folder/'state.json')['updates'])==4
    # Labels or CSV corrections are applied by refitting the real classifier, not merely changing the displayed prediction.
    model=PathologyProbe.load(state['classifier']['path'])
    assert model.metadata['contract']['feedback'][qs[0]['tile_id']]['corrected_label']=='STR'


def test_final_explanations_reuse_matching_history_only(tmp_path,monkeypatch):
    c,m,bank,cache,state=fixture(tmp_path);run_probe(c,m,cache,state)
    guide=fit_guide(m[m.role=='train_fit'],m[m.role=='train_dev'],cache)
    state['checkpoints']['feedback']={'path':'fixture-qwen','model_version':'qwen200'}
    stage=state['classifier']['result_stage'];folder=Path(state['results'][stage]['path'])
    pred=pd.read_csv(folder/f'preds_vlm_{stage}.csv').set_index('tile_id')
    rows=list(m[m.role=='eval'].itertuples());row=rows[0];p=pred.loc[row.tile_id]
    e=guide.retrieve(row,p['pred'],p['top2']);prior=guide.finish(row,p['pred'],p['top2'],render(e))
    history=Path(c['run_dir'])/'history'
    atomic_json(history/'reference.json',{'model_version':'qwen200'});write_jsonl(history/'qwen_rationales.jsonl',[prior])
    calls=[];lookup={r.local_path:r for r in rows}
    class Student:
        def __init__(self,*a):pass
        def generate_many(self,paths,prompts,predictions):
            calls.extend(paths)
            return [render(guide.retrieve(lookup[path],pred.loc[lookup[path].tile_id,'pred'],pred.loc[lookup[path].tile_id,'top2'])) for path in paths]
    monkeypatch.setattr('tb.model.Student',Student)
    result=explain_probe(c,m,cache,state)
    assert result['n_reused']==1 and row.local_path not in calls
    rr=read_jsonl(folder/f'vlm_rationales_{stage}.jsonl')
    assert len(rr)==9 and all(r['explanation_model_version']=='qwen200' for r in rr)
    before=len(calls);explain_probe(c,m,cache,state);assert len(calls)==before
    assert all(r['classifier_version']==state['classifier']['model_version'] for r in rr)


def test_inherit_actual_feedback_and_qwen_result(tmp_path):
    c=load_config(ROOT/'configs/final_4080.yaml');source=tmp_path/'followup';root=tmp_path/'final';root.mkdir()
    old_queue=tmp_path/'old_feedback';new_queue=tmp_path/'new_feedback'
    c.update(inherit_run=str(source),inherit_feedback_dir=str(old_queue),run_dir=str(root),feedback_dir=str(new_queue))
    c['probe']['minimum_inherited_feedback']=1
    m=load_manifest(False,c);pool=m[m.role=='pool'].head(1)
    qs=select(pool[['tile_id']].assign(pred='MUS',top2='STR',margin=.1),set(),1,'qwen-before')
    write_jsonl(old_queue/'queries.jsonl',qs);submit(old_queue/'feedback_records.jsonl',qs[0],pool.iloc[0].label,'simulated_dataset_label')
    cp=source/'checkpoints/fb200/step_200';cp.mkdir(parents=True);atomic_json(cp/'checkpoint.json',{})
    results=source/'results/actual';results.mkdir(parents=True)
    write_jsonl(results/'vlm_rationales_fb_200.jsonl',[{'tile_id':'saved-explanation'}])
    atomic_json(source/'region_guide.json',{'version':'saved-guide'})
    atomic_json(source/'run_manifest.json',{'descriptor_sha256':file_hash(c['descriptors']),
        'effective_config':{'model':c['model'],'feedback_dir':str(old_queue)},
        'checkpoints':{'feedback':{'path':str(cp),'model_version':'qwen200'}},
        'results':{'fb_200':{'path':str(results),'model_version':'qwen200'}}})
    state={'checkpoints':{}};inherit(c,state)
    assert state['inherited_feedback_count']==1
    assert read_jsonl(new_queue/'feedback_records.jsonl')==read_jsonl(old_queue/'feedback_records.jsonl')
    assert read_json(root/'region_guide.json')['version']=='saved-guide'
    assert read_jsonl(root/'history/qwen_rationales.jsonl')[0]['tile_id']=='saved-explanation'
    # A second prepare must not erase corrections collected since inheritance.
    submit(new_queue/'feedback_records.jsonl',qs[0],'STR','expert_label');inherit(c,state)
    assert len(read_jsonl(new_queue/'feedback_records.jsonl'))==2

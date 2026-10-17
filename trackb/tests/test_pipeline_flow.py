"""CPU orchestration with explicit fake inference/training. No model-quality claim."""
from pathlib import Path
from types import SimpleNamespace
import sys
import pandas as pd
import numpy as np
from tb import pipeline, model, train_lora
from tb.common import ROOT,load_config,atomic_json,read_json,read_jsonl,write_jsonl
from tb.data import CLASSES
from tb.build_sft import r1
from tb.grounding import evidence,render
from tb.feedback import submit

def setup(tmp_path,monkeypatch):
    c=load_config(ROOT/'configs/smoke_4080.yaml');c['run_dir']=str(tmp_path/'run');c['feedback_dir']=str(tmp_path/'feedback')
    root=Path(c['run_dir']);root.mkdir();atomic_json(root/'threshold.json',{'threshold':.1})
    rows=[]
    for role in ('train_fit','train_dev','pool','eval'):
        for i,cl in enumerate(CLASSES): rows.append({'tile_id':f'{role}_{cl}','label':cl,'path':f'{role}/{cl}.png','fname':f'{role}_{cl}.png','local_path':f'{role}/{cl}.png','role':role,'split':'train' if role.startswith('train_') else role})
    m=pd.DataFrame(rows);m.attrs['manifest_version']='fixture'
    import yaml
    desc=yaml.safe_load((ROOT/'configs/descriptors_crc.yaml').read_text())['descriptors']
    class Cache:
        descriptors=desc
        def read(self,name,row,kind):
            a=np.zeros((9,len(desc)));a[0,next(i for i,d in enumerate(desc) if d['label']==row.label)]=1.;return a
    cache=Cache()
    class FakeStudent:
        def __init__(self,*args,**kw):pass
        def generate(self,path,prompt,sample=False):
            label=Path(path).stem;row=SimpleNamespace(label=label)
            return render(evidence(cache.read('conch',row,'regions'),desc,label))
    def scores(student,rows,*a,**kw):
        return rows[['tile_id']].assign(pred='MUS',top2='STR',margin=np.linspace(.01,.5,len(rows))),[]
    calls=[]
    def train(c,targets,dev,stage,initial,resume,stop_after):
        assert set(dev.role)=={'train_dev'};calls.append((stage,targets))
        return {'path':str(root/stage),'model_version':stage,'dev_balanced_accuracy':.7+len(calls)*.01}
    monkeypatch.setattr(model,'Student',FakeStudent);monkeypatch.setattr(pipeline,'score_rows',scores)
    monkeypatch.setattr(train_lora,'train',train);monkeypatch.setattr(train_lora,'seed_all',lambda seed:None)
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda:None)))
    def evaluate(c,m,state,stage,cp,*a):
        return {'run_id':c['run_id'],'stage':stage,'model_version':cp['model_version'],'n_eval':9,'balanced_accuracy':.5,
                'pair_error_STR_MUS':0.,'pair_error_DEB_MUC':0.,'pair_error_NORM_TUM':0.}
    monkeypatch.setattr(pipeline,'stage_eval',evaluate)
    write_jsonl(root/'targets/r2.jsonl',r1(m[m.role=='train_fit']))
    cp=root/'r2';cp.mkdir();state={'checkpoints':{'r2':{'path':str(cp),'model_version':'r2','dev_balanced_accuracy':.6}},'stages':{}}
    args=SimpleNamespace(resume=True,stop_after=None)
    return c,m,cache,state,args,calls

def test_two_r3_rounds_resume_without_regeneration(tmp_path,monkeypatch):
    c,m,cache,state,args,calls=setup(tmp_path,monkeypatch)
    pipeline.run_r3(c,args,m,m[m.role=='train_fit'],m[m.role=='train_dev'],cache,state)
    assert len(calls)==2 and len(state['r3_rounds'])==2
    for stage,targets in calls:
        assert all(t['tile_id'].startswith('train_fit') for t in targets)
        assert sum(t['supervision_source']=='student_self_generated_teacher_filtered' for t in targets)==9
        assert len(targets)==18 # 50% R2 replay, duplicate generations removed
    pipeline.run_r3(c,args,m,m[m.role=='train_fit'],m[m.role=='train_dev'],cache,state)
    assert len(calls)==2

def test_feedback_queries_then_accumulated_updates_and_revision(tmp_path,monkeypatch):
    c,m,cache,state,args,calls=setup(tmp_path,monkeypatch)
    c['feedback']['source']='simulated_dataset_label';c['feedback']['nodes']=[2,4]
    assert pipeline.run_feedback(c,args,m,m[m.role=='train_dev'],cache,state)=='complete'
    assert len(calls)==2
    assert [len({t['tile_id'] for t in targets if t['tile_id'].startswith('pool')}) for _,targets in calls]==[2,4]
    folder=Path(c['feedback_dir']);qs=read_jsonl(folder/'queries.jsonl');records=read_jsonl(folder/'feedback_records.jsonl')
    assert len(qs)==len(records)==4
    assert all(r['timestamp']>=next(q['selected_at'] for q in qs if q['tile_id']==r['tile_id']) for r in records)
    submit(folder/'feedback_records.jsonl',qs[0],'TUM','expert_label')
    pipeline.run_feedback(c,args,m,m[m.role=='train_dev'],cache,state)
    assert len(calls)==3
    corrected=[t for t in calls[-1][1] if t['tile_id']==qs[0]['tile_id']]
    assert corrected and all(t['answer']=='TUM' for t in corrected)
    assert len(read_json(folder/'state.json')['updates'])==3

def test_expert_flow_does_not_reveal_labels(tmp_path,monkeypatch):
    c,m,cache,state,args,calls=setup(tmp_path,monkeypatch)
    assert pipeline.run_feedback(c,args,m,m[m.role=='train_dev'],cache,state)=='awaiting_feedback'
    folder=Path(c['feedback_dir']);assert not (folder/'feedback_records.jsonl').exists()
    assert len(read_jsonl(folder/'queries.jsonl'))==2 and not calls

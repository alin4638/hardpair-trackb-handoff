"""CPU checks for continuation, retrieved explanations, and real feedback control flow."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import yaml
from tb.common import ROOT,load_config,atomic_json,read_json,read_jsonl,write_jsonl,file_hash
from tb.data import CLASSES,load_manifest
from tb.region_guide import RegionGuide,fit_guide
from tb.grounding import render
from tb.build_sft import r2
from tb.continuation import inherit
from tb.feedback import import_labels,select
from tb.pipeline import feedback_config
from tb.evaluate import score_rows,metrics
from tb.scoring import normalize_scores


class FixtureCache:
    def __init__(self,tmp_path):
        self.c={'run_dir':str(tmp_path)};self.identity={'fixture':True}
        self.descriptors=yaml.safe_load((ROOT/'configs/descriptors_crc.yaml').read_text())['descriptors']
    def read(self,teacher,row,kind):
        s=np.zeros((9,len(self.descriptors)))
        s[4,[i for i,d in enumerate(self.descriptors) if d['label']==row.label]]=1
        return s


def calibrated(tmp_path):
    cache=FixtureCache(tmp_path)
    fit=pd.DataFrame({'tile_id':CLASSES,'label':CLASSES,'role':'train_fit'})
    guide=fit_guide(fit,fit.assign(role='train_dev'),cache)
    return cache,guide


def test_pair_calibration_and_relative_crop(tmp_path):
    cache,guide=calibrated(tmp_path)
    assert len(guide.artifact['pairs'])==72
    assert guide.artifact['n_fit']==guide.artifact['n_dev']==9
    scores=np.zeros((9,27));own=[i for i,d in enumerate(cache.descriptors) if d['label']=='STR']
    other=[i for i,d in enumerate(cache.descriptors) if d['label']=='MUS']
    scores[0,own]=.9;scores[0,other]=.85;scores[4,own]=.8;scores[4,other]=.1
    chosen=guide.choose(scores,scores,'STR','MUS')
    assert chosen['region']=='1,1'  # contrast wins over the highest absolute support at 0,0
    row=SimpleNamespace(tile_id='a',label='STR')
    e=guide.retrieve(row,'STR','MUS')
    good=guide.finish(row,'STR','MUS',render(e))
    assert good['student_grounding_pass'] and good['explanation_source']=='evidence_conditioned_student'
    bad=guide.finish(row,'STR','MUS',render(e).replace('1,1','0,0'))
    assert not bad['student_grounding_pass'] and bad['explanation_source']=='retrieved_teacher_template'
    assert bad['student_rationale']!=bad['rationale']
    absent=guide.finish(row,'MUS','STR','fabricated')
    assert absent['evidence_status']=='insufficient_support' and not absent['cited_regions']
    with pytest.raises(ValueError):fit_guide(pd.DataFrame({'role':['eval']}),pd.DataFrame(),cache)


def test_guided_targets_and_inference(tmp_path):
    from PIL import Image
    cache,guide=calibrated(tmp_path)
    im=tmp_path/'tile.png';Image.new('RGB',(224,224)).save(im)
    rows=pd.DataFrame({'tile_id':['a'],'path':['tile.png'],'local_path':[str(im)],'role':['train_fit'],'label':['STR']})
    targets=r2(rows,cache,0,guide)
    assert len(targets)==1 and targets[0]['evidence']['support'] in targets[0]['prompt']
    assert targets[0]['evidence']['guide_version']==guide.version
    class Student:
        c={'grounding':{'mode':'guided'},'model':{'generate_batch':8}}
        def scores(self,path):
            scores=np.zeros(9);scores[CLASSES.index('STR')]=2;scores[CLASSES.index('MUS')]=1
            return {**normalize_scores(scores),'candidate_token_lengths':[1]*9}
        def generate_many(self,images,prompts,predictions):return ['bad output']*len(images)
    path=tmp_path/'pred.csv';rrpath=tmp_path/'rationale.jsonl'
    preds,rr=score_rows(Student(),rows,path,'model',rrpath,cache)
    assert rr[0]['pred']=='STR' and rr[0]['grounding_pass'] and not rr[0]['student_grounding_pass']
    result=metrics(rows,preds,rr)
    assert result['retrieved_support_rate']==1 and result['student_grounding_rate']==0
    _,again=score_rows(Student(),rows,path,'model',rrpath,cache)
    assert again==rr and len(read_jsonl(rrpath))==1


def test_inherit_moved_checkpoint_and_schedule(tmp_path):
    c=load_config(ROOT/'configs/followup_4080.yaml')
    source=tmp_path/'parent';root=tmp_path/'child';root.mkdir()
    old_queue=tmp_path/'old_feedback';old_queue.mkdir()
    cp=source/'checkpoints/r2/step_0001000';cp.mkdir(parents=True)
    atomic_json(cp/'checkpoint.json',{'model_version':'actual-r2'})
    atomic_json(source/'run_manifest.json',{'descriptor_sha256':file_hash(c['descriptors']),
        'effective_config':{'model':c['model'],'feedback_dir':str(old_queue)},'checkpoints':{
            'r2':{'path':r'C:\old\trackb\outputs\parent\checkpoints\r2\step_0001000',
                  'model_version':'actual-r2','dev_balanced_accuracy':.9}}})
    write_jsonl(source/'targets/r2.jsonl',[{'path':'a.png','local_path':r'C:\old\a.png'}])
    write_jsonl(old_queue/'queries.jsonl',[{'tile_id':'previous-selection'}])
    c.update(inherit_run=str(source),run_dir=str(root),feedback_dir=str(tmp_path/'new_feedback'),inherit_queries=True)
    state={'checkpoints':{}};inherit(c,state)
    assert state['checkpoints']['r2_parent']['path']==str(cp)
    assert state['inherited_from']['selected_queries']==1
    assert read_jsonl(root/'targets/r2.jsonl')[0]['local_path']==str(Path(c['data_root'])/'a.png')
    assert [feedback_config(c,n)['train']['steps'] for n in (50,100,200)]==[100,200,400]
    assert c['train']['steps']==300


def test_bulk_labels_selected_only(tmp_path):
    m=load_manifest(False);pool=m[m.role=='pool'].head(2)
    qs=select(pool[['tile_id']].assign(pred='MUS',top2='STR',margin=.1),set(),2,'r2')
    frame=pool[['tile_id','label']].rename(columns={'label':'corrected_label'})
    path=tmp_path/'labels.jsonl';records=import_labels(frame,qs,m,path,'simulated_dataset_label')
    assert len(records)==2 and all(r['feedback_source']=='simulated_dataset_label' for r in records)
    invalid=pd.concat([frame,m[m.role=='eval'].head(1)[['tile_id','label']].rename(columns={'label':'corrected_label'})])
    with pytest.raises(ValueError,match='selected pool'):import_labels(invalid,qs,m,path,'expert_label')
    assert len(read_jsonl(path))==2

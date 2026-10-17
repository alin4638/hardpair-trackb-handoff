from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pandas as pd
import pytest
from PIL import Image
from tb.common import ROOT,load_config,read_json,read_jsonl,digest
from tb.data import load_manifest,training_rows,assert_no_leak,LeakError,freeze_roles,CLASSES
from tb.crops import regions,crop_grid
from tb.grounding import evidence,render,validate,calibrate
from tb.build_sft import r1,r2,feedback_targets,mix,CLASS_PROMPT
from tb.feedback import select,submit,current,validate_records
from tb.evaluate import metrics,score_rows
from tb.scoring import assistant_labels,normalize_scores
from tb.hard import mine

@pytest.fixture
def manifest(): return load_manifest(False)

@pytest.fixture
def descriptors():
    import yaml
    return yaml.safe_load((ROOT/'configs/descriptors_crc.yaml').read_text())['descriptors']

def test_frozen_manifest_and_roles(manifest,tmp_path):
    old=pd.read_csv(ROOT/'data/manifest_standin.csv',comment='#')
    pd.testing.assert_frame_equal(manifest[old.columns],old)
    assert manifest.role.value_counts().to_dict()=={'train_fit':16200,'eval':3590,'pool':3590,'train_dev':1800}
    assert manifest[manifest.role=='train_dev'].label.value_counts().eq(200).all()
    r=freeze_roles(old,tmp_path/'roles.csv')
    pd.testing.assert_frame_equal(r,manifest[['tile_id','role']])
    r.loc[0,'role']='eval';r.to_csv(tmp_path/'roles.csv',index=False)
    with pytest.raises(ValueError):freeze_roles(old,tmp_path/'roles.csv')

def test_training_leak_guards(manifest,tmp_path):
    fit=training_rows(manifest);assert len(fit)==16200
    for role in ('train_dev','eval','pool'):
        bad=manifest[manifest.role==role].head(1)
        with pytest.raises(LeakError):assert_no_leak(pd.concat([fit,bad]),manifest)
        with pytest.raises(LeakError):assert_no_leak(bad.assign(tile_id='fake'),manifest)
    pool=manifest[manifest.role=='pool'].head(1)
    with pytest.raises(ValueError):training_rows(manifest,pool_ids=pool.tile_id,log_path=tmp_path/'l.csv')
    with pytest.raises(LeakError):training_rows(manifest,extra=manifest[manifest.role=='train_dev'].head(1))

def test_class_correction_and_accumulation(manifest,tmp_path):
    pool=manifest[manifest.role=='pool'].head(3)
    pred=pool[['tile_id']].assign(pred='MUS',margin=[.3,.1,.2])
    query=select(pred,set(),2,'checkpoint1')
    assert [q['tile_id'] for q in query]==pool.tile_id.tolist()[1:]
    path=tmp_path/'feedback.jsonl'
    for q in query:submit(path,q,'STR','simulated_dataset_label')
    submit(path,query[0],'TUM','expert_label')
    submit(path,query[0],'MUS','simulated_dataset_label')
    active=validate_records(read_jsonl(path),query,manifest)
    assert len(active)==2 and active[query[0]['tile_id']]['corrected_label']=='TUM'
    fit=training_rows(manifest).head(2);replay=r1(fit)
    t=feedback_targets(manifest,active,replay,tmp_path/'labels.csv')
    primary=[x for x in t if x['tile_id'] in active]
    assert {x['tile_id']:x['answer'] for x in primary}=={k:r['corrected_label'] for k,r in active.items()}
    assert all(x['prompt']==CLASS_PROMPT for x in primary)
    assert len(t)==4
    forged=read_jsonl(path);forged[0]['query_model_version']='wrong'
    with pytest.raises(ValueError):validate_records(forged,query,manifest)

def test_crop_mapping():
    grid=regions((448,448));assert len(grid)==9
    assert grid[4]['box']==[56,56,168,168] and grid[4]['original_box']==[112,112,336,336]
    assert grid[-1]['box']==[112,112,224,224]
    crops,_=crop_grid(Image.new('RGB',(224,224)))
    assert all(c.size==(112,112) for c in crops)

def test_grounding_rejects_wrong_label_region_and_descriptor(descriptors):
    scores=np.zeros((9,len(descriptors)));i=next(i for i,d in enumerate(descriptors) if d['label']=='STR')
    scores[4,i]=.7
    e=evidence(scores,descriptors,'STR','MUS');text=render(e)
    assert validate(text,'STR',scores,descriptors,.1)['grounding_pass']
    assert not validate(text,'MUS',scores,descriptors,.1)['grounding_pass']
    assert not validate(text.replace('1,1','0,0'),'STR',scores,descriptors,.1)['grounding_pass']
    assert not validate(text.replace(e['support'],'invented feature'),'STR',scores,descriptors,.1)['grounding_pass']
    assert not validate(text,'STR',scores,descriptors,.8)['grounding_pass']

def test_r2_only_fit_and_dev_calibration(manifest,descriptors):
    class FakeCache:
        def __init__(self):self.descriptors=descriptors
        def read(self,teacher,row,kind):
            s=np.zeros((9,len(descriptors)));i=next(i for i,d in enumerate(descriptors) if d['label']==row.label);s[0,i]=1;return s
    cache=FakeCache();targets=r2(manifest[manifest.role=='train_fit'].head(3),cache,.1)
    assert len(targets)==3 and all(t['supervision_source']=='automatic_CONCH_region_evidence' for t in targets)
    with pytest.raises(ValueError):r2(manifest[manifest.role=='eval'].head(1),cache,0)
    c=load_config(ROOT/'configs/laptop_4080.yaml')
    result=calibrate(manifest[manifest.role=='train_dev'].head(9),cache,c)
    assert result['n_dev']==9
    with pytest.raises(ValueError):calibrate(manifest[manifest.role=='eval'].head(9),cache,c)

def test_oof_mining():
    labels=np.repeat(CLASSES,10);rng=np.random.default_rng(3)
    rows=pd.DataFrame({'tile_id':[f't{i}' for i in range(90)],'label':labels,'role':'train_fit'})
    x=np.repeat(np.eye(9),10,axis=0)+rng.normal(0,.01,(90,9))
    hard=mine(rows,x)
    assert len(hard)==90 and hard.tile_id.is_unique
    assert hard.true_rival_margin.gt(0).all() and set(hard.weight)=={1,3}
    with pytest.raises(ValueError):mine(rows.assign(role='eval'),x)

def test_assistant_mask_and_scores():
    assert assistant_labels([1,2,99,3,4,0],[1,2,99],[1,1,1,1,1,0])==[-100,-100,-100,3,4,-100]
    with pytest.raises(ValueError):assistant_labels([1,2],[9],[1,1])
    with pytest.raises(ValueError):assistant_labels([1,2],[1,2],[1,1])
    pred=normalize_scores([-10000,-10001,-10002,-10003,-10004,-10005,-10006,-10007,-10008])
    assert pred['pred']=='ADI' and pred['top2']=='BACK'
    assert abs(sum(pred[f'p_{c}'] for c in CLASSES)-1)<1e-12
    with pytest.raises(ValueError):normalize_scores([np.nan]*9)

def test_metrics_denominators():
    rows=pd.DataFrame({'tile_id':CLASSES,'label':CLASSES})
    preds=rows.rename(columns={'label':'pred'});preds.loc[preds.pred=='STR','pred']='MUS'
    report=metrics(rows,preds)
    assert report['balanced_accuracy']==pytest.approx(8/9)
    assert report['pair_error_STR_MUS']==.5 and report['pair_counts_STR_MUS']=={'errors':1,'denominator':2}
    assert report['format_rate'] is None
    with pytest.raises(ValueError):metrics(rows,preds.head(8))

def test_prediction_resume_and_provenance(tmp_path):
    im=tmp_path/'im.png';Image.new('RGB',(224,224)).save(im)
    rows=pd.DataFrame({'tile_id':['a','b'],'path':['a.png','b.png'],'local_path':[str(im)]*2})
    class FakeStudent:
        calls=0
        def scores(self,path):self.calls+=1;return {**normalize_scores(range(9)),'candidate_token_lengths':[1]*9}
    s=FakeStudent();out=tmp_path/'pred.csv'
    score_rows(s,rows,out,'model1');score_rows(s,rows,out,'model1');assert s.calls==2
    with pytest.raises(ValueError):score_rows(s,rows,out,'model2')
    Image.new('RGB',(224,224),'red').save(im)
    with pytest.raises(ValueError):score_rows(s,rows,out,'model1')

def test_cache_rejects_changed_teacher_and_image(manifest,tmp_path):
    from tb.cache import Cache
    from tb.common import atomic_json,file_hash
    c=load_config(ROOT/'configs/smoke_4080.yaml');c['run_dir']=str(tmp_path)
    cache=Cache(c,manifest);im=tmp_path/'tile.png';Image.new('RGB',(224,224)).save(im)
    row=SimpleNamespace(tile_id='fixture',path='NCT-CRC-HE-100K/ADI/fixture.png',local_path=str(im))
    dest=cache.path('conch',row.tile_id,'regions');dest.parent.mkdir(parents=True)
    atomic_json(cache.root/'conch/revision.json',{'sha':'teacher1'})
    np.savez_compressed(dest,values=np.zeros((9,27)),tile_id=row.tile_id,path=row.path,image_sha256=file_hash(im),model_revision='teacher1')
    assert cache.read('conch',row,'regions').shape==(9,27)
    atomic_json(cache.root/'conch/revision.json',{'sha':'teacher2'})
    with pytest.raises(ValueError,match='revision'):cache.read('conch',row,'regions')
    Image.new('RGB',(224,224),'red').save(im)
    with pytest.raises(ValueError,match='Image changed'):cache.read('conch',row,'regions')

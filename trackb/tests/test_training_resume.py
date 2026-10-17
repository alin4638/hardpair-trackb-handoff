"""Actual CPU autograd/AdamW/RNG recovery using a tiny random model, no pretrained weights."""
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from tb.common import load_config,ROOT,read_json
from tb.data import CLASSES
from tb import train_lora as engine

torch=pytest.importorskip('torch')

class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__();self.visual=torch.nn.Linear(2,2);self.adapter=torch.nn.Linear(2,1);self.dropout=torch.nn.Dropout(.2)
        self.config=SimpleNamespace(image_token_id=999)
    def forward(self,x,labels,**kw):
        y=self.adapter(self.dropout(self.visual(x)))
        return SimpleNamespace(loss=(y-labels).square().mean())
    def save_pretrained(self,path,**kw):
        Path(path).mkdir(parents=True,exist_ok=True);torch.save(self.state_dict(),Path(path)/'weights.pt')

class TinyStudent:
    def __init__(self,c,checkpoint=None,trainable=False):
        self.c=c;self.dtype=torch.bfloat16;self.base_revision='random-fixture';self.visual_names={'visual.weight','visual.bias'}
        self.model=Tiny();self.proc=SimpleNamespace(save_pretrained=lambda p:Path(p).mkdir(parents=True,exist_ok=True))
        if checkpoint:self.model.load_state_dict(torch.load(Path(checkpoint)/'adapter/weights.pt',weights_only=True))
    def inputs(self,path,prompt,answer):
        x=float(path);return {'x':torch.tensor([[x,1.]]),'input_ids':torch.tensor([[999,1,2]]),'labels':torch.tensor([[float(answer)]])}

def test_optimizer_scheduler_rng_visual_exact_resume(tmp_path,monkeypatch):
    monkeypatch.setattr(engine,'Student',TinyStudent)
    monkeypatch.setattr(torch,'autocast',lambda *a,**k:nullcontext())
    for name in ('manual_seed_all','set_rng_state_all','reset_peak_memory_stats','synchronize','empty_cache'):
        monkeypatch.setattr(torch.cuda,name,lambda *a,**k:None)
    monkeypatch.setattr(torch.cuda,'get_rng_state_all',lambda:[])
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:0)
    monkeypatch.setattr(engine,'score_rows',lambda student,rows,*a,**k:(rows[['tile_id','label']].rename(columns={'label':'pred'}),[]))
    c=load_config(ROOT/'configs/smoke_4080.yaml');c['train'].update(steps=6,save_every=3,eval_every=3,grad_accum=2)
    dev=pd.DataFrame({'tile_id':CLASSES,'label':CLASSES})
    targets=[{'tile_id':str(i),'local_path':str(i),'prompt':'fixture','answer':str(i%2),'weight':i+1} for i in range(3)]
    c['run_dir']=str(tmp_path/'full');engine.train(c,targets,dev,'fixture')
    full=tmp_path/'full/checkpoints/fixture/step_0000006'
    c['run_dir']=str(tmp_path/'resumed')
    with pytest.raises(engine.TrainingPaused):engine.train(c,targets,dev,'fixture',stop_after=3)
    latest=tmp_path/'resumed/checkpoints/fixture/latest.json';latest.unlink() # emulate crash before latest-pointer write
    engine.train(c,targets,dev,'fixture',resume=True)
    resumed=tmp_path/'resumed/checkpoints/fixture/step_0000006'
    a=torch.load(full/'adapter/weights.pt',weights_only=True);b=torch.load(resumed/'adapter/weights.pt',weights_only=True)
    assert all(torch.equal(a[k],b[k]) for k in a)
    sa=torch.load(full/'training.pt',weights_only=False);sb=torch.load(resumed/'training.pt',weights_only=False)
    assert sa['step']==sb['step']==6 and sa['scheduler']==sb['scheduler']
    assert sa['python_rng']==sb['python_rng'] and torch.equal(sa['torch_rng'],sb['torch_rng'])
    assert np.array_equal(sa['numpy_rng'][1],sb['numpy_rng'][1])
    assert set(torch.load(resumed/'visual.pt',weights_only=True))=={'visual.weight','visual.bias'}
    # Corrupted state must fail before training or loading the optimizer.
    (resumed/'visual.pt').write_bytes(b'broken')
    with pytest.raises(ValueError,match='Corrupt checkpoint'):engine.train(c,targets,dev,'fixture',resume=True)

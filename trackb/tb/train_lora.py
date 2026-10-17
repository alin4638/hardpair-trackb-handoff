"""Single-GPU training, optimizer-boundary recovery, dev-only checkpoint selection."""
from pathlib import Path
import random
import time
import os
import gc
import numpy as np
from .common import atomic_json,read_json,append_jsonl,digest,file_hash
from .model import Student
from .evaluate import score_rows,metrics


class TrainingPaused(RuntimeError):
    """A requested optimizer-boundary stop; safe to resume unchanged configuration."""

def seed_all(seed):
    import torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)

def save(student,optimizer,scheduler,scaler,step,path,contract):
    import torch
    path=Path(path);tmp=path.with_name(path.name+'.incomplete');tmp.mkdir(parents=True,exist_ok=True)
    student.model.save_pretrained(tmp/'adapter',safe_serialization=True)
    student.proc.save_pretrained(tmp/'processor')
    torch.save({n:p.detach().cpu() for n,p in student.model.named_parameters() if n in student.visual_names},tmp/'visual.pt')
    torch.save({'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),
                'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),
                'cuda_rng':torch.cuda.get_rng_state_all(),'step':step},tmp/'training.pt')
    atomic_json(tmp/'checksums.json',{str(p.relative_to(tmp)):file_hash(p) for p in tmp.rglob('*') if p.is_file() and p.name not in ('checkpoint.json','checksums.json')})
    atomic_json(tmp/'checkpoint.json',{'step':step,'contract':contract,'base_revision':student.base_revision,
                'base_model':student.c['model']['name'],'visual_names':sorted(student.visual_names),
                'model_version':path.name+'-'+digest({'contract':contract,'base_revision':student.base_revision})[:12]})
    if path.exists(): raise FileExistsError(f'Checkpoint already exists: {path}')
    os.replace(tmp,path)

def train(c,targets,dev,stage,initial=None,resume=False,stop_after=None):
    import torch
    run=Path(c['run_dir']);folder=run/'checkpoints'/stage;folder.mkdir(parents=True,exist_ok=True)
    if not targets: raise ValueError('No training targets')
    contract={'stage':stage,'targets':digest([{k:v for k,v in t.items() if k!='local_path'} for t in targets]),
              'model':c['model'],'train':c['train'],'seed':c['seed'],'initial':str(initial),
              'dev_ids':dev.tile_id.tolist()}
    complete=sorted(p for p in folder.glob('step_[0-9]*') if p.is_dir() and not p.name.endswith('.incomplete') and (p/'checkpoint.json').exists())
    latest={'path':str(complete[-1])} if complete else None
    if latest and not resume: raise ValueError('Existing stage; pass --resume or choose a new run_dir')
    cp=Path(latest['path']) if latest else None
    if cp and read_json(cp/'checkpoint.json')['contract']!=contract: raise ValueError('Resume config/target mismatch')
    if cp:
        for relative,expected in read_json(cp/'checksums.json').items():
            if file_hash(cp/relative)!=expected: raise ValueError(f'Corrupt checkpoint component: {relative}')
    seed_all(c['seed']);student=Student(c,cp or initial,trainable=True)
    params=[p for p in student.model.parameters() if p.requires_grad]
    if not params: raise RuntimeError('No trainable parameters')
    optimizer=torch.optim.AdamW(params,lr=c['train']['lr'])
    steps=c['train']['steps'];scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:max(0.,1-s/steps))
    scaler=torch.amp.GradScaler('cuda',enabled=student.dtype==torch.float16)
    start=0
    if cp:
        # Only load checkpoints created by this trusted local run: optimizer/RNG use pickle.
        state=torch.load(cp/'training.pt',map_location='cpu',weights_only=False)
        optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);scaler.load_state_dict(state['scaler'])
        random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng']);start=state['step']
    atomic_json(folder/'trainable.json',{'parameters':{n:list(p.shape) for n,p in student.model.named_parameters() if p.requires_grad},
                'dtype':str(student.dtype),'base_revision':student.base_revision})
    best=read_json(folder/'best.json');weights=[t.get('weight',1) for t in targets]
    def evaluate_checkpoint(path):
        nonlocal best
        meta=read_json(path/'checkpoint.json');version=meta['model_version']
        df,_=score_rows(student,dev,folder/f'dev_{path.name}.csv',version)
        score=metrics(dev,df)['balanced_accuracy']
        if score is None: raise ValueError('Development selection requires all nine classes')
        if best is None or score>best['dev_balanced_accuracy']:
            best={'path':str(path),'model_version':version,'dev_balanced_accuracy':score,'step':meta['step']}
            atomic_json(folder/'best.json',best)
        atomic_json(path/'dev_metrics.json',metrics(dev,df))
    if cp and (start%c['train']['eval_every']==0 or start==steps) and not (cp/'dev_metrics.json').exists(): evaluate_checkpoint(cp)
    for step in range(start+1,steps+1):
        student.model.train();optimizer.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats();t0=time.perf_counter();loss_total=0;visual_tokens=[]
        for _ in range(c['train']['grad_accum']):
            t=random.choices(targets,weights=weights,k=1)[0]
            x=student.inputs(t['local_path'],t['prompt'],t['answer'])
            visual_tokens.append(int((x['input_ids']==student.model.config.image_token_id).sum()))
            with torch.autocast('cuda',dtype=student.dtype): loss=student.model(**x).loss/c['train']['grad_accum']
            if not torch.isfinite(loss): raise RuntimeError('Non-finite loss; latest complete checkpoint is recoverable')
            scaler.scale(loss).backward();loss_total+=float(loss.detach())
        scaler.unscale_(optimizer);norm=torch.nn.utils.clip_grad_norm_(params,1.)
        if not torch.isfinite(norm): raise RuntimeError('Non-finite gradient; latest complete checkpoint is recoverable')
        scaler.step(optimizer);scaler.update();scheduler.step();torch.cuda.synchronize()
        append_jsonl(run/'logs'/f'{stage}_train.jsonl',{'step':step,'loss':loss_total,'seconds':time.perf_counter()-t0,
                     'peak_vram_gib':torch.cuda.max_memory_allocated()/2**30,'visual_tokens':visual_tokens,'lr':scheduler.get_last_lr()[0]})
        if step%c['train']['save_every']==0 or step%c['train']['eval_every']==0 or step==steps or step==stop_after:
            path=folder/f'step_{step:07d}'
            save(student,optimizer,scheduler,scaler,step,path,contract);atomic_json(folder/'latest.json',{'path':str(path)})
            if step%c['train']['eval_every']==0 or step==steps: evaluate_checkpoint(path)
            if step==stop_after and step<steps: raise TrainingPaused(f'Checkpoint saved at {path}; rerun with --resume')
    del student,optimizer,scheduler,scaler,params;gc.collect();torch.cuda.empty_cache()
    if best is None: raise RuntimeError('No dev-evaluated checkpoint')
    return best

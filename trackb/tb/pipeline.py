"""Stage orchestration; GPU work is lazy-imported and never run by --dry-run."""
from pathlib import Path
import gc
import json
import shutil
import subprocess
import platform
import importlib.metadata
from copy import deepcopy
import pandas as pd
from .common import ROOT,atomic_json,read_json,read_jsonl,write_jsonl,append_jsonl,file_hash,digest
from .data import load_manifest,freeze_roles,training_rows,CLASSES
from .cache import Cache
from . import build_sft
from .grounding import calibrate,generation_prompt,validate
from .evaluate import score_rows,metrics,upsert_csv
from .feedback import current,submit,select,validate_records

STAGES=['prepare','cache','evidence','explain','zeroshot','r1','r2','r3','feedback','report']

def feedback_config(c,n):
    result=deepcopy(c)
    schedule=result['feedback'].get('train',{})
    result['train'].update({k:v for k,v in schedule.items() if k not in ('steps_per_label','min_steps','max_steps')})
    if 'steps_per_label' in schedule:
        result['train']['steps']=max(schedule['min_steps'],min(schedule['max_steps'],int(n*schedule['steps_per_label'])))
    return result

def diagnostics(c):
    info={'python':platform.python_version(),'platform':platform.platform(),'packages':{},'data_root':c['data_root'],
          'disk_free_gib':shutil.disk_usage(Path(c['run_dir']).resolve().parent).free/2**30}
    for name in ('torch','torchvision','transformers','peft','accelerate','timm','numpy','pandas','bitsandbytes'):
        try: info['packages'][name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: info['packages'][name]=None
    try:
        import torch
        info['cuda_available']=torch.cuda.is_available();info['cuda_runtime']=torch.version.cuda
        if torch.cuda.is_available():
            p=torch.cuda.get_device_properties(0)
            info.update(gpu_name=p.name,vram_gib=p.total_memory/2**30,bf16=torch.cuda.is_bf16_supported())
    except ImportError: info['cuda_available']=False
    info['encoder_source_commits']={}
    for name in ('CONCH','MUSK'):
        try: info['encoder_source_commits'][name]=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT/'third_party'/name,text=True,stderr=subprocess.DEVNULL).strip()
        except (OSError,subprocess.CalledProcessError): info['encoder_source_commits'][name]=None
    return info

def prepare(c,check_images=True):
    src=Path(c['manifest']);m=pd.read_csv(src)
    if len(m)!=25180 or m.split.value_counts().to_dict()!={'train':18000,'eval':3590,'pool':3590}:
        raise ValueError('B baseline must preserve 18000/3590/3590 rows')
    freeze_roles(m,c['roles']);m=load_manifest(check_images,c)
    return {'counts':m.role.value_counts().to_dict(),'manifest_version':m.attrs['manifest_version'],'images_checked':check_images}

def dev_rows(m,c):
    rows=m[m.role=='train_dev']
    if c['train']['dev_limit']:
        n=max(1,c['train']['dev_limit']//9);rows=rows.groupby('label',group_keys=False).head(n)
    return rows

def fit_rows(m,c):
    extra=None
    if c.get('extra_manifest'):
        extra=pd.read_csv(c['extra_manifest']);extra['fname']=extra.path.map(lambda p:Path(p).name)
        extra['role']='extra_train';extra['local_path']=extra.path.map(lambda p:str(Path(c['data_root'])/p))
        if not extra.tile_id.is_unique or not extra.path.is_unique: raise ValueError('Duplicate extra rows')
        if any(Path(p).is_absolute() or '..' in Path(p).parts for p in extra.path): raise ValueError('Unsafe extra paths')
        if any(not Path(p).exists() for p in extra.local_path): raise FileNotFoundError('Missing extra image')
    rows=training_rows(m,extra=extra)
    limit=c['train']['sample_limit']
    return rows.groupby('label',group_keys=False).head(max(1,limit//9)) if limit else rows

def release(student):
    import torch
    del student;gc.collect();torch.cuda.empty_cache()

def checkpoint(state,key):
    if key not in state['checkpoints']: raise FileNotFoundError(f'Run stage {key} first')
    cp=state['checkpoints'][key]
    if not Path(cp['path']).exists(): raise FileNotFoundError(f'Checkpoint missing: {cp["path"]}')
    return cp

def build_log(c,stage,rows,**extra):
    append_jsonl(Path(c['run_dir'])/'target_build_log.jsonl',{'stage':stage,'training_targets':len(rows),
                  'unique_tiles':len({t['tile_id'] for t in rows}), 'targets_sha256':digest(rows),**extra})

def save_state(c,state): atomic_json(Path(c['run_dir'])/'run_manifest.json',state)

def stage_eval(c,m,state,stage,cp=None,grounded=False,cache=None,threshold=0):
    from .model import Student
    student=Student(c,cp['path'] if cp else None)
    version=cp['model_version'] if cp else 'base-'+c['model']['name'].replace('/','_')+'-'+student.base_revision
    result_dir=Path(c['run_dir'])/'results'/'by_model'/digest({'stage':stage,'version':version})[:20]
    result_dir.mkdir(parents=True,exist_ok=True)
    rows=m[m.role=='eval']
    if c['train']['sample_limit']: rows=rows.groupby('label',group_keys=False).head(max(1,c['train']['sample_limit']//9))
    pred_path=result_dir/f'preds_vlm_{stage}.csv';rrpath=result_dir/f'vlm_rationales_{stage}.jsonl' if grounded else None
    try: preds,rr=score_rows(student,rows,pred_path,version,rrpath,cache,threshold)
    finally: del student;gc.collect();__import__('torch').cuda.empty_cache()
    report=metrics(rows,preds,rr);report.update(run_id=c['run_id'],stage=stage,model_version=version,manifest_version=m.attrs['manifest_version'])
    atomic_json(result_dir/'metrics.json',report)
    pd.DataFrame([{'class':cl,'recall':report['recall'][cl],'n':report['class_counts'].get(cl,0)} for cl in CLASSES]).to_csv(result_dir/'per_class_recall.csv',index=False)
    pd.DataFrame.from_dict(report['confusion_matrix'],orient='index').to_csv(result_dir/'confusion_matrix.csv')
    cols=['run_id','stage','model_version','manifest_version','n_eval','balanced_accuracy',
          'pair_error_STR_MUS','pair_error_DEB_MUC','pair_error_NORM_TUM','format_rate','grounding_pass_rate','n_rationale',
          'retrieved_support_rate','student_grounding_rate']
    upsert_csv(Path(c['run_dir'])/'results/vlm_summary.csv',{k:report[k] for k in cols},['run_id','stage','model_version'])
    # R1/zero-shot have an empty rationale artifact with n_rationale=0.
    if rrpath is None:
        rrpath=result_dir/f'vlm_rationales_{stage}.jsonl';write_jsonl(rrpath,[])
    # Public names point only to fully written stage outputs.
    for p in [pred_path,rrpath]: shutil.copy2(p,Path(c['run_dir'])/'results'/p.name)
    state.setdefault('results',{})[stage]={'path':str(result_dir),'model_version':version}
    save_state(c,state)
    return report

def run(c,args):
    root=Path(c['run_dir']);root.mkdir(parents=True,exist_ok=True)
    state=read_json(root/'run_manifest.json',{'run_id':c['run_id'],'checkpoints':{},'stages':{}})
    config_hash=digest(c)
    if state.get('config_hash') and state['config_hash']!=config_hash: raise ValueError('Effective config changed; use a new run_dir')
    if state.get('manifest_sha256') and state['manifest_sha256']!=file_hash(c['manifest']): raise ValueError('Manifest changed during run')
    state.update(config_hash=config_hash,effective_config=c,manifest_sha256=file_hash(c['manifest']))
    atomic_json(root/'effective_config.json',c)
    try: head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    except (OSError,subprocess.CalledProcessError): head='unavailable'
    code_hash=digest({str(p.relative_to(ROOT)):file_hash(p) for folder in ('tb','scripts','configs') for p in sorted((ROOT/folder).glob('**/*')) if p.is_file() and p.suffix in ('.py','.yaml')})
    if state.get('code_hash') and state['code_hash']!=code_hash: raise ValueError('Code changed during run; use a new run_dir or explicitly migrate artifacts')
    state.update(code_head=head,code_hash=code_hash,descriptor_sha256=file_hash(c['descriptors']))
    stage=args.stage
    state['stages'][stage]={'status':'running','validation_status':'gpu_pending' if stage not in ('prepare','evidence','report') else 'code_ready'};save_state(c,state)
    if stage=='prepare':
        evidence=prepare(c,not args.skip_images);env=diagnostics(c);atomic_json(root/'environment.json',env)
        state['manifest_version']=evidence['manifest_version']
        from .continuation import inherit
        inherit(c,state)
        state['stages'][stage]={'status':'cpu_checked','evidence':evidence,'gpu_status':'gpu_pending'}
        save_state(c,state);return state['stages'][stage]
    if stage=='report': return report(c,state)
    if stage=='evidence':
        from .region_guide import fit_guide
        m=load_manifest(True,c);cache=Cache(c,m);fit=fit_rows(m,c)
        guide=fit_guide(fit[fit.role=='train_fit'],dev_rows(m,c),cache,
                        c.get('grounding',{}).get('minimum_descriptor_auc',.55),
                        c.get('grounding',{}).get('musk_weight',.15))
        targets=build_sft.r2(fit,cache,0,guide)
        if not targets: raise ValueError('No supported regional targets; inspect region_guide.json')
        write_jsonl(root/'targets/r2.jsonl',targets)
        build_log(c,'evidence',targets,guide_version=guide.version,input_tiles=len(fit))
        state['stages'][stage]={'status':'cpu_checked','guide_version':guide.version,'targets':len(targets)}
        save_state(c,state);return state['stages'][stage]
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('This stage requires the teammate CUDA machine; use --dry-run in CPU development')
    m=load_manifest(True,c);dev=dev_rows(m,c);fit=fit_rows(m,c)
    cache=Cache(c,m)
    guide=None
    if c.get('grounding',{}).get('mode')=='guided' and stage in ('r2','r3','feedback','explain'):
        from .region_guide import RegionGuide
        guide=RegionGuide.load(cache)
    if stage=='cache':
        if args.cache_kind=='features':
            base=fit[fit.role=='train_fit'];cache.build(base,'features')
            from .hard import mine
            hard=mine(base,cache.features(base),c['cache']['hard_fraction']);hard.to_csv(root/'hard_train.csv',index=False)
        else:
            # Full dev and evaluation images are inference-only; no label enters encoder inputs.
            rows=pd.concat([fit,dev,m[m.role.isin(['pool','eval'])]])
            if c['train']['sample_limit']:
                rows=pd.concat([fit,dev,m[m.role.isin(['pool','eval'])].groupby(['role','label'],group_keys=False).head(max(1,c['train']['sample_limit']//9))])
            cache.build(rows,'regions')
            atomic_json(root/'threshold.json',calibrate(dev,cache,c))
            export_examples(fit,cache,root/'examples')
        state['stages'][f'cache_{args.cache_kind}']={'status':'gpu_verified','evidence':str(cache.root)}
    elif stage=='zeroshot': stage_eval(c,m,state,'zeroshot')
    elif stage=='explain':
        best=max((v for k,v in state['checkpoints'].items() if k in ('r1','r2','r2_parent','r3','feedback')),
                 key=lambda v:v['dev_balanced_accuracy'])
        stage_eval(c,m,state,'guided_explain',best,True,cache)
    elif stage in ('r1','r2'):
        from .train_lora import train
        if stage=='r1':
            hard=pd.read_csv(root/'hard_train.csv');targets=build_sft.r1(fit,hard);initial=None;name='r1_sft'
        else:
            threshold=read_json(root/'threshold.json')
            if threshold is None: raise FileNotFoundError('Run cache --cache-kind regions first')
            targets=build_sft.r2(fit,cache,threshold['threshold'],guide);name='r2_grounded'
            initial=checkpoint(state,c.get('r2',{}).get('initial_checkpoint','r1'))['path']
        write_jsonl(root/'targets'/f'{stage}.jsonl',targets);build_log(c,stage,targets,input_tiles=len(fit),accepted_targets=len(targets))
        train_targets=targets
        fraction=(c.get('r2') or {}).get('class_replay_fraction',0)
        if stage=='r2' and fraction:
            # Keep the class-code prompt that scoring uses alive during grounded training; r2.jsonl stays
            # grounded-only, so R3/feedback replay semantics are unchanged.
            train_targets=build_sft.mix(targets,build_sft.r1(fit,pd.read_csv(root/'hard_train.csv')),fraction,c['seed'])
            write_jsonl(root/'targets/r2_train_mix.jsonl',train_targets)
            build_log(c,'r2_train_mix',train_targets,grounded_targets=len(targets),class_replay_targets=len(train_targets)-len(targets),class_replay_fraction=fraction)
        best=train(c,train_targets,dev,name,initial,args.resume,args.stop_after);state['checkpoints'][stage]=best;save_state(c,state)
        stage_eval(c,m,state,name,best,stage=='r2',cache,read_json(root/'threshold.json',{}).get('threshold',0))
    elif stage=='r3': run_r3(c,args,m,fit[fit.role=='train_fit'],dev,cache,state)
    elif stage=='feedback':
        result=run_feedback(c,args,m,dev,cache,state)
        if result=='awaiting_feedback': return {'status':result,'queries':str(Path(c['feedback_dir'])/'queries.jsonl')}
    state['stages'][stage]={'status':'gpu_verified','evidence':str(root/'logs'), 'note':'GPU execution completed; does not imply efficacy'}
    save_state(c,state);return state['stages'][stage]

def run_r3(c,args,m,fit,dev,cache,state):
    from .model import Student
    from .train_lora import train,seed_all
    root=Path(c['run_dir']);threshold=read_json(root/'threshold.json')['threshold']
    guide=None
    if c.get('grounding',{}).get('mode')=='guided':
        from .region_guide import RegionGuide
        guide=RegionGuide.load(cache)
    parent=getattr(args,'r3_from','r2');base=checkpoint(state,parent)
    cycle='r3' if parent=='r2' else 'r3_after_'+digest(base['model_version'])[:12]
    fit_ids=set(m.loc[m.role=='train_fit','tile_id'])
    replay=[t for t in read_jsonl(root/'targets/r2.jsonl') if t['tile_id'] in fit_ids]
    if not replay: raise FileNotFoundError('R2 targets required')
    selected=state.get(cycle+'_selected',base)
    for iteration in range(1,c['r3']['rounds']+1):
        key=f'{cycle}_round_{iteration}'
        if key in state['checkpoints']: continue
        seed_all(c['seed']+iteration)
        work=root/'r3'/key;work.mkdir(parents=True,exist_ok=True)
        stage_contract={'initial':selected['model_version'],'iteration':iteration}
        old=read_json(work/'contract.json')
        if old and old!=stage_contract: raise ValueError('R3 generation parent changed')
        atomic_json(work/'contract.json',stage_contract)
        student=Student(c,selected['path'])
        try:
            # Rank current mistakes and low-margin examples using current student predictions.
            scores,_=score_rows(student,fit,work/'fit_predictions.csv',selected['model_version'])
            ranked=fit.merge(scores[['tile_id','pred','top2','margin']],on='tile_id')
            ranked['wrong']=ranked.label!=ranked.pred
            ranked=ranked.sort_values(['wrong','margin','tile_id'],ascending=[False,True,True]).head(c['r3']['max_tiles'])
            accepted=[];generated=0
            for row in ranked.itertuples():
                e=guide.retrieve(row,row.label,row.top2 if row.pred==row.label else row.pred) if guide else None
                if e is not None and not e['accepted']: continue
                prompt=guide.prompt(e) if guide else generation_prompt(cache.descriptors)
                p=work/(digest(row.tile_id)+'.json');stored=read_json(p)
                if stored is None:
                    # Seed per tile makes crash recovery independent of completed generation count.
                    seed_all(c['seed']+iteration+int(digest(row.tile_id)[:8],16))
                    answers=student.generate_many([row.local_path],[prompt],True,n=c['r3']['candidates'])
                    stored={'tile_id':row.tile_id,'answers':answers};atomic_json(p,stored)
                generated+=len(stored['answers']);kept=0
                for text in dict.fromkeys(stored['answers']):
                    if guide:
                        check={'grounding_pass':guide.finish(row,row.label,e['rival'],text)['student_grounding_pass']}
                    else:
                        check=validate(text,row.label,cache.read('conch',row,'regions'),cache.descriptors,threshold,cache.read('musk',row,'regions'))
                    if check['grounding_pass']:
                        accepted.append(build_sft.target(row,prompt,text,'student_self_generated_teacher_filtered'))
                        kept+=1
                    if kept>=c['r3']['max_per_tile']: break
        finally: del student;gc.collect();__import__('torch').cuda.empty_cache()
        if not accepted:
            build_log(c,key,accepted,generated=generated,accepted=0,parent=selected['model_version'])
            state.setdefault('r3_rounds',{})[key]={'status':'no_accepted_candidates','generated':generated}
            save_state(c,state);raise RuntimeError('R3 accepted zero candidates; inspect generation records and improve supervision before retrying')
        targets=build_sft.mix(accepted,replay,c['r3']['replay_fraction'],c['seed']+iteration)
        build_log(c,key,targets,generated=generated,accepted=len(accepted),replay_targets=len(targets)-len(accepted),parent=selected['model_version'])
        write_jsonl(root/'targets'/f'{key}.jsonl',targets)
        best=train(c,targets,dev,key,selected['path'],args.resume,args.stop_after)
        state['checkpoints'][key]=best
        if best['dev_balanced_accuracy']>selected['dev_balanced_accuracy']: selected=best
        state[cycle+'_selected']=selected;state['checkpoints']['r3']=selected
        state.setdefault('r3_rounds',{})[key]={'status':'gpu_verified','accepted':len(accepted),'generated':generated,'selected':selected['model_version']}
        save_state(c,state)
    stage_eval(c,m,state,'r3_selftrain',state['checkpoints']['r3'],True,cache,threshold)

def run_feedback(c,args,m,dev,cache,state):
    from .model import Student
    from .train_lora import train
    root=Path(c['run_dir']);folder=Path(c['feedback_dir']);folder.mkdir(parents=True,exist_ok=True)
    records_path=folder/'feedback_records.jsonl';queries_path=folder/'queries.jsonl'
    # Replay remains fit-only; expert pool labels never turn into automatic regional targets.
    replay=read_jsonl(root/'targets/r2.jsonl')
    if not replay: raise FileNotFoundError('Feedback requires valid R2 replay')
    fb=read_json(folder/'state.json')
    if fb is None:
        choices=[v for k,v in state['checkpoints'].items() if k in ('r1','r2','r2_parent','r3')]
        if not choices: raise FileNotFoundError('No dev-selected checkpoint available')
        anchor=max(choices,key=lambda v:v['dev_balanced_accuracy'])
        fb={'anchor':anchor,'current':anchor,'trained_feedback_hash':None,'updates':[]}
        atomic_json(folder/'state.json',fb)
    threshold=read_json(root/'threshold.json')['threshold']
    def curve(stage,cp,active):
        result=stage_eval(c,m,state,stage,cp,True,cache,threshold)
        counts={s:sum(r['feedback_source']==s for r in active.values()) for s in ('expert_label','simulated_dataset_label')}
        sources=[s for s,n in counts.items() if n] or [c['feedback']['source']]
        vals={k:result[k] for k in ('run_id','stage','model_version','n_eval','balanced_accuracy','pair_error_STR_MUS','pair_error_DEB_MUC','pair_error_NORM_TUM')}
        vals.update(anchor_checkpoint=fb['anchor']['model_version'],n_feedback=len(active),
                    n_changed=sum(r['corrected_label']!=r['original_prediction'] for r in active.values()),**{f'n_{k}':v for k,v in counts.items()})
        for source in sources:
            upsert_csv(root/'results/vlm_feedback_curve.csv',{**vals,'feedback_source':source,'n_source_feedback':counts[source]},['run_id','stage','model_version','feedback_source'])
    if not fb.get('baseline_done'):
        curve('fb_0',fb['anchor'],{});fb['baseline_done']=True;atomic_json(folder/'state.json',fb)
    while True:
        queries=read_jsonl(queries_path);records=read_jsonl(records_path);active=validate_records(records,queries,m)
        h=digest(active)
        if active and fb['trained_feedback_hash']!=h:
            name=f'fb_{len(active)}';update=f'{name}_{h[:12]}'
            targets=build_sft.feedback_targets(m,active,replay,folder/f'{update}_labels.csv',c['feedback']['replay_fraction'],
                                               c['feedback'].get('class_replay_fraction',0))
            write_jsonl(root/'targets'/f'{update}.jsonl',targets);build_log(c,update,targets,n_feedback=len(active))
            update_config=feedback_config(c,len(active))
            atomic_json(root/'targets'/f'{update}_train_config.json',update_config['train'])
            best=train(update_config,targets,dev,update,fb['current']['path'],args.resume,args.stop_after)
            curve(name,best,active)
            fb.update(current=best,trained_feedback_hash=h)
            fb['updates'].append({'stage':name,'model_version':best['model_version'],'feedback_hash':h,'n_feedback':len(active),
                                  'n_changed':sum(r['corrected_label']!=r['original_prediction'] for r in active.values()),
                                  'train_config':update_config['train']})
            atomic_json(root/'targets/active_feedback.json',{'target_file':str(root/'targets'/f'{update}.jsonl'),'feedback_hash':h,'current_labels':{k:v['corrected_label'] for k,v in active.items()},'supersedes':fb['updates'][-2]['feedback_hash'] if len(fb['updates'])>1 else None})
            atomic_json(folder/'effective_feedback.json',active)
            atomic_json(folder/'state.json',fb)
            state['checkpoints']['feedback']=best;save_state(c,state)
        node=next((n for n in c['feedback']['nodes'] if n>len(active)),None)
        if node is None: return 'complete'
        outstanding=[q for q in queries if q['tile_id'] not in active]
        if not outstanding:
            student=Student(c,fb['current']['path'])
            try:
                pool=m[(m.role=='pool')&~m.tile_id.isin(active)].drop(columns='label')
                pred,_=score_rows(student,pool,folder/f'pool_{len(active)}_{digest(fb["current"])[0:12]}.csv',fb['current']['model_version'])
            finally: del student;gc.collect();__import__('torch').cuda.empty_cache()
            outstanding=select(pred,set(active),node-len(active),fb['current']['model_version'])
            if not outstanding: return 'pool_exhausted'
            write_jsonl(queries_path,queries+outstanding)  # committed selection BEFORE label lookup
        if c['feedback']['source']=='expert_label':
            state['stages']['feedback']={'status':'awaiting_feedback','outstanding':len(outstanding)};save_state(c,state)
            return 'awaiting_feedback'
        labels=m.set_index('tile_id').label
        for q in outstanding: submit(records_path,q,labels[q['tile_id']],'simulated_dataset_label')

def report(c,state):
    root=Path(c['run_dir']);dest=root/'return_bundle';dest.mkdir(exist_ok=True)
    for name in ('environment.json','effective_config.json','run_manifest.json','target_build_log.jsonl','threshold.json','region_guide.json','inheritance.json','model_revision.json','packages.txt','nvidia-smi.txt'):
        if (root/name).exists(): shutil.copy2(root/name,dest/name)
    for name in ('logs','results','examples'):
        if (root/name).exists(): shutil.copytree(root/name,dest/name,dirs_exist_ok=True)
    if Path(c['feedback_dir']).exists(): shutil.copytree(c['feedback_dir'],dest/'feedback',dirs_exist_ok=True)
    ids={key:{**v,'checkpoint':read_json(Path(v['path'])/'checkpoint.json')} for key,v in state['checkpoints'].items()}
    atomic_json(dest/'checkpoint_identifiers.json',ids)
    summary=root/'results/vlm_summary.csv'
    if summary.exists():
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        df=pd.read_csv(summary);fig,ax=plt.subplots(figsize=(9,4));ax.bar(df.stage,df.balanced_accuracy);ax.set_ylabel('Balanced accuracy (fixed eval)');ax.tick_params(axis='x',rotation=30);fig.tight_layout();fig.savefig(dest/'progress.png');plt.close(fig)
    state['stages']['report']={'status':'cpu_checked','evidence':str(root/'return_bundle.zip')}
    save_state(c,state);shutil.copy2(root/'run_manifest.json',dest/'run_manifest.json')
    zip_path=shutil.make_archive(str(root/'return_bundle'),'zip',dest)
    state['stages']['report']={'status':'cpu_checked','evidence':zip_path};save_state(c,state)
    return state['stages']['report']


def export_examples(rows,cache,dest):
    from PIL import Image,ImageDraw
    from .grounding import evidence
    from .crops import regions
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    cases=[]
    for label in ('STR','DEB','NORM'):
        selected=rows[rows.label==label].head(1)
        if selected.empty: continue
        row=next(selected.itertuples());scores=cache.read('conch',row,'regions')
        e=evidence(scores,cache.descriptors,label)
        with Image.open(row.local_path) as im: image=im.convert('RGB').resize((224,224))
        draw=ImageDraw.Draw(image);draw.rectangle(regions()[e['region_index']]['box'],outline='red',width=3)
        image.save(dest/f'{row.tile_id}.png');cases.append({'tile_id':row.tile_id,**e})
    atomic_json(dest/'cases.json',cases)

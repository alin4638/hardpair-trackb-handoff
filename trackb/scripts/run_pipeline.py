"""Track B pipeline. Run from trackb; paths in YAML are relative to trackb."""
import argparse
import json
import logging
from pathlib import Path
import sys
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tb.common import load_config,atomic_json,read_json
from tb.pipeline import STAGES,run
from tb.feedback import lock

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',required=True,choices=STAGES)
    p.add_argument('--config',default=str(ROOT/'configs/laptop_4080.yaml'))
    p.add_argument('--dry-run',action='store_true',help='Validate config and list prerequisite status; never load models or write artifacts')
    p.add_argument('--r3-from',choices=['r2','feedback'],default='r2',help='Start a new R3 cycle from R2 or the actual feedback checkpoint')
    p.add_argument('--stop-after',type=int,help='Stop cleanly after this optimizer step; unchanged config can resume')
    p.add_argument('--resume',action='store_true',help='Resume latest complete optimizer boundary; stage config must match')
    p.add_argument('--cache-kind',choices=['features','regions'],default='features')
    p.add_argument('--skip-images',action='store_true',help='prepare only: freeze/check contracts without local images; GPU data check remains pending')
    return p

def plan(c,args):
    root=Path(c['run_dir']);state=read_json(root/'run_manifest.json',{'checkpoints':{}})
    required=[Path(c['manifest']),Path(c['manifest']).with_suffix('.meta.json')]
    if args.stage not in ('prepare','report'): required += [Path(c['roles']),Path(c['data_root'])]
    if args.stage=='prepare' and c.get('inherit_run'): required += [Path(c['inherit_run'])/'run_manifest.json']
    if args.stage=='evidence': required += [Path(c.get('cache_dir') or root/'cache')/'identity.json']
    if args.stage=='r1': required += [root/'hard_train.csv']
    if args.stage in ('r2','r3','feedback'): required += [root/'threshold.json']
    if args.stage in ('r3','feedback'): required += [root/'targets/r2.jsonl']
    if c.get('grounding',{}).get('mode')=='guided' and args.stage in ('r2','r3','feedback','explain'):
        required += [root/'region_guide.json']
    parent={'r2':c.get('r2',{}).get('initial_checkpoint','r1'),'r3':args.r3_from}.get(args.stage)
    if parent:
        cp=state['checkpoints'].get(parent)
        required += [Path(cp['path']) if cp else root/'checkpoints'/f'MISSING_SELECTED_{parent}']
    if args.stage in ('feedback','explain') and not state['checkpoints']:
        required += [root/'checkpoints'/'MISSING_SELECTED_CHECKPOINT_RUN_PREPARE']
    return {'stage':args.stage,'gpu_required':args.stage not in ('prepare','evidence','report'),
            'run_dir':str(root),'logs':str(root/'logs'),'resume':args.resume,
            'prerequisites':[{'path':str(p),'exists':p.exists()} for p in required],
            'cache_note':'Individual teacher image caches and resolved model access checked at execution',
            'gpu_status':'gpu_pending'}

def main():
    args=parser().parse_args();c=load_config(args.config)
    if args.skip_images and args.stage!='prepare': raise ValueError('--skip-images is prepare-only')
    if args.dry_run:
        print(json.dumps(plan(c,args),indent=2));return 0
    root=Path(c['run_dir']);logs=root/'logs';logs.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    logging.basicConfig(level=logging.INFO,handlers=[logging.FileHandler(logs/f'{args.stage}_{stamp}.log',encoding='utf-8'),logging.StreamHandler()])
    logging.info('Plan: %s',json.dumps(plan(c,args)))
    try:
        with lock(root/'pipeline.lock'):
            result=run(c,args);logging.info('Completed: %s',json.dumps(result));print(json.dumps(result,indent=2))
        return 0
    except __import__('tb.train_lora',fromlist=['TrainingPaused']).TrainingPaused as e:
        state=read_json(root/'run_manifest.json')
        state['stages'][args.stage]={'status':'checkpoint_saved','validation_status':'gpu_pending','note':str(e)}
        atomic_json(root/'run_manifest.json',state)
        logging.info('%s',e);print(e);return 0
    except Exception:
        logging.exception('Stage failed; resume the latest complete checkpoint after resolving the cause')
        state=read_json(root/'run_manifest.json')
        if state:
            state['stages'][args.stage]={'status':'failed','validation_status':'gpu_pending','error_log':str(logs/f'{args.stage}_{stamp}.log')}
            atomic_json(root/'run_manifest.json',state)
        return 1
if __name__=='__main__': sys.exit(main())

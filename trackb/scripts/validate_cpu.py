"""Run CPU-only evidence checks; never instantiate pretrained models."""
from pathlib import Path
import subprocess
import sys
import tempfile
import json
import platform
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tb.common import load_config,atomic_json,file_hash
from tb.pipeline import STAGES

def main():
    dest=ROOT/'docs/validation/followup_2026-10-18';dest.mkdir(parents=True,exist_ok=True)
    commands=[[sys.executable,'-m','pytest','-q','tests','--junitxml='+str(dest/'cpu-tests.xml')],
              [sys.executable,'-m','compileall','-q','tb','scripts','app.py'],
              [sys.executable,'scripts/run_pipeline.py','--help']]
    results=[]
    with tempfile.TemporaryDirectory(prefix='trackb_cpu_') as temp:
        c=load_config(ROOT/'configs/laptop_4080.yaml');c['run_dir']=str(Path(temp)/'run');c['feedback_dir']=str(Path(temp)/'feedback')
        # The real frozen sidecars are read/validated; no image/model data is loaded.
        config=Path(temp)/'cpu.json';config.write_text(json.dumps(c))
        commands += [[sys.executable,'scripts/run_pipeline.py','--stage','prepare','--config',str(config),'--skip-images']]
        for stage in STAGES:
            commands.append([sys.executable,'scripts/run_pipeline.py','--stage',stage,'--config',str(config),'--dry-run'])
        for filename in ('followup_smoke.yaml','followup_4080.yaml'):
            for stage in ('prepare','evidence','r2','feedback','report'):
                commands.append([sys.executable,'scripts/run_pipeline.py','--stage',stage,'--config',str(ROOT/'configs'/filename),'--dry-run'])
        for cmd in commands:
            p=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True)
            results.append({'command':cmd,'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
            if p.returncode:
                atomic_json(dest/'cpu-evidence.json',results);print(p.stdout,p.stderr);return p.returncode
    atomic_json(dest/'cpu-evidence.json',{'executed_at':datetime.now(timezone.utc).isoformat(),'platform':platform.platform(),
        'python':platform.python_version(),'gpu_execution':False,'commands':results,
        'source_hashes':{str(p.relative_to(ROOT)):file_hash(p) for folder in ('tb','scripts','tests') for p in sorted((ROOT/folder).glob('*.py'))}})
    print(f'PASS: {len(commands)} CPU commands; evidence at {dest}');return 0
if __name__=='__main__':sys.exit(main())

"""Write machine diagnostics before loading any model."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tb.common import load_config,atomic_json
from tb.pipeline import diagnostics
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/laptop_4080.yaml');a=p.parse_args()
c=load_config(a.config);root=Path(c['run_dir']);root.mkdir(parents=True,exist_ok=True)
info=diagnostics(c);atomic_json(root/'environment.json',info);print(info)

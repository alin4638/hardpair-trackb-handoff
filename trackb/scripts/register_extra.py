"""Explicitly register real-label extra NCT rows for dev-observed confusion classes."""
import argparse
from pathlib import Path
import sys
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tb.common import load_config,atomic_json,file_hash
from tb.data import load_manifest,CLASSES
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--config',default='configs/laptop_4080.yaml');p.add_argument('--classes',nargs='+',required=True,choices=CLASSES)
p.add_argument('--per-class',type=int,default=500);p.add_argument('--out',required=True);p.add_argument('--reason',required=True,help='Dev error report motivating this addition')
a=p.parse_args();c=load_config(a.config);m=load_manifest(False,c);used=set(m.path);rows=[]
for label in a.classes:
    candidates=sorted((Path(c['data_root'])/'NCT-CRC-HE-100K'/label).glob('*.tif'))
    n=0
    for f in candidates:
        rel=f.relative_to(c['data_root']).as_posix()
        if rel in used: continue
        rows.append({'tile_id':'extra_'+f.stem,'path':rel,'label':label,'split':'train_extra'});n+=1
        if n>=a.per_class:break
if not rows:raise ValueError('No extra images found')
out=Path(a.out)
if out.exists():raise FileExistsError('Extra manifest is immutable; choose a new path')
out.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(out,index=False)
atomic_json(out.with_suffix('.meta.json'),{'reason':a.reason,'source':'NCT directory real label','sha256':file_hash(out),'n':len(rows)})
print(out)

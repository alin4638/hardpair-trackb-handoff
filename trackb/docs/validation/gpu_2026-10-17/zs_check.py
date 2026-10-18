import sys;from pathlib import Path
ROOT=Path(r'C:\dev\handoff\hardpair-trackb-handoff\trackb');sys.path.insert(0,str(ROOT))
import numpy as np,yaml
from tb.common import load_config,read_json
from tb.data import load_manifest,CLASSES
from tb.cache import Cache
from tb.pipeline import fit_rows,dev_rows
from tb.build_sft import CLASS_NAMES
from tb.encoders import load
from PIL import Image
from tb.crops import crop_grid
c=load_config(str(ROOT/'configs/smoke_4080.yaml'));m=load_manifest(True,c)
cache=Cache.__new__(Cache);cache.c=c;cache.root=Path(c['run_dir'])/'cache'
desc=yaml.safe_load(Path(c['descriptors']).read_text())['descriptors']
fit=fit_rows(m,c);fit=fit[fit.role=='train_fit']
F=np.stack([cache.read('conch',r,'features') for r in fit.itertuples()]);y=np.array([CLASSES.index(l) for l in fit.label])
rev=read_json(cache.root/'conch/revision.json')['sha']
enc=load('conch',device='cuda',revision=rev)
names=enc.embed_texts([f'an H&E image of {CLASS_NAMES[k]}' for k in CLASSES])
D=enc.embed_texts([d['text'] for d in desc]);dl=np.array([CLASSES.index(d['label']) for d in desc])
print('whole-tile, class-name prompts acc:',((F@names.T).argmax(1)==y).mean())
s=F@D.T;cls=np.stack([s[:,dl==k].max(1) for k in range(9)],1);print('whole-tile, descriptors(max) acc:',(cls.argmax(1)==y).mean())
cls=np.stack([s[:,dl==k].mean(1) for k in range(9)],1);print('whole-tile, descriptors(mean) acc:',(cls.argmax(1)==y).mean())
# crops with class names
acc=[];accd=[]
for r in fit.itertuples():
    with Image.open(r.local_path) as im: crops,_=crop_grid(im)
    E=enc.embed_images(crops);acc.append(((E@names.T).mean(0).argmax())==CLASSES.index(r.label))
    sd=(E@D.T);accd.append(np.argmax([sd[:,dl==k].max() for k in range(9)])==CLASSES.index(r.label))
print('crops, class-name prompts (mean over crops) acc:',np.mean(acc),' crops descriptors(max) acc:',np.mean(accd))

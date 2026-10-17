"""Per-image atomic teacher caches with portable, content-addressed provenance."""
from pathlib import Path
import gc
import time
import logging
import numpy as np
import yaml
from PIL import Image
from .common import digest, file_hash, atomic_json, read_json, append_jsonl
from .crops import crop_grid

class Cache:
    def __init__(self,c,manifest):
        self.c=c; self.root=Path(c['run_dir'])/'cache'
        self.descriptors=yaml.safe_load(Path(c['descriptors']).read_text())['descriptors']
        self.identity={'manifest_version':manifest.attrs['manifest_version'],
                       'descriptors':file_hash(c['descriptors']), 'geometry':'224-112-56-v1',
                       'implementation':file_hash(Path(__file__).with_name('encoders.py')),
                       'configured_revisions':c['cache']['revisions']}
        self.root.mkdir(parents=True,exist_ok=True)
        old=read_json(self.root/'identity.json')
        if old and old!=self.identity: raise ValueError('Cache contract changed; choose a new run_dir')
        atomic_json(self.root/'identity.json',self.identity)
    def path(self,teacher,tile_id,kind):
        return self.root/teacher/kind/(digest(tile_id)+'.npz')
    def read(self,teacher,row,kind):
        path=self.path(teacher,row.tile_id,kind)
        if not path.exists(): raise FileNotFoundError(f'Missing {teacher}/{kind} for {row.tile_id}; run cache --cache-kind {kind}')
        with np.load(path,allow_pickle=False) as z:
            if str(z['tile_id'])!=row.tile_id or str(z['path'])!=row.path: raise ValueError('Cache identity mismatch')
            if str(z['image_sha256'])!=file_hash(row.local_path): raise ValueError('Image changed since caching')
            pinned=read_json(self.root/teacher/'revision.json')
            if not pinned or str(z['model_revision'])!=pinned['sha']: raise ValueError('Teacher revision mismatch')
            values=z['values'].copy()
            if not np.isfinite(values).all(): raise ValueError('Non-finite cached scores')
            if kind=='regions' and values.shape!=(9,len(self.descriptors)): raise ValueError('Region score shape mismatch')
            return values
    def build(self,rows,kind):
        import torch
        from .encoders import load
        from huggingface_hub import model_info
        repos={'conch':'MahmoodLab/CONCH','musk':'xiangjx/musk'}
        for teacher in self.c['cache']['teachers']:
            revision_file=self.root/teacher/'revision.json'
            pinned=read_json(revision_file)
            if pinned is None:
                pinned={'repo':repos[teacher],'sha':model_info(repos[teacher],revision=self.c['cache']['revisions'][teacher]).sha}
                atomic_json(revision_file,pinned)
            pending=[]
            for row in rows.itertuples():
                if self.path(teacher,row.tile_id,kind).exists(): self.read(teacher,row,kind)
                else: pending.append(row)
            if not pending: continue
            enc=None
            try:
                torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                enc=load(teacher,device='cuda',revision=pinned['sha'])
                text=enc.embed_texts([d['text'] for d in self.descriptors]) if kind=='regions' else None
                batch_size=self.c['cache']['batch_size'] if kind=='features' else 1
                for offset in range(0,len(pending),batch_size):
                    batch=pending[offset:offset+batch_size];images=[];coordinates=[]
                    for row in batch:
                        with Image.open(row.local_path) as im:
                            if kind=='regions':
                                images,coordinates=crop_grid(im)
                            else: images.append(im.convert('RGB'))
                    values=enc.embed_images(images,batch_size=self.c['cache']['batch_size'])
                    values=[values@text.T] if kind=='regions' else values
                    for row,value in zip(batch,values):
                        dest=self.path(teacher,row.tile_id,kind);dest.parent.mkdir(parents=True,exist_ok=True)
                        tmp=dest.with_suffix('.tmp')
                        with tmp.open('wb') as f:
                            np.savez_compressed(f,values=value,tile_id=row.tile_id,path=row.path,
                                image_sha256=file_hash(row.local_path),model_revision=pinned['sha'],
                                processor=f'{teacher}-native-v1',manifest_version=self.identity['manifest_version'])
                        tmp.replace(dest)
                        if coordinates: atomic_json(dest.with_suffix('.coords.json'),coordinates)
                    completed=offset+len(batch)
                    if completed%100<batch_size or completed==len(pending):
                        torch.cuda.synchronize();elapsed=time.perf_counter()-start
                        event={'teacher':teacher,'kind':kind,'new_tiles':completed,'pending_at_start':len(pending),
                               'seconds':elapsed,'tiles_per_second':completed/elapsed,
                               'peak_vram_gib':torch.cuda.max_memory_allocated()/2**30,'model_revision':pinned['sha']}
                        append_jsonl(Path(self.c['run_dir'])/'logs/cache.jsonl',event)
                        logging.info('Cache progress: %s',event)
            finally:
                del enc;gc.collect();torch.cuda.empty_cache()
    def features(self,rows):
        return np.stack([self.read('conch',r,'features') for r in rows.itertuples()])

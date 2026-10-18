"""GPU check of the new Student paths against the smoke_4080_v1 R2 checkpoint (scratch run_dir)."""
import sys, time, copy
from pathlib import Path
ROOT = Path(r'C:\dev\handoff\hardpair-trackb-handoff\trackb'); sys.path.insert(0, str(ROOT))
import torch, yaml
from tb.common import load_config
from tb.model import Student
from tb.grounding import generation_prompt
c = copy.deepcopy(load_config(str(ROOT / 'configs/smoke_4080.yaml'))); c['run_dir'] = sys.argv[1]; Path(c['run_dir']).mkdir(parents=True, exist_ok=True)
cp = ROOT / 'outputs/smoke_4080_v1/checkpoints/r2_grounded/step_0000010'
desc = yaml.safe_load(Path(c['descriptors']).read_text())['descriptors']
imgs = sorted((ROOT / 'data/raw/CRC-VAL-HE-7K').glob('*/*.tif'))[::450][:8]
st = Student(c, str(cp)); print('merged:', st.merged)
preds = [st.scores(p) for p in imgs]
prompts = [generation_prompt(desc, p['pred'], p['top2']) for p in preds]
t = time.perf_counter(); single = [st.generate(i, q, prediction=p['pred']) for i, q, p in zip(imgs, prompts, preds)]; torch.cuda.synchronize(); ts = time.perf_counter() - t
t = time.perf_counter(); batch = st.generate_many(imgs, prompts, predictions=[p['pred'] for p in preds]); torch.cuda.synchronize(); tb_ = time.perf_counter() - t
print(f'greedy single {ts/len(imgs):.2f}s/tile, batched {tb_/len(imgs):.2f}s/tile, identical {sum(a == b for a, b in zip(single, batch))}/{len(imgs)}')
for a, b in zip(single, batch):
    if a != b: print('DIFF\n', a, '\n---\n', b); break
torch.manual_seed(0)
t = time.perf_counter(); samples = st.generate_many([imgs[0]], [generation_prompt(desc)], True, n=4); torch.cuda.synchronize()
print(f'sampled 4 in {time.perf_counter()-t:.2f}s; unique {len(set(samples))}/4; peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB')
for s in samples: print(repr(s[:160]))

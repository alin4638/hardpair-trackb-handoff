"""Profile VLM scoring/generation and compare batched vs per-candidate scoring (read-only on run artifacts)."""
import sys, time
from pathlib import Path
ROOT = Path(r'C:\dev\handoff\hardpair-trackb-handoff\trackb')
sys.path.insert(0, str(ROOT))
import copy
import numpy as np, torch
from PIL import Image
from tb.common import load_config
from tb.model import Student
from tb.data import CLASSES
from tb.build_sft import CLASS_PROMPT
from tb.scoring import assistant_labels, normalize_scores
from tb.grounding import generation_prompt
import yaml

c = load_config(str(ROOT / 'configs/smoke_4080.yaml'))
c = copy.deepcopy(c); c['run_dir'] = str(Path(sys.argv[1]))  # scratch run_dir so model_revision.json is not touched
Path(c['run_dir']).mkdir(parents=True, exist_ok=True)
cp = ROOT / 'outputs/smoke_4080_v1/checkpoints/r2_grounded/step_0000010'
desc = yaml.safe_load(Path(c['descriptors']).read_text())['descriptors']
imgs = sorted((ROOT / 'data/raw/CRC-VAL-HE-7K').glob('*/*.tif'))[::400][:12]

def batched(st, image):
    msgs = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': CLASS_PROMPT}]}]
    prefix = st.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    with Image.open(image) as im: im = im.convert('RGB').resize((224, 224))
    pref = st.proc(text=[prefix], images=[im], return_tensors='pt')['input_ids'][0].tolist()
    tok = st.proc.tokenizer; old = tok.padding_side; tok.padding_side = 'right'
    try: x = st.proc(text=[prefix + l for l in CLASSES], images=[im] * 9, padding=True, return_tensors='pt').to('cuda')
    finally: tok.padding_side = old
    labels = torch.tensor([assistant_labels(x['input_ids'][i].tolist(), pref, x['attention_mask'][i].tolist()) for i in range(9)], device='cuda')
    with torch.inference_mode(), torch.autocast('cuda', dtype=st.dtype):
        start = len(pref) - 1
        logits = st.model(**x).logits[:, start:-1].float()
        lab = labels[:, start + 1:]; mask = lab != -100
        logp = logits.log_softmax(-1).gather(-1, lab.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        s = (logp * mask).sum(1)
    return [float(v) for v in s], mask.sum(1).tolist()

for name, ckpt in [('base', None), ('r2', cp)]:
    st = Student(c, str(ckpt) if ckpt else None)
    torch.cuda.synchronize(); st.scores(imgs[0]); batched(st, imgs[0])
    t = time.perf_counter(); old = [st.scores(p) for p in imgs]; torch.cuda.synchronize(); t_old = (time.perf_counter() - t) / len(imgs)
    t = time.perf_counter(); new = [batched(st, p) for p in imgs]; torch.cuda.synchronize(); t_new = (time.perf_counter() - t) / len(imgs)
    # compare raw log-liks via the normalized probabilities
    diffs = []; agree = 0
    for o, (s, l) in zip(old, new):
        n = normalize_scores(s)
        diffs.append(max(abs(o[f'p_{k}'] - n[f'p_{k}']) for k in CLASSES)); agree += o['pred'] == n['pred']
        assert o['candidate_token_lengths'] == l, (o['candidate_token_lengths'], l)
    print(f'{name}: per-tile old={t_old:.3f}s batched={t_new:.3f}s maxprobdiff={max(diffs):.2e} pred_agree={agree}/{len(imgs)} lens={l}')
    prompt = generation_prompt(desc, 'STR', 'MUS')
    t = time.perf_counter(); out = st.generate(imgs[0], prompt, prediction='STR'); torch.cuda.synchronize(); dt = time.perf_counter() - t
    ntok = len(st.proc.tokenizer(out)['input_ids'])
    print(f'{name}: greedy generate {dt:.2f}s ~{ntok} tokens -> {ntok/dt:.1f} tok/s; peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB')
    if ckpt:
        st.model = st.model.merge_and_unload()
        t = time.perf_counter(); out2 = st.generate(imgs[0], prompt, prediction='STR'); torch.cuda.synchronize(); dt2 = time.perf_counter() - t
        print(f'{name} merged: generate {dt2:.2f}s same_text={out2 == out}')
        t = time.perf_counter(); merged_old = [st.scores(p) for p in imgs[:4]]; torch.cuda.synchronize()
        print(f'{name} merged: old scoring {(time.perf_counter()-t)/4:.3f}s/tile, pred agree {sum(a["pred"]==b["pred"] for a,b in zip(merged_old, old))}/4')
    del st; torch.cuda.empty_cache()

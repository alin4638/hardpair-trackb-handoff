"""Headless Streamlit check on a throwaway copy of the smoke run's state/feedback (no GPU work launched)."""
import sys, shutil, json, io
from pathlib import Path
ROOT = Path(r'C:\dev\handoff\hardpair-trackb-handoff\trackb'); sys.path.insert(0, str(ROOT))
import yaml, pandas as pd
from streamlit.testing.v1 import AppTest
from tb.common import read_jsonl, load_config
from tb.data import load_manifest
from tb.feedback import import_labels

work = Path(sys.argv[1]); shutil.rmtree(work, ignore_errors=True); (work / 'run').mkdir(parents=True)
src_run = ROOT / 'outputs/followup_smoke_v1'
for name in ('run_manifest.json', 'region_guide.json'): shutil.copy2(src_run / name, work / 'run' / name)
shutil.copytree(ROOT / 'data/feedback_followup_smoke_v1', work / 'feedback')
cfg = yaml.safe_load((ROOT / 'configs/followup_smoke.yaml').read_text())
cfg.update(run_dir=str(work / 'run'), feedback_dir=str(work / 'feedback'),
           cache_dir=str(ROOT / 'outputs/laptop_4080_v1/cache'), inherit_run=str(ROOT / 'outputs/laptop_4080_v1'),
           manifest=str(ROOT / 'data/manifest.csv'), roles=str(ROOT / 'data/roles.csv'),
           data_root=str(ROOT / 'data/raw'), descriptors=str(ROOT / 'configs/descriptors_crc.yaml'))
cfg_path = work / 'app_config.yaml'; cfg_path.write_text(yaml.safe_dump(cfg))
sys.argv = ['app.py', '--config', str(cfg_path)]
import os; os.chdir(ROOT)

def run():
    at = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=300); at.run()
    assert not at.exception, at.exception
    return at
out = {}
at = run()
out['metrics'] = {m.label: m.value for m in at.metric}
out['feedback_model_version_text'] = [w.value for w in at.markdown][:6]
out['buttons'] = {b.label: b.disabled for b in at.button}
# Results tab: R2 stage, first tile
sb = at.selectbox[0]; out['result_stages'] = list(sb.options)
sb.select('r2_grounded').run()
folder = Path(json.load(open(work / 'run/run_manifest.json'))['results']['r2_grounded']['path'])
saved = {r['tile_id']: r for r in read_jsonl(folder / 'vlm_rationales_r2_grounded.jsonl')}
seen = set()
for tid in list(at.selectbox[1].options):
    src = saved[tid]['explanation_source']
    if src in seen: continue
    seen.add(src); at.selectbox[1].select(tid).run()
    shown = [t.value for t in at.text]
    out.setdefault('result_tiles', []).append({'tile_id': tid, 'explanation_source': src, 'cited_regions': saved[tid]['cited_regions'],
        'student_grounding_pass': saved[tid].get('student_grounding_pass'), 'model_version_shown': at.code[0].value,
        'shown_text_matches_file': bool(shown) and shown[0] == saved[tid]['rationale'], 'n_images': len(at.get('imgs'))})
# Feedback tab: query versions + a single expert correction via the form
fsel = at.selectbox[2]; tid = list(fsel.options)[0]; fsel.select(tid).run()
q = next(q for q in read_jsonl(work / 'feedback/queries.jsonl') if q['tile_id'] == tid)
out['feedback_tile'] = {'tile_id': tid, 'query_model_version': q['query_model_version'], 'original_prediction': q['original_prediction'],
                        'evidence_text': at.text[-1].value if at.text else None}
before = len(read_jsonl(work / 'feedback/feedback_records.jsonl'))
at.selectbox[3].select('TUM'); at.button[-1] if False else None
form_btn = [b for b in at.button if b.label == '提交类别反馈'][0]; form_btn.click().run()
recs = read_jsonl(work / 'feedback/feedback_records.jsonl')
out['form_submit'] = {'records_before': before, 'records_after': len(recs), 'last': recs[-1]}
at = run(); out['metrics_after_submit'] = {m.label: m.value for m in at.metric}
# CSV import path (same function as the app's import button)
c = load_config(str(cfg_path)); m = load_manifest(False, c); queries = read_jsonl(work / 'feedback/queries.jsonl')
ids = list(dict.fromkeys(x['tile_id'] for x in queries))
good = pd.DataFrame({'tile_id': ids[1:3], 'corrected_label': ['MUC', 'BACK']})
rows = import_labels(good, queries, m, work / 'feedback/feedback_records.jsonl', 'expert_label')
out['csv_import_ok'] = [(r['tile_id'], r['corrected_label'], r['feedback_source'], r['revision']) for r in rows]
for bad_id in (m[m.role == 'eval'].tile_id.iloc[0], m[(m.role == 'pool') & ~m.tile_id.isin(ids)].tile_id.iloc[0]):
    try: import_labels(pd.DataFrame({'tile_id': [bad_id], 'corrected_label': ['STR']}), queries, m, work / 'feedback/feedback_records.jsonl', 'expert_label'); out.setdefault('csv_reject', []).append((bad_id, 'ACCEPTED?!'))
    except ValueError as e: out.setdefault('csv_reject', []).append((bad_id, str(e)))
at = run(); out['metrics_after_csv'] = {m_.label: m_.value for m_ in at.metric}
out['pending_table_rows'] = len(at.dataframe[0].value) if at.dataframe else None
(work / 'run/pipeline.lock').write_text('test'); at = run()
out['launch_button_disabled_when_locked'] = {b.label: b.disabled for b in at.button}.get('执行反馈更新 / 继续队列')
print(json.dumps(out, indent=1, ensure_ascii=False, default=str))

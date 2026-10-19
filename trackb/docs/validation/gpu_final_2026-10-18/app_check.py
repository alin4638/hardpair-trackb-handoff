"""Read-only headless Streamlit check of the final run (configs/final_4080.yaml). Clicks no buttons, writes nothing."""
import sys, json, os
from pathlib import Path
ROOT = Path(r'C:\dev\handoff\hardpair-trackb-handoff\trackb'); sys.path.insert(0, str(ROOT))
from streamlit.testing.v1 import AppTest
from tb.common import read_jsonl

os.chdir(ROOT); sys.argv = ['app.py', '--config', 'configs/final_4080.yaml']
state = json.load(open(ROOT / 'outputs/final_4080_v1/run_manifest.json'))

def run(at=None):
    at = at or AppTest.from_file(str(ROOT / 'app.py'), default_timeout=600); at.run()
    assert not at.exception, at.exception
    return at

at = run(); out = {}
out['metrics'] = {m.label: m.value for m in at.metric}
out['captions'] = [c.value for c in at.caption][:4]
out['buttons_disabled'] = {b.label: b.disabled for b in at.button}
stage = at.selectbox[0]
out['result_stages'] = list(stage.options); out['default_stage'] = stage.value
out['model_version_shown'] = at.code[0].value
entry = state['results'][stage.value]; name = stage.value
saved = {r['tile_id']: r for r in read_jsonl(Path(entry['path']) / f'vlm_rationales_{name}.jsonl')}
seen = set(); out['result_tiles'] = []
for tid in list(at.selectbox[1].options):
    key = (saved[tid].get('explanation_source'), saved[tid].get('reused_parent_explanation'))
    if key in seen: continue
    seen.add(key); at.selectbox[1].select(tid).run()
    shown = [t.value for t in at.text]
    out['result_tiles'].append({'tile_id': tid, 'explanation_source': key[0], 'reused': key[1],
        'classifier_version': saved[tid].get('classifier_version'), 'explanation_model_version': saved[tid].get('explanation_model_version'),
        'shown_text_matches_file': bool(shown) and shown[0] == saved[tid]['rationale']})
    if len(seen) == 4: break
fsel = at.selectbox[2]; tid = list(fsel.options)[-1]; fsel.select(tid).run()
out['header_markdown'] = [w.value for w in at.markdown][:6]
out['feedback_tab'] = {'n_queried': len(fsel.options), 'tile_id': tid,
                       'versions_shown': [j.value for j in at.json if 'query_model_version' in j.value or 'current_classifier' in j.value]}
print(json.dumps(out, indent=1, ensure_ascii=False, default=str))

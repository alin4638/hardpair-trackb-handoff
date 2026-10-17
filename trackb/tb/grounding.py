"""Teacher-relative evidence, dev-only calibration, strict structured citation checks."""
import re
import numpy as np
from .data import CLASSES
from .hard import PAIRS

FORMAT = re.compile(r'^Diagnosis: (ADI|BACK|DEB|LYM|MUC|MUS|NORM|STR|TUM)\nSupporting: (.+) \[region ([0-2],[0-2])\]\nAgainst (ADI|BACK|DEB|LYM|MUC|MUS|NORM|STR|TUM): Relative support for (.+) is weaker than the supporting descriptor \[region ([0-2],[0-2])\]\nConclusion: (.+)$')

def evidence(scores,descriptors,label,rival=None):
    own=[i for i,d in enumerate(descriptors) if d['label']==label]
    if rival is None:
        rival=next((b if label==a else a for a,b in PAIRS if label in (a,b)),None)
        if rival is None:
            rival=max((c for c in CLASSES if c!=label),key=lambda c:float(scores[:,[i for i,d in enumerate(descriptors) if d['label']==c]].max()))
    other=[i for i,d in enumerate(descriptors) if d['label']==rival]
    # strongest supporting crop; rival is evaluated on the SAME crop
    r,k=np.unravel_index(scores[:,own].argmax(),(9,len(own)));a=own[k]
    b=max(other,key=lambda j:scores[r,j])
    return {'label':label,'rival':rival,'region':f'{r//3},{r%3}', 'support':descriptors[a]['text'],
            'against':descriptors[b]['text'],'support_id':descriptors[a]['descriptor_id'],
            'against_id':descriptors[b]['descriptor_id'],'margin':float(scores[r,a]-scores[r,b]),
            'region_index':int(r),'support_index':int(a),'against_index':int(b)}

def render(e):
    return (f"Diagnosis: {e['label']}\nSupporting: {e['support']} [region {e['region']}]\n"
            f"Against {e['rival']}: Relative support for {e['against']} is weaker than the supporting descriptor [region {e['region']}]\n"
            f"Conclusion: The cited region relatively supports {e['label']} over {e['rival']}.")

def validate(text,pred,scores,descriptors,threshold,musk=None,expected_rival=None):
    match=FORMAT.fullmatch(text.strip())
    if not match: return {'format_pass':False,'grounding_pass':False,'cited_regions':[]}
    label,support,reg,rival,against,reg2,conclusion=match.groups()
    result={'format_pass':label==pred and rival!=pred and (expected_rival is None or rival==expected_rival) and conclusion==f'The cited region relatively supports {label} over {rival}.',
            'grounding_pass':False,'cited_regions':[reg,reg2], 'teacher_source':'CONCH; MUSK auxiliary'}
    a=next((i for i,d in enumerate(descriptors) if d['text']==support and d['label']==label),None)
    b=next((i for i,d in enumerate(descriptors) if d['text']==against and d['label']==rival),None)
    if a is None or b is None or reg!=reg2 or not result['format_pass']: return result
    r=int(reg[0])*3+int(reg[-1]); margin=float(scores[r,a]-scores[r,b])
    result.update(grounding_pass=margin>threshold,teacher_margin=margin,
                  musk_margin=float(musk[r,a]-musk[r,b]) if musk is not None else None)
    return result

def calibrate(dev,cache,c):
    if set(dev.role)!={'train_dev'}: raise ValueError('Calibration uses train_dev only')
    trials=[]
    for row in dev.itertuples():
        s=cache.read('conch',row,'regions')
        for label in CLASSES:
            e=evidence(s,cache.descriptors,label)
            trials.append((e['margin'],label==row.label))
    table=[]
    for t in c['cache']['threshold_candidates']:
        accepted=[ok for margin,ok in trials if margin>t]
        table.append({'threshold':float(t),'accepted':len(accepted),'precision':sum(accepted)/len(accepted) if accepted else 0.0,
                      'true_accepted':sum(accepted)})
    valid=[r for r in table if r['accepted'] and r['precision']>=c['cache']['minimum_dev_precision']]
    # Explicitly record fallback; this is a class-match proxy, not expert region quality.
    best=max(valid or table,key=lambda r:(r['true_accepted'] if valid else r['precision'],r['threshold']))
    return {'threshold':best['threshold'],'calibration':'train_dev class-match proxy',
            'precision_constraint_met':bool(valid),'n_dev':len(dev),'trials':table}

def generation_prompt(descriptors,pred=None,rival=None):
    catalog='\n'.join(f"{d['label']}: {d['text']}" for d in descriptors)
    fixed=f'Use Diagnosis: {pred} and Against {rival}.' if pred else 'Choose the diagnosis from the image.'
    return (f'Examine this H&E tile. {fixed} Regions are 0-based row,col in a 3x3 overlapping 112px grid on the 224px image. '
            'Use exact descriptors from the catalog. Output exactly four lines:\nDiagnosis: <class>\n'
            'Supporting: <descriptor> [region row,col]\nAgainst <rival>: Relative support for <rival descriptor> is weaker than the supporting descriptor [region row,col]\n'
            'Conclusion: The cited region relatively supports <class> over <rival>.\nCatalog:\n'+catalog)

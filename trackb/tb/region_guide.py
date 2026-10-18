"""Retrieve discriminative regional descriptors and calibrate ordered class pairs.

Teacher score centering uses train_fit; thresholds use train_dev. No new GPU
encoding is needed. Explanations expose retrieval provenance and missing support.
"""
from pathlib import Path
import logging
import numpy as np
from sklearn.metrics import roc_auc_score
from .common import atomic_json, read_json, digest
from .data import CLASSES
from .grounding import render

class RegionGuide:
    def __init__(self, cache, artifact):
        self.cache = cache
        self.artifact = artifact
        self.descriptors = cache.descriptors
        self.version = digest(artifact)[:16]

    @classmethod
    def load(cls, cache):
        path = Path(cache.c['run_dir']) / 'region_guide.json'
        artifact = read_json(path)
        if artifact is None:
            raise FileNotFoundError('Run --stage evidence to calibrate regional evidence first')
        if artifact['cache_identity'] != cache.identity:
            raise ValueError('Regional guide belongs to a different teacher/descriptor cache')
        return cls(cache, artifact)

    def normalized(self, teacher, scores):
        stats = self.artifact['normalization'][teacher]
        return (scores - np.asarray(stats['mean'])) / np.asarray(stats['std'])

    def choose(self, conch, musk, label, rival):
        """Select the same crop for support and rival, using relative rather than absolute support."""
        z = self.normalized('conch', conch)
        aux = self.normalized('musk', musk)
        allowed = self.artifact['usable_descriptors']
        own = [i for i,d in enumerate(self.descriptors) if d['label']==label and d['descriptor_id'] in allowed]
        other = [i for i,d in enumerate(self.descriptors) if d['label']==rival and d['descriptor_id'] in allowed]
        # Each class keeps at least its most discriminative descriptor during fitting.
        candidates=[]
        for r in range(9):
            a=max(own,key=lambda j:z[r,j]); b=max(other,key=lambda j:z[r,j])
            margin=float(z[r,a]-z[r,b]); auxiliary=float(aux[r,a]-aux[r,b])
            candidates.append((margin + self.artifact['musk_weight'] * auxiliary, r, a, b, margin, auxiliary))
        _,r,a,b,margin,auxiliary=max(candidates)
        threshold=self.artifact.get('pairs',{}).get(f'{label}/{rival}',{}).get('threshold',0.)
        raw=float(conch[r,a]-conch[r,b])
        return {'label':label,'rival':rival,'region':f'{r//3},{r%3}',
                'region_index':r,'support_index':a,'against_index':b,
                'support_id':self.descriptors[a]['descriptor_id'],'against_id':self.descriptors[b]['descriptor_id'],
                'support':self.descriptors[a]['text'],'against':self.descriptors[b]['text'],
                'margin':raw,'calibrated_margin':margin,'musk_margin':float(musk[r,a]-musk[r,b]),
                'musk_calibrated_margin':auxiliary,'threshold':threshold,
                'accepted':bool(margin>threshold and raw>0), 'guide_version':self.version}

    def retrieve(self, row, label, rival):
        return self.choose(self.cache.read('conch',row,'regions'),self.cache.read('musk',row,'regions'),label,rival)

    def explanation(self,row,label,rival):
        e=self.retrieve(row,label,rival)
        if e['accepted']:
            text=render(e); regions=[e['region']]
        else:
            text=(f'Diagnosis: {label}\nRegional evidence: No sufficiently discriminative region was retrieved '
                  f'for {label} versus {rival}. Review the image before accepting the explanation.')
            regions=[]
        return {'tile_id':row.tile_id,'pred':label,'rationale':text,'cited_regions':regions,
                'format_pass':bool(e['accepted']),'grounding_pass':bool(e['accepted']),
                'teacher_source':'CONCH calibrated descriptors; MUSK auxiliary',
                'explanation_source':'retrieved_teacher_template' if e['accepted'] else 'insufficient_evidence',
                'evidence_status':'supported' if e['accepted'] else 'insufficient_support','evidence':e}

    def prompt(self,e):
        return ('Classify this H&E tile and explain the supplied regional comparison. Use the exact descriptor '
                'and coordinate below. These are automatic image-text scores, not expert annotations.\n'
                'Regions are 0-based row,col in a 3x3 overlapping 112px grid on the 224px image.\n'
                f"Diagnosis candidate: {e['label']}; rival: {e['rival']}.\n"
                f"Region [{e['region']}]: {e['support']}. Rival descriptor: {e['against']}.\n"
                'Return exactly four lines:\nDiagnosis: <class>\nSupporting: <support descriptor> [region row,col]\n'
                'Against <rival>: Relative support for <rival descriptor> is weaker than the supporting descriptor '
                '[region row,col]\nConclusion: The cited region relatively supports <class> over <rival>.')

    def finish(self, row, label, rival, generated=None):
        result=self.explanation(row,label,rival)
        # This output grammar intentionally requires the retrieved descriptors verbatim.
        passed=generated is not None and result['evidence']['accepted'] and generated.strip()==render(result['evidence'])
        result.update(student_rationale=generated,student_grounding_pass=bool(passed))
        if passed:
            result.update(rationale=generated.strip(),explanation_source='evidence_conditioned_student')
        return result


def fit_guide(fit,dev,cache,minimum_auc=.55,musk_weight=.15):
    if set(fit.role)!={'train_fit'} or set(dev.role)!={'train_dev'}:
        raise ValueError('Regional calibration needs train_fit statistics and train_dev threshold selection')
    desc=cache.descriptors; matrices={}
    for teacher in ('conch','musk'):
        logging.info('Reading %s cached regional scores for %d train_fit tiles',teacher,len(fit))
        matrices[teacher]=np.stack([cache.read(teacher,row,'regions') for row in fit.itertuples()])
    normalization={name:{'mean':x.mean(axis=(0,1)).tolist(),'std':np.maximum(x.std(axis=(0,1)),1e-4).tolist()}
                   for name,x in matrices.items()}
    pooled=matrices['conch'].max(axis=1)
    auc={d['descriptor_id']:float(roc_auc_score(fit.label.eq(d['label']),pooled[:,i])) for i,d in enumerate(desc)}
    usable=[]
    for label in CLASSES:
        ids=[d['descriptor_id'] for d in desc if d['label']==label]
        usable.extend([i for i in ids if auc[i]>=minimum_auc] or [max(ids,key=auc.get)])
    artifact={'version':'regional-guide-v1','calibration':'train_dev class-pair proxy, not expert region annotations',
              'cache_identity':cache.identity,'normalization':normalization,
              'descriptor_auc_train_fit':auc,'usable_descriptors':usable,'minimum_auc':minimum_auc,
              'musk_weight':musk_weight,'n_fit':len(fit),'n_dev':len(dev),'pairs':{}}
    guide=RegionGuide(cache,artifact)
    observations={(a,b):[] for a in CLASSES for b in CLASSES if a!=b}
    for count,row in enumerate(dev.itertuples(),1):
        if count%200==0: logging.info('Calibrating regional pairs: %d/%d train_dev tiles',count,len(dev))
        conch=cache.read('conch',row,'regions');musk=cache.read('musk',row,'regions')
        for rival in CLASSES:
            if rival==row.label: continue
            for label,other,positive in ((row.label,rival,True),(rival,row.label,False)):
                e=guide.choose(conch,musk,label,other)
                observations[label,other].append((e['calibrated_margin'],e['margin']>0,positive))
    for (label,rival),values in observations.items():
        v=np.asarray(values);positive=v[:,2].astype(bool);raw_ok=v[:,1].astype(bool)
        # Optimize pair discrimination, rather than calling nine independent class hypotheses positives.
        candidates=np.unique(np.r_[0.,np.maximum(0,np.quantile(v[:,0],np.linspace(0,1,101)))])
        records=[]
        for t in candidates:
            accept=(v[:,0]>t)&raw_ok
            tpr=float(accept[positive].mean());fpr=float(accept[~positive].mean())
            records.append({'threshold':float(t),'balanced_pair_accuracy':.5*(tpr+1-fpr),
                            'true_acceptance':tpr,'false_acceptance':fpr,'n_positive':int(positive.sum()),
                            'n_negative':int((~positive).sum())})
        artifact['pairs'][f'{label}/{rival}']=max(records,key=lambda x:(x['balanced_pair_accuracy'],x['true_acceptance']))
    atomic_json(Path(cache.c['run_dir'])/'region_guide.json',artifact)
    return RegionGuide(cache,artifact)

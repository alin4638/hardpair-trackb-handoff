"""Whole-tile pathology classifiers and diverse, label-blind feedback selection."""
from pathlib import Path
import logging
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from .common import atomic_json,read_json,digest
from .data import CLASSES
from .feedback import select

TEACHERS=('conch','musk')


def feature_bank(c,m,cache):
    """Read existing per-tile caches once, then reuse the assembled arrays on CPU."""
    folder=Path(c['run_dir'])/'features';folder.mkdir(parents=True,exist_ok=True)
    contract={'ids':m.tile_id.tolist(),'cache_identity':cache.identity}
    meta=read_json(folder/'index.json')
    if meta is not None and meta!=contract: raise ValueError('Feature bank IDs/cache changed')
    bank={}
    for teacher in TEACHERS:
        path=folder/f'{teacher}.npy'
        if path.exists() and meta is not None:
            bank[teacher]=np.load(path,allow_pickle=False)
        else:
            logging.info('Assembling %s whole-tile features: %d tiles',teacher,len(m))
            bank[teacher]=np.stack([cache.read(teacher,r,'features') for r in m.itertuples()]).astype(np.float32)
            tmp=path.with_suffix('.tmp')
            with tmp.open('wb') as f: np.save(f,bank[teacher],allow_pickle=False)
            tmp.replace(path)
        if bank[teacher].ndim!=2 or len(bank[teacher])!=len(m): raise ValueError('Invalid feature bank shape')
    atomic_json(folder/'index.json',contract)
    return bank


class PathologyProbe:
    def __init__(self,arrays,metadata):
        self.arrays=arrays;self.metadata=metadata
        self.version=metadata['model_version']

    @classmethod
    def load(cls,path):
        path=Path(path)
        with np.load(path/'weights.npz',allow_pickle=False) as z: arrays={k:z[k].copy() for k in z.files}
        return cls(arrays,read_json(path/'classifier.json'))

    def probabilities(self,bank,name='concat'):
        x=np.concatenate([bank[t] for t in TEACHERS],axis=1) if name=='concat' else bank[name]
        a=self.arrays
        x=(x-a[name+'_mean'])/a[name+'_scale']
        logits=x@a[name+'_coef'].T+a[name+'_intercept']
        logits-=logits.max(axis=1,keepdims=True)
        p=np.exp(logits);return p/p.sum(axis=1,keepdims=True)

    def predict(self,ids,bank):
        p=self.probabilities(bank);order=np.argsort(-p,axis=1,kind='stable')
        conch=self.probabilities(bank,'conch').argmax(axis=1)
        musk=self.probabilities(bank,'musk').argmax(axis=1)
        frame=pd.DataFrame({'tile_id':list(ids),'pred':[CLASSES[i] for i in order[:,0]],
            'top2':[CLASSES[i] for i in order[:,1]],'margin':p[np.arange(len(p)),order[:,0]]-p[np.arange(len(p)),order[:,1]],
            'conch_pred':[CLASSES[i] for i in conch],'musk_pred':[CLASSES[i] for i in musk],
            'encoder_disagreement':conch!=musk,'classification_backend':'pathology_probe','model_version':self.version})
        for i,cl in enumerate(CLASSES):frame['p_'+cl]=p[:,i]
        return frame


def subset(bank,indices):return {k:v[np.asarray(indices)] for k,v in bank.items()}


def fit_probe(c,m,bank,active,cache_identity,previous=None):
    settings=c['probe'];root=Path(c['run_dir'])/'classifiers'
    fit_mask=m.role.eq('train_fit').to_numpy();fit_idx=np.flatnonzero(fit_mask)
    by_id={r.tile_id:i for i,r in enumerate(m.itertuples())}
    feedback_ids=sorted(active)
    if any(t not in by_id or m.iloc[by_id[t]].role!='pool' for t in feedback_ids):
        raise ValueError('Classifier feedback must use selected pool tiles')
    if set(m.iloc[fit_idx].label)!=set(CLASSES):raise ValueError('All nine train_fit classes are required')
    idx=np.r_[fit_idx,[by_id[t] for t in feedback_ids]].astype(int)
    y=np.r_[m.iloc[fit_idx].label.to_numpy(),[active[t]['corrected_label'] for t in feedback_ids]]
    if not set(y)<=set(CLASSES):raise ValueError('Unknown feedback class')
    weights=np.r_[np.ones(len(fit_idx)),np.full(len(feedback_ids),settings['feedback_weight'])]
    contract={'backend':'standardized_pathology_logistic_v1','settings':settings,'cache_identity':cache_identity,
              'train_fit_ids':m.iloc[fit_idx].tile_id.tolist(),
              'train_fit_labels':m.iloc[fit_idx].label.tolist(),'feedback':active}
    version='probe-'+digest(contract)[:16];folder=root/version
    if (folder/'classifier.json').exists():return PathologyProbe.load(folder)
    arrays={};logs={};start=time.perf_counter()
    with threadpool_limits(limits=settings.get('cpu_threads',4)):
        for name in ('concat',*TEACHERS):
            logging.info('Fitting %s classifier with %d fit + %d feedback tiles',name,len(fit_idx),len(feedback_ids))
            x=np.concatenate([bank[t] for t in TEACHERS],axis=1) if name=='concat' else bank[name]
            # Fixed source-domain scaling; feedback updates the decision boundary, not data normalization.
            scaler=StandardScaler().fit(x[fit_idx]);xx=scaler.transform(x[idx]).astype(np.float64)
            clf=LogisticRegression(C=settings['C'],max_iter=settings['max_iter'],tol=settings['tol'],
                                   random_state=c['seed'],warm_start=True)
            if previous is not None:
                clf.classes_=np.array(CLASSES);clf.coef_=previous.arrays[name+'_coef'].copy()
                clf.intercept_=previous.arrays[name+'_intercept'].copy()
            t0=time.perf_counter();clf.fit(xx,y,sample_weight=weights)
            # Store fixed CLASSES ordering rather than depending on an estimator's implicit order.
            order=[list(clf.classes_).index(cl) for cl in CLASSES]
            arrays.update({name+'_mean':scaler.mean_,name+'_scale':scaler.scale_,
                           name+'_coef':clf.coef_[order],name+'_intercept':clf.intercept_[order]})
            logs[name]={'seconds':time.perf_counter()-t0,'iterations':int(clf.n_iter_.max()),
                        'converged':bool(clf.n_iter_.max()<settings['max_iter'])}
    folder.mkdir(parents=True,exist_ok=True)
    tmp=folder/'weights.tmp'
    with tmp.open('wb') as f:np.savez_compressed(f,**arrays)
    tmp.replace(folder/'weights.npz')
    metadata={'model_version':version,'path':str(folder),'classification_backend':'pathology_probe',
              'n_feedback':len(active),'n_train_fit':len(fit_idx),'feedback_hash':digest(active),
              'seconds':time.perf_counter()-start,'heads':logs,'contract':contract}
    atomic_json(folder/'classifier.json',metadata)
    return PathologyProbe(arrays,metadata)


def diverse_queries(preds,bank,n,version,hard_fraction=.5,diversity_weight=.25):
    """Inputs contain predictions/features only. Reserve half for MUS/STR boundary, then global disagreement."""
    if 'label' in preds:raise ValueError('Query selection must not receive true labels')
    n=min(n,len(preds))
    if n<=0:return []
    x=np.concatenate([bank[t] for t in TEACHERS],axis=1)
    x=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
    frame=preds.reset_index(drop=True).copy();frame['pos']=np.arange(len(frame))
    selected=[]
    def choose(candidates,count):
        candidates=candidates[~candidates.pos.isin(selected)].sort_values(
            ['encoder_disagreement','margin','tile_id'],ascending=[False,True,True]).head(max(count*4,count))
        positions=candidates.pos.to_numpy();chosen=[]
        if not len(positions):return
        # Diversity within a shortlist of uncertain/disagreeing cases avoids querying near duplicates.
        importance=np.linspace(1.,0.,len(positions))
        for _ in range(min(count,len(positions))):
            prior=selected+chosen
            novelty=1-np.max(x[positions]@x[prior].T,axis=1) if prior else np.ones(len(positions))
            score=(1-diversity_weight)*importance+diversity_weight*novelty
            score[np.isin(positions,chosen)]=-np.inf
            chosen.append(int(positions[np.argmax(score)]))
        selected.extend(chosen)
    hard=frame[frame.pred.isin(['MUS','STR'])&frame.top2.isin(['MUS','STR'])]
    quota=round(n*hard_fraction)
    for cl,count in (('MUS',quota//2),('STR',quota-quota//2)):choose(hard[hard.pred==cl],count)
    choose(hard,quota-len(selected))
    choose(frame,n-len(selected))
    chosen=frame.iloc[selected]
    queries=select(chosen,set(),n,version)
    for q in queries:q['selection_strategy']='MUS_STR_quota_then_disagreement_margin_diversity'
    return queries

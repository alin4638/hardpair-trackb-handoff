"""Out-of-fold train_fit mining; no train_dev/pool/eval fits."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from .data import CLASSES
PAIRS = [('STR','MUS'),('DEB','MUC'),('NORM','TUM')]

def mine(rows, features, fraction=0.25):
    if set(rows.role) != {'train_fit'}: raise ValueError('OOF mining accepts train_fit only')
    if len(rows) != len(features): raise ValueError('Feature identity/row mismatch')
    y = rows.label.to_numpy(); p = np.zeros((len(y),len(CLASSES)))
    for train, test in StratifiedKFold(5,shuffle=True,random_state=0).split(features,y):
        clf = LogisticRegression(max_iter=2000, random_state=0).fit(features[train],y[train])
        p[np.ix_(test,[CLASSES.index(c) for c in clf.classes_])] = clf.predict_proba(features[test])
    out=[]
    for i,row in enumerate(rows.itertuples()):
        rival=next((b if row.label==a else a for a,b in PAIRS if row.label in (a,b)),None)
        if rival is None: rival=CLASSES[max((j for j in range(9) if CLASSES[j]!=row.label),key=lambda j:p[i,j])]
        out.append({'tile_id':row.tile_id,'label':row.label,'pred':CLASSES[p[i].argmax()],
                    'rival':rival,'true_rival_margin':float(p[i,CLASSES.index(row.label)]-p[i,CLASSES.index(rival)])})
    out=pd.DataFrame(out).sort_values(['true_rival_margin','tile_id']).reset_index(drop=True)
    out['rank']=np.arange(1,len(out)+1);out['weight']=1
    out.loc[(out['pred']!=out.label)|(out['rank']<=max(1,int(len(out)*fraction))),'weight']=3
    return out

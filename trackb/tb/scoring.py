"""Backend-independent boundary validation used by the actual GPU scorer."""
import numpy as np
from .data import CLASSES

def assistant_labels(ids,prefix,attention):
    if ids[:len(prefix)]!=prefix: raise ValueError('Tokenizer prefix boundary changed')
    if len(ids)!=len(attention): raise ValueError('Attention mask length mismatch')
    labels=[token if i>=len(prefix) and attention[i] else -100 for i,token in enumerate(ids)]
    if all(x==-100 for x in labels): raise ValueError('Empty assistant supervision')
    return labels

def normalize_scores(scores):
    a=np.asarray(scores,dtype=np.float64)
    if a.shape!=(9,) or not np.isfinite(a).all(): raise ValueError('Nine finite class likelihoods required')
    p=np.exp(a-a.max());p/=p.sum();order=np.argsort(-p,kind='stable')
    return {'pred':CLASSES[order[0]],**{f'p_{c}':float(p[i]) for i,c in enumerate(CLASSES)},
            'top2':CLASSES[order[1]],'margin':float(p[order[0]]-p[order[1]])}

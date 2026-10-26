import numpy as np
from sklearn.model_selection import train_test_split

def make_splits(targets, seed=314159):
    y=np.asarray(targets); ids=np.flatnonzero(np.isin(y,[3,5])); labels=(y[ids]==5).astype(int)
    if len(ids)!=10000: raise ValueError('Expected exactly 10,000 official cat/dog training examples')
    train, rest=train_test_split(ids,test_size=2000,stratify=labels,random_state=seed)
    val, analysis=train_test_split(rest,test_size=1000,stratify=y[rest],random_state=seed+1)
    # Probes use disjoint subsets of model training data. Evaluation uses official test only.
    pt,pv=train_test_split(train,test_size=1600,stratify=y[train],random_state=seed+2)
    return {k:sorted(v.tolist()) for k,v in dict(train=train,val=val,analysis=analysis,probe_train=pt,probe_val=pv).items()}

def validate_splits(s, targets):
    y=np.asarray(targets)
    for k,n in [('train',8000),('val',1000),('analysis',1000),('probe_train',6400),('probe_val',1600)]:
        assert len(s[k])==n and len(set(s[k]))==n
        assert np.isin(y[s[k]],[3,5]).all()
        assert (y[s[k]]==3).sum()==n//2
    for a,b in [('train','val'),('train','analysis'),('val','analysis'),('probe_train','probe_val')]:
        assert not set(s[a]) & set(s[b])
    assert set(s['probe_train']) | set(s['probe_val']) == set(s['train'])

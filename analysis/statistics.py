import numpy as np
from scipy import stats

def summarize(values):
    x=np.asarray(values,float);n=len(x);mean=float(x.mean());sd=float(x.std(ddof=1)) if n>1 else None
    half=float(stats.t.ppf(.975,n-1)*sd/np.sqrt(n)) if n>1 else None
    return dict(n_seeds=n,mean=mean,std=sd,ci_low=mean-half if half is not None else None,ci_high=mean+half if half is not None else None)

def paired_seed_test(a,b):
    d=np.asarray(a)-np.asarray(b);out=summarize(d)
    sd=out['std'];out['cohens_dz']=float(d.mean()/sd) if sd and sd>0 else None
    out['p_value']=float(stats.ttest_1samp(d,0).pvalue) if len(d)>1 and np.std(d)>0 else None
    return out

def holm(pvalues):
    p=np.asarray(pvalues,float);order=np.argsort(p);adjusted=np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order]);out=np.empty_like(p);out[order]=np.minimum(adjusted,1);return out

def paired_bootstrap(y,a,b,repeats=2000,seed=0):
    """Within-model paired image bootstrap, stratified by class. Not a seed-level CI."""
    y=np.asarray(y);a=np.asarray(a);b=np.asarray(b);rng=np.random.default_rng(seed)
    ids=[np.flatnonzero(y==k) for k in [0,1]];delta=[]
    for _ in range(repeats):
        idx=np.concatenate([rng.choice(i,len(i),replace=True) for i in ids])
        delta.append(np.mean((a[idx]==y[idx]).astype(float)-(b[idx]==y[idx]).astype(float)))
    return tuple(np.quantile(delta,[.025,.975]).tolist())

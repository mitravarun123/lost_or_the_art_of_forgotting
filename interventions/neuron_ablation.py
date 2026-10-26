import numpy as np
import torch

def choose(scores,k,method,rng,magnitude=None):
    scores=np.asarray(scores);n=len(scores);k=min(n,max(0,k))
    high=np.argsort(-scores,kind='stable')[:k]
    if method in ['selective','gradient','high']:return high
    if method=='low':return np.argsort(scores,kind='stable')[:k]
    if method=='random':return rng.choice(n,k,replace=False)
    if method=='matched':
        # Quantile stratification matches empirical activation/weight magnitude distribution.
        mag=np.asarray(magnitude);order=np.argsort(mag,kind='stable');bins=np.empty(n,int)
        bins[order]=np.minimum(np.arange(n)*20//n,19);chosen=[]
        for b in np.unique(bins[high]):
            count=int((bins[high]==b).sum());pool=np.flatnonzero(bins==b)
            chosen.extend(rng.choice(pool,count,replace=False).tolist())
        return np.array(chosen,dtype=int)
    raise ValueError(method)

def neuron_mask(width,indices):
    m=torch.ones(width);m[indices]=0;return m

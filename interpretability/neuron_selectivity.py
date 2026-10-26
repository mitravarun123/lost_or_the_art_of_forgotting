import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from training.evaluate import predict

def selectivity(model,loader,device):
    p=predict(model,loader,device);y=p['y'];grads=[torch.zeros(d) for d in model.dims[1:-1]];n=0
    model.eval()
    for x,_ in loader:
        z,c=model(x.to(device),return_cache=True)
        g=torch.autograd.grad((z[:,0]-z[:,1]).sum(),c['post'])
        for l,(a,da) in enumerate(zip(c['post'],g)):grads[l]+=(a*da).detach().abs().sum(0).cpu()
        n+=len(x)
    frames=[]
    for l,a in enumerate(p['representations']):
        cat=a[y==0];dog=a[y==1];diff=cat.mean(0)-dog.mean(0)
        pooled=np.sqrt((cat.var(0,ddof=1)+dog.var(0,ddof=1))/2)
        auc=np.array([roc_auc_score(y,a[:,j]) for j in range(a.shape[1])])
        frame=pd.DataFrame(dict(layer=l,neuron=np.arange(a.shape[1]),cat_mean=cat.mean(0),dog_mean=dog.mean(0),
            selectivity=diff,absolute_difference=abs(diff),standardized_difference=diff/np.maximum(pooled,1e-8),
            variance=a.var(0),sparsity=(a==0).mean(0),magnitude=a.mean(0),auc_dog=auc,auc_direction_free=np.maximum(auc,1-auc),gradient=grads[l].numpy()/n))
        frames.append(frame)
    return pd.concat(frames,ignore_index=True),p

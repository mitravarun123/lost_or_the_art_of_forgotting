import numpy as np
import torch
from analysis.metrics import metrics

@torch.no_grad()
def predict(model,loader,device,intervention=None,patch_layer=None):
    model.eval(); zs=[];ys=[];reps=[[] for _ in model.layers[:-1]]
    intervention=intervention or {}
    for x,y in loader:
        x=x.to(device); kw=dict(intervention)
        if patch_layer is not None:
            _,clean=model(x,return_cache=True)
            kw['patches']={patch_layer:(clean['post'][patch_layer],torch.ones(model.dims[patch_layer+1],dtype=torch.bool))}
        z,c=model(x,return_cache=True,**kw)
        zs.append(z.cpu());ys.append(y.cpu())
        for a,r in zip(c['post'],reps):r.append(a.cpu())
    return dict(logits=torch.cat(zs).numpy(),y=torch.cat(ys).numpy(),
        representations=[torch.cat(r).numpy() for r in reps])

def evaluate(model,loader,device,intervention=None,baseline=None,patch_layer=None):
    p=predict(model,loader,device,intervention,patch_layer)
    return metrics(p['y'],p['logits'],baseline),p

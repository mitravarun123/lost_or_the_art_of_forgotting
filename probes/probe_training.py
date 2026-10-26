import copy
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from probes.linear_probe import linear_probe
from probes.nonlinear_probe import nonlinear_probe
from training.checkpoint import seed_all
from analysis.metrics import metrics

def fit_probe(train,val,test,family,cfg,seed,device):
    seed_all(seed);xt,yt=train;xv,yv=val;xe,ye=test
    mean=xt.mean(0);std=np.maximum(xt.std(0),1e-6)
    # Fit preprocessing only on probe training data; preserve dimensionality for both families.
    xt=torch.tensor((xt-mean)/std,dtype=torch.float32);xv=torch.tensor((xv-mean)/std,dtype=torch.float32,device=device)
    xe=torch.tensor((xe-mean)/std,dtype=torch.float32);yv=torch.tensor(yv,device=device)
    model=(linear_probe if family=='linear' else nonlinear_probe)(xt.shape[1]).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=0.001,weight_decay=0.01)
    loader=DataLoader(TensorDataset(xt,torch.tensor(yt)),batch_size=cfg['batch_size'],shuffle=True,generator=torch.Generator().manual_seed(seed))
    best=float('inf');state=None;stale=0
    for epoch in range(cfg['probe_epochs']):
        model.train()
        for x,y in loader:
            opt.zero_grad(set_to_none=True);loss=nn.functional.cross_entropy(model(x.to(device)),y.to(device));loss.backward();opt.step()
        model.eval()
        with torch.no_grad():loss=float(nn.functional.cross_entropy(model(xv),yv))
        if loss<best:best=loss;state=copy.deepcopy(model.state_dict());stale=0
        else:stale+=1
        if stale>=cfg['probe_patience']:break
    model.load_state_dict(state);model.eval()
    with torch.no_grad():z=torch.cat([model(x.to(device)).cpu() for x in xe.split(cfg['batch_size'])]).numpy()
    return metrics(ye,z),dict(model={k:v.cpu() for k,v in model.state_dict().items()},mean=mean,std=std,family=family,width=xt.shape[1],seed=seed,validation_loss=best),z

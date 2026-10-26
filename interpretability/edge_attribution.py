import torch

def edge_scores(model,loader,device,max_batches=None):
    """Exact streaming mean g_j W_ji a_i. Absolute score uses |g|^T |a|, not abs(mean)."""
    names=['signed','absolute','cat_signed','dog_signed','cat_absolute','dog_absolute']
    scores={k:[torch.zeros_like(l.weight,device='cpu') for l in model.layers] for k in names};counts=[0,0];n=0
    model.eval()
    for batch,(x,y) in enumerate(loader):
        if max_batches is not None and batch>=max_batches:break
        z,c=model(x.to(device),return_cache=True)
        g=torch.autograd.grad((z[:,0]-z[:,1]).sum(),c['pre'])
        for l,(a,da,linear) in enumerate(zip(c['inputs'],g,model.layers)):
            w=linear.weight.detach()
            scores['signed'][l]+=(w*(da.T@a)).detach().cpu()
            scores['absolute'][l]+=(w.abs()*(da.abs().T@a.abs())).detach().cpu()
            for k,name in [(0,'cat'),(1,'dog')]:
                sel=(y==k).to(device)
                scores[name+'_signed'][l]+=(w*(da[sel].T@a[sel])).detach().cpu()
                scores[name+'_absolute'][l]+=(w.abs()*(da[sel].abs().T@a[sel].abs())).detach().cpu()
        n+=len(y)
        for k in range(2):counts[k]+=int((y==k).sum())
    for name in names:
        denom=counts[0] if name.startswith('cat') else counts[1] if name.startswith('dog') else n
        for a in scores[name]:a.div_(max(denom,1))
    scores['counts']=dict(total=n,cat=counts[0],dog=counts[1]);return scores

import torch

def restore_edges(ablation,scores,fraction,rng,random=False):
    masks=ablation['edge_masks'];flat=torch.cat([m.flatten() for m in masks.values()]);removed=torch.where(flat==0)[0]
    k=round(len(removed)*fraction);score=torch.cat([s.flatten() for s in scores])
    ids=torch.as_tensor(rng.choice(removed.numpy(),k,replace=False)) if random else removed[torch.argsort(score[removed],descending=True,stable=True)[:k]]
    flat[ids]=1;out={};start=0
    for l,m in masks.items():out[l]=flat[start:start+m.numel()].reshape(m.shape);start+=m.numel()
    return {'edge_masks':out},k/max(len(removed),1)

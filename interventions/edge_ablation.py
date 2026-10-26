import torch

def edge_masks(shapes,indices):
    flat=torch.ones(sum(s.numel() for s in shapes));flat[indices]=0;out={};start=0
    for l,s in enumerate(shapes):
        out[l]=flat[start:start+s.numel()].reshape(s.shape);start+=s.numel()
    return out

def affected_connections(model,intervention):
    removed=[torch.zeros_like(l.weight,dtype=torch.bool,device='cpu') for l in model.layers]
    for l,m in intervention.get('edge_masks',{}).items():removed[l]|=~m.cpu().bool()
    for l,m in intervention.get('neuron_masks',{}).items():
        # Zero activation suppresses outgoing contributions, not the incoming computation.
        removed[l+1][:,~m.cpu().bool()]=True
    weights=[l.weight.detach().cpu().abs() for l in model.layers]
    return dict(affected_connection_fraction=sum(int(m.sum()) for m in removed)/sum(m.numel() for m in removed),
        removed_weight_fraction=sum(float(w[m].sum()) for w,m in zip(weights,removed))/max(sum(float(w.sum()) for w in weights),1e-12))

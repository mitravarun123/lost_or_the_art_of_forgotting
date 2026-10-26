import torch

def patch(clean_activations,layer,indices=None):
    a=clean_activations[layer];mask=torch.ones(a.shape[1],dtype=torch.bool)
    if indices is not None:mask.zero_();mask[indices]=True
    return {layer:(a,mask)}

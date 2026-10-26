import torch
from torch import nn
from torch.nn import functional as F

class MLP(nn.Module):
    dimensions=(3072,512,256,128,2)
    def __init__(self, dimensions=None):
        super().__init__(); self.dims=tuple(dimensions or self.dimensions)
        self.layers=nn.ModuleList([nn.Linear(a,b) for a,b in zip(self.dims[:-1],self.dims[1:])])
    def forward(self,x,neuron_masks=None,edge_masks=None,patches=None,return_cache=False):
        neuron_masks=neuron_masks or {};edge_masks=edge_masks or {};patches=patches or {}
        a=x.flatten(1);cache={'inputs':[],'pre':[],'post':[]}
        for l,layer in enumerate(self.layers):
            cache['inputs'].append(a)
            weight=layer.weight if l not in edge_masks else layer.weight*edge_masks[l].to(a.device)
            z=F.linear(a,weight,layer.bias);cache['pre'].append(z)
            if l<len(self.layers)-1:
                a=F.relu(z)
                if l in neuron_masks: a=a*neuron_masks[l].to(a.device)
                if l in patches:
                    clean,selection=patches[l]; selection=selection.to(a.device)
                    a=torch.where(selection.bool(),clean.to(a.device),a)
                cache['post'].append(a)
            else:a=z
        return (a,cache) if return_cache else a
    def edge_contributions(self,inputs,layer,source_indices=None):
        # Explicit per-minibatch utility; discovery instead uses matrix products.
        a=inputs if source_indices is None else inputs[:,source_indices]
        w=self.layers[layer].weight if source_indices is None else self.layers[layer].weight[:,source_indices]
        return a[:,None,:]*w[None,:,:]

import torch
from models.mlp import MLP
from interventions.edge_ablation import affected_connections
from interventions.restoration import restore_edges
import numpy as np

def test_zero_and_complete_neurons():
    m=MLP((4,3,2,2));x=torch.randn(3,4)
    assert torch.equal(m(x),m(x,neuron_masks={0:torch.ones(3)}))
    z,c=m(x,neuron_masks={1:torch.zeros(2)},return_cache=True)
    assert not c['post'][1].any();assert torch.equal(z,m.layers[-1].bias.expand_as(z))

def test_edges_linear_and_identity():
    m=MLP((4,3,2));x=torch.randn(2,4);mask=torch.ones_like(m.layers[0].weight);mask[1,2]=0
    z,c=m(x,edge_masks={0:mask},return_cache=True)
    expected=torch.nn.functional.linear(x,m.layers[0].weight*mask,m.layers[0].bias)
    assert torch.equal(c['pre'][0],expected)
    assert torch.equal(m(x),m(x,edge_masks={i:torch.ones_like(l.weight) for i,l in enumerate(m.layers)}))
    assert torch.equal(m.layers[0].weight,m.state_dict()['layers.0.weight'])

def test_full_restoration_and_patching():
    m=MLP((4,3,2,2));x=torch.randn(4,4);clean,c=m(x,return_cache=True)
    masks={i:torch.zeros_like(l.weight) for i,l in enumerate(m.layers)}
    restored,f=restore_edges({'edge_masks':masks},[l.weight.abs() for l in m.layers],1,np.random.default_rng(1))
    assert f==1 and torch.equal(clean,m(x,**restored))
    patched=m(x,neuron_masks={0:torch.zeros(3)},patches={0:(c['post'][0],torch.ones(3,dtype=torch.bool))})
    assert torch.equal(clean,patched)

def test_affected_outgoing_connections():
    m=MLP((4,3,2));result=affected_connections(m,{'neuron_masks':{0:torch.tensor([0.,1.,1.])}})
    assert result['affected_connection_fraction']==2/18

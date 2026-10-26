import torch
from torch.utils.data import DataLoader,TensorDataset
from models.mlp import MLP
from interpretability.edge_attribution import edge_scores
from interpretability.circuit_discovery import candidate,inspect_connectivity,connected_paths

def test_streaming_attribution_matches_explicit():
    torch.manual_seed(5);m=MLP((4,3,2));x=torch.randn(6,4);y=torch.arange(6)%2
    s=edge_scores(m,DataLoader(TensorDataset(x,y),batch_size=2),'cpu')
    z,c=m(x,return_cache=True);g=torch.autograd.grad((z[:,0]-z[:,1]).sum(),c['pre'])
    for l in range(2):
        tensor=m.edge_contributions(c['inputs'][l],l)*g[l][:,:,None]
        assert torch.allclose(s['signed'][l],tensor.mean(0),atol=1e-6)
        assert torch.allclose(s['absolute'][l],tensor.abs().mean(0),atol=1e-6)

def test_graph_validity():
    scores=[torch.rand(3,4),torch.rand(2,3)];masks=candidate(scores,.5)
    assert sum(int(m.sum()) for m in masks)==9
    info=inspect_connectivity(masks);assert info['edges']==9
    paths=connected_paths(scores,2);assert inspect_connectivity(paths)['has_input_output_path']
    assert all(m.shape==s.shape for m,s in zip(paths,scores))

def test_disconnected_detection():
    masks=[torch.tensor([[1,0],[0,0]],dtype=torch.bool),torch.tensor([[0,1]],dtype=torch.bool)]
    info=inspect_connectivity(masks);assert not info['connected'] and not info['has_input_output_path']

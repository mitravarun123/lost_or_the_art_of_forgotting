import torch
from models.mlp import MLP
from training.evaluate import predict
from torch.utils.data import DataLoader,TensorDataset

def test_exact_standard_equivalence():
    torch.manual_seed(1);m=MLP();x=torch.randn(2,3072);z,c=m(x,return_cache=True);a=x
    for l in m.layers[:-1]:a=torch.relu(l(a))
    assert torch.equal(z,m.layers[-1](a));assert z.shape==(2,2)
    assert [a.shape[1] for a in c['post']]==[512,256,128]
    assert torch.equal(z,m(x))

def test_extraction():
    m=MLP((4,3,2,2));x=torch.randn(7,4);y=torch.arange(7)%2
    p=predict(m,DataLoader(TensorDataset(x,y),batch_size=3),'cpu')
    _,c=m(x,return_cache=True)
    assert torch.allclose(torch.tensor(p['representations'][0]),c['post'][0],atol=1e-6)

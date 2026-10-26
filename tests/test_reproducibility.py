import torch
from training.checkpoint import seed_all,save_torch,load_torch
from models.mlp import MLP
from run import configuration
from analysis.statistics import holm,summarize

def test_seed_and_serialization(tmp_path):
    seed_all(3);a=MLP((4,3,2));seed_all(3);b=MLP((4,3,2))
    assert all(torch.equal(x,y) for x,y in zip(a.parameters(),b.parameters()))
    save_torch(tmp_path/'a.pt',dict(version=1,model=a.state_dict()));b.load_state_dict(load_torch(tmp_path/'a.pt')['model'])
    assert configuration('mac')['version']==configuration('colab')['version']==1
    assert len(configuration('colab')['seeds'])==5

def test_statistics():
    assert summarize([1])['ci_low'] is None
    assert list(holm([.01,.04,.03]))==[.03,.06,.06]

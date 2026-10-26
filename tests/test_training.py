"""Training/resume software check; no fixture is used as research evidence."""
import pandas as pd
import torch
from torch.utils.data import TensorDataset,DataLoader
from training.train import train
from training.checkpoint import load_torch
from training.evaluate import evaluate

class Fixture:
    manifest={'source':'synthetic software fixture, not CIFAR-10'}
    def __init__(self):
        torch.manual_seed(1);self.ds=TensorDataset(torch.randn(16,3072),torch.arange(16)%2)
    def loader(self,name,shuffle=False,seed=0):
        return DataLoader(self.ds,batch_size=8,shuffle=shuffle,generator=torch.Generator().manual_seed(seed))

def test_training_selection_and_resume(tmp_path):
    cfg=dict(learning_rate=.001,weight_decay=.01,clip=1.,epochs=1,patience=8)
    data=Fixture();model=train(data,cfg,42,torch.device('cpu'),tmp_path)
    first=load_torch(tmp_path/'last.pt');assert first['epoch']==0
    cfg['epochs']=2;model=train(data,cfg,42,torch.device('cpu'),tmp_path)
    last=load_torch(tmp_path/'last.pt');best=load_torch(tmp_path/'best.pt')
    assert last['epoch']==1 and len(pd.read_csv(tmp_path/'history.csv'))==2
    assert best['epoch']==int(pd.read_csv(tmp_path/'history.csv').val_loss.idxmin())
    assert all(torch.equal(v.cpu(),best['model'][k]) for k,v in model.state_dict().items())

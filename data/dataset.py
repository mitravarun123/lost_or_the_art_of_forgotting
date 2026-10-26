import hashlib, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import CIFAR10
from data.splits import make_splits, validate_splits
from training.checkpoint import save_json

def filter_classes(images, targets):
    targets=np.asarray(targets); keep=np.isin(targets,[3,5])
    return images[keep], (targets[keep]==5).astype(np.int64)

class Data:
    def __init__(self, root, manifest_dir, split_seed, batch_size, download=False):
        self.batch_size=batch_size
        tr=CIFAR10(str(root),train=True,download=download)
        te=CIFAR10(str(root),train=False,download=download)
        p=Path(manifest_dir)/'splits.json'
        fingerprint=hashlib.sha256(np.asarray(tr.targets,dtype=np.int64).tobytes()).hexdigest()
        if p.exists():
            m=json.loads(p.read_text())
            if m['split_seed']!=split_seed or m['targets_sha256']!=fingerprint: raise ValueError('Split manifest conflict')
            s=m['indices']
        else:
            s=make_splits(tr.targets,split_seed)
            m=dict(version=1,split_seed=split_seed,targets_sha256=fingerprint,indices=s,
                test_indices=np.flatnonzero(np.isin(te.targets,[3,5])).tolist(), source='torchvision CIFAR10 official train/test')
            # Float64 channel moments over training partition only, without a giant float64 copy.
            total=np.zeros(3); sq=np.zeros(3); n=0
            for start in range(0,len(s['train']),128):
                a=tr.data[s['train'][start:start+128]].astype(np.float64)/255
                total+=a.sum((0,1,2));sq+=(a*a).sum((0,1,2));n+=a.shape[0]*1024
            mean=total/n; std=np.sqrt(sq/n-mean*mean)
            m['mean']=mean.tolist();m['std']=std.tolist();save_json(p,m)
        validate_splits(s,tr.targets)
        self.manifest=m; self.datasets={}
        mean=np.array(m['mean'],dtype=np.float32);std=np.array(m['std'],dtype=np.float32)
        for name,idx in s.items():
            # Probe subsets share model-training tensors below rather than extra image copies.
            if name.startswith('probe_'): continue
            a=((tr.data[idx].astype(np.float32)/255-mean)/std).transpose(0,3,1,2).reshape(-1,3072)
            y=(np.asarray(tr.targets)[idx]==5).astype(np.int64)
            self.datasets[name]=TensorDataset(torch.from_numpy(a.copy()),torch.from_numpy(y))
        lookup={v:i for i,v in enumerate(s['train'])}
        from torch.utils.data import Subset
        for name in ['probe_train','probe_val']:
            self.datasets[name]=Subset(self.datasets['train'],[lookup[i] for i in s[name]])
        idx=m['test_indices']; a=((te.data[idx].astype(np.float32)/255-mean)/std).transpose(0,3,1,2).reshape(-1,3072)
        self.datasets['test']=TensorDataset(torch.from_numpy(a.copy()),torch.tensor((np.asarray(te.targets)[idx]==5).astype(np.int64)))
    def loader(self,name,shuffle=False,seed=0):
        return DataLoader(self.datasets[name],batch_size=self.batch_size,shuffle=shuffle,num_workers=0,
            generator=torch.Generator().manual_seed(seed))

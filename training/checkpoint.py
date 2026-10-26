import json, random, platform, sys, os
from pathlib import Path
import numpy as np
import torch

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def device(name='auto'):
    if name != 'auto': return torch.device(name)
    return torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

def save_json(path, obj):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2, default=lambda x: x.item() if hasattr(x,'item') else str(x), allow_nan=False))
    tmp.replace(path)

def save_torch(path, obj):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp'); torch.save(obj,tmp); tmp.replace(path)

def load_torch(path):
    return torch.load(path,map_location='cpu',weights_only=False)

def environment():
    import importlib.metadata as md
    return dict(python=sys.version, platform=platform.platform(), torch=torch.__version__,
        packages={p:md.version(p) for p in ['torch','torchvision','numpy','scipy','scikit-learn','pandas','matplotlib','PyYAML']},
        cuda=torch.version.cuda, mps=torch.backends.mps.is_available(),
        mps_fallback=os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK'),
        determinism='Seeded; deterministic algorithms warn on unsupported CUDA/MPS kernels; hardware equality is not guaranteed.')

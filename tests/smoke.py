"""Software integration check only; synthetic fixtures are NEVER research evidence."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset
from models.mlp import MLP
from training.checkpoint import seed_all,save_json,save_torch,load_torch
from training.evaluate import predict
from interpretability.neuron_selectivity import selectivity
from interpretability.edge_attribution import edge_scores
from interpretability.circuit_discovery import candidate,inspect_connectivity
from probes.probe_training import fit_probe
from analysis.plots import generate

def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);seed_all(11)
    x=torch.randn(48,3072);y=torch.arange(48)%2;loader=DataLoader(TensorDataset(x,y),batch_size=8)
    model=MLP();opt=torch.optim.AdamW(model.parameters(),lr=.001)
    for xb,yb in loader:
        opt.zero_grad();loss=torch.nn.functional.cross_entropy(model(xb),yb);loss.backward();opt.step()
    clean=predict(model,loader,'cpu');sel,_=selectivity(model,loader,'cpu');scores=edge_scores(model,loader,'cpu',2)
    neuron=predict(model,loader,'cpu',{'neuron_masks':{0:torch.cat([torch.zeros(10),torch.ones(502)])}})
    circuit=candidate(scores['absolute'],.01);edge=predict(model,loader,'cpu',{'edge_masks':{l:1-m.float() for l,m in enumerate(circuit)}})
    a=neuron['representations'][2];yy=neuron['y'];cfg=dict(batch_size=8,probe_epochs=2,probe_patience=2)
    for family in ['linear','nonlinear']:
        result,state,z=fit_probe((a[:24],yy[:24]),(a[24:36],yy[24:36]),(a[36:],yy[36:]),family,cfg,11,'cpu')
        save_torch(out/f'{family}_software_test.pt',state)
    save_torch(out/'model_software_test.pt',model.state_dict());model.load_state_dict(load_torch(out/'model_software_test.pt'))
    generate(out)
    # Rendering measured software-test activation values is explicitly separated from research plots.
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots();ax.plot(sel.groupby('layer').absolute_difference.mean());ax.set(title='SYNTHETIC SOFTWARE TEST — not research',xlabel='Layer',ylabel='Fixture activation difference')
    fig.savefig(out/'software_test.png',dpi=100);plt.close(fig)
    save_json(out/'status.json',dict(status='passed',kind='synthetic software integration only',checks=['training','activation extraction','selectivity','neuron ablation','streaming edge attribution','edge ablation','linear probe','nonlinear probe','serialization','plot rendering'],graph=inspect_connectivity(circuit)))
    print(f'Software smoke test passed: {out.resolve()}')

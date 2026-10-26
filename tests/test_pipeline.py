"""All-stage software integration with synthetic fixtures, never research results."""
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset,DataLoader
from experiments.common import Context,write_rows
from models.mlp import MLP
from analysis.metrics import metrics
from training.checkpoint import save_json

class FixtureData:
    batch_size=8
    manifest={'source':'SYNTHETIC SOFTWARE TEST ONLY'}
    def __init__(self):
        torch.manual_seed(1)
        self.datasets={name:TensorDataset(torch.randn(16,4),torch.arange(16)%2) for name in ['analysis','test','val','probe_train','probe_val']}
    def loader(self,name,shuffle=False,seed=0):
        return DataLoader(self.datasets[name],batch_size=8,shuffle=shuffle,generator=torch.Generator().manual_seed(seed))

def test_all_experiment_modules_and_exports(tmp_path):
    cfg=dict(split_seed=3,analysis_batches=1,neuron_budgets=[.2],edge_budgets=[.1],random_repeats=1,
        probe_conditions='targeted',probe_families=['linear','nonlinear'],probe_epochs=2,probe_patience=2,batch_size=8,
        restoration_fractions=[.5,1.],bootstrap_repeats=10,seeds=[42],learning_rate=.001,epochs=1,patience=1,clip=1.)
    ctx=Context(FixtureData(),cfg,42,'cpu',tmp_path);ctx.model=MLP((4,8,6,4,2))
    p=ctx.clean();ctx.save_prediction('baseline',p);write_rows(ctx,'e1',[dict(seed=42,**metrics(p['y'],p['logits']))])
    import importlib
    for name in ['e2_neurons','e3_neuron_ablation','e4_circuit_discovery','e5_circuit_ablation','e6_comparison','e7_recoverability','e8_restoration']:
        result=importlib.import_module('experiments.'+name).run(ctx)
        assert len(result)>0
    from experiments.e9_robustness import run
    assert len(run(tmp_path,cfg))>0
    from analysis.plots import generate
    from analysis.tables import generate as tables
    generate(tmp_path);tables(tmp_path,cfg)
    assert (tmp_path/'figures/F11_restoration_target_0.png').exists()
    assert (tmp_path/'tables/T8.csv').exists()
    assert not list((tmp_path/'figures').glob('*.pdf'))
    assert (tmp_path/'figures/figure_manifest.json').exists()
    probes=pd.read_csv(ctx.out/'e7.csv')
    assert (probes[probes.layer==3].representation_status=='unaffected_pre_readout_reference').all()
    restore=pd.read_csv(ctx.out/'e8.csv')
    full=restore[(restore.method=='selected')&(restore.restored_fraction==1)]
    assert np.allclose(full.accuracy,metrics(p['y'],p['logits'])['accuracy'])

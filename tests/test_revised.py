"""Revised-scope integration; all tensors here are software fixtures only."""
import ast,json,importlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from models.mlp import MLP
from experiments.common import Context,write_rows
from analysis.metrics import metrics
from training.evaluate import evaluate
from tests.test_pipeline import FixtureData
from run import configuration


def test_revised_profile_and_notebook():
    root=Path(__file__).resolve().parents[1]
    cfg=configuration('colab',root/'configs/revised.yaml')
    assert cfg['seeds']==[42] and cfg['neuron_layers']==[2] and cfg['edge_layers']==[3]
    assert cfg['neuron_methods']==['selective','random'] and cfg['probe_layers']==[2]
    nb=json.loads((root/'notebooks/REVISED_MASTER_Colab_Pilot.ipynb').read_text())
    code=[]
    for c in nb['cells']:
        if c['cell_type']=='code':
            s=''.join(c['source']);ast.parse(s);code.append(s)
    s='\n'.join(code)
    assert "'configs/revised.yaml'" in s and 'MAX_EXPERIMENT_MINUTES = 45' in s
    assert 'LAST_REVISED_RUN.txt' in s and "'revised_results'" in s


def test_revised_full_fixture_pipeline_and_eight_figures(tmp_path):
    root=Path(__file__).resolve().parents[1]
    cfg=configuration('colab',root/'configs/revised.yaml');cfg.update(probe_epochs=1,analysis_batches=1,bootstrap_repeats=5)
    (tmp_path/'config.json').write_text(json.dumps(cfg))
    ctx=Context(FixtureData(),cfg,42,'cpu',tmp_path);ctx.model=MLP((4,8,6,4,2))
    opt=torch.optim.AdamW(ctx.model.parameters(),lr=.001);history=[]
    for epoch in range(2):
        total=correct=count=0
        for x,y in ctx.data.loader('probe_train'):
            opt.zero_grad();z=ctx.model(x);loss=torch.nn.functional.cross_entropy(z,y);loss.backward();opt.step()
            total+=float(loss.detach())*len(y);correct+=int((z.argmax(1)==y).sum());count+=len(y)
        val,_=evaluate(ctx.model,ctx.data.loader('val'),'cpu')
        history.append(dict(epoch=epoch,train_loss=total/count,train_accuracy=correct/count,val_loss=val['cross_entropy'],val_accuracy=val['accuracy']))
    (ctx.out/'checkpoints').mkdir();pd.DataFrame(history).to_csv(ctx.out/'checkpoints/history.csv',index=False)
    p=ctx.clean();ctx.save_prediction('baseline',p);write_rows(ctx,'e1',[dict(seed=42,**metrics(p['y'],p['logits']))])
    for module in ['e2_neurons','e3_neuron_ablation','e4_circuit_discovery','e5_circuit_ablation','e6_comparison','e7_recoverability','e8_restoration']:
        importlib.import_module('experiments.'+module).run(ctx)
    from experiments.e9_robustness import run
    run(tmp_path,cfg)
    n,e,p,r=[pd.read_csv(ctx.out/f'{name}.csv') for name in ['e3','e5','e7','e8']]
    assert len(n)==16 and set(n.layer)=={2} and set(n.method)=={'selective','random'}
    assert len(e)==16 and set(e.layer)=={3} and set(e.method)=={'high','random'}
    assert len(p)==33 and set(p.probe_layer)=={2} and set(p.family)=={'linear'}
    assert (p[p.kind=='edge'].representation_status=='unaffected_pre_readout_reference').all()
    assert (p[p.kind=='neuron'].representation_status=='post_intervention').all()
    assert len(r)==16 and np.allclose(r.accuracy,r.clean_accuracy)
    from analysis.plots import generate
    generate(tmp_path)
    figures=sorted((tmp_path/'figures').glob('*.png'))
    assert len(figures)==8
    assert not list((tmp_path/'figures').glob('*.pdf'))
    status=json.loads((tmp_path/'figures/figure_status.json').read_text());assert all(status['families'].values())
    manifest=json.loads((tmp_path/'figures/figure_manifest.json').read_text())
    assert all(not v['text_outside_layout'] for v in manifest.values())

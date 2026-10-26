import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from models.mlp import MLP
from training.checkpoint import load_torch,save_torch,save_json
from training.evaluate import predict
from analysis.metrics import metrics
from interventions.edge_ablation import affected_connections

class Context:
    def __init__(self,data,cfg,seed,device,root):
        self.data=data;self.cfg=cfg;self.seed=seed;self.device=device;self.root=Path(root);self.out=self.root/f'seed_{seed}'
        self.out.mkdir(parents=True,exist_ok=True);self.model=None;self._clean={}
    def get_model(self):
        if self.model is None:
            self.model=MLP().to(self.device);self.model.load_state_dict(load_torch(self.out/'checkpoints/best.pt')['model']);self.model.eval()
        return self.model
    def clean(self,split='test'):
        if split not in self._clean:self._clean[split]=predict(self.get_model(),self.data.loader(split),self.device)
        return self._clean[split]
    def save_prediction(self,name,p):
        path=self.out/'predictions'/f'{name}.npz';path.parent.mkdir(parents=True,exist_ok=True)
        from scipy.special import softmax
        np.savez_compressed(path,logits=p['logits'],y=p['y'],probabilities=softmax(p['logits'],axis=1),
            predictions=p['logits'].argmax(1),cat_dog_margin=p['logits'][:,0]-p['logits'][:,1])
    def evaluate_condition(self,name,intervention,meta,patch_layer=None):
        marker=self.out/'conditions'/f'{name}.json'
        if marker.exists():return json.loads(marker.read_text())
        clean=self.clean();p=predict(self.get_model(),self.data.loader('test'),self.device,intervention,patch_layer)
        row={**meta,**metrics(p['y'],p['logits'],clean['logits']),**affected_connections(self.get_model(),intervention),
            'condition':name,'seed':self.seed}
        clean_metrics=metrics(clean['y'],clean['logits'])
        for label in ['cat','dog']:row[label+'_margin_degradation']=clean_metrics[label+'_margin']-row[label+'_margin']
        for l,(a,b) in enumerate(zip(clean['representations'],p['representations'])):
            row[f'representation_delta_l{l}']=float(np.linalg.norm(b-a)/max(np.linalg.norm(a),1e-12))
            row[f'activation_norm_l{l}']=float(np.linalg.norm(b,axis=1).mean())
        target=int(meta.get('target',0))
        row['target_degradation']=row['cat_degradation' if target==0 else 'dog_degradation']
        row['non_target_degradation']=row['dog_degradation' if target==0 else 'cat_degradation']
        self.save_prediction(name,p)
        save_torch(self.out/'masks'/f'{name}.pt',dict(intervention=intervention,metadata=meta))
        save_json(marker,row);return row
    def conditions(self,experiment):
        path=self.out/f'{experiment}.csv'
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

def write_rows(ctx,name,rows):
    frame=pd.DataFrame(rows);frame.to_csv(ctx.out/f'{name}.csv',index=False);return frame

from pathlib import Path
import numpy as np
import pandas as pd
from analysis.statistics import summarize,paired_seed_test,holm,paired_bootstrap
from training.checkpoint import load_torch

def run(root,cfg):
    root=Path(root);rows=[];tests=[];image=[]
    for experiment in ['e1','e3','e5','e7','e8']:
        files=sorted(root.glob(f'seed_*/{experiment}.csv'))
        if not files:continue
        frame=pd.concat([pd.read_csv(f) for f in files],ignore_index=True)
        keys=[k for k in ['kind','layer','target','method','budget','family','probe_layer','restored_fraction'] if k in frame]
        groups=frame.groupby(keys,dropna=False) if keys else [((),frame)]
        for key,group in groups:
            key=key if isinstance(key,tuple) else (key,);meta=dict(zip(keys,key))
            for metric in ['accuracy','balanced_accuracy','cat_recall','dog_recall','probe_improvement']:
                if metric not in group:continue
                # Average random-control repetitions within each model before across-seed inference.
                vals=group.groupby('seed')[metric].mean()
                rows.append(dict(experiment=experiment,metric=metric,**meta,**summarize(vals)))
        if experiment in ['e3','e5']:
            targeted='selective' if experiment=='e3' else 'high'
            for key,g in frame.groupby(['layer','target','budget']):
                pivot=g.groupby(['seed','method']).accuracy.mean().unstack()
                if targeted in pivot and 'random' in pivot:
                    p=pivot[[targeted,'random']].dropna()
                    tests.append(dict(experiment=experiment,layer=key[0],target=key[1],budget=key[2],comparison=targeted+' minus random',**paired_seed_test(p[targeted],p.random)))
        if experiment in ['e3','e5']:
            for f in files:
                seed_dir=f.parent;base=np.load(seed_dir/'predictions/baseline.npz')
                for _,r in pd.read_csv(f).iterrows():
                    if r.method not in ['selective','high']:continue
                    p=np.load(seed_dir/'predictions'/f'{r.condition}.npz')
                    lo,hi=paired_bootstrap(base['y'],p['logits'].argmax(1),base['logits'].argmax(1),cfg['bootstrap_repeats'])
                    image.append(dict(seed=r.seed,condition=r.condition,ci_low=lo,ci_high=hi,unit='paired images within one trained model',effect='ablated accuracy minus clean accuracy'))
    stats=pd.DataFrame(tests)
    if len(stats):
        valid=stats.p_value.notna();stats.loc[valid,'p_holm']=holm(stats.loc[valid,'p_value'])
    pd.DataFrame(rows).to_csv(root/'robustness.csv',index=False);stats.to_csv(root/'paired_tests.csv',index=False)
    pd.DataFrame(image).to_csv(root/'paired_image_bootstrap.csv',index=False)
    # Edge-support Jaccard across independently trained models is descriptive: neuron permutation can confound it.
    instability=[];dirs=sorted(root.glob('seed_*'))
    for i,a in enumerate(dirs):
        for b in dirs[i+1:]:
            for file in a.glob('circuits/*.pt'):
                other=b/'circuits'/file.name
                if not other.exists():continue
                x=load_torch(file);y=load_torch(other)
                intersection=sum(int((u&v).sum()) for u,v in zip(x,y));union=sum(int((u|v).sum()) for u,v in zip(x,y))
                instability.append(dict(seed_a=a.name,seed_b=b.name,circuit=file.stem,jaccard=intersection/max(union,1),caveat='Unaligned hidden-unit identities; not semantic circuit equivalence'))
    pd.DataFrame(instability).to_csv(root/'circuit_stability.csv',index=False)
    return pd.DataFrame(rows)

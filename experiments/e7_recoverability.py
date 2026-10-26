import numpy as np
import pandas as pd
from training.evaluate import predict
from training.checkpoint import load_torch,save_torch,save_json
from probes.probe_training import fit_probe
from analysis.metrics import metrics
from experiments.common import write_rows

def run(ctx):
    rows=[];frame=pd.concat([ctx.conditions('e3'),ctx.conditions('e5')],ignore_index=True)
    if ctx.cfg['probe_conditions']=='targeted':
        frame=frame[((frame.kind=='neuron')&(frame.method=='selective'))|((frame.kind=='edge')&(frame.method=='high'))]
    # Cache only clean representations; intervention representations discarded per condition.
    clean={s:ctx.clean(s) for s in ['probe_train','probe_val','test']}
    def fit(name,reps,start,meta):
        for layer in ctx.cfg.get('probe_layers',range(start,3)):
            if layer<start:continue
            for family in ctx.cfg['probe_families']:
                key=f'{name}_l{layer}_{family}';marker=ctx.out/'probe_metrics'/f'{key}.json'
                if marker.exists():
                    import json
                    rows.append(json.loads(marker.read_text()));continue
                datasets=[(reps[s]['representations'][layer],reps[s]['y']) for s in ['probe_train','probe_val','test']]
                result,state,z=fit_probe(*datasets,family,ctx.cfg,ctx.seed,ctx.device)
                original=metrics(clean['test']['y'],clean['test']['logits'])
                classifier=metrics(reps['test']['y'],reps['test']['logits'])
                row=dict(seed=ctx.seed,condition=name,probe_layer=layer,family=family,**meta,**result,
                    original_classifier_accuracy=original['accuracy'],ablated_classifier_accuracy=classifier['accuracy'],
                    ablated_classifier_balanced_accuracy=classifier['balanced_accuracy'],probe_improvement=result['balanced_accuracy']-classifier['balanced_accuracy'])
                save_torch(ctx.out/'probes'/f'{key}.pt',state);save_json(marker,row)
                ctx.save_prediction('probe_'+key,dict(y=datasets[-1][1],logits=z));rows.append(row)
                if len(rows)%50==0:print(f'E7 | seed {ctx.seed} | {len(rows)} probe fits recorded',flush=True)
    fit('clean',clean,0,dict(kind='clean',layer=-1,target=-1,method='clean',budget=0,representation_status='clean_reference'))
    for _,row in frame.iterrows():
        intervention=load_torch(ctx.out/'masks'/f'{row.condition}.pt')['intervention']
        reps={s:predict(ctx.get_model(),ctx.data.loader(s),ctx.device,intervention) for s in ['probe_train','probe_val','test']}
        # A linear edge layer l affects hidden representation l; global edges affect all.
        # Output-only ablation has no subsequent hidden representation: mark readout reference explicitly.
        start=max(0,int(row.layer))
        output_only=row.kind=='edge' and start==3
        if output_only:start=2
        fit(row.condition,reps,start,dict(kind=row.kind,layer=int(row.layer),target=int(row.target),method=row.method,budget=row.budget,
            representation_status='unaffected_pre_readout_reference' if output_only else 'post_intervention'))
    out=pd.DataFrame(rows)
    for family in ctx.cfg['probe_families']:
        for layer in range(3):
            ref=out[(out.condition=='clean')&(out.family==family)&(out.probe_layer==layer)]
            if len(ref):
                sel=(out.family==family)&(out.probe_layer==layer)
                out.loc[sel,'clean_probe_balanced_accuracy']=float(ref.iloc[0].balanced_accuracy)
                out.loc[sel,'relative_to_clean_probe']=out.loc[sel,'balanced_accuracy']-float(ref.iloc[0].balanced_accuracy)
    out.to_csv(ctx.out/'e7.csv',index=False);return out

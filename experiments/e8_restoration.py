import numpy as np
from interventions.restoration import restore_edges
from training.checkpoint import load_torch
from experiments.common import write_rows

def run(ctx):
    if ctx.cfg.get('restoration_mode')=='neuron_patching':
        from experiments.revised_restoration import run
        return run(ctx)
    frame=ctx.conditions('e5');scores=load_torch(ctx.out/'edge_scores.pt');rows=[]
    # Restore every high-attribution ablation; locations/budgets predeclared, no test selection.
    for _,row in frame[frame.method=='high'].iterrows():
        intervention=load_torch(ctx.out/'masks'/f'{row.condition}.pt')['intervention']
        for fraction in ctx.cfg['restoration_fractions']:
            for method in ['selected','random']:
                for repeat in range(ctx.cfg['random_repeats'] if method=='random' else 1):
                    restored,actual=restore_edges(intervention,scores['cat_absolute' if row.target==0 else 'dog_absolute'],fraction,
                        np.random.default_rng(ctx.seed+repeat),method=='random')
                    name=f'restore_{row.condition}_{method}_f{fraction}_r{repeat}'
                    rows.append(ctx.evaluate_condition(name,restored,dict(source_condition=row.condition,kind='restoration',layer=int(row.layer),target=int(row.target),
                        method=method,budget=row.budget,restored_fraction=actual,repeat=repeat,ablated_accuracy=row.accuracy)))
        for layer in range(max(0,int(row.layer)),3):
            name=f'patch_{row.condition}_l{layer}'
            rows.append(ctx.evaluate_condition(name,intervention,dict(source_condition=row.condition,kind='restoration',layer=layer,target=int(row.target),
                method='activation_patch',budget=row.budget,restored_fraction=None,repeat=0,ablated_accuracy=row.accuracy),patch_layer=layer))
        rows.append(dict(seed=ctx.seed,source_condition=row.condition,method='fully_ablated',accuracy=row.accuracy,cat_recall=row.cat_recall,dog_recall=row.dog_recall,
            margin=row.margin,layer=int(row.layer),target=int(row.target),budget=row.budget,ablated_accuracy=row.accuracy))
        from analysis.metrics import metrics
        p=ctx.clean();rows.append(dict(seed=ctx.seed,source_condition=row.condition,method='clean',layer=int(row.layer),target=int(row.target),budget=row.budget,**metrics(p['y'],p['logits']),ablated_accuracy=row.accuracy))
    result=write_rows(ctx,'e8',rows)
    if len(result):
        result['accuracy_recovery']=result.accuracy-result.ablated_accuracy
        for source,g in result.groupby('source_condition'):
            base=g[g.method=='fully_ablated'].iloc[0];clean=g[g.method=='clean'].iloc[0]
            sel=result.source_condition==source
            for metric in ['cat_recall','dog_recall','margin']:
                result.loc[sel,metric+'_recovery']=result.loc[sel,metric]-float(base[metric])
            denom=float(clean.accuracy-base.accuracy)
            result.loc[sel,'normalized_accuracy_recovery']=result.loc[sel,'accuracy_recovery']/denom if abs(denom)>1e-8 else float('nan')
            selected=g[g.method=='selected']
            for idx,s in selected.iterrows():
                random=g[(g.method=='random')&(g.restored_fraction==s.restored_fraction)]
                if len(random):result.loc[idx,'accuracy_over_random_restoration']=s.accuracy-random.accuracy.mean()
        result.to_csv(ctx.out/'e8.csv',index=False)
    return result

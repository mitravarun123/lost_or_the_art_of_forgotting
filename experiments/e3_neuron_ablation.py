import numpy as np
import pandas as pd
from interventions.neuron_ablation import choose,neuron_mask
from experiments.common import write_rows

def run(ctx):
    scores=pd.read_csv(ctx.out/'e2.csv');rows=[]
    for layer in ctx.cfg.get('neuron_layers',range(3)):
        frame=scores[scores.layer==layer].sort_values('neuron')
        for target in [0,1]:
            # Signed selectivity favors the selected class, not merely absolute difference.
            score=frame.selectivity.to_numpy()*(1 if target==0 else -1)
            for budget in ctx.cfg['neuron_budgets']:
                k=max(1,round(len(frame)*budget))
                for method in ctx.cfg.get('neuron_methods',['selective','gradient','random','matched','low']):
                    for repeat in range(ctx.cfg['random_repeats'] if method in ['random','matched'] else 1):
                        rng=np.random.default_rng(ctx.seed+10000*layer+100*target+repeat)
                        ranking=frame.gradient.to_numpy() if method=='gradient' else score
                        ids=choose(ranking,k,method,rng,frame.magnitude.to_numpy())
                        mask=neuron_mask(len(frame),ids)
                        name=f'neuron_l{layer}_t{target}_{method}_b{budget}_r{repeat}'
                        rows.append(ctx.evaluate_condition(name,{'neuron_masks':{layer:mask}},
                            dict(kind='neuron',layer=layer,target=target,method=method,budget=budget,repeat=repeat,ablated_count=k)))
    return write_rows(ctx,'e3',rows)

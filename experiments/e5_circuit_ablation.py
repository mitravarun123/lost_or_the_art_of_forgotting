import numpy as np
import torch
from interventions.neuron_ablation import choose
from interventions.edge_ablation import edge_masks
from training.checkpoint import load_torch
from experiments.common import write_rows

def run(ctx):
    scores=load_torch(ctx.out/'edge_scores.pt');weights=[l.weight.detach().cpu() for l in ctx.get_model().layers]
    magnitude=torch.cat([w.abs().flatten() for w in weights]).numpy();rows=[]
    for target,label in [(0,'cat'),(1,'dog')]:
        flat=torch.cat([s.flatten() for s in scores[label+'_absolute']]).numpy()
        # Global and every layer-specific intervention, including readout edges.
        for layer in ctx.cfg.get('edge_layers',[-1,0,1,2,3]):
            offset=0 if layer==-1 else sum(w.numel() for w in weights[:layer])
            count=len(flat) if layer==-1 else weights[layer].numel()
            local=flat[offset:offset+count];mag=magnitude[offset:offset+count]
            for budget in ctx.cfg['edge_budgets']:
                for method in ctx.cfg.get('edge_methods',['high','random','low','matched']):
                    for repeat in range(ctx.cfg['random_repeats'] if method in ['random','matched'] else 1):
                        rng=np.random.default_rng(ctx.seed+100*target+repeat+1000*(layer+1))
                        ids=choose(local,max(1,round(count*budget)),method,rng,mag)+offset
                        name=f'edge_l{layer}_t{target}_{method}_b{budget}_r{repeat}'
                        intervention={'edge_masks':edge_masks(weights,ids)}
                        result=ctx.evaluate_condition(name,intervention,
                            dict(kind='edge',layer=layer,target=target,method=method,budget=budget,repeat=repeat,ablated_count=len(ids)))
                        result['fraction_edges_removed']=result['affected_connection_fraction']
                        if method=='high':
                            from training.evaluate import predict
                            from interpretability.circuit_discovery import inspect_connectivity
                            selected=[~m.bool() for m in intervention['edge_masks'].values()]
                            result.update({'support_'+k:v for k,v in inspect_connectivity(selected).items()})
                            retained={'edge_masks':{l:1-m for l,m in intervention['edge_masks'].items()}}
                            kept=predict(ctx.get_model(),ctx.data.loader('test'),ctx.device,retained)['logits']
                            clean=ctx.clean()['logits']
                            removed=np.load(ctx.out/'predictions'/f'{name}.npz')['logits']
                            margin=lambda z:z[:,target]-z[:,1-target]
                            denom=max(float(np.mean(abs(margin(clean)))),1e-8)
                            result['faithfulness']=1-float(np.mean(abs(margin(kept)-margin(clean))))/denom
                            result['completeness']=float(np.mean(abs(margin(removed)-margin(clean))))/denom
                            result['margin_normalizer']=denom
                        rows.append(result)
    return write_rows(ctx,'e5',rows)

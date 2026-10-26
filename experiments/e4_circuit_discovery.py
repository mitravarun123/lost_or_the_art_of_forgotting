import pandas as pd
import torch
from interpretability.edge_attribution import edge_scores
from interpretability.circuit_discovery import candidate,inspect_connectivity,connected_paths
from training.checkpoint import save_torch,save_json
from experiments.common import write_rows
from analysis.metrics import metrics
from training.evaluate import predict

def run(ctx):
    scores=edge_scores(ctx.get_model(),ctx.data.loader('analysis',True,ctx.cfg['split_seed']),ctx.device,ctx.cfg['analysis_batches'])
    save_torch(ctx.out/'edge_scores.pt',scores);rows=[]
    for target,label in [(0,'cat'),(1,'dog')]:
        attribution=scores[label+'_absolute']
        for budget in ctx.cfg['edge_budgets']:
            for extraction in ctx.cfg.get('circuit_extractions',['ranked','connected_paths']):
                masks=candidate(attribution,budget) if extraction=='ranked' else connected_paths(attribution,max(1,round(budget*1000)))
                info=inspect_connectivity(masks)
                name=f'circuit_t{target}_b{budget}_{extraction}'
                save_torch(ctx.out/'circuits'/f'{name}.pt',masks);save_json(ctx.out/'circuits'/f'{name}.json',info)
                # Validate on analysis, never tune candidates with official test outcomes.
                clean=ctx.clean('analysis')['logits'];retain={l:m.float() for l,m in enumerate(masks)}
                rem={l:1-m.float() for l,m in enumerate(masks)}
                kept=predict(ctx.get_model(),ctx.data.loader('analysis'),ctx.device,{'edge_masks':retain})['logits']
                removed=predict(ctx.get_model(),ctx.data.loader('analysis'),ctx.device,{'edge_masks':rem})['logits']
                import numpy as np
                margin=lambda z:z[:,target]-z[:,1-target]
                denom=max(float(np.mean(abs(margin(clean)))),1e-8)
                # Definitions in docs/METHODOLOGY.md; both can fall outside [0,1].
                faith=1-float(np.mean(abs(margin(kept)-margin(clean))))/denom
                complete=float(np.mean(abs(margin(removed)-margin(clean))))/denom
                rows.append(dict(seed=ctx.seed,target=target,budget=budget,extraction=extraction,**info,
                    fraction_retained=sum(int(m.sum()) for m in masks)/sum(m.numel() for m in masks),faithfulness=faith,completeness=complete,analysis_examples=scores['counts']['total']))
    # Save a bounded per-layer edge ranking as CSV; dense scores remain in CPU tensor artifact.
    records=[]
    for l,s in enumerate(scores['absolute']):
        for idx in torch.argsort(s.flatten(),descending=True,stable=True)[:100].tolist():
            j,i=divmod(idx,s.shape[1]);records.append(dict(layer=l,source=i,destination=j,absolute=float(s[j,i]),signed=float(scores['signed'][l][j,i])))
    pd.DataFrame(records).to_csv(ctx.out/'edge_rankings.csv',index=False)
    if not rows:
        rows=[dict(seed=ctx.seed,mode='attribution_ranking_only',analysis_examples=scores['counts']['total'],note='No candidate circuit extraction or connectivity claim in this reduced profile')]
    return write_rows(ctx,'e4',rows)

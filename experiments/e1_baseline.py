import pandas as pd
from training.train import train
from experiments.common import write_rows
from analysis.metrics import metrics

def run(ctx):
    path=ctx.out/'checkpoints'
    if not (path/'best.pt').exists() or not (ctx.out/'e1.csv').exists():
        ctx.model=train(ctx.data,ctx.cfg,ctx.seed,ctx.device,path)
    p=ctx.clean();ctx.save_prediction('baseline',p)
    return write_rows(ctx,'e1',[dict(seed=ctx.seed,**metrics(p['y'],p['logits']))])

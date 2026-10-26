import pandas as pd
from experiments.common import write_rows

def run(ctx):
    frame=pd.concat([ctx.conditions('e3'),ctx.conditions('e5')],ignore_index=True)
    frame.to_csv(ctx.out/'e6.csv',index=False)
    # Descriptive performance matching only: selection on validation accuracy, test shown afterward.
    from training.checkpoint import load_torch
    from training.evaluate import predict
    from analysis.metrics import metrics
    values=[]
    for _,row in frame.iterrows():
        mask=load_torch(ctx.out/'masks'/f'{row.condition}.pt')['intervention']
        p=predict(ctx.get_model(),ctx.data.loader('val'),ctx.device,mask)
        values.append(metrics(p['y'],p['logits'])['balanced_accuracy'])
    frame['validation_balanced_accuracy']=values
    matched=[]
    for target in [0,1]:
        for layer in range(3):
            neurons=frame[(frame.kind=='neuron')&(frame.method=='selective')&(frame.target==target)&(frame.layer==layer)]
            edges=frame[(frame.kind=='edge')&(frame.method=='high')&(frame.target==target)&(frame.layer==layer+1)]
            for _,n in neurons.iterrows():
                if edges.empty:continue
                e=edges.loc[(edges.validation_balanced_accuracy-n.validation_balanced_accuracy).abs().idxmin()]
                matched.append(dict(seed=ctx.seed,target=target,layer=layer,neuron_condition=n.condition,edge_condition=e.condition,
                    validation_gap=abs(e.validation_balanced_accuracy-n.validation_balanced_accuracy),neuron_accuracy=n.accuracy,edge_accuracy=e.accuracy))
    pd.DataFrame(matched).to_csv(ctx.out/'performance_matches.csv',index=False)
    return frame

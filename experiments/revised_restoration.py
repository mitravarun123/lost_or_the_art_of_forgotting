"""Same-input patching for final-hidden neuron masks, including random controls."""
from experiments.common import write_rows
from training.checkpoint import load_torch
from analysis.metrics import metrics


def run(ctx):
    frame=ctx.conditions('e3');rows=[]
    clean=ctx.clean();clean_metrics=metrics(clean['y'],clean['logits'])
    for _,source in frame.iterrows():
        intervention=load_torch(ctx.out/'masks'/f'{source.condition}.pt')['intervention']
        row=ctx.evaluate_condition('patch_'+source.condition,intervention,
            dict(source_condition=source.condition,kind='restoration',layer=int(source.layer),
                 target=int(source.target),method=source.method,budget=float(source.budget),
                 repeat=int(source['repeat']),restoration='same_input_final_hidden_patch',
                 ablated_accuracy=float(source.accuracy)),patch_layer=int(source.layer))
        row['clean_accuracy']=clean_metrics['accuracy']
        row['accuracy_recovery']=row['accuracy']-float(source.accuracy)
        row['cat_recall_recovery']=row['cat_recall']-float(source.cat_recall)
        row['dog_recall_recovery']=row['dog_recall']-float(source.dog_recall)
        # Restoring the entire same-input final hidden state reconstitutes the clean readout.
        # This is a computation-restoration control, not evidence of a unique causal circuit.
        rows.append(row)
    return write_rows(ctx,'e8',rows)

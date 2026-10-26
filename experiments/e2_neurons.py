import numpy as np
from interpretability.neuron_selectivity import selectivity

def run(ctx):
    frame,p=selectivity(ctx.get_model(),ctx.data.loader('analysis'),ctx.device)
    frame.to_csv(ctx.out/'e2.csv',index=False)
    np.savez_compressed(ctx.out/'analysis_activations.npz',y=p['y'],**{f'layer_{i}':a for i,a in enumerate(p['representations'])})
    return frame

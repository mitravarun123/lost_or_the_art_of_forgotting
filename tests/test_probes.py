import numpy as np
from probes.probe_training import fit_probe

def test_probe_heldout_and_train_only_scaling():
    rng=np.random.default_rng(3);x=rng.normal(size=(90,4)).astype('float32');y=(x[:,0]>0).astype('int64')
    cfg=dict(batch_size=16,probe_epochs=3,probe_patience=2)
    for family in ['linear','nonlinear']:
        result,state,z=fit_probe((x[:50],y[:50]),(x[50:70],y[50:70]),(x[70:]+10,y[70:]),family,cfg,1,'cpu')
        np.testing.assert_allclose(state['mean'],x[:50].mean(0));assert z.shape==(20,2);assert 0<=result['balanced_accuracy']<=1

import numpy as np
from data.dataset import filter_classes
from data.splits import make_splits,validate_splits

def test_class_mapping():
    x=np.arange(12).reshape(6,2);a,y=filter_classes(x,[0,3,5,3,5,9])
    assert y.tolist()==[0,1,0,1] and a.tolist()==x[[1,2,3,4]].tolist()

def test_no_leakage():
    targets=np.repeat(np.arange(10),5000);s=make_splits(targets);validate_splits(s,targets)
    assert s==make_splits(targets)
    assert s!=make_splits(targets,7)

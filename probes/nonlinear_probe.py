from torch import nn

def nonlinear_probe(width):
    return nn.Sequential(nn.Linear(width,64),nn.ReLU(),nn.Linear(64,2))

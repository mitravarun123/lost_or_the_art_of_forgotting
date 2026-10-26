from torch import nn

def linear_probe(width):
    return nn.Linear(width,2)

from training.evaluate import predict

def extract(model,loader,device,intervention=None):
    """Streaming minibatch forward; returns compact hidden activations, never example-edge arrays."""
    return predict(model,loader,device,intervention)

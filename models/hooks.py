from contextlib import contextmanager

@contextmanager
def capture_outputs(module):
    outputs=[]
    handle=module.register_forward_hook(lambda m,i,o: outputs.append(o.detach().cpu()))
    try: yield outputs
    finally: handle.remove()

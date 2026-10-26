import numpy as np
import torch

def candidate(scores,budget):
    flat=torch.cat([s.flatten() for s in scores]);k=max(1,round(flat.numel()*budget));ids=torch.argsort(flat,descending=True,stable=True)[:k]
    selected=torch.zeros(flat.numel(),dtype=torch.bool);selected[ids]=True;parts=[];start=0
    for score in scores:
        parts.append(selected[start:start+score.numel()].reshape(score.shape));start+=score.numel()
    return parts

def inspect_connectivity(masks):
    """Weak components plus input-to-output reachability for a layered DAG."""
    edges=[];parent={}
    def find(x):
        parent.setdefault(x,x)
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    for l,m in enumerate(masks):
        for j,i in torch.nonzero(m,as_tuple=False).tolist():
            u=(l,i);v=(l+1,j);edges.append((u,v));ru,rv=find(u),find(v);parent[ru]=rv
    components={}
    for u in list(parent):components.setdefault(find(u),[]).append(u)
    forward=set((0,i) for i in range(masks[0].shape[1]));back=set((len(masks),j) for j in range(masks[-1].shape[0]))
    for u,v in edges:
        if u in forward:forward.add(v)
    for u,v in reversed(edges):
        if v in back:back.add(u)
    path_edges=[(u,v) for u,v in edges if u in forward and v in back]
    return dict(edges=len(edges),weak_components=len(components),largest_component=max(map(len,components.values()),default=0),
        input_output_path_edges=len(path_edges),has_input_output_path=bool(path_edges),
        connected=len(components)==1 and bool(edges))

def connected_paths(scores,paths=8):
    """Union of highest-scoring complete input-to-output paths (dynamic programming).
    Paths share the chosen target output, so each target-specific union is connected.
    Two output unions may remain separate; connectivity is reported explicitly.
    """
    masks=[torch.zeros_like(s,dtype=torch.bool) for s in scores]
    for target in range(scores[-1].shape[0]):
        work=[s.clone() for s in scores]
        for _ in range(paths):
            best=torch.zeros(scores[0].shape[1]);backs=[]
            for s in work:
                vals,idx=(s+best[None,:]).max(1);backs.append(idx);best=vals
            j=target
            for l in reversed(range(len(scores))):
                i=int(backs[l][j]);masks[l][j,i]=True;work[l][j,i]=0;j=i
    return masks

import copy, time
import torch
from torch import nn
import pandas as pd
from models.mlp import MLP
from training.checkpoint import seed_all, save_torch, load_torch, save_json
from training.evaluate import evaluate

def train(data,cfg,seed,device,out):
    seed_all(seed);out.mkdir(parents=True,exist_ok=True)
    model=MLP().to(device); opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    start=0;best=float('inf');stale=0;history=[]
    if (out/'last.pt').exists():
        state=load_torch(out/'last.pt');model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
        for state_opt in opt.state.values():
            for k,v in state_opt.items():
                if torch.is_tensor(v): state_opt[k]=v.to(device)
        start=state['epoch']+1;best=state['best'];stale=state['stale'];history=state['history']
        torch.set_rng_state(state['rng']);
        if device.type=='cuda' and state.get('cuda_rng') is not None:torch.cuda.set_rng_state_all(state['cuda_rng'])
    for epoch in range(start,cfg['epochs']):
        if stale>=cfg['patience']:break
        model.train();total=correct=count=0;t0=time.time()
        for x,y in data.loader('train',True,seed+epoch):
            x=x.to(device);y=y.to(device);opt.zero_grad(set_to_none=True);z=model(x)
            loss=nn.functional.cross_entropy(z,y);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),cfg['clip']);opt.step()
            total+=loss.item()*len(y);correct+=(z.argmax(1)==y).sum().item();count+=len(y)
        val,_=evaluate(model,data.loader('val'),device)
        history.append(dict(epoch=epoch,train_loss=total/count,train_accuracy=correct/count,val_loss=val['cross_entropy'],val_accuracy=val['accuracy'],seconds=time.time()-t0))
        if val['cross_entropy']<best:
            best=val['cross_entropy'];stale=0
            save_torch(out/'best.pt',dict(version=1,model={k:v.cpu() for k,v in model.state_dict().items()},seed=seed,epoch=epoch,config=cfg,split=data.manifest))
        else:stale+=1
        save_torch(out/'last.pt',dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch,best=best,stale=stale,history=history,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else None))
        pd.DataFrame(history).to_csv(out/'history.csv',index=False)
        print(f'seed={seed} epoch={epoch+1} train={total/count:.4f} val={val["cross_entropy"]:.4f}',flush=True)
    model.load_state_dict(load_torch(out/'best.pt')['model']);return model

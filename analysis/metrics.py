import numpy as np
from scipy.special import softmax, logsumexp
from sklearn.metrics import accuracy_score,balanced_accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,roc_auc_score

def metrics(y,logits,baseline=None):
    y=np.asarray(y);z=np.asarray(logits);pred=z.argmax(1);p=softmax(z,axis=1)
    recall=recall_score(y,pred,labels=[0,1],average=None,zero_division=0)
    out=dict(accuracy=float(accuracy_score(y,pred)),balanced_accuracy=float(balanced_accuracy_score(y,pred)),
        cat_recall=float(recall[0]),dog_recall=float(recall[1]),precision=float(precision_score(y,pred,average='macro',zero_division=0)),
        macro_f1=float(f1_score(y,pred,average='macro',zero_division=0)),
        cross_entropy=float(np.mean(logsumexp(z,axis=1)-z[np.arange(len(y)),y])),
        roc_auc=float(roc_auc_score(y,p[:,1])), margin=float(np.mean(z[:,0]-z[:,1])),
        cat_margin=float(np.mean((z[:,0]-z[:,1])[y==0])),dog_margin=float(np.mean((z[:,1]-z[:,0])[y==1])),
        confusion_matrix=confusion_matrix(y,pred,labels=[0,1]).tolist())
    if baseline is not None:
        base=np.asarray(baseline);bpred=base.argmax(1)
        out['prediction_flips']=float(np.mean(pred!=bpred))
        out['margin_change']=float(np.mean((z[:,0]-z[:,1])-(base[:,0]-base[:,1])))
        for label,name in [(0,'cat'),(1,'dog')]:
            out[name+'_degradation']=float(np.mean(bpred[y==label]==label)-np.mean(pred[y==label]==label))
    return out

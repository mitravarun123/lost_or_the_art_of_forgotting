"""Eight PNG figure families for the predeclared revised one-model pilot."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from analysis.plots import panel_grid, shared_legend, seed_curve, save, PALETTE, label


def generate(root):
    root=Path(root);out=root/'figures';out.mkdir(parents=True,exist_ok=True)
    n,e,p,r=(read(root,x) for x in ['e3','e5','e7','e8'])
    source=sorted(root.glob('seed_*/checkpoints/history.csv'))
    if source:
        h=pd.read_csv(source[0]);fig,axes=panel_grid(2,'P1 · Baseline training and validation',
            'One training seed. Checkpoint selected by lowest validation loss; the epoch cap is a pilot constraint, not proof of convergence.',cols=2)
        for ax,metric in zip(axes,['loss','accuracy']):
            ax.plot(h.epoch+1,h['train_'+metric],color=PALETTE[0],label='Training')
            ax.plot(h.epoch+1,h['val_'+metric],color=PALETTE[1],ls='--',label='Validation')
            best=h.loc[h.val_loss.idxmin()]
            ax.axvline(best.epoch+1,color='#555555',ls=':',label='Selected checkpoint')
            ax.set(xlabel='Epoch',ylabel='Cross entropy' if metric=='loss' else 'Accuracy',title='Loss' if metric=='loss' else 'Accuracy')
        shared_legend(fig,axes);save(fig,out,'P1_training_curves')
    files=sorted(root.glob('seed_*/predictions/baseline.npz'))
    if files:
        from scipy.special import softmax
        from sklearn.metrics import confusion_matrix
        with np.load(files[0]) as data:y=data['y'];z=data['logits']
        fig,axes=panel_grid(2,'P2 · Baseline on the held-out official test set',
            'Model selection uses validation data only. Confidence is the predicted probability assigned to the true class.',cols=2)
        cm=confusion_matrix(y,z.argmax(1),labels=[0,1]);ax=axes[0];ax.imshow(cm,cmap='Blues');ax.grid(False)
        for i in range(2):
            for j in range(2):ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=17,color='white' if cm[i,j]>.5*cm.max() else '#222222')
        ax.set(title='Confusion matrix',xlabel='Predicted class',ylabel='True class',xticks=[0,1],yticks=[0,1],xticklabels=['Cat','Dog'],yticklabels=['Cat','Dog'])
        prob=softmax(z,axis=1)
        for k,name in [(0,'Cat'),(1,'Dog')]:axes[1].hist(prob[y==k,k],bins=np.linspace(0,1,26),alpha=.5,density=True,color=PALETTE[k],label=name)
        axes[1].set(title='Class-specific confidence',xlabel='Probability of true class',ylabel='Density');shared_legend(fig,axes);save(fig,out,'P2_baseline_performance')
    s=read(root,'e2')
    if len(s):
        s=s[s.layer==2].sort_values('neuron');fig,axes=panel_grid(2,'P3 · Final-hidden-layer neuron selectivity',
            'Analysis examples only. S = mean Cat activation − mean Dog activation. Rankings are fixed before test-set evaluation.',cols=2)
        axes[0].bar(s.neuron,s.selectivity,color=np.where(s.selectivity>=0,PALETTE[0],PALETTE[1]),width=1)
        axes[0].axhline(0,color='#333333',lw=.8);axes[0].set(title='All final-hidden neurons',xlabel='Neuron index',ylabel='Cat − Dog mean activation')
        top=s.iloc[np.argsort(-abs(s.selectivity.to_numpy()))[:12]];xx=np.arange(len(top))
        axes[1].bar(xx-.2,top.cat_mean,width=.4,color=PALETTE[0],label='Cat');axes[1].bar(xx+.2,top.dog_mean,width=.4,color=PALETTE[1],label='Dog')
        axes[1].set(title='Twelve most selective neurons',xlabel='Neuron index, ordered by absolute selectivity',ylabel='Mean activation',xticks=xx,xticklabels=top.neuron.astype(int).astype(str))
        shared_legend(fig,axes);save(fig,out,'P3_neuron_selectivity')
    for frame,name,title in [(n,'P4_neuron_ablation','P4 · Final-hidden neuron ablation'),(e,'P5_edge_ablation','P5 · Output-weight edge ablation')]:
        if frame.empty:continue
        fig,axes=panel_grid(4,title,'One seed; three random masks per budget/target, averaged within the model. No cross-seed confidence intervals. Lower panels show recall for the named target class.',cols=2)
        for column,target in enumerate([0,1]):
            g=frame[frame.target==target];cls='Cat' if target==0 else 'Dog'
            seed_curve(axes[column],g,'budget','accuracy','method',True,True)
            axes[column].set(title=f'Target: {cls}',xlabel='Fraction ablated at this location',ylabel='Test accuracy')
            seed_curve(axes[column+2],g,'budget','cat_recall' if target==0 else 'dog_recall','method',True,True)
            axes[column+2].set(title=f'Target: {cls}',xlabel='Fraction ablated at this location',ylabel=f'{cls} recall')
        shared_legend(fig,axes);save(fig,out,name)
    if len(p):
        q=p[p.condition!='clean'];fig,axes=panel_grid(4,'P6 · Classifier performance versus linear readouts',
            'Neuron panels probe changed hidden states. Output-weight panels probe unchanged hidden states and are readout references, not evidence of survival after hidden-state damage.',cols=2)
        clean=p[p.condition=='clean'];clean_probe=float(clean.balanced_accuracy.iloc[0]);base=read(root,'e1')
        for row,kind in enumerate(['neuron','edge']):
            for column,target in enumerate([0,1]):
                ax=axes[row*2+column];g=q[(q.kind==kind)&(q.target==target)].copy()
                if g.empty:continue
                records=[]
                for _,item in g.iterrows():
                    common=dict(training_seed=item.training_seed,budget=item.budget)
                    records.extend([{**common,'curve':label(item.method)+' — classifier','value':item.ablated_classifier_balanced_accuracy},
                                    {**common,'curve':label(item.method)+' — linear probe','value':item.balanced_accuracy}])
                seed_curve(ax,pd.DataFrame(records),'budget','value','curve',True,True)
                ax.axhline(clean_probe,color='#333333',ls=':',label='Clean-representation linear probe')
                if len(base):ax.axhline(base.balanced_accuracy.iloc[0],color='#777777',ls='--',label='Clean original classifier')
                ax.set(title=f'{"Changed hidden states" if kind=="neuron" else "Unchanged hidden-state reference"}\nTarget: {"Cat" if target==0 else "Dog"}',xlabel='Ablation fraction',ylabel='Test balanced accuracy')
        shared_legend(fig,axes);save(fig,out,'P6_information_recoverability')
        fig,axes=panel_grid(2,'P7 · Functional loss versus post-intervention decodability',
            'Neuron interventions only: the probed final-hidden state is actually modified. Above the diagonal, the learned linear probe outperforms the modified original readout.',cols=2)
        for ax,target in zip(axes,[0,1]):
            for method,color in [('selective',PALETTE[0]),('random',PALETTE[1])]:
                g=q[(q.kind=='neuron')&(q.target==target)&(q.method==method)]
                ax.scatter(g.ablated_classifier_balanced_accuracy,g.balanced_accuracy,s=45,color=color,alpha=.75,label=label(method))
            ax.plot([0,1],[0,1],ls=':',color='#777777',label='Equal performance');ax.set(title=f'Target: {"Cat" if target==0 else "Dog"}',xlabel='Ablated classifier balanced accuracy',ylabel='Post-intervention linear-probe balanced accuracy',xlim=(-.02,1.02),ylim=(-.02,1.02));ax.set_aspect('equal')
        shared_legend(fig,axes);save(fig,out,'P7_function_vs_recoverability')
    if len(r):
        fig,axes=panel_grid(2,'P8 · Same-input activation restoration',
            'The entire final hidden state is patched from the clean run of the same input. Recovery is expected as a computation-restoration sanity check; it does not identify a unique circuit.',cols=2)
        for ax,target in zip(axes,[0,1]):
            g=r[r.target==target];records=[]
            for _,item in g.iterrows():
                common=dict(training_seed=item.training_seed,budget=item.budget)
                records.extend([{**common,'curve':label(item.method)+' — ablated','value':item.ablated_accuracy},
                                {**common,'curve':label(item.method)+' — patched','value':item.accuracy}])
            seed_curve(ax,pd.DataFrame(records),'budget','value','curve',True,True)
            ax.set(title=f'Target: {"Cat" if target==0 else "Dog"}',xlabel='Original neuron ablation fraction',ylabel='Test accuracy')
        shared_legend(fig,axes);save(fig,out,'P8_activation_restoration')
    families={f'P{i}':any(out.glob(f'P{i}_*.png')) for i in range(1,9)}
    (out/'figure_status.json').write_text(json.dumps(dict(format='PNG only',default_dpi=600,families=families,scope='one-seed pilot; no cross-seed claims'),indent=2)+'\n')


def read(root,name):
    from analysis.plots import read as shared_read
    return shared_read(root,name)

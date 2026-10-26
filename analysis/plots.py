"""PNG-only figures with explicit conditions, readable panels and seed-level intervals.

No measured figure is created without its source CSV/NPZ. ``LOI_FIGURE_DPI``
can lower resolution for software tests; research/notebook default is 600 DPI.
"""
import json
import math
import os
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import PercentFormatter, MaxNLocator
import numpy as np
import pandas as pd
from scipy.stats import t

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.titlesize': 12,
    'axes.labelsize': 11, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'legend.fontsize': 10, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.titlepad': 12, 'axes.labelpad': 8, 'axes.grid': True,
    'grid.alpha': .18, 'grid.linewidth': .7, 'lines.linewidth': 2,
    'figure.facecolor': 'white', 'savefig.facecolor': 'white',
})
LABELS = {
    'selective': 'Class selectivity', 'gradient': 'Activation × gradient',
    'random': 'Uniform random', 'matched': 'Magnitude-matched random',
    'low': 'Low-ranked', 'high': 'High edge attribution',
    'linear': 'Linear probe', 'nonlinear': 'Nonlinear probe (64 units)',
    'neuron': 'Neuron ablation', 'edge': 'Edge ablation',
    'selected': 'Attribution-based restoration', 'activation_patch': 'Same-input activation patch',
    'fully_ablated': 'Fully ablated classifier', 'clean': 'Unmodified reference',
    'connected_paths': 'Complete-path support', 'ranked': 'Top-ranked support',
}
PALETTE = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#333333']


def label(value):
    return LABELS.get(str(value), str(value).replace('_', ' '))


def save(fig, out, name):
    """Save PNG only, and record resolution and text-boundary checks for review."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    dpi = int(os.environ.get('LOI_FIGURE_DPI', '600'))
    if not 72 <= dpi <= 1200:
        raise ValueError('LOI_FIGURE_DPI must be between 72 and 1200')
    for ax in fig.axes:
        for getter, setter in [(ax.get_xlabel, ax.set_xlabel), (ax.get_ylabel, ax.set_ylabel)]:
            value = getter()
            if len(value)>42 and '\n' not in value:
                setter(textwrap.fill(value, width=38))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    # Axis locators may create off-range tick artists which Matplotlib never draws.
    # Exclude those artists rather than flagging invisible tick labels as clipping.
    ignored = set()
    for ax in fig.axes:
        if not ax.get_visible():
            ignored.update(id(a) for a in ax.findobj(match=matplotlib.text.Text))
        for axis, bounds in [(ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())]:
            low, high = sorted(bounds)
            for tick in axis.get_major_ticks()+axis.get_minor_ticks():
                if tick.get_loc() < low-1e-10 or tick.get_loc() > high+1e-10:
                    ignored.update([id(tick.label1), id(tick.label2)])
    # constrained_layout reserves titles, axes, colour bars and the shared legend.
    for artist in fig.findobj(match=matplotlib.text.Text):
        if id(artist) in ignored or not artist.get_visible() or not artist.get_text().strip():
            continue
        box = artist.get_window_extent(renderer)
        if not np.isfinite(box.bounds).all():
            continue
        if box.x0 < -2 or box.y0 < -2 or box.x1 > fig.bbox.x1+2 or box.y1 > fig.bbox.y1+2:
            outside.append(artist.get_text())
    path = out / f'{name}.png'
    fig.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=.22)
    from PIL import Image
    with Image.open(path) as image:
        size = list(image.size)
    record = dict(file=path.name, dpi=dpi, pixels=size,
                  text_outside_layout=outside,
                  note='Boundary check only; visual inspection is still required for text/marker overlap.')
    manifest = out / 'figure_manifest.json'
    entries = json.loads(manifest.read_text()) if manifest.exists() else {}
    entries[path.name] = record
    manifest.write_text(json.dumps(entries, indent=2)+'\n')
    plt.close(fig)


def read(root, experiment):
    frames = []
    for path in sorted(Path(root).glob(f'seed_*/{experiment}.csv')):
        if path.stat().st_size > 2:
            frames.append(pd.read_csv(path).assign(training_seed=int(path.parent.name.split('_')[-1])))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def panel_grid(count, title, note='', cols=3, height=3.7):
    cols = min(cols, max(1, count))
    if count==4 and cols==3:cols=2
    rows = math.ceil(count/cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.1*cols+2.1, height*rows+1.0),
                             squeeze=False, layout='constrained')
    width = fig.get_figwidth()
    fig.suptitle(textwrap.fill(title, width=min(120, int(width*7))), fontsize=16, fontweight='bold')
    if note:
        fig.supxlabel('\n'.join(textwrap.wrap(note, width=min(145, int(width*11)))), fontsize=10, color='#465260')
    flat = axes.ravel()
    for ax in flat[count:]:
        ax.set_visible(False)
    return fig, flat[:count]


def shared_legend(fig, axes):
    unique = {}
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        for h, name in zip(handles, labels):
            unique.setdefault(name, h)
    if unique:
        fig.legend(unique.values(), unique.keys(), loc='outside right upper',
                   frameon=False, title='Condition', borderaxespad=1.2)


def seed_curve(ax, frame, x, y, group, percent_x=False, probability_y=False):
    for index, (name, g) in enumerate(frame.groupby(group, dropna=False, sort=False)):
        # Collapse repeated random masks within a seed before using models as replicates.
        per_seed = g.groupby(['training_seed', x], dropna=False)[y].mean().reset_index()
        s = per_seed.groupby(x)[y].agg(['mean', 'std', 'count']).sort_index()
        if s.empty:
            continue
        xx = s.index.to_numpy(dtype=float); yy = s['mean'].to_numpy(dtype=float)
        color = PALETTE[index % len(PALETTE)]
        ax.plot(xx, yy, marker='o', ms=5, color=color, label=label(name))
        valid = (s['count'] > 1).to_numpy()
        half = (t.ppf(.975, s['count']-1)*s['std']/np.sqrt(s['count'])).to_numpy()
        if valid.any():
            ax.fill_between(xx, yy-half, yy+half, where=valid, color=color, alpha=.14)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ticks = np.sort(frame[x].dropna().unique().astype(float))
    if 0 < len(ticks) <= 8:
        ax.set_xticks(ticks)
    if percent_x:
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=1))
    if probability_y:
        ax.set_ylim(-.025, 1.025); ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.margins(x=.07)


def intervention_panels(frame, out, prefix, title, y='accuracy', ylabel='Test accuracy'):
    if frame.empty or y not in frame:
        return
    for target, g in frame.groupby('target'):
        layers = sorted(g.layer.unique())
        fig, axes = panel_grid(len(layers), f'{title} — target: {"Cat" if target==0 else "Dog"}',
            'Each panel is one intervention location. Lines: mean across training seeds; shading: 95% Student-t interval. Random repetitions are averaged within each seed. No interval for one seed.')
        for ax, layer in zip(axes, layers):
            sub = g[g.layer==layer]
            seed_curve(ax, sub, 'budget', y, 'method', True, y in ['accuracy','cat_recall','dog_recall'])
            location = 'All weight layers' if layer==-1 else ('Output weights' if layer==3 else f'{"Hidden activations" if prefix.startswith("F4") else "Weights into hidden layer"} {int(layer)+1}')
            ax.set(title=location, xlabel='Fraction ablated at this location', ylabel=ylabel)
        shared_legend(fig, axes); save(fig, out, f'{prefix}_target_{int(target)}')


def architecture(out):
    fig, ax = plt.subplots(figsize=(15, 3.6), layout='constrained'); ax.axis('off')
    fig.suptitle('F1 · Research design: functional loss versus recoverability', fontsize=17, fontweight='bold')
    boxes = [
        ('DATA', 'CIFAR-10: Cat / Dog\nTrain · validation · analysis\nOfficial test held out'),
        ('MODEL', 'MLP: 3072 → 512 → 256\n→ 128 → 2; hidden ReLU\nFive independent seeds'),
        ('INTERVENE', 'Neuron activation masks\nWeight-edge masks\nAnalysis-based selection'),
        ('EVALUATE', 'Classification and margins\nPost-intervention probes\nOriginal-circuit restoration'),
    ]
    for i, (head, text) in enumerate(boxes):
        x=.015+i*.25
        ax.add_patch(FancyBboxPatch((x,.22),.215,.58,boxstyle='round,pad=0.008',
                     transform=ax.transAxes,facecolor='#eaf2f8',edgecolor='#46708f',linewidth=1.4))
        ax.text(x+.1075,.69,head,transform=ax.transAxes,ha='center',fontsize=11,fontweight='bold',color='#174966')
        ax.text(x+.1075,.47,text,transform=ax.transAxes,ha='center',va='center',fontsize=10.5,linespacing=1.7)
        if i<3:
            ax.annotate('',xy=(x+.243,.51),xytext=(x+.221,.51),xycoords='axes fraction',arrowprops=dict(arrowstyle='->',lw=1.8,color='#46708f'))
    fig.supxlabel('Protocol schematic only. Classification failure does not establish information erasure.',fontsize=11)
    save(fig,out,'F1_architecture')


def learning(root,out):
    files=sorted(root.glob('seed_*/checkpoints/history.csv'))
    if not files:return
    fig,axes=panel_grid(2,'F2 · Learning curves by independent training seed',
                         'Colours identify training seeds. Solid = model training; dashed = validation. Checkpoint selection uses validation loss.',cols=2)
    for i,path in enumerate(files):
        h=pd.read_csv(path);color=PALETTE[i%len(PALETTE)];seed=path.parents[1].name.replace('seed_','')
        for ax,metric in zip(axes,['loss','accuracy']):
            ax.plot(h.epoch+1,h['train_'+metric],color=color,label=f'Seed {seed}: train')
            ax.plot(h.epoch+1,h['val_'+metric],color=color,ls='--',label=f'Seed {seed}: validation')
            ax.set(title='Cross-entropy loss' if metric=='loss' else 'Classification accuracy',xlabel='Epoch',ylabel='Cross entropy' if metric=='loss' else 'Accuracy')
            if metric=='accuracy':ax.yaxis.set_major_formatter(PercentFormatter(1))
    shared_legend(fig,axes);save(fig,out,'F2_learning_curves')


def selectivity(root,out,frame):
    if frame.empty:return
    # Neuron identities are not aligned across independently trained models.
    seed=int(frame.training_seed.min());g=frame[frame.training_seed==seed]
    layers=sorted(g.layer.unique());fig,axes=panel_grid(len(layers),f'F3 · Neuron selectivity — training seed {seed}',
        'S = mean activation for Cat − mean activation for Dog, using analysis examples only. Units from different models are not aligned.',height=3.5)
    for ax,layer in zip(axes,layers):
        sub=g[g.layer==layer].sort_values('neuron')
        ax.bar(sub.neuron,sub.selectivity,width=1,color=np.where(sub.selectivity>=0,'#0072B2','#D55E00'))
        ax.axhline(0,color='#333333',lw=.8);ax.set(title=f'Hidden layer {int(layer)+1} ({len(sub)} neurons)',xlabel='Neuron index',ylabel='Cat − Dog activation')
    save(fig,out,'F3_selectivity')
    width=max(int(g[g.layer==l].neuron.max())+1 for l in layers)
    matrix=np.full((len(layers),width),np.nan)
    for row,layer in enumerate(layers):
        sub=g[g.layer==layer];matrix[row,sub.neuron.astype(int)]=sub.selectivity
    fig,axes=panel_grid(1,f'F3a · Selectivity heatmap — training seed {seed}',
        'Grey cells are absent neurons, not zero selectivity. Blue = Cat preference; red = Dog preference.',cols=1,height=3)
    ax=axes[0];cmap=plt.get_cmap('RdBu').copy();cmap.set_bad('#e4e7eb');limit=max(float(np.nanmax(abs(matrix))),1e-8)
    im=ax.imshow(matrix,aspect='auto',cmap=cmap,vmin=-limit,vmax=limit,interpolation='nearest')
    ax.set(xlabel='Neuron index',ylabel='Hidden layer',yticks=range(len(layers)),yticklabels=[str(int(l)+1) for l in layers]);ax.grid(False)
    fig.colorbar(im,ax=ax,label='Cat − Dog mean activation',pad=.04);save(fig,out,'F3a_heatmap')
    file=root/f'seed_{seed}/analysis_activations.npz'
    if file.exists():
        with np.load(file) as p:
            fig,axes=panel_grid(len(layers),f'F3b · Class-conditioned activation distributions — seed {seed}',
                'The most selective neuron in each layer is chosen using analysis data only. Histograms show analysis examples.',height=3.6)
            for l,ax in zip(layers,axes):
                a=p[f'layer_{int(l)}'];y=p['y'];score=abs(a[y==0].mean(0)-a[y==1].mean(0));j=int(score.argmax())
                for k,name in [(0,'Cat'),(1,'Dog')]:
                    ax.hist(a[y==k,j],bins=25,density=True,alpha=.5,color=PALETTE[k],label=name)
                ax.set(title=f'Hidden layer {int(l)+1}, neuron {j}',xlabel='Post-ReLU activation',ylabel='Density')
            shared_legend(fig,axes);save(fig,out,'F3b_class_activations')


def circuits(root,out,frame):
    if frame.empty:return
    for target,g in frame.groupby('target'):
        fig,axes=panel_grid(3,f'F5 · Candidate circuit validation — {"Cat" if target==0 else "Dog"}',
            'Analysis-set measurements. Actual retained-edge fraction is shown because complete-path extraction can deduplicate edges. Attribution alone is not causal evidence.')
        for ax,y,ylab in zip(axes,['faithfulness','completeness','input_output_path_edges'],
                             ['Margin faithfulness (higher is closer)','Removal sensitivity / completeness','Edges on input-to-output paths']):
            seed_curve(ax,g,'fraction_retained',y,'extraction',True)
            ax.set(xlabel='Actual fraction of weight edges retained',ylabel=ylab,title=ylab.split(' (')[0])
        shared_legend(fig,axes);save(fig,out,f'F5_circuit_connectivity_target_{int(target)}')
    files=sorted(root.glob('seed_*/edge_rankings.csv'))
    if files:
        f=pd.read_csv(files[0]);layers=sorted(f.layer.unique())
        fig,axes=panel_grid(len(layers),f'F5b · Highest-attribution edges — {files[0].parent.name.replace("_"," ")}',
            'One panel per weight layer. Each dot is an observed edge; colour encodes its absolute gradient-times-contribution score.')
        for ax,l in zip(axes,layers):
            sub=f[f.layer==l];im=ax.scatter(sub.source,sub.destination,c=np.log10(sub.absolute.clip(lower=1e-12)),cmap='viridis',s=18,alpha=.8)
            ax.set(title='Output weights' if l==3 else f'Weights into hidden layer {int(l)+1}',xlabel='Source unit index',ylabel='Destination unit index')
            fig.colorbar(im,ax=ax,label='log₁₀ absolute attribution',shrink=.8,pad=.03)
        save(fig,out,'F5b_edge_attribution')


def comparisons(out,n,e):
    if n.empty or e.empty:return
    # Restrict to targeted conditions; do not quietly mix them with random controls.
    n=n[n.method=='selective'].copy();e=e[e.method=='high'].copy()
    for target in sorted(n.target.unique()):
        fig,axes=panel_grid(3,f'F7 · Neuron versus edge interventions — {"Cat" if target==0 else "Dog"}',
            'Neuron layer L is compared with its outgoing weight layer L+1. Points show measured conditions; counts are not treated as equivalent intervention budgets.')
        for ax,l in zip(axes,range(3)):
            for g,name,color in [(n[(n.target==target)&(n.layer==l)],'Neuron ablation',PALETTE[0]),(e[(e.target==target)&(e.layer==l+1)],'Outgoing-edge ablation',PALETTE[1])]:
                ax.scatter(g.affected_connection_fraction,g.accuracy,s=25,alpha=.55,label=name,color=color)
            ax.set(title=f'Hidden layer {l+1} / outgoing weights',xlabel='Fraction of all weight connections affected',ylabel='Test accuracy',ylim=(-.025,1.025))
            ax.xaxis.set_major_formatter(PercentFormatter(1));ax.yaxis.set_major_formatter(PercentFormatter(1))
        shared_legend(fig,axes);save(fig,out,f'F7_neuron_vs_edge_target_{int(target)}')
        fig,axes=panel_grid(3,f'F7b · Representation perturbation — {"Cat" if target==0 else "Dog"}',
            'Final-hidden relative perturbation = ||A′ − A||F / max(||A||F, 10⁻¹²). Each point is one targeted intervention and training seed.')
        for ax,l in zip(axes,range(3)):
            for g,name,color in [(n[(n.target==target)&(n.layer==l)],'Neuron ablation',PALETTE[0]),(e[(e.target==target)&(e.layer==l+1)],'Outgoing-edge ablation',PALETTE[1])]:
                ax.scatter(g.removed_weight_fraction,g.representation_delta_l2,s=25,alpha=.5,color=color,label=name)
            ax.set(title=f'Hidden layer {l+1} / outgoing weights',xlabel='Fraction of absolute weight magnitude affected',ylabel='Final-hidden relative perturbation');ax.xaxis.set_major_formatter(PercentFormatter(1))
        shared_legend(fig,axes);save(fig,out,f'F7b_representation_target_{int(target)}')


def probes(root,out,p):
    if p.empty:return
    if 'representation_status' in p:
        p=p[p.representation_status!='unaffected_pre_readout_reference']
    q=p[p.condition!='clean'].copy()
    # Primary overview: global high-edge and final-hidden selective-neuron conditions.
    selected=q[((q.kind=='edge')&(q.method=='high')&(q.layer==-1))|((q.kind=='neuron')&(q.method=='selective')&(q.layer==2))]
    selected=selected[selected.probe_layer==2]
    for target,g in selected.groupby('target'):
        conditions=list(g.groupby(['kind','family'],sort=True))
        fig,axes=panel_grid(len(conditions),f'F8 · Original readout versus learned probes — {"Cat" if target==0 else "Dog"}',
            'Final-hidden representations only. High-attribution global edge masks or final-hidden selective neuron masks. Clean probes use the same family and dimension.')
        for ax,((kind,family),sub) in zip(axes,conditions):
            sub=sub.copy();sub['curve']='Post-intervention probe'
            seed_curve(ax,sub,'budget','balanced_accuracy','curve',True,True)
            for metric,title,color,style in [('ablated_classifier_balanced_accuracy','Ablated classifier','#D55E00','--'),('clean_probe_balanced_accuracy','Clean-representation probe','#333333',':')]:
                if metric in sub:
                    per=sub.groupby(['training_seed','budget'])[metric].mean().reset_index();mean=per.groupby('budget')[metric].mean()
                    ax.plot(mean.index,mean.values,color=color,ls=style,marker='s',ms=4,label=title)
            ax.set(title=f'{label(kind)} · {label(family)}',xlabel='Ablated fraction at selected location',ylabel='Test balanced accuracy')
        shared_legend(fig,axes);save(fig,out,f'F8_probe_performance_target_{int(target)}')
    for family,g in q[q.probe_layer==2].groupby('family'):
        fig,axes=panel_grid(2,f'F9 · Functional performance and recoverability — {label(family)}',
            'Final-hidden post-intervention probes only; unaffected pre-readout references excluded. Above the diagonal: probe performs better than the modified original classifier.',cols=2)
        for ax,target in zip(axes,[0,1]):
            for kind,color in [('neuron',PALETTE[0]),('edge',PALETTE[1])]:
                sub=g[(g.target==target)&(g.kind==kind)]
                ax.scatter(sub.ablated_classifier_balanced_accuracy,sub.balanced_accuracy,s=20,alpha=.35,color=color,label=label(kind))
            ax.plot([0,1],[0,1],ls=':',color='#777777',label='Equal performance');ax.set(title=f'Target: {"Cat" if target==0 else "Dog"}',xlabel='Ablated classifier: balanced accuracy',ylabel='Post-intervention probe: balanced accuracy',xlim=(-.025,1.025),ylim=(-.025,1.025));ax.set_aspect('equal');ax.xaxis.set_major_formatter(PercentFormatter(1));ax.yaxis.set_major_formatter(PercentFormatter(1))
        shared_legend(fig,axes);save(fig,out,f'F9_function_vs_recoverability_{family}')
    # No aggregation over different probe dimensions/locations in layer-wise figures.
    for (kind,target,family),g in q[q.method.isin(['selective','high'])].groupby(['kind','target','family']):
        locations=sorted(g.layer.unique());fig,axes=panel_grid(len(locations),f'F10 · Layer-wise decodability — {label(kind)} · {"Cat" if target==0 else "Dog"} · {label(family)}',
            'Each curve fixes the intervention budget. Hidden states at and after the affected layer are evaluated; output-only masks are excluded.')
        for ax,location in zip(axes,locations):
            sub=g[g.layer==location].copy();sub['budget_label']=sub.budget.map(lambda b:f'{100*b:g}% ablation')
            seed_curve(ax,sub,'probe_layer','balanced_accuracy','budget_label',False,True)
            ax.set(title='All weight layers' if location==-1 else f'Intervention location {int(location)+1}',xlabel='Probed hidden layer (1–3)',ylabel='Probe balanced accuracy',xticks=[0,1,2],xticklabels=['1','2','3'])
        shared_legend(fig,axes);save(fig,out,f'F10_layer_recovery_{kind}_target_{int(target)}_{family}')
    matched=[]
    for file in sorted(root.glob('seed_*/performance_matches.csv')):
        if file.stat().st_size<5:continue
        for _,pair in pd.read_csv(file).iterrows():
            seed=int(file.parent.name.split('_')[-1])
            for family in q.family.unique():
                a=q[(q.training_seed==seed)&(q.condition==pair.neuron_condition)&(q.family==family)&(q.probe_layer==2)]
                b=q[(q.training_seed==seed)&(q.condition==pair.edge_condition)&(q.family==family)&(q.probe_layer==2)]
                if len(a) and len(b):matched.append(dict(seed=seed,family=family,target=pair.target,validation_gap=pair.validation_gap,neuron_probe=float(a.iloc[0].balanced_accuracy),edge_probe=float(b.iloc[0].balanced_accuracy)))
    if matched:
        mf=pd.DataFrame(matched);mf.to_csv(root/'performance_matched_probes.csv',index=False)
        for family,g in mf.groupby('family'):
            fig,axes=panel_grid(1,f'F9b · Validation-matched recoverability — {label(family)}',
                'Pairs selected using validation accuracy only. Colour reports the residual validation matching gap; no test-based matching.',cols=1)
            ax=axes[0];im=ax.scatter(g.neuron_probe,g.edge_probe,c=g.validation_gap,cmap='viridis',s=45,edgecolor='white',lw=.5)
            ax.plot([0,1],[0,1],ls=':',color='#777777');ax.set(xlabel='Neuron ablation: final-hidden probe balanced accuracy',ylabel='Edge ablation: final-hidden probe balanced accuracy',xlim=(-.025,1.025),ylim=(-.025,1.025));ax.set_aspect('equal');fig.colorbar(im,ax=ax,label='Absolute validation accuracy gap',pad=.04)
            save(fig,out,f'F9b_performance_matched_recoverability_{family}')


def restoration(out,r):
    if r.empty:return
    # Restore a fixed, explicitly labelled largest predeclared budget at each location.
    for target,g in r.groupby('target'):
        budget=float(g.budget.max());g=g[np.isclose(g.budget,budget)]
        layers=sorted(g.layer.unique());fig,axes=panel_grid(len(layers),f'F11 · Circuit restoration — {"Cat" if target==0 else "Dog"} · {100*budget:g}% initial ablation',
            'Selected/random edge restoration of the same original ablation. The highest configured budget is shown, not a test-selected condition. Other budgets remain in E8 CSV.')
        for ax,location in zip(axes,layers):
            sub=g[(g.layer==location)&(g.method.isin(['selected','random']))]
            seed_curve(ax,sub,'restored_fraction','accuracy','method',True,True)
            for method,color,style in [('clean','#333333',':'),('fully_ablated','#777777','--')]:
                ref=g[(g.layer==location)&(g.method==method)]
                if len(ref):ax.axhline(ref.groupby('training_seed').accuracy.mean().mean(),color=color,ls=style,label=label(method))
            ax.set(title='All weight layers' if location==-1 else ('Output weights' if location==3 else f'Weight layer {int(location)+1}'),xlabel='Fraction of removed edges restored',ylabel='Test accuracy')
        shared_legend(fig,axes);save(fig,out,f'F11_restoration_target_{int(target)}')
    patch=r[r.method=='activation_patch']
    if len(patch):
        for target,g in patch.groupby('target'):
            budget=float(g.budget.max());g=g[np.isclose(g.budget,budget)].copy()
            # Source condition records original intervention location; patch layer is a separate axis.
            g['source_location']=g.source_condition.str.extract(r'edge_l(-?\d+)_')[0].map(lambda l:'All weight layers' if l=='-1' else f'Original edge layer {int(l)+1}')
            fig,axes=panel_grid(1,f'F11b · Same-input activation restoration — {"Cat" if target==0 else "Dog"}',
                f'Initial ablation budget = {100*budget:g}%. Same input under clean and ablated network. Downstream masked edges may still prevent recovery.',cols=1)
            seed_curve(axes[0],g,'layer','accuracy','source_location',False,True);axes[0].set(xlabel='Patched hidden layer',ylabel='Test accuracy',xticks=[0,1,2],xticklabels=['1','2','3'])
            shared_legend(fig,axes);save(fig,out,f'F11b_activation_patching_target_{int(target)}')


def robustness(root,out):
    path=root/'robustness.csv'
    if not path.exists() or path.stat().st_size<5:return
    f=pd.read_csv(path);f=f[(f.metric=='accuracy')&(f.n_seeds>1)]
    f=f[((f.experiment=='e3')&(f.method=='selective'))|((f.experiment=='e5')&(f.method=='high'))]
    for (experiment,target),g in f.groupby(['experiment','target']):
        layers=sorted(g.layer.unique());fig,axes=panel_grid(len(layers),f'F12 · Cross-seed robustness — {experiment.upper()} · {"Cat" if target==0 else "Dog"}',
            'Each error bar is a 95% Student-t interval over independent training seeds. Random masks and test images are not counted as additional training seeds.')
        for ax,l in zip(axes,layers):
            sub=g[g.layer==l].sort_values('budget');ax.errorbar(sub.budget,sub['mean'],yerr=[sub['mean']-sub.ci_low,sub.ci_high-sub['mean']],fmt='o-',capsize=4,color=PALETTE[0],label=f'{int(sub.n_seeds.min())} training seeds')
            ax.set(title='All weight layers' if l==-1 else f'Intervention location {int(l)+1}',xlabel='Ablation budget',ylabel='Test accuracy');ax.xaxis.set_major_formatter(PercentFormatter(1));ax.yaxis.set_major_formatter(PercentFormatter(1))
        shared_legend(fig,axes);save(fig,out,f'F12_robustness_{experiment}_target_{int(target)}')


def baseline(root,out):
    files=sorted(root.glob('seed_*/predictions/baseline.npz'))
    if not files:return
    from scipy.special import softmax
    from sklearn.metrics import confusion_matrix
    fig,axes=panel_grid(len(files),'Baseline · Confusion matrices for each training seed',
        'Each test image is counted once per independently trained model. Counts are not pooled across models.',cols=3)
    for ax,file in zip(axes,files):
        with np.load(file) as p:cm=confusion_matrix(p['y'],p['logits'].argmax(1),labels=[0,1])
        im=ax.imshow(cm,cmap='Blues',vmin=0,vmax=max(cm.max(),1));ax.grid(False)
        for i in range(2):
            for j in range(2):ax.text(j,i,f'{cm[i,j]:,}',ha='center',va='center',fontsize=15,fontweight='bold',color='white' if cm[i,j]>.5*cm.max() else '#192c3b')
        ax.set(title=file.parent.parent.name.replace('_',' '),xticks=[0,1],yticks=[0,1],xticklabels=['Cat','Dog'],yticklabels=['Cat','Dog'],xlabel='Predicted class',ylabel='True class')
    save(fig,out,'baseline_confusion_matrices')
    fig,axes=panel_grid(len(files),'Baseline · True-class confidence distributions',
        'Predicted probability assigned to the actual class. Each panel uses the official held-out test images for one trained model.',cols=3)
    for ax,file in zip(axes,files):
        with np.load(file) as p:
            prob=softmax(p['logits'],axis=1)
            for k,name in [(0,'Cat'),(1,'Dog')]:ax.hist(prob[p['y']==k,k],bins=np.linspace(0,1,26),alpha=.5,density=True,color=PALETTE[k],label=name)
        ax.set(title=file.parent.parent.name.replace('_',' '),xlabel='Probability assigned to the true class',ylabel='Density',xlim=(0,1))
    shared_legend(fig,axes);save(fig,out,'baseline_confidence')


def generate(root):
    root=Path(root)
    if (root/'config.json').exists() and json.loads((root/'config.json').read_text()).get('report_mode')=='revised':
        from analysis.revised_plots import generate as revised_generate
        return revised_generate(root)
    out=root/'figures';out.mkdir(parents=True,exist_ok=True)
    architecture(out);learning(root,out)
    selectivity(root,out,read(root,'e2'))
    n,e,p,r=(read(root,exp) for exp in ['e3','e5','e7','e8'])
    for f,prefix,title in [(n,'F4_neuron_ablation','F4 · Neuron ablation'),(e,'F6_edge_ablation','F6 · Edge ablation')]:
        for metric,suffix,ylab in [('accuracy','','Test accuracy'),('cat_recall','_cat_recall','Cat recall'),('dog_recall','_dog_recall','Dog recall'),('target_degradation','_target_degradation','Target-class recall degradation')]:
            intervention_panels(f,out,prefix+suffix,title,metric,ylab)
    circuits(root,out,read(root,'e4'));comparisons(out,n,e);probes(root,out,p);restoration(out,r);robustness(root,out);baseline(root,out)
    available={p.name for p in out.glob('*.png')}
    required={f'F{i}':any(name.startswith(f'F{i}_') or name.startswith(f'F{i}a_') or name.startswith(f'F{i}b_') for name in available) for i in range(1,13)}
    (out/'figure_status.json').write_text(json.dumps({'format':'PNG only','default_dpi':600,'families':required,'missing_reason':'Families without executed source data or cross-seed replications are not fabricated.'},indent=2)+'\n')

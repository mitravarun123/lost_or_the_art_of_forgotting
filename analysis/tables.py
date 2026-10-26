from pathlib import Path
import pandas as pd

TABLES={'T2':'e1','T3':'e2','T4':'e3','T5':'e4','T6':'e5','T7':'e6','T8':'e7','T9':'e8'}
def generate(root,cfg):
    root=Path(root);out=root/'tables';out.mkdir(exist_ok=True)
    architecture=pd.DataFrame({'parameter':['Architecture','Optimizer','Learning rate','Batch size','Maximum epochs','Patience','Clip','Seeds'],
        'value':['3072–512–256–128–2','AdamW',cfg['learning_rate'],cfg['batch_size'],cfg['epochs'],cfg['patience'],cfg['clip'],str(cfg['seeds'])]})
    architecture.to_csv(out/'T1.csv',index=False);(out/'T1.tex').write_text(architecture.to_latex(index=False,escape=True))
    for name,exp in TABLES.items():
        files=sorted(root.glob(f'seed_*/{exp}.csv'))
        if files:
            frame=pd.concat([pd.read_csv(f).assign(training_seed=int(f.parent.name.split('_')[-1])) for f in files],ignore_index=True)
            frame.to_csv(out/f'{name}.csv',index=False)
            # Compact numeric summaries for manuscript; full records in CSV.
            keys=[c for c in ['method','kind','layer','budget','family','probe_layer','extraction'] if c in frame]
            cols=[c for c in ['accuracy','balanced_accuracy','cat_recall','dog_recall','faithfulness','completeness','absolute_difference','probe_improvement'] if c in frame]
            if cols:
                if keys:
                    # First collapse repeats and conditions within a training seed.
                    byseed=frame.groupby(keys+['training_seed'],dropna=False)[cols].mean().reset_index()
                    summary=byseed.groupby(keys,dropna=False)[cols].agg(['mean','std']).reset_index()
                else:summary=frame[cols].agg(['mean','std']).T.reset_index()
            else:summary=frame.select_dtypes('number').describe().T.reset_index()
            (out/f'{name}.tex').write_text(summary.to_latex(index=False,float_format='%.4f',escape=True,longtable=True))
        else:(out/f'{name}.tex').write_text(r'\textit{Not run.}\par')
    file=root/'robustness.csv'
    if file.exists() and file.stat().st_size>2:
        f=pd.read_csv(file);f.to_csv(out/'T10.csv',index=False);(out/'T10.tex').write_text(f.to_latex(index=False,float_format='%.4f',escape=True,longtable=True))
    else:(out/'T10.tex').write_text(r'\textit{Not run.}\par')

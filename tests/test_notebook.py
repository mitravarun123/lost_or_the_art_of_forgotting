"""Static execution-contract checks for the user-run Colab notebook."""
import ast
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def test_master_notebook_is_parseable_and_runs_full_profile():
    notebook=json.loads((ROOT/'notebooks/MASTER_Colab_Whole_Experiment.ipynb').read_text())
    sources=[]
    for cell in notebook['cells']:
        if cell['cell_type']=='code':
            source=''.join(cell['source']);ast.parse(source);sources.append(source)
    source='\n'.join(sources)
    assert "EXPERIMENT = 'all'" in source
    assert "FIGURE_DPI = 600" in source
    assert "--skip-report" in source and "--report-only" in source
    assert "files.upload()" in source and "drive.mount(" in source
    assert "LAST_RUN.txt" in source and "LAST_PROJECT_ZIP.txt" in source
    assert "archive.write(path, path.name)" in source
    assert 'restart' not in source.lower() or 'runtime' in source.lower()
    assert "full_complete" in source and "e9.done.json" in source

def test_png_export_resolution_and_no_pdf(tmp_path,monkeypatch):
    from analysis.plots import architecture
    from PIL import Image
    monkeypatch.setenv('LOI_FIGURE_DPI','600')
    architecture(tmp_path)
    with Image.open(tmp_path/'F1_architecture.png') as image:
        assert image.width>=8000 and image.height>=1500
        assert abs(image.info['dpi'][0]-600)<1
    assert not list(tmp_path.glob('*.pdf'))
    report=json.loads((tmp_path/'figure_manifest.json').read_text())
    assert not report['F1_architecture.png']['text_outside_layout']

def test_report_only_does_not_load_dataset(tmp_path):
    import subprocess,sys
    from run import configuration
    (tmp_path/'config.json').write_text(json.dumps(configuration('colab')))
    completed=subprocess.run([sys.executable,str(ROOT/'run.py'),'--report-only',str(tmp_path),'--figure-dpi','100'],cwd=ROOT,capture_output=True,text=True)
    assert completed.returncode==0, completed.stderr
    assert (tmp_path/'figures/F1_architecture.png').exists()
    assert (tmp_path/'e10.done.json').exists()
    assert not list(tmp_path.glob('seed_*'))

def test_cross_seed_figure_has_explicit_conditions(tmp_path):
    import pandas as pd
    from analysis.plots import robustness
    rows=[dict(experiment=experiment,metric='accuracy',method=method,target=target,layer=layer,budget=budget,n_seeds=5,mean=.7,std=.05,ci_low=.64,ci_high=.76)
          for experiment,method in [('e3','selective'),('e5','high')]
          for target in [0,1] for layer in [0,1,2] for budget in [.01,.05]]
    pd.DataFrame(rows).to_csv(tmp_path/'robustness.csv',index=False)
    out=tmp_path/'figures';out.mkdir()
    robustness(tmp_path,out)
    assert len(list(out.glob('F12_robustness_*.png')))==4
    assert not list(out.glob('*.pdf'))

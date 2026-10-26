import argparse,json,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('run');args=p.parse_args();root=Path(args.run).resolve();paper=Path(__file__).resolve().parents[1]/'paper'
if not (root/'config.json').exists():raise SystemExit('Expected an executed run directory containing config.json')
for folder in ['figures','tables']:
    (paper/folder).mkdir(exist_ok=True)
    for file in (root/folder).glob('*'):
        dest=paper/folder/file.name
        if dest.exists() and dest.read_bytes()!=file.read_bytes():raise SystemExit(f'Refusing to overwrite {dest}; archive/remove it explicitly first')
        shutil.copy2(file,dest)
(paper/'EXPORT_SOURCE.json').write_text(json.dumps(dict(source=str(root),config=json.loads((root/'config.json').read_text()),status=json.loads((root/'execution_status.json').read_text()) if (root/'execution_status.json').exists() else 'not recorded'),indent=2))
print('Artifacts exported; manuscript prose was not changed. Inspect measured records before writing conclusions.')

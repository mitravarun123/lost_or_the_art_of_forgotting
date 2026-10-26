#!/usr/bin/env python3
import os
# Must precede importing torch. PyTorch CPU fallback is opt-in for unsupported MPS kernels.
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse, hashlib, importlib, json, subprocess, sys, time, traceback
from pathlib import Path
import yaml
from training.checkpoint import device,environment,save_json
from data.dataset import Data
from experiments.common import Context
from analysis.plots import generate
from analysis.tables import generate as tables

MODULES={'e1':'e1_baseline','e2':'e2_neurons','e3':'e3_neuron_ablation','e4':'e4_circuit_discovery','e5':'e5_circuit_ablation','e6':'e6_comparison','e7':'e7_recoverability','e8':'e8_restoration'}
def configuration(profile,override=None):
    base=Path(__file__).parent/'configs'
    cfg=yaml.safe_load((base/'base.yaml').read_text());cfg.update(yaml.safe_load((base/f'{profile}.yaml').read_text()) or {})
    if override:cfg.update(yaml.safe_load(Path(override).read_text()) or {})
    return cfg

def main():
    parser=argparse.ArgumentParser(description='CIFAR-10 cat/dog causal intervention study')
    parser.add_argument('--profile',choices=['mac','colab'],default='mac');parser.add_argument('--experiment',default='all',choices=['all']+list(MODULES)+['e9','e10'])
    parser.add_argument('--device',default='auto',choices=['auto','cpu','mps','cuda']);parser.add_argument('--config');parser.add_argument('--data-root',default='datasets')
    parser.add_argument('--figure-dpi',type=int,default=600);parser.add_argument('--report-only',metavar='RUN_DIRECTORY');parser.add_argument('--skip-report',action='store_true');
    parser.add_argument('--output',default='results');parser.add_argument('--download',action='store_true');parser.add_argument('--smoke-test',action='store_true')
    args=parser.parse_args()
    if not 72 <= args.figure_dpi <= 1200:parser.error('--figure-dpi must be between 72 and 1200')
    os.environ['LOI_FIGURE_DPI']=str(args.figure_dpi)
    if args.report_only:
        report_root=Path(args.report_only)
        if not (report_root/'config.json').exists():parser.error('--report-only requires a run directory with config.json')
        report_cfg=json.loads((report_root/'config.json').read_text())
        generate(report_root);tables(report_root,report_cfg)
        save_json(report_root/'e10.done.json',dict(status='complete',format='PNG',dpi=args.figure_dpi))
        print(f'PNG report generated: {report_root.resolve()}/figures',flush=True)
        return
    if args.smoke_test:
        from tests.smoke import run
        run(Path(args.output)/'software_smoke');return
    cfg=configuration(args.profile,args.config)
    # Source hash prevents accidental reuse after implementation changes.
    source=hashlib.sha256()
    for file in sorted(Path(__file__).parent.rglob('*.py')):
        if any(p in ['.venv','results','__pycache__'] for p in file.parts):continue
        source.update(str(file.relative_to(Path(__file__).parent)).encode());source.update(file.read_bytes())
    cfg['source_sha256']=source.hexdigest();cfg['profile']=args.profile
    digest=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()[:12];root=Path(args.output)/f'{args.profile}_{digest}';root.mkdir(parents=True,exist_ok=True)
    save_json(root/'config.json',cfg);save_json(root/'environment.json',environment());dev=device(args.device)
    print(f'Run directory: {root.resolve()} | device: {dev}',flush=True)
    deps=yaml.safe_load((Path(__file__).parent/'configs/experiments.yaml').read_text());order=[]
    def add(name):
        for dependency in deps[name]:add(dependency)
        if name not in order:order.append(name)
    for name in cfg['experiments'] if args.experiment=='all' else [args.experiment]:add(name)
    # Plotting-only requests can inspect existing output without touching data.
    data=None
    report_generated=False
    for experiment in order:
        if experiment in MODULES:
            if data is None:
                try:data=Data(args.data_root,root,cfg['split_seed'],cfg['batch_size'],args.download)
                except Exception as exc:
                    save_json(root/'data.failed.json',dict(error=str(exc),traceback=traceback.format_exc()))
                    raise RuntimeError(f'CIFAR-10 loading failed; see {root}/data.failed.json. Use --download or supply the official extracted dataset under --data-root.') from exc
            for seed in cfg['seeds']:
                ctx=Context(data,cfg,seed,dev,root);marker=ctx.out/f'{experiment}.done.json'
                if marker.exists():print(f'Reuse {experiment}, seed {seed}',flush=True);continue
                started=time.time()
                print(f'Start {experiment.upper()} | seed {seed}',flush=True)
                try:
                    importlib.import_module('experiments.'+MODULES[experiment]).run(ctx)
                    save_json(marker,dict(status='complete',seconds=time.time()-started,device=str(dev)))
                    print(f'Complete {experiment.upper()} | seed {seed} | {time.time()-started:.1f}s',flush=True)
                except Exception as exc:
                    save_json(ctx.out/f'{experiment}.failed.json',dict(error=str(exc),traceback=traceback.format_exc(),device=str(dev)))
                    raise RuntimeError(f'{experiment}, seed {seed} failed. See {ctx.out}. Retry with --device cpu if MPS operation is unsupported.') from exc
        elif experiment=='e9':
            from experiments.e9_robustness import run
            print('Start E9 | cross-seed statistics and paired bootstraps',flush=True)
            run(root,cfg)
            print('Complete E9',flush=True)
            save_json(root/'e9.done.json',dict(status='complete'))
        elif not args.skip_report:
            generate(root);tables(root,cfg);report_generated=True
            save_json(root/'e10.done.json',dict(status='complete',format='PNG',dpi=args.figure_dpi))
    if not args.skip_report and not report_generated:
        generate(root);tables(root,cfg)
        save_json(root/'e10.done.json',dict(status='complete',format='PNG',dpi=args.figure_dpi))
    status={f'seed_{seed}':{e:('executed' if (root/f'seed_{seed}'/f'{e}.done.json').exists() else 'not run') for e in MODULES} for seed in cfg['seeds']}
    save_json(root/'execution_status.json',status)
    print(f'Completed requested pipeline. Outputs: {root.resolve()}',flush=True)
if __name__=='__main__':main()

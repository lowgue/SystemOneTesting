"""Single entry point: doctor, run, report, portable data and Clef server."""
import argparse
import json
from pathlib import Path
import sys
import subprocess
import psutil
from .core import ROOT,DEFAULT_CONTRACT,load_json,pack_dataset,unpack_dataset


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    default_config=ROOT/'benchmarks/configs/local.json';default_manifest=ROOT/'benchmarks/datasets/portable_vehicles.json'
    def inputs(p):
        p.add_argument('--config',type=Path,default=default_config);p.add_argument('--manifest',type=Path,default=default_manifest)
        p.add_argument('--contract',type=Path,default=DEFAULT_CONTRACT);p.add_argument('--data-root',type=Path,default=ROOT)
        p.add_argument('--models',nargs='+',help='Model IDs from config')
    doctor=sub.add_parser('doctor',help='Validate files, dependencies and environment without loading weights or sending inference');inputs(doctor)
    run=sub.add_parser('run',help='Run models sequentially with frozen configuration');inputs(run)
    run.add_argument('--output',type=Path,required=True);run.add_argument('--repetitions',type=int,default=3);run.add_argument('--threads',type=int,default=8)
    run.add_argument('--device',default='cpu');run.add_argument('--offline',action='store_true');run.add_argument('--resume',action='store_true');run.add_argument('--dry-run',action='store_true')
    run.add_argument('--limit',type=int);run.add_argument('--machine-id');run.add_argument('--max-edge',type=int,default=1024);run.add_argument('--seed',type=int,default=20261026)
    report=sub.add_parser('report',help='Report any number of models; keep runs and machines separate');report.add_argument('runs',type=Path,nargs='+');report.add_argument('--output',type=Path,required=True)
    pack=sub.add_parser('pack-data',help='Export verified image/manifest ZIP, without model weights');pack.add_argument('--manifest',type=Path,required=True);pack.add_argument('--data-root',type=Path,default=ROOT);pack.add_argument('--output',type=Path,required=True);pack.add_argument('--contract',type=Path,default=DEFAULT_CONTRACT)
    unpack=sub.add_parser('unpack-data',help='Verify image hashes before importing a bundle');unpack.add_argument('bundle',type=Path);unpack.add_argument('--destination',type=Path,required=True);unpack.add_argument('--contract',type=Path,default=DEFAULT_CONTRACT)
    serve=sub.add_parser('clef-server',help='Download pinned weights with hf CLI and start a Clef-capable llama-server')
    serve.add_argument('--weights',type=Path,default=ROOT/'benchmarks/configs/clef_weights.json');serve.add_argument('--llama-server');serve.add_argument('--host',default='127.0.0.1');serve.add_argument('--port',type=int,default=11437)
    serve.add_argument('--context',type=int,default=4096);serve.add_argument('--threads',type=int,default=8);serve.add_argument('--gpu-layers',type=int,default=0);serve.add_argument('--metadata',type=Path,default=ROOT/'.artifacts/clef-server.json');serve.add_argument('--dry-run',action='store_true')
    studio=sub.add_parser('studio',help='Open the local web studio for images, benchmarks, results and diagnostics')
    studio.add_argument('--port',type=int,default=7860);studio.add_argument('--open',action='store_true')
    args=parser.parse_args(argv)
    try:
        if args.command=='studio':
            from .webui.server import serve
            serve(args.port,args.open);return 0
        if args.command=='doctor':
            from .runner import preflight,environment
            specs,doc,issues=preflight(load_json(args.config),args.manifest,load_json(args.contract),args.data_root,args.models)
            result=dict(status='ready' if not issues else 'needs-setup',models=[s['id'] for s in specs],events=len(doc['items']),issues=issues,environment=environment(),note='No model load or inference; HTTP endpoint availability and protocol support require a smoke run.')
            print(json.dumps(result,indent=2,ensure_ascii=False));return 0 if not issues else 1
        if args.command=='run':
            if args.max_edge<1:raise ValueError('max-edge must be positive')
            from .runner import run
            result=run(args)
        elif args.command=='report':
            from .report import generate
            result={'report':generate(args.runs,args.output)}
        elif args.command=='pack-data':result=pack_dataset(args.manifest,args.data_root,args.output,load_json(args.contract))
        elif args.command=='unpack-data':result={'manifest':str(unpack_dataset(args.bundle,args.destination,load_json(args.contract)))}
        elif args.command=='clef-server':
            from .clef import serve
            result=serve(args)
        print(json.dumps(result,indent=2,ensure_ascii=False));return 0
    except (ValueError,RuntimeError,OSError,KeyError,ImportError,subprocess.SubprocessError,psutil.Error) as exc:
        print(f'ERROR: {exc}',file=sys.stderr);return 2

if __name__=='__main__':sys.exit(main())

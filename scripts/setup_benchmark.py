"""Create a Python environment and optionally download model weights."""
import argparse
from pathlib import Path
import subprocess
import sys
import venv

ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--backend',choices=['http','cpu'],default='http')
    ap.add_argument('--venv',type=Path,default=ROOT/'.venv')
    ap.add_argument('--download',nargs='*',metavar='MODEL',help='Optional: download and cache model weights (e.g. --download laya d1-omni clef-gguf)')
    args=ap.parse_args()
    if sys.version_info<(3,12):ap.error('Use Python 3.12 or newer; Python 3.12 is the tested baseline')
    target=args.venv.resolve();python=target/('Scripts/python.exe' if sys.platform=='win32' else 'bin/python')
    if not python.exists():venv.EnvBuilder(with_pip=True).create(target)
    if args.backend=='cpu':subprocess.run(['git','submodule','update','--init','--recursive'],cwd=ROOT,check=True)
    subprocess.run([str(python),'-m','pip','install','-r',str(ROOT/'requirements'/f'{args.backend}.txt'),'-e',str(ROOT)],cwd=ROOT,check=True)
    print(f'Environment ready: {python}')
    if args.download is not None:
        models = args.download if len(args.download) > 0 else ['laya', 'd1-omni', 'clef-gguf']
        print(f'Downloading requested models: {models}...')
        subprocess.run([str(python),str(ROOT/'scripts/download_models.py'),'--models',*models],cwd=ROOT,check=True)
    else:
        print('No model weights downloaded. Use scripts/download_models.py or --download to cache weights.')
    print(f'Validate setup with: {python} -m vision_bench doctor')

if __name__=='__main__':main()

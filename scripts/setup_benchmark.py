"""Create a Python environment without downloading model weights."""
import argparse
from pathlib import Path
import subprocess
import sys
import venv

ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--backend',choices=['http','cpu'],default='http');ap.add_argument('--venv',type=Path,default=ROOT/'.venv');args=ap.parse_args()
    if sys.version_info<(3,12):ap.error('Use Python 3.12 or newer; Python 3.12 is the tested baseline')
    target=args.venv.resolve();python=target/('Scripts/python.exe' if sys.platform=='win32' else 'bin/python')
    if not python.exists():venv.EnvBuilder(with_pip=True).create(target)
    if args.backend=='cpu':subprocess.run(['git','submodule','update','--init','--recursive'],cwd=ROOT,check=True)
    subprocess.run([str(python),'-m','pip','install','-r',str(ROOT/'requirements'/f'{args.backend}.txt'),'-e',str(ROOT)],cwd=ROOT,check=True)
    print(f'Ready: {python} -m vision_bench doctor')
    print('No model weights have been downloaded. See docs/COMPARISON_SETUP.md.')

if __name__=='__main__':main()

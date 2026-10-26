"""Prepare Ollama Clef and launch the local browser playground (Python 3.12+)."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import venv
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
API = 'http://127.0.0.1:11434'
MODEL = 'clef-flash:9b'


def get_json(url):
    with urllib.request.urlopen(url, timeout=3) as response:
        return json.load(response)


def wait_ready(url, process, timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f'O processo encerrou com código {process.returncode}.')
        try:
            return get_json(url)
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(0.3)
    raise RuntimeError(f'Tempo esgotado aguardando {url}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    if sys.version_info < (3, 12):
        parser.error('Instale Python 3.12 ou superior.')
    ollama = shutil.which('ollama')
    if not ollama:
        raise RuntimeError('Instale o Ollama em https://ollama.com/download e execute novamente.')
    python = ROOT / '.venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if not python.exists():
        venv.EnvBuilder(with_pip=True).create(ROOT / '.venv')
    if subprocess.run([str(python), '-c', 'from PIL import Image'], capture_output=True).returncode:
        subprocess.run([str(python), '-m', 'pip', 'install', 'Pillow==12.3.0'], check=True)
    server = None
    web = None
    try:
        try:
            version = get_json(API + '/api/version')
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            env = dict(os.environ, OLLAMA_HOST='127.0.0.1:11434')
            server = subprocess.Popen([ollama, 'serve'], env=env)
            version = wait_ready(API + '/api/version', server)
        numbers = tuple(int(p) for p in version['version'].split('-')[0].split('.')[:3])
        if numbers < (0, 35, 1):
            raise RuntimeError('Atualize o Ollama: Clef requer versão 0.35.1 ou superior.')
        env = dict(os.environ, OLLAMA_HOST='127.0.0.1:11434')
        models = get_json(API + '/api/tags')['models']
        if not any(m.get('name') == MODEL for m in models):
            print(f'Baixando {MODEL}; o primeiro uso pode demorar.', flush=True)
            subprocess.run([ollama, 'pull', MODEL], env=env, check=True)
        url = f'http://127.0.0.1:{args.port}'
        try:
            get_json(url + '/api/status')
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        else:
            raise RuntimeError(f'Porta {args.port} ocupada; use --port com outra porta.')
        env.update(OLLAMA_MODEL=MODEL, OLLAMA_SYSTEMONE_URL=API + '/v1/systemone')
        web = subprocess.Popen([str(python), str(ROOT / 'web/server.py'), '--model',
                                'clef-ollama', '--host', '127.0.0.1', '--port', str(args.port)],
                               cwd=ROOT, env=env)
        status = wait_ready(url + '/api/status', web)
        if status.get('model') != 'clef-ollama':
            raise RuntimeError('A porta está sendo usada por outro serviço.')
        print(f'Pronto: {url} — envie uma imagem. Ctrl+C para encerrar.', flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        web.wait()
    finally:
        for process in (web, server):
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    except (RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        print(f'Erro: {exc}', file=sys.stderr)
        sys.exit(1)

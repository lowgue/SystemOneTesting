"""One owned worker at a time; durable output and cancellable subprocesses."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import threading
import uuid

import psutil

from ..core import ROOT, load_json, write_json


class Jobs:
    def __init__(self, root=None):
        self.root = Path(root or ROOT / '.artifacts/studio')
        self.lock = threading.RLock()
        self.process = None
        self.current = None
        self.log = None

    def refresh(self):
        if self.process is not None and self.process.poll() is not None:
            path = self.root / self.current / 'status.json'
            status = load_json(path)
            if status['state'] in {'queued', 'running'}:
                status.update(state='failed', error=f'O processo terminou com código {self.process.returncode}. Consulte o registro.')
                write_json(path, status)
            if self.log is not None:
                self.log.close()
                self.log = None

    def start(self, request, image=None):
        with self.lock:
            self.refresh()
            if self.process is not None and self.process.poll() is None:
                raise RuntimeError('Uma tarefa está em andamento. Aguarde ou cancele antes de iniciar outra.')
            job_id = uuid.uuid4().hex
            folder = self.root / job_id
            folder.mkdir(parents=True)
            if image is not None:
                image.save(folder / 'input.png', format='PNG')
            write_json(folder / 'request.json', request)
            write_json(folder / 'status.json', dict(id=job_id, kind=request['kind'], state='queued', results=[]))
            self.log = (folder / 'worker.log').open('w', encoding='utf-8')
            env = dict(os.environ, PYTHONUNBUFFERED='1', MPLBACKEND='Agg')
            try:
                self.process = subprocess.Popen(
                    [sys.executable, '-m', 'vision_bench.webui.worker', str(folder)],
                    cwd=ROOT, stdout=self.log, stderr=subprocess.STDOUT, env=env,
                )
            except OSError:
                self.log.close()
                self.log = None
                write_json(folder / 'status.json', dict(id=job_id, kind=request['kind'], state='failed', error='Não foi possível iniciar o processo.'))
                raise
            self.current = job_id
            return self.status(job_id)

    def folder(self, job_id):
        if len(job_id) != 32 or any(c not in '0123456789abcdef' for c in job_id):
            raise ValueError('Identificador de tarefa inválido.')
        path = self.root / job_id
        if not path.is_dir():
            raise ValueError('Tarefa não encontrada.')
        return path

    def status(self, job_id):
        with self.lock:
            self.refresh()
            folder = self.folder(job_id)
            result = load_json(folder / 'status.json')
            request = load_json(folder / 'request.json')
            if request.get('kind') in {'live', 'benchmark'}:
                result['settings'] = {
                    'config': Path(request['config']).stem,
                    'contract': {'description_ollama_v1': 'ollama', 'description_v1': 'original'}.get(Path(request['contract']).stem, Path(request['contract']).stem),
                    'models': request['models'],
                    'device': request['device'],
                    'threads': request['threads'],
                }
            log = folder / 'worker.log'
            if log.exists():
                with log.open('rb') as stream:
                    stream.seek(max(0, log.stat().st_size - 8000))
                    result['log'] = stream.read().decode('utf-8', errors='replace')
            if (folder / 'run/status.json').exists():
                result['progress'] = load_json(folder / 'run/status.json')
            return result

    def cancel(self, job_id):
        with self.lock:
            self.folder(job_id)
            if self.current != job_id:
                raise ValueError('A tarefa não está ativa nesta sessão.')
            self.refresh()
            if self.process is not None and self.process.poll() is None:
                parent = psutil.Process(self.process.pid)
                children = parent.children(recursive=True)
                for process in reversed(children):
                    try:
                        process.terminate()
                    except psutil.NoSuchProcess:
                        pass
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
                _, alive = psutil.wait_procs(children, timeout=2)
                for process in alive:
                    try:
                        process.kill()
                    except psutil.NoSuchProcess:
                        pass
                path = self.root / job_id / 'status.json'
                result = load_json(path)
                result['state'] = 'cancelled'
                write_json(path, result)
            return self.status(job_id)

    def close(self):
        if self.current:
            self.cancel(self.current)
        if self.log:
            self.log.close()

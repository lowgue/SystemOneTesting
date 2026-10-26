"""Loopback-only studio HTTP server. No model imports or inference on startup."""
from __future__ import annotations

import argparse
import base64
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
import mimetypes
from pathlib import Path
import re
import sys
import threading
from urllib.parse import unquote, urlsplit
import urllib.request
import webbrowser

from PIL import Image, UnidentifiedImageError
import psutil

from ..core import ROOT, load_json, resolve_image
from ..runner import validate_config
from .jobs import Jobs

STATIC = Path(__file__).with_name('static')
MAX_BODY = 24 * 1024 * 1024


class Studio:
    def __init__(self, config_dir=None, jobs_root=None):
        self.config_dir = Path(config_dir or ROOT / 'benchmarks/configs')
        self.jobs = Jobs(jobs_root)
        self.contracts = {
            'ollama': ROOT / 'benchmarks/contracts/description_ollama_v1.json',
            'original': ROOT / 'benchmarks/contracts/description_v1.json',
        }
        self.manifests = {'portable': ROOT / 'benchmarks/datasets/portable_vehicles.json'}
        self.sources = {
            'studio': self.jobs.root,
            'reports': ROOT / '.artifacts/reports',
            'scientific': ROOT / 'eval/figures/scientific_20261026_final',
        }

    def configs(self):
        result = {}
        for path in sorted(self.config_dir.glob('*.json')):
            try:
                doc = load_json(path)
                validate_config(doc)
                result[path.stem] = path
            except (ValueError, KeyError):
                pass
        return result

    def catalog(self):
        return dict(configs=[dict(id=key, models=[dict(id=s['id'], adapter=s['adapter'],
            simulated=s['adapter']=='mock') for s in load_json(path)['models']]) for key,path in self.configs().items()],
            contracts=[dict(id=k, name=load_json(v)['name'], domains=list(load_json(v)['domains'])) for k,v in self.contracts.items()],
            datasets=[dict(id=k, events=len(load_json(v)['items'])) for k,v in self.manifests.items()],
            current_job=self.jobs.current)

    def prepare(self, body):
        if not isinstance(body, dict):
            raise ValueError('Envie um objeto JSON.')
        config = self.configs().get(body.get('config'))
        contract = self.contracts.get(body.get('contract', 'ollama'))
        if not config or not contract:
            raise ValueError('Selecione uma configuração e um contrato da lista.')
        models = body.get('models')
        if not isinstance(models, list) or not models or not all(isinstance(m,str) for m in models):
            raise ValueError('Selecione pelo menos um modelo.')
        specs = validate_config(load_json(config), models)
        doc = load_json(contract)
        for spec in specs:
            if spec.get('max_choices') and any(len(q.get('criteria',{})) > spec['max_choices']
                for d in doc['domains'].values() for q in d['questions'].values()):
                raise ValueError('Este modelo requer o contrato compartilhado Ollama (até 26 alternativas).')
        device = body.get('device', 'cpu')
        if not isinstance(device,str) or not re.fullmatch(r'cpu|cuda(?::\d+)?',device):
            raise ValueError('Dispositivo deve ser cpu, cuda ou cuda:N.')
        threads = body.get('threads', 8)
        if type(threads) is not int or not 1 <= threads <= 256:
            raise ValueError('Threads deve estar entre 1 e 256.')
        request = dict(kind=body.get('kind', 'live'), config=str(config), contract=str(contract),
                       models=[s['id'] for s in specs], device=device, threads=threads)
        image = None
        if request['kind'] == 'live':
            domain = body.get('domain', 'vehicle')
            if domain not in doc['domains']:
                raise ValueError('Domínio inválido.')
            request['domain'] = domain
            encoded = body.get('image_data', '')
            if not isinstance(encoded,str) or not encoded:
                raise ValueError('Escolha uma imagem antes de iniciar.')
            try:
                data = base64.b64decode(encoded.split(',')[-1], validate=True)
                with Image.open(BytesIO(data)) as source:
                    if source.width * source.height > 25_000_000:
                        raise ValueError('Use uma imagem com até 25 megapixels.')
                    image = source.convert('RGB')
                image.thumbnail((2048,2048))
            except (ValueError, UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
                raise ValueError(f'Imagem inválida: {exc}') from exc
        elif request['kind'] == 'benchmark':
            manifest = self.manifests.get(body.get('dataset','portable'))
            reps = body.get('repetitions',3)
            limit = body.get('limit', 0)
            if not manifest or type(reps) is not int or not 1 <= reps <= 100 or type(limit) is not int or not 0 <= limit <= 100000:
                raise ValueError('Dataset, repetições ou limite inválidos.')
            request.update(manifest=str(manifest), repetitions=reps, limit=limit)
        else:
            raise ValueError('Tipo de tarefa inválido.')
        return request, image

    def artifacts(self):
        files = []
        for source, root in self.sources.items():
            if not root.exists():
                continue
            for path in sorted(root.rglob('*')):
                rel = path.relative_to(root).as_posix()
                if path.is_file() and path.suffix in {'.png','.pdf','.csv','.md','.json','.jsonl'}:
                    if source == 'studio' and not ('/report/' in rel or '/run/' in rel):
                        continue
                    files.append(dict(source=source, name=rel, url=f'/files/{source}/{rel}'))
        return files

    def diagnostics(self):
        memory = psutil.virtual_memory()
        result = dict(python=sys.version.split()[0], ram_total_gb=round(memory.total/2**30,1),
                      ram_available_gb=round(memory.available/2**30,1), inference_tested=False)
        # Probe only the configured Ollama origin; never load weights or run inference.
        import os
        value = os.environ.get('OLLAMA_SYSTEMONE_URL','http://127.0.0.1:11434/v1/systemone')
        parsed = urlsplit(value)
        if parsed.hostname not in {'127.0.0.1','localhost','::1'} or parsed.scheme != 'http' or parsed.username or parsed.password:
            result.update(ollama='not-probed',error='Diagnóstico automático disponível apenas para Ollama HTTP local.')
            return result
        origin = f'{parsed.scheme}://{parsed.netloc}'
        try:
            for key, route in [('version','version'),('models','tags'),('loaded','ps')]:
                with urllib.request.urlopen(origin+'/api/'+route,timeout=3) as response:
                    data = json.load(response)
                result[key] = data.get('models',[]) if key != 'version' else data.get('version')
            result['ollama'] = 'online'
            result['clef_installed'] = any(m.get('name') == 'clef-flash:9b' for m in result['models'])
        except (OSError,ValueError) as exc:
            result.update(ollama='unavailable',error=str(exc))
        return result


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, studio, **kwargs):
        self.studio = studio
        super().__init__(*args, **kwargs)

    def send(self, data, content_type='application/json; charset=utf-8', status=200):
        if not isinstance(data,bytes):
            data = json.dumps(data,ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header('Content-Type',content_type)
        self.send_header('Content-Length',str(len(data)))
        self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError,ConnectionResetError):
            pass

    def do_GET(self):
        path = unquote(urlsplit(self.path).path)
        try:
            if path == '/api/catalog':
                return self.send(self.studio.catalog())
            if path == '/api/status':
                return self.send(dict(app='vision-studio'))
            if path == '/api/diagnostics':
                return self.send(self.studio.diagnostics())
            if path == '/api/artifacts':
                return self.send(self.studio.artifacts())
            if path.startswith('/api/jobs/'):
                return self.send(self.studio.jobs.status(path.split('/')[-1]))
            if path.startswith('/files/'):
                parts = path.split('/',3)
                allowed = {item['url'] for item in self.studio.artifacts()}
                if path not in allowed:
                    raise ValueError('Arquivo não disponível.')
                target = resolve_image(self.studio.sources[parts[2]],parts[3])
                return self.send(target.read_bytes(),mimetypes.guess_type(target.name)[0] or 'application/octet-stream')
            assets = {'/':'index.html','/app.js':'app.js','/style.css':'style.css'}
            if path in assets:
                target = STATIC / assets[path]
                return self.send(target.read_bytes(),mimetypes.guess_type(target.name)[0]+'; charset=utf-8')
            return self.send(dict(error='Página não encontrada.'),status=404)
        except (ValueError,KeyError,FileNotFoundError) as exc:
            self.send(dict(error=str(exc)),status=404)

    def do_POST(self):
        try:
            origin = self.headers.get('Origin')
            if origin and origin != 'http://' + self.headers.get('Host',''):
                return self.send(dict(error='Origem não permitida.'),status=403)
            length = int(self.headers.get('Content-Length','0'))
            if not 0 < length <= MAX_BODY:
                return self.send(dict(error='Envie uma requisição com até 24 MiB.'),status=413)
            body = json.loads(self.rfile.read(length))
            if self.path == '/api/shutdown':
                self.send(dict(status='stopping'))
                threading.Thread(target=self.server.shutdown,daemon=True).start()
                return
            if self.path == '/api/jobs':
                request,image = self.studio.prepare(body)
                return self.send(self.studio.jobs.start(request,image),status=202)
            if self.path.startswith('/api/jobs/') and self.path.endswith('/cancel'):
                return self.send(self.studio.jobs.cancel(self.path.split('/')[3]))
            self.send(dict(error='Rota não encontrada.'),status=404)
        except (ValueError,KeyError,TypeError) as exc:
            self.send(dict(error=str(exc)),status=400)
        except RuntimeError as exc:
            self.send(dict(error=str(exc)),status=409)
        except OSError as exc:
            self.send(dict(error=str(exc)),status=500)


def make_server(port=7860, studio=None):
    studio = studio or Studio()
    return ThreadingHTTPServer(('127.0.0.1',port),partial(Handler,studio=studio))


def serve(port=7860, open_browser=False):
    studio = Studio()
    server = make_server(port,studio)
    url = f'http://127.0.0.1:{server.server_port}'
    print(f'Laboratório visual: {url}',flush=True)
    print('Abra o diagnóstico para verificar o Ollama. Nenhuma inferência foi iniciada.',flush=True)
    if open_browser:
        threading.Timer(.5,lambda:webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        studio.jobs.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=7860)
    parser.add_argument('--open',action='store_true')
    args = parser.parse_args()
    serve(args.port,args.open)


if __name__ == '__main__':
    main()

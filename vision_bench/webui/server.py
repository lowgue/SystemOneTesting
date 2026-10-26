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


from .services import Studio


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

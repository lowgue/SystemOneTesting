"""FastAPI application: REST API, live events and compiled React frontend."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from ..core import ROOT, load_json, resolve_image
from .services import Studio


MODEL = 'clef-flash:9b'


def ollama_origin():
    value = os.environ.get('OLLAMA_SYSTEMONE_URL', 'http://127.0.0.1:11434/v1/systemone')
    parsed = urlsplit(value)
    if parsed.scheme != 'http' or parsed.hostname not in {'localhost','127.0.0.1','::1','ollama','host.docker.internal'} or parsed.username or parsed.password:
        raise ValueError('Configure um endpoint HTTP do Ollama local ou do serviço Docker.')
    return f'{parsed.scheme}://{parsed.netloc}'


async def json_body(request):
    try:
        body=await request.json()
    except (ValueError,UnicodeDecodeError):
        raise ValueError('Envie JSON válido.')
    if not isinstance(body,dict):
        raise ValueError('Envie um objeto JSON.')
    return body


def create_app(studio=None, frontend=None):
    studio = studio or Studio()
    frontend = Path(frontend or os.environ.get('VISION_FRONTEND_DIR', ROOT/'frontend/dist'))

    @asynccontextmanager
    async def lifespan(app):
        yield
        await run_in_threadpool(studio.jobs.close)

    app = FastAPI(title='System One Testing', version='0.2.0', lifespan=lifespan)
    app.state.studio = studio

    @app.middleware('http')
    async def boundaries(request, call_next):
        if request.method in {'POST','PUT','DELETE'}:
            origin = request.headers.get('origin')
            if origin and urlsplit(origin).netloc != request.headers.get('host'):
                return JSONResponse({'error':'Origem não permitida.'},status_code=403)
            # Consume streaming bodies with an enforced cap, including chunked requests.
            chunks = bytearray()
            async for chunk in request.stream():
                chunks.extend(chunk)
                if len(chunks) > 24*1024*1024:
                    return JSONResponse({'error':'Limite de 24 MiB excedido.'},status_code=413)
            request._body = bytes(chunks)
        response = await call_next(request)
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response

    @app.exception_handler(ValueError)
    async def invalid(request, exc):
        return JSONResponse({'error':str(exc)},status_code=400)

    @app.exception_handler(RuntimeError)
    async def conflict(request, exc):
        return JSONResponse({'error':str(exc)},status_code=409)

    @app.get('/api/health')
    def health():
        return {'status':'ok','app':'vision-studio','frontend_built':(frontend/'index.html').is_file()}

    @app.get('/api/catalog')
    def catalog():
        return studio.catalog()

    @app.get('/api/diagnostics')
    def diagnostics():
        return studio.diagnostics()

    @app.get('/api/jobs')
    def history():
        studio.jobs.refresh()
        results=[]
        for path in sorted(studio.jobs.root.glob('*/status.json'),key=lambda p:p.stat().st_mtime,reverse=True)[:100]:
            value=load_json(path)
            results.append({key:value.get(key) for key in ['id','kind','state','errors','started_utc','phase','dataset','simulated']})
        return results

    @app.post('/api/jobs',status_code=202)
    async def start(request: Request):
        try:
            body=await json_body(request)
        except (ValueError, UnicodeDecodeError):
            raise ValueError('Envie JSON válido.')
        prepared,image=await run_in_threadpool(studio.prepare,body)
        return await run_in_threadpool(studio.jobs.start,prepared,image)

    @app.post('/api/datasets',status_code=201)
    async def new_dataset(request: Request):
        body=await json_body(request)
        return await run_in_threadpool(studio.datasets.create,body.get('name','Minhas fotos'),body.get('domain','vehicle'))

    @app.post('/api/datasets/{dataset_id}/images')
    async def upload_photo(dataset_id: str,request: Request):
        body=await json_body(request)
        return await run_in_threadpool(studio.datasets.add_image,dataset_id,body.get('filename'),body.get('image_data'))

    @app.post('/api/datasets/{dataset_id}/finalize')
    async def finalize_dataset(dataset_id: str,request: Request):
        body=await json_body(request)
        contract=studio.contracts.get(body.get('contract','ollama'))
        if not contract:raise ValueError('Contrato inválido.')
        text=body.get('csv','')
        if not isinstance(text,str):raise ValueError('O gabarito deve ser texto CSV.')
        return await run_in_threadpool(studio.datasets.finalize,dataset_id,text,load_json(contract))

    @app.post('/api/datasets/{dataset_id}/discard')
    def discard_dataset(dataset_id: str):
        return studio.datasets.discard(dataset_id)

    @app.get('/api/contracts/{contract_id}')
    def contract_questions(contract_id: str):
        path=studio.contracts.get(contract_id)
        if not path:raise ValueError('Contrato inválido.')
        return load_json(path)['domains']

    @app.post('/api/protocol-preview')
    def protocol_preview(body: dict):
        from ..protocols import expand_specs, item_contract, describe
        request,_=studio.prepare({**body,'kind':'benchmark'})
        contract=load_json(request['contract'])
        specs=[s for s in load_json(request['config'])['models'] if s['id'] in request['models']]
        specs=expand_specs(specs,request.get('comparison','baseline'))
        items=load_json(request['manifest'])['items']
        if request['limit']:items=items[:request['limit']]
        examples={}
        for item in items:
            key=(item['domain'],tuple(item.get('task_fields',[])),item.get('task_field'),item.get('question'))
            examples.setdefault(key,item)
        return dict(variants=len(specs),photos=min(request['limit'] or len(items),len(items)),
                    planned=min(request['limit'] or len(items),len(items))*len(specs)*request['repetitions'],
                    protocols=[dict(model=s['id'],examples=[dict(domain=it['domain'],**describe(s,item_contract(contract,it),it['domain'])) for it in examples.values()]) for s in specs])

    @app.get('/api/datasets/{dataset_id}/photos/{photo_id}')
    def dataset_photo(dataset_id: str,photo_id: str,thumbnail: bool=False):
        manifest_path=studio.dataset_paths().get(dataset_id)
        if not manifest_path:raise ValueError('Dataset não encontrado.')
        item=next((it for it in load_json(manifest_path)['items'] if it['id']==photo_id),None)
        if not item:raise ValueError('Foto não encontrada.')
        root=ROOT if dataset_id in studio.manifests else manifest_path.parent
        path=resolve_image(root,item['rel_path'])
        if thumbnail:
            from io import BytesIO
            from PIL import Image
            buffer=BytesIO()
            with Image.open(path) as image:
                image=image.convert('RGB');image.thumbnail((200,150));image.save(buffer,format='JPEG',quality=75)
            return Response(buffer.getvalue(),media_type='image/jpeg')
        return FileResponse(path)

    @app.get('/api/datasets/{dataset_id}/bundle')
    def dataset_bundle(dataset_id: str):
        from ..core import pack_dataset, sha256
        manifest=studio.dataset_paths().get(dataset_id)
        if not manifest:raise ValueError('Dataset não encontrado.')
        contract=studio.contracts.get(load_json(manifest).get('required_contract','ollama'))
        root=ROOT if dataset_id in studio.manifests else manifest.parent
        folder=studio.jobs.root.parent/'dataset-bundles';folder.mkdir(exist_ok=True,parents=True)
        bundle=folder/(sha256(manifest)+'.zip')
        with studio.datasets.lock:
            if not bundle.exists():
                try:pack_dataset(manifest,root,bundle,load_json(contract))
                except Exception:
                    bundle.unlink(missing_ok=True)
                    raise
        return FileResponse(bundle,filename=dataset_id+'.zip',media_type='application/zip')

    @app.get('/api/jobs/{job_id}/evaluation')
    def evaluation(job_id: str):
        path=studio.jobs.folder(job_id)/'report/evaluation.json'
        if not path.is_file():
            return JSONResponse({'error':'O relatório estará disponível ao concluir a avaliação.'},status_code=404)
        return load_json(path)

    @app.post('/api/ollama/pull',status_code=202)
    def pull():
        return studio.jobs.start({'kind':'ollama-pull','origin':ollama_origin(),'model':MODEL})

    @app.get('/api/jobs/{job_id}')
    def status(job_id: str):
        return studio.jobs.status(job_id)

    @app.post('/api/jobs/{job_id}/cancel')
    def cancel(job_id: str):
        return studio.jobs.cancel(job_id)

    @app.get('/api/jobs/{job_id}/events')
    async def events(job_id: str,request: Request):
        studio.jobs.folder(job_id)
        async def stream():
            while not await request.is_disconnected():
                state=await run_in_threadpool(studio.jobs.status,job_id)
                yield 'data: '+json.dumps(state,ensure_ascii=False)+'\n\n'
                if state['state'] not in {'queued','running'}:
                    break
                await asyncio.sleep(.6)
        return StreamingResponse(stream(),media_type='text/event-stream',headers={'Cache-Control':'no-cache','X-Accel-Buffering':'no'})

    @app.get('/api/artifacts')
    def artifacts():
        return studio.artifacts()

    @app.get('/files/{source}/{relative:path}')
    def file(source: str,relative: str):
        url=f'/files/{source}/{relative}'
        if not any(item['url']==url for item in studio.artifacts()):
            return JSONResponse({'error':'Arquivo não disponível.'},status_code=404)
        target=resolve_image(studio.sources[source],relative)
        return FileResponse(target)

    if (frontend/'assets').is_dir():
        app.mount('/assets',StaticFiles(directory=frontend/'assets'),name='assets')

    @app.get('/')
    def index():
        if not (frontend/'index.html').is_file():
            return JSONResponse({'error':'Frontend não compilado. Use Docker Compose ou execute npm ci e npm run build na pasta frontend.'},status_code=503)
        return FileResponse(frontend/'index.html')

    return app


app = create_app()


def serve(port=7860, open_browser=False):
    import uvicorn
    if open_browser:
        import threading
        import webbrowser
        threading.Timer(1,lambda:webbrowser.open(f'http://127.0.0.1:{port}')).start()
    uvicorn.run(app,host='127.0.0.1',port=port,workers=1,timeout_graceful_shutdown=5)


def main():
    import argparse
    parser=argparse.ArgumentParser(description='System One web application (FastAPI + React)')
    parser.add_argument('--port',type=int,default=7860)
    parser.add_argument('--open',action='store_true')
    args=parser.parse_args()
    serve(args.port,args.open)


if __name__=='__main__':
    main()

"""Real React/FastAPI browser integration with simulated and failing local backends."""
from contextlib import ExitStack
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from unittest.mock import patch
from pathlib import Path
import tempfile
import threading
import time
import socket

from PIL import Image
from playwright.sync_api import sync_playwright, expect
import uvicorn

from vision_bench.core import ROOT
from vision_bench.webui.api import create_app
from vision_bench.webui.services import Studio


def main():
    output=ROOT/'.artifacts/validation/react-web';output.mkdir(parents=True,exist_ok=True)
    with ExitStack() as stack:
        temp=Path(stack.enter_context(tempfile.TemporaryDirectory()))
        class Failure(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_GET(self):
                data=json.dumps({'version':'0.40.6'} if self.path=='/api/version' else {'models':[]}).encode()
                self.send_response(200);self.end_headers();self.wfile.write(data)
            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length']))
                if self.path=='/api/pull':
                    self.send_response(200);self.end_headers()
                    self.wfile.write(b'{"status":"success"}\n')
                    return
                data=json.dumps({'error':'runner failed: insufficient memory, requested 12634 MB'}).encode()
                self.send_response(500);self.end_headers();self.wfile.write(data)
        fault=ThreadingHTTPServer(('127.0.0.1',0),Failure)
        threading.Thread(target=fault.serve_forever,daemon=True).start()
        stack.callback(fault.server_close);stack.callback(fault.shutdown)
        configs=temp/'configs';configs.mkdir()
        (configs/'simulation.json').write_text(json.dumps({'models':[{'id':'SIMULATED-a','adapter':'mock'},{'id':'SIMULATED-b','adapter':'mock'}]}))
        (configs/'ollama.json').write_text(json.dumps({'models':[dict(id='clef-flash-ollama',adapter='systemone_http',url=f'http://127.0.0.1:{fault.server_port}/v1/systemone',request_model='clef-flash:9b',image_encoding='base64',max_choices=26)]}))
        stack.enter_context(patch.dict(os.environ,{'OLLAMA_SYSTEMONE_URL':f'http://127.0.0.1:{fault.server_port}/v1/systemone'}))
        app=create_app(Studio(configs,temp/'jobs'))
        listener=socket.socket();listener.bind(('127.0.0.1',0))
        port=listener.getsockname()[1]
        server=uvicorn.Server(uvicorn.Config(app,host='127.0.0.1',port=port,log_level='warning'))
        thread=threading.Thread(target=server.run,kwargs={'sockets':[listener]},daemon=True);thread.start()
        def stop():server.should_exit=True;thread.join(timeout=15);listener.close()
        stack.callback(stop)
        deadline=time.monotonic()+10
        while not server.started:
            if time.monotonic()>deadline:raise RuntimeError('API startup timeout')
            time.sleep(.05)
        photo=temp/'image.png';Image.new('RGB',(600,400),'#739bbe').save(photo)
        playwright=stack.enter_context(sync_playwright())
        browser=playwright.chromium.launch(headless=True,channel='chromium');stack.callback(browser.close)
        page=browser.new_page(viewport={'width':1440,'height':1000})
        errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto(f'http://127.0.0.1:{port}');page.wait_for_load_state('networkidle')
        expect(page.get_by_role('heading',name='Avaliar foto',exact=True)).to_be_visible()
        page.locator('#config').select_option('simulation')
        page.locator('#image').set_input_files(photo)
        page.get_by_role('button',name='Avaliar foto',exact=True).click()
        expect(page.locator('.answers-panel [role=status]')).to_have_text('Concluído',timeout=20000)
        expect(page.locator('.answer')).to_have_count(2)
        with page.expect_download() as result:page.get_by_role('button',name='Baixar JSON').click()
        result.value.save_as(output/'interactive.json')
        page.screenshot(path=str(output/'desktop.png'),full_page=True)
        page.get_by_role('link',name='Avaliar dataset',exact=True).click()
        page.locator('#repetitions').fill('1');page.locator('#limit').fill('2')
        page.get_by_role('button',name='Iniciar avaliação do dataset').click()
        expect(page.locator('.benchmark-layout [role=status]')).to_have_text('Concluído',timeout=30000)
        page.get_by_role('link',name='Resultados',exact=True).click()
        expect(page.locator('.artifacts')).to_contain_text('comparison_1.png',timeout=10000)
        page.locator('#filter').fill('.csv');expect(page.locator('.artifacts')).to_contain_text('metrics.csv')
        page.get_by_role('link',name='Avaliar dataset',exact=True).click()
        page.get_by_role('button',name='Importar minhas fotos',exact=True).click()
        a=temp/'a.png';b=temp/'b.png'
        Image.new('RGB',(40,30),'black').save(a);Image.new('RGB',(40,30),'white').save(b)
        page.locator('#dataset-name').fill('Meu dataset de teste')
        page.locator('#dataset-images').set_input_files([a,b])
        page.locator('#dataset-csv').set_input_files({'name':'gabarito.csv','mimeType':'text/csv','buffer':'arquivo;presença;marca;cor;carroceria\na.png;sim;Audi;preto;sedã\nb.png;;;branco;\n'.encode()})
        page.get_by_role('button',name='Preparar dataset',exact=True).click()
        expect(page.locator('.dataset-summary')).to_contain_text('2 fotos',timeout=15000)
        expect(page.locator('.dataset-summary')).to_contain_text('2 com gabarito')
        page.get_by_role('button',name='Iniciar avaliação do dataset',exact=True).click()
        expect(page.locator('[aria-label="Relatório do dataset"]')).to_contain_text('Meu dataset de teste',timeout=30000)
        expect(page.locator('.metric-grid').first).to_contain_text('80,0%')
        expect(page.locator('.metric-grid').first).to_contain_text('4 de 5 atributos anotados')
        expect(page.locator('.photo-results')).to_contain_text('a.png')
        page.locator('#photo-filter').select_option('mistakes')
        expect(page.locator('.photo-results tbody tr').first).to_contain_text('b.png')
        page.screenshot(path=str(output/'dataset.png'),full_page=True)
        page.set_viewport_size({'width':390,'height':844})
        page.screenshot(path=str(output/'dataset-mobile.png'),full_page=True)
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'),'Dataset mobile overflow'
        page.set_viewport_size({'width':1440,'height':1000})
        page.get_by_role('link',name='Ambiente e modelos',exact=True).click()
        expect(page.get_by_role('heading',name='Ambiente e modelos',exact=True)).to_be_visible()
        expect(page.locator('.diagnostic-list:visible')).to_contain_text('RAM total',timeout=15000)
        page.get_by_role('button',name='Baixar ou atualizar Clef-Flash 9B').click()
        expect(page.get_by_role('status')).to_have_text('Concluído',timeout=15000)
        page.get_by_role('link',name='Avaliar foto',exact=True).click()
        page.locator('#config').select_option('ollama')
        page.get_by_role('button',name='Avaliar foto',exact=True).click()
        expect(page.locator('.answers-panel [role=status]')).to_have_text('Concluído com falhas',timeout=15000)
        expect(page.locator('.answer .error')).to_contain_text('12634 MB')
        page.reload();expect(page.locator('.answer .error')).to_contain_text('12634 MB',timeout=15000)
        page.set_viewport_size({'width':390,'height':844})
        page.screenshot(path=str(output/'mobile.png'),full_page=True)
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'),'Mobile overflow'
        assert not errors,errors
        (output/'summary.json').write_text(json.dumps({'status':'passed','page_errors':errors,'simulated':True}))
        print('React + FastAPI browser: passed (single photo, dataset import, partial CSV ground truth, automatic accuracy report, photo filters, 2 models, SSE, exports, history, diagnostics, errors, reload, mobile).')


if __name__=='__main__':main()

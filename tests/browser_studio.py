"""Optional real-browser smoke: simulated models and an intentionally failing local HTTP backend."""
import argparse
from contextlib import ExitStack
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading

from PIL import Image
from playwright.sync_api import sync_playwright, expect

from vision_bench.core import ROOT
from vision_bench.webui.server import Studio, make_server


def serve(server, stack):
    thread=threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    def close():
        server.shutdown()
        server.server_close()
        thread.join()
    stack.callback(close)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'.artifacts/validation/studio')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    class FailingOllama(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def do_POST(self):
            self.rfile.read(int(self.headers.get('Content-Length','0')))
            body=json.dumps({'error':'runner failed: insufficient memory, requested 12634 MB'}).encode()
            self.send_response(500)
            self.send_header('Content-Length',str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    with ExitStack() as stack:
        temp=Path(stack.enter_context(tempfile.TemporaryDirectory()))
        backend=ThreadingHTTPServer(('127.0.0.1',0),FailingOllama)
        serve(backend,stack)
        configs=temp/'configs'
        configs.mkdir()
        (configs/'simulation.json').write_text(json.dumps({'models':[
            {'id':'SIMULATED-a','adapter':'mock'},{'id':'SIMULATED-b','adapter':'mock'}]}))
        (configs/'ollama-error.json').write_text(json.dumps({'models':[dict(
            id='clef-flash-ollama',adapter='systemone_http',url=f'http://127.0.0.1:{backend.server_port}/v1/systemone',
            request_model='clef-flash:9b',image_encoding='base64',max_choices=26)]}))
        studio=Studio(configs,temp/'jobs')
        stack.callback(studio.jobs.close)
        server=make_server(0,studio)
        serve(server,stack)
        origin=f'http://127.0.0.1:{server.server_port}'
        photo=temp/'test.png'
        Image.new('RGB',(600,400),'#739bbe').save(photo)
        p=stack.enter_context(sync_playwright())
        browser=p.chromium.launch(headless=True,channel="chromium")
        stack.callback(browser.close)
        page=browser.new_page(viewport={'width':1440,'height':1000})
        errors=[]
        page.on('pageerror',lambda error:errors.append(str(error)))
        page.goto(origin)
        page.wait_for_load_state('networkidle')
        expect(page.get_by_role('heading',name='Teste de imagem',exact=True)).to_be_visible()
        page.locator('#config').select_option('simulation')
        page.locator('#image').set_input_files(photo)
        expect(page.locator('#preview')).to_be_visible()
        page.get_by_role('button',name='Avaliar imagem',exact=True).click()
        expect(page.locator('#live-state')).to_have_text('Concluído',timeout=20000)
        expect(page.locator('#answers article')).to_have_count(2)
        expect(page.locator('#answers')).to_contain_text('Simulação de infraestrutura')
        with page.expect_download() as download:
            page.get_by_role('button',name='Baixar JSON').click()
        saved=args.output/'interactive.json'
        download.value.save_as(saved)
        assert len(json.loads(saved.read_text())['results'])==2
        page.screenshot(path=str(args.output/'desktop.png'),full_page=True)
        page.get_by_role('link',name='Benchmark',exact=True).click()
        page.locator('#limit').fill('2')
        page.locator('#repetitions').fill('1')
        page.get_by_role('button',name='Iniciar benchmark').click()
        expect(page.locator('#benchmark-state')).to_have_text('Concluído',timeout=30000)
        page.get_by_role('link',name='Resultados',exact=True).click()
        expect(page.locator('#artifacts')).to_contain_text('comparison_1.png',timeout=10000)
        page.locator('#artifact-filter').fill('.csv')
        expect(page.locator('#artifacts')).to_contain_text('metrics.csv')
        page.screenshot(path=str(args.output/'results.png'),full_page=True)
        page.get_by_role('link',name='Diagnóstico',exact=True).click()
        expect(page.locator('#diagnostics')).to_contain_text('RAM total',timeout=15000)
        page.get_by_role('link',name='Teste de imagem',exact=True).click()
        page.locator('#config').select_option('ollama-error')
        page.get_by_role('button',name='Avaliar imagem',exact=True).click()
        expect(page.locator('#live-state')).to_have_text('Concluído com falhas',timeout=15000)
        expect(page.locator('#answers')).to_contain_text('Memória insuficiente')
        expect(page.locator('#answers')).to_contain_text('12634 MB')
        page.screenshot(path=str(args.output/'error.png'),full_page=True)
        page.set_viewport_size({'width':390,'height':844})
        page.screenshot(path=str(args.output/'mobile.png'),full_page=True)
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), 'Mobile overflow'
        assert not errors,errors
        print('Browser smoke OK: upload, 2 models, JSON, benchmark, report, diagnostics, detailed error, mobile.')
        (args.output/'summary.json').write_text(json.dumps({'status':'passed','page_errors':errors,'simulated':True},indent=2))


if __name__=='__main__':
    main()

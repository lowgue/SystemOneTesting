"""ASGI application integration: REST, events, model download and shutdown."""
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from vision_bench.webui.api import create_app
from vision_bench.webui.services import Studio


class WebAPITests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        configs=self.root/'configs';configs.mkdir()
        (configs/'mock.json').write_text(json.dumps({'models':[{'id':'SIMULATED-a','adapter':'mock'}]}))
        frontend=self.root/'frontend';frontend.mkdir();(frontend/'index.html').write_text('<html>React app</html>')
        self.studio=Studio(configs,self.root/'jobs')
        self.client=TestClient(create_app(self.studio,frontend))
        self.client.__enter__()
        image=BytesIO();Image.new('RGB',(10,10)).save(image,format='PNG')
        self.body={'kind':'live','config':'mock','contract':'ollama','models':['SIMULATED-a'],'image_data':base64.b64encode(image.getvalue()).decode()}

    def tearDown(self):
        self.client.__exit__(None,None,None)
        self.temp.cleanup()

    def finish(self,job_id):
        deadline=time.monotonic()+30
        while time.monotonic()<deadline:
            state=self.client.get('/api/jobs/'+job_id).json()
            if state['state'] not in {'running','queued'}:return state
            time.sleep(.1)
        self.fail('Worker timeout')

    def test_rest_event_stream_history_and_frontend(self):
        self.assertEqual(self.client.get('/').status_code,200)
        self.assertTrue(self.client.get('/api/health').json()['frontend_built'])
        self.assertEqual(self.client.get('/api/catalog').status_code,200)
        response=self.client.post('/api/jobs',json=self.body)
        self.assertEqual(response.status_code,202,response.text)
        job_id=response.json()['id']
        self.assertEqual(response.json()['settings'], {
            'config':'mock', 'contract':'ollama', 'models':['SIMULATED-a'],
            'device':'cpu', 'threads':8,
        })
        response=self.client.get('/api/jobs/'+job_id+'/events')
        self.assertIn('text/event-stream',response.headers['content-type'])
        states=[json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]
        self.assertEqual(states[-1]['state'],'complete')
        self.assertTrue(states[-1]['results'][0]['simulated'])
        self.assertEqual(self.client.get('/api/jobs').json()[0]['id'],job_id)

    def test_input_and_origin_validation(self):
        self.assertEqual(self.client.post('/api/jobs',json={**self.body,'models':[]}).status_code,400)
        self.assertEqual(self.client.post('/api/jobs',content=b'bad-json').status_code,400)
        self.assertEqual(self.client.post('/api/jobs',json=self.body,headers={'Origin':'https://unrelated.example'}).status_code,403)
        self.assertEqual(self.client.get('/files/studio/secret.json').status_code,404)

    def test_protocol_preview_and_verified_dataset_export(self):
        body={**self.body,'kind':'benchmark','dataset':'historical-color-eval','contract':'historical','comparison':'all','limit':2,'repetitions':3}
        response=self.client.post('/api/protocol-preview',json=body)
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(response.json()['planned'],6)
        self.assertEqual(list(response.json()['protocols'][0]['examples'][0]['request']['questions']),['color'])
        self.assertIsNone(self.studio.jobs.current)
        response=self.client.get('/api/datasets/portable/bundle')
        self.assertEqual(response.status_code,200)
        import zipfile
        with zipfile.ZipFile(BytesIO(response.content)) as bundle:
            manifest=json.loads(bundle.read('manifest.json'))
            self.assertEqual(len(manifest['items']),22)
            import hashlib
            for item in manifest['items']:self.assertEqual(hashlib.sha256(bundle.read(item['rel_path'])).hexdigest(),item['sha256'])

    def test_model_pull_streaming_through_owned_worker(self):
        observed=[]
        class Ollama(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                observed.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
                self.send_response(200);self.end_headers()
                for item in [{'status':'pulling','total':100,'completed':50},{'status':'success'}]:
                    self.wfile.write((json.dumps(item)+'\n').encode());self.wfile.flush()
        server=ThreadingHTTPServer(('127.0.0.1',0),Ollama)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            with patch.dict(os.environ,{'OLLAMA_SYSTEMONE_URL':f'http://127.0.0.1:{server.server_port}/v1/systemone'}):
                response=self.client.post('/api/ollama/pull',json={})
            self.assertEqual(response.status_code,202,response.text)
            state=self.finish(response.json()['id'])
            self.assertEqual(state['state'],'complete',state)
            self.assertEqual(state['download']['status'],'success')
            self.assertEqual(observed,[{'model':'clef-flash:9b','stream':True}])
        finally:
            server.shutdown();server.server_close();thread.join()

    def test_missing_frontend_reports_actionable_error(self):
        with TestClient(create_app(self.studio,self.root/'absent')) as client:
            response=client.get('/')
            self.assertEqual(response.status_code,503)
            self.assertIn('Docker Compose',response.json()['error'])


if __name__=='__main__':unittest.main()

"""Upload, ground truth and automatic evaluation through the real ASGI API."""
import base64
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import threading
from io import BytesIO
import json
from pathlib import Path
import tempfile
import time
import unittest
from fastapi.testclient import TestClient
from PIL import Image
from vision_bench.webui.api import create_app
from vision_bench.webui.services import Studio


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        config=self.root/'configs';config.mkdir()
        (config/'mock.json').write_text(json.dumps({'models':[{'id':'SIMULATED-a','adapter':'mock'}]}))
        self.studio=Studio(config,self.root/'jobs')
        self.client=TestClient(create_app(self.studio));self.client.__enter__()
        self.dataset=self.client.post('/api/datasets',json={'name':'Meu dataset','domain':'vehicle'}).json()['id']
        for filename,color in [('a.jpg','black'),('b.jpg','white')]:
            buffer=BytesIO();Image.new('RGB',(30,20),color).save(buffer,format='PNG')
            encoded=base64.b64encode(buffer.getvalue()).decode()
            response=self.client.post(f'/api/datasets/{self.dataset}/images',json={'filename':filename,'image_data':encoded})
            self.assertEqual(response.status_code,200,response.text)
        self.csv='arquivo;presença;marca;cor;carroceria\na.jpg;sim;Audi;preto;sedã\nb.jpg;;;branco;\n'

    def tearDown(self):
        self.client.__exit__(None,None,None);self.temp.cleanup()

    def finalize(self,text):
        return self.client.post(f'/api/datasets/{self.dataset}/finalize',json={'csv':text,'contract':'ollama'})

    def test_import_ground_truth_evaluate_and_export(self):
        response=self.finalize(self.csv)
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(response.json()['complete_labels'],1)
        self.assertEqual(response.json()['annotated'],2)
        catalog=self.client.get('/api/catalog').json()
        self.assertTrue(any(d['id']==self.dataset and d['events']==2 for d in catalog['datasets']))
        request=dict(kind='benchmark',config='mock',contract='ollama',models=['SIMULATED-a'],dataset=self.dataset,repetitions=2,threads=1)
        response=self.client.post('/api/jobs',json=request)
        self.assertEqual(response.status_code,202,response.text)
        job=response.json()['id'];deadline=time.monotonic()+30
        while time.monotonic()<deadline:
            state=self.client.get('/api/jobs/'+job).json()
            if state['state'] not in {'queued','running'}:break
            time.sleep(.1)
        self.assertEqual(state['state'],'complete',state)
        self.assertEqual(state['dataset'],self.dataset)
        evaluation=self.client.get(f'/api/jobs/{job}/evaluation')
        self.assertEqual(evaluation.status_code,200,evaluation.text)
        result=evaluation.json()['models'][0]
        self.assertEqual(result['photos'],2)
        self.assertEqual(result['requests'],4)
        self.assertEqual(result['exact_accuracy'],1)
        self.assertEqual(result['field_accuracy'],.8)
        self.assertEqual(result['field_n'],5)
        for filename in ['evaluation.json','models.csv','photos.csv','attributes.csv','accuracy.png','accuracy.pdf']:
            self.assertEqual(self.client.get(f'/files/studio/{job}/report/{filename}').status_code,200)
        photo=self.client.get(f'/api/datasets/{self.dataset}/photos/photo-000001?thumbnail=true')
        self.assertEqual(photo.headers['content-type'],'image/jpeg')

    def test_warmup_failure_does_not_abort_measured_dataset(self):
        class Failure(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length']))
                self.send_response(500);self.end_headers();self.wfile.write(b'failed')
        server=ThreadingHTTPServer(('127.0.0.1',0),Failure)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        config=self.root/'configs/http.json'
        config.write_text(json.dumps({'models':[{'id':'failing','adapter':'systemone_http','url':f'http://127.0.0.1:{server.server_port}/v1/systemone'}]}))
        try:
            self.finalize(self.csv)
            response=self.client.post('/api/jobs',json=dict(kind='benchmark',config='http',contract='ollama',models=['failing'],dataset=self.dataset,repetitions=1))
            self.assertEqual(response.status_code,202,response.text)
            job=response.json()['id'];deadline=time.monotonic()+30
            while time.monotonic()<deadline:
                state=self.client.get('/api/jobs/'+job).json()
                if state['state'] not in {'queued','running'}:break
                time.sleep(.1)
            self.assertEqual(state['state'],'complete',state)
            result=self.client.get(f'/api/jobs/{job}/evaluation').json()['models'][0]
            self.assertEqual(result['errors'],2)
            self.assertEqual(result['field_accuracy'],0)
            self.assertEqual(result['warmup_failures'],1)
            self.assertIsNone(result['latency']['p50_ms'])
        finally:
            server.shutdown();server.server_close();thread.join()

    def test_retry_is_idempotent_and_only_drafts_can_be_discarded(self):
        buffer=BytesIO();Image.new('RGB',(30,20),'black').save(buffer,format='PNG')
        encoded=base64.b64encode(buffer.getvalue()).decode()
        response=self.client.post(f'/api/datasets/{self.dataset}/images',json={'filename':'a.jpg','image_data':encoded})
        self.assertEqual(response.json()['events'],2)
        self.finalize(self.csv)
        self.assertEqual(self.client.post(f'/api/datasets/{self.dataset}/discard',json={}).status_code,400)
        draft=self.client.post('/api/datasets',json={'name':'Rascunho','domain':'person'}).json()['id']
        self.assertEqual(self.client.post(f'/api/datasets/{draft}/discard',json={}).status_code,200)
        self.assertFalse((self.studio.datasets.root/draft).exists())

    def test_no_ground_truth_remains_unlabeled(self):
        result=self.finalize('').json()
        self.assertEqual(result['annotated'],0)
        self.assertEqual(result['label_status'],'unlabeled')

    def test_invalid_labels_do_not_publish_partial_dataset(self):
        for csv in ['arquivo,cor\nmissing.jpg,black\n','arquivo,cor\na.jpg,invented-color\n','arquivo,cor\na.jpg,black\na.jpg,white\n']:
            with self.subTest(csv=csv):
                response=self.finalize(csv)
                self.assertEqual(response.status_code,400,response.text)
                self.assertNotIn(self.dataset,[d['id'] for d in self.client.get('/api/catalog').json()['datasets']])
        self.assertEqual(self.finalize(self.csv).status_code,200)

    def test_finalized_datasets_are_immutable_and_paths_are_confined(self):
        self.finalize(self.csv)
        self.assertEqual(self.client.post(f'/api/datasets/{self.dataset}/images',json={'filename':'new.jpg','image_data':'bad'}).status_code,400)
        self.assertEqual(self.client.post('/api/datasets/import-invalid/images',json={'filename':'../../outside.jpg','image_data':'bad'}).status_code,400)
        self.assertEqual(self.client.get(f'/api/datasets/{self.dataset}/photos/unknown').status_code,400)

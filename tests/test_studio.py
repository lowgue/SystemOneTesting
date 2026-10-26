"""Studio integration checks: real HTTP and workers, simulated models only."""
import base64
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from unittest.mock import patch

from PIL import Image

from vision_bench.core import load_json
from vision_bench.webui.server import Studio, make_server


class StudioTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        config = self.root / 'configs'
        config.mkdir()
        (config / 'fixture.json').write_text(json.dumps({'models':[
            {'id':'SIMULATED-a','adapter':'mock'}, {'id':'SIMULATED-b','adapter':'mock'}]}))
        self.studio = Studio(config, self.root / 'jobs')
        self.server = make_server(0, self.studio)
        self.thread = threading.Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start()
        self.origin = f'http://127.0.0.1:{self.server.server_port}'
        buffer = BytesIO()
        Image.new('RGB',(12,12),'blue').save(buffer,format='PNG')
        self.body = dict(kind='live', config='fixture', contract='ollama',
                         models=['SIMULATED-a','SIMULATED-b'], domain='vehicle',
                         image_data=base64.b64encode(buffer.getvalue()).decode(), threads=1)

    def tearDown(self):
        self.studio.jobs.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temp.cleanup()

    def call(self,path,body=None,headers=None):
        data=json.dumps(body).encode() if body is not None else None
        request=urllib.request.Request(self.origin+path,data=data,headers=headers or {})
        with urllib.request.urlopen(request,timeout=10) as response:
            return json.load(response)

    def finish(self,job):
        deadline=time.monotonic()+40
        while time.monotonic()<deadline:
            result=self.studio.jobs.status(job['id'])
            if result['state'] not in {'queued','running'}:
                return result
            time.sleep(.1)
        self.fail('Worker did not finish')

    def test_startup_never_loads_models_and_serves_offline_assets(self):
        with patch('vision_bench.adapters.create_adapter',side_effect=AssertionError('must not load')):
            self.assertEqual(self.call('/api/catalog')['configs'][0]['id'],'fixture')
            self.assertEqual(self.call('/api/status')['app'],'vision-studio')
            for path in ['/','/app.js','/style.css']:
                with urllib.request.urlopen(self.origin+path) as response:
                    self.assertEqual(response.status,200)
                    self.assertTrue(response.read())

    def test_live_shared_image_and_contract_two_models(self):
        job=self.call('/api/jobs',self.body)
        result=self.finish(job)
        self.assertEqual(result['state'],'complete',result)
        self.assertEqual(len(result['results']),2)
        self.assertTrue(all(r['simulated'] and r['valid'] for r in result['results']))
        self.assertIn('image_sha256',result)
        self.assertIn('contract_sha256',result)
        self.assertTrue(all('load_ms' in r and 'elapsed_ms' in r for r in result['results']))
        self.assertEqual(self.call('/api/jobs/'+job['id'])['id'],job['id'])

    def test_missing_python_dependencies_do_not_block_other_models(self):
        from vision_bench.webui.worker import run
        import builtins
        config=self.root/'configs/fixture.json'
        config.write_text(json.dumps({'models':[
            {'id':'blip-unavailable','adapter':'blip','model':'unused','revision':'0'*40},
            {'id':'SIMULATED-a','adapter':'mock'}]}))
        request,image=self.studio.prepare({**self.body,'models':['blip-unavailable','SIMULATED-a']})
        folder=self.root/'isolated'
        folder.mkdir()
        image.save(folder/'input.png')
        (folder/'request.json').write_text(json.dumps(request))
        (folder/'status.json').write_text(json.dumps({'state':'queued','results':[]}))
        real_import=builtins.__import__
        def unavailable(name,*args,**kwargs):
            if name=='torch':
                raise ModuleNotFoundError('No module named torch')
            return real_import(name,*args,**kwargs)
        with patch('builtins.__import__',side_effect=unavailable):
            run(folder)
        result=load_json(folder/'status.json')
        self.assertEqual(result['state'],'complete',result)
        self.assertEqual(result['errors'],1)
        self.assertIn('No module named torch',result['results'][0]['error'])
        self.assertTrue(result['results'][1]['valid'])

    def test_invalid_inputs_and_origin_are_rejected_before_worker(self):
        for change in [dict(models=['missing']),dict(config='../secrets'),dict(image_data='bad'),
                       dict(domain='unsupported'),dict(threads=0),dict(device='shell;command')]:
            with self.subTest(change=change), self.assertRaises(urllib.error.HTTPError) as caught:
                self.call('/api/jobs',{**self.body,**change})
            self.assertEqual(caught.exception.code,400)
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.call('/api/jobs',self.body,{'Origin':'https://unrelated.example'})
        self.assertEqual(caught.exception.code,403)
        self.assertIsNone(self.studio.jobs.current)

    def test_busy_cancel_and_persisted_state(self):
        real_popen=subprocess.Popen
        def sleeping_worker(*args,**kwargs):
            return real_popen([sys.executable,'-c','import time; time.sleep(30)'],**kwargs)
        with patch('vision_bench.webui.jobs.subprocess.Popen',side_effect=sleeping_worker):
            job=self.call('/api/jobs',self.body)
            with self.assertRaises(urllib.error.HTTPError) as caught:
                self.call('/api/jobs',self.body)
            self.assertEqual(caught.exception.code,409)
            result=self.call('/api/jobs/'+job['id']+'/cancel',{})
            self.assertEqual(result['state'],'cancelled')
            self.assertIsNotNone(self.studio.jobs.process.poll())
            self.assertEqual(load_json(self.studio.jobs.folder(job['id'])/'status.json')['state'],'cancelled')

    def test_benchmark_generates_report_and_exposes_only_output_files(self):
        job=self.call('/api/jobs',{**self.body,'kind':'benchmark','dataset':'portable','repetitions':1,'limit':2})
        result=self.finish(job)
        self.assertEqual(result['state'],'complete',result)
        artifacts=self.call('/api/artifacts')
        charts=[f for f in artifacts if f['name'].endswith('.png') and job['id'] in f['name']]
        self.assertTrue(charts)
        with urllib.request.urlopen(self.origin+charts[0]['url']) as response:
            self.assertTrue(response.read().startswith(b'\x89PNG'))
        self.assertFalse(any('input.png' in f['name'] or 'request.json' in f['name'] for f in artifacts))
        for path in ['/files/studio/../../README.md','/files/studio/'+job['id']+'/input.png','/api/jobs/not-a-job']:
            with self.assertRaises(urllib.error.HTTPError) as caught:
                self.call(path)
            self.assertEqual(caught.exception.code,404)


if __name__=='__main__':
    unittest.main()

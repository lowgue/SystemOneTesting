"""Portable framework integration tests: no Hub downloads, real models or paid API calls."""
from contextlib import redirect_stdout,redirect_stderr
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from io import StringIO
import json
import base64
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
import zipfile

from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from vision_bench.adapters import SystemOneHTTP,endpoint
from vision_bench.cli import main
from vision_bench.core import DEFAULT_CONTRACT,load_json,sha256,validate_manifest,pack_dataset,unpack_dataset,finalize,normalize_text
from vision_bench.runner import validate_config,preflight


class PortableTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.contract=load_json(DEFAULT_CONTRACT)
        Image.new('RGB',(32,24),'white').save(self.root/'photo.png')
        self.manifest=self.root/'manifest.json'
        self.manifest.write_text(json.dumps(dict(items=[dict(id='fixture',domain='vehicle',rel_path='photo.png',sha256=sha256(self.root/'photo.png'))])))
        self.config=self.root/'models.json';self.config.write_text(json.dumps(dict(models=[dict(id='SIMULATED-test',adapter='mock')])))
    def tearDown(self):self.temp.cleanup()
    def invoke(self,argv):
        with redirect_stdout(StringIO()) as out,redirect_stderr(StringIO()) as err:code=main(argv)
        return code,out.getvalue(),err.getvalue()
    def common(self):return ['--config',str(self.config),'--manifest',str(self.manifest),'--data-root',str(self.root)]
    def test_dataset_roundtrip_and_hash(self):
        bundle=self.root/'dataset.zip';p=pack_dataset(self.manifest,self.root,bundle,self.contract)
        self.assertEqual(p['unique_images'],1)
        manifest=unpack_dataset(bundle,self.root/'imported',self.contract)
        self.assertEqual(validate_manifest(manifest,manifest.parent,self.contract)['items'][0]['sha256'],sha256(self.root/'photo.png'))
    def test_corrupt_archive_fails_before_extraction(self):
        bundle=self.root/'bad.zip'
        doc=load_json(self.manifest)
        with zipfile.ZipFile(bundle,'w') as f:f.writestr('manifest.json',json.dumps(doc));f.writestr('photo.png','not the image')
        with self.assertRaisesRegex(ValueError,'checksum'):unpack_dataset(bundle,self.root/'imported',self.contract)
        self.assertFalse((self.root/'imported').exists())
    def test_zip_traversal_rejected(self):
        bundle=self.root/'traversal.zip'
        with zipfile.ZipFile(bundle,'w') as f:f.writestr('manifest.json',self.manifest.read_text());f.writestr('../escape.png','unsafe')
        with self.assertRaisesRegex(ValueError,'escapes'):unpack_dataset(bundle,self.root/'imported',self.contract)
        self.assertFalse((self.root/'escape.png').exists())
    def test_manifest_image_missing_is_actionable(self):
        (self.root/'photo.png').unlink()
        with self.assertRaisesRegex(ValueError,'import the dataset bundle'):validate_manifest(self.manifest,self.root,self.contract)
    def test_dry_run_does_not_create_output_or_load_adapter(self):
        with patch('vision_bench.runner.create_adapter',side_effect=AssertionError('must not load')):
            code,_,error=self.invoke(['run',*self.common(),'--output',str(self.root/'run'),'--dry-run'])
        self.assertEqual(code,0,error);self.assertFalse((self.root/'run').exists())
    def test_full_cli_resume_and_report(self):
        run=self.root/'run'
        argv=['run',*self.common(),'--output',str(run),'--repetitions','2']
        code,_,error=self.invoke(argv);self.assertEqual(code,0,error)
        before=(run/'predictions.jsonl').read_text()
        code,_,error=self.invoke([*argv,'--resume']);self.assertEqual(code,0,error)
        self.assertEqual(before,(run/'predictions.jsonl').read_text())
        code,_,error=self.invoke(['report',str(run),'--output',str(self.root/'report')]);self.assertEqual(code,0,error)
        self.assertIn('SIMULAÇÃO',(self.root/'report/README.md').read_text())
        code,_,error=self.invoke([*argv,'--resume','--threads','2']);self.assertEqual(code,2);self.assertIn('configuration changed',error)
    def test_invalid_config_and_secret_urls(self):
        with self.assertRaisesRegex(ValueError,'full commit'):validate_config(dict(models=[dict(id='laya',adapter='laya',revision='main')]))
        with self.assertRaisesRegex(ValueError,'literal credentials'):validate_config(dict(models=[dict(id='x',adapter='systemone_http',token='secret')]))
        for url in ['http://user:password@localhost/v1/systemone','http://localhost/v1/systemone?token=secret']:
            with self.assertRaises(ValueError):endpoint({'url':url})
    def test_three_models_share_contract_and_report(self):
        self.config.write_text(json.dumps(dict(models=[dict(id=f'SIMULATED-{i}',adapter='mock') for i in range(3)])))
        target=self.root/'three'
        code,_,error=self.invoke(['run',*self.common(),'--output',str(target),'--repetitions','2'])
        self.assertEqual(code,0,error)
        rows=[json.loads(line) for line in (target/'predictions.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows),6)
        self.assertEqual({r['model'] for r in rows},{f'SIMULATED-{i}' for i in range(3)})
        code,_,error=self.invoke(['report',str(target),'--output',str(self.root/'three-report')])
        self.assertEqual(code,0,error)
        self.assertEqual(len(load_json(self.root/'three-report/metrics.json')),3)
    def test_parser_and_gate_preserve_errors(self):
        self.assertEqual(normalize_text(self.contract,'vehicle','body','car'),'parse_error')
        self.assertEqual(normalize_text(self.contract,'person','helmet','no helmet'),'no')
        result=finalize(self.contract,'vehicle',dict(presence='yes',make='unknown',color='parse_error',body='suv'))
        self.assertFalse(result['valid']);self.assertEqual(result['facts'],{'body':'suv'})
    def test_ollama_contract_enforces_choice_limit(self):
        spec=dict(id='ollama',adapter='systemone_http',url='http://127.0.0.1:11434/v1/systemone',max_choices=26)
        _,_,issues=preflight(dict(models=[spec]),self.manifest,self.contract,self.root)
        self.assertTrue(any('exceeds 26' in issue for issue in issues))
        contract=load_json(ROOT/'benchmarks/contracts/description_ollama_v1.json')
        _,_,issues=preflight(dict(models=[spec]),self.manifest,contract,self.root)
        self.assertEqual(issues,[])
        validate_manifest(ROOT/'benchmarks/datasets/portable_vehicles.json',ROOT,contract)
    def test_cli_errors_have_nonzero_exit(self):
        code,_,_=self.invoke(['run',*self.common(),'--output',str(self.root/'run'),'--limit','0']);self.assertEqual(code,2)
    def test_http_transport_and_wrapped_response(self):
        contract=self.contract;observed={}
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                observed.update(body)
                answers={k:{'type':'choice','choice':next(iter(q['criteria']))} for k,q in body['questions'].items()}
                payload=json.dumps(dict(success=True,result=dict(answers=answers,usage={'input_tokens':12}))).encode()
                self.send_response(200);self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            spec=dict(id='http',adapter='systemone_http',url=f'http://127.0.0.1:{server.server_port}/v1/systemone',url_env='PORTABLE_TEST_ENDPOINT')
            adapter=SystemOneHTTP(spec,contract)
            answers,raw=adapter.predict(Image.new('RGB',(32,32)),'vehicle')
            self.assertEqual(set(answers),set(contract['domains']['vehicle']['questions']))
            self.assertTrue(observed['images'][0].startswith('data:image/png;base64,'))
            self.assertEqual(len(observed['questions']),4);self.assertEqual(raw['usage']['input_tokens'],12)
            ollama=SystemOneHTTP({**spec,'image_encoding':'base64','request_model':'clef-flash:9b'},contract)
            ollama.predict(Image.new('RGB',(16,16)),'vehicle')
            self.assertEqual(observed['model'],'clef-flash:9b')
            self.assertTrue(base64.b64decode(observed['images'][0],validate=True).startswith(b'\x89PNG'))
            self.assertNotIn('data:',observed['images'][0])
            self.config.write_text(json.dumps(dict(models=[spec])))
            code,_,error=self.invoke(['run',*self.common(),'--output',str(self.root/'http-run'),'--repetitions','1'])
            self.assertEqual(code,0,error)
            row=json.loads((self.root/'http-run/predictions.jsonl').read_text())
            self.assertEqual(row['resource_scope'],'client_only');self.assertIsNone(row['error'])
            with patch.dict(os.environ,{'PORTABLE_TEST_ENDPOINT':'http://127.0.0.1:1/v1/systemone'}):
                code,_,error=self.invoke(['run',*self.common(),'--output',str(self.root/'http-run'),'--repetitions','1','--resume'])
            self.assertEqual(code,2);self.assertIn('Resume rejected',error)
        finally:server.shutdown();server.server_close();thread.join()
    def test_http_error_does_not_leak_response_credentials(self):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length']))
                self.send_response(403);self.end_headers();self.wfile.write(b'SECRET-test-token')
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            with patch.dict(os.environ,{'BENCH_TEST_TOKEN':'SECRET-test-token'}):
                adapter=SystemOneHTTP(dict(url=f'http://127.0.0.1:{server.server_port}/v1/systemone',token_env='BENCH_TEST_TOKEN'),self.contract)
                with self.assertRaisesRegex(RuntimeError,'HTTP 403') as ctx:adapter.predict(Image.new('RGB',(16,16)),'vehicle')
                self.assertNotIn('SECRET',str(ctx.exception))
        finally:server.shutdown();server.server_close();thread.join()

if __name__=='__main__':unittest.main()

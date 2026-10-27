"""Adapters are loaded lazily: HTTP benchmarks do not import torch or transformers."""
from __future__ import annotations
import base64
from io import BytesIO
import importlib
import os
from pathlib import Path
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
import json
import time

from .core import ROOT,normalize_text


def endpoint(spec):
    url=os.environ.get(spec.get('url_env',''),spec.get('url',''))
    if spec.get('account_env'):
        account=os.environ.get(spec['account_env'],'')
        if not re.fullmatch(r'[A-Za-z0-9_-]+',account):raise ValueError('Set the configured Cloudflare account environment variable')
        url=f'https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/@cf/cloudflare/clef-flash'
    parsed=urllib.parse.urlsplit(url)
    if parsed.scheme not in {'http','https'} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('Endpoint must be HTTP(S), without embedded credentials, query or fragment')
    return url

class MockAdapter:
    """Contract fixture only. The model ID makes simulated results explicit."""
    def __init__(self,spec,contract,**kwargs):self.contract=contract
    def predict(self,image,domain):
        qs=self.contract['domains'][domain]['questions'];a={k:next(iter(q['criteria'])) for k,q in qs.items()}
        return a,{'simulated':True,'answers':{k:{'choice':v} for k,v in a.items()}}
    def synchronize(self):pass
    def close(self):pass

class SystemOneHTTP(MockAdapter):
    def __init__(self,spec,contract,**kwargs):
        self.spec=spec;self.contract=contract;self.url=endpoint(spec)
        self.diagnostics=kwargs.get('diagnostics',False)
        env=spec.get('token_env');self.token=os.environ.get(env,'') if env else ''
        if env and not self.token:raise ValueError(f'Set {env} for this backend')
    def predict(self,image,domain):
        buffer=BytesIO();image.save(buffer,format='PNG')
        encoded=base64.b64encode(buffer.getvalue()).decode('ascii')
        image_value=encoded if self.spec.get('image_encoding')=='base64' else 'data:image/png;base64,'+encoded
        payload=dict(state='Describe only the visible attributes of the target in this image.',images=[image_value],questions=self.contract['domains'][domain]['questions'])
        if self.spec.get('request_model'):payload['model']=self.spec['request_model']
        headers={'Content-Type':'application/json'}
        if self.token:headers['Authorization']='Bearer '+self.token
        request=urllib.request.Request(self.url,data=json.dumps(payload).encode(),headers=headers,method='POST')
        try:
            with urllib.request.urlopen(request,timeout=self.spec.get('timeout_s',300)) as response:raw=json.load(response)
        except urllib.error.HTTPError as exc:
            # Detailed local diagnostics are opt-in for the studio. Remote errors
            # stay sanitized; hosted error bodies may contain credentials.
            if self.diagnostics and urllib.parse.urlsplit(self.url).hostname in {'127.0.0.1','localhost','::1','ollama','host.docker.internal'} and not self.token:
                detail=exc.read(16384).decode('utf-8',errors='replace')
                try:detail=str(json.loads(detail).get('error',detail))
                except (ValueError,AttributeError):pass
                detail=detail[:2000]
                if any(t in detail.lower() for t in ['insufficient memory','out of memory','not enough memory']):
                    detail='Memória insuficiente ao carregar o modelo. Detalhes do servidor: '+detail
                raise RuntimeError(f'HTTP {exc.code}: {detail}') from None
            raise RuntimeError(f'HTTP {exc.code}; verify System One route, model capability and context budget') from None
        except urllib.error.URLError as exc:
            raise RuntimeError('Não foi possível conectar ao servidor. Verifique o serviço Ollama/llama-server e a configuração do endpoint.') from None
        if not isinstance(raw,dict) or raw.get('success') is False:raise RuntimeError('System One backend reported failure')
        raw=raw.get('result',raw)
        if not isinstance(raw,dict) or not isinstance(raw.get('answers'),dict):raise ValueError('Backend returned no answers map')
        answers={k:str(raw['answers'].get(k,{}).get('choice','parse_error')) for k in self.contract['domains'][domain]['questions']}
        for value in raw['answers'].values():
            if isinstance(value,dict) and value.get('truncated'):raise ValueError('Backend truncated a question or options')
        return answers,raw

class BLIP(MockAdapter):
    def __init__(self,spec,contract,device='cpu',offline=False,**kwargs):
        from transformers import AutoProcessor,AutoModelForVisualQuestionAnswering
        self.contract=contract;self.device=device;self.spec=spec
        self.processor=AutoProcessor.from_pretrained(spec['model'],revision=spec['revision'],local_files_only=offline)
        self.model=AutoModelForVisualQuestionAnswering.from_pretrained(spec['model'],revision=spec['revision'],local_files_only=offline).to(device).eval()
    def synchronize(self):
        if self.device.startswith('cuda'):
            import torch
            torch.cuda.synchronize(self.device)
    def predict(self,image,domain):
        import torch
        from .protocols import prompts
        ps=prompts(self.spec,self.contract,domain);keys=list(ps);raw={};timings={}
        groups=[[key] for key in keys] if self.spec.get('profile')=='short_serial' else [keys]
        for group in groups:
            self.synchronize();start=time.perf_counter()
            inputs=self.processor(images=[image]*len(group),text=[ps[k] for k in group],padding=True,return_tensors='pt').to(self.device)
            with torch.inference_mode():tokens=self.model.generate(**inputs,max_new_tokens=64 if self.spec.get('profile')=='structured_json' else 12,do_sample=False)
            self.synchronize()
            raw.update(zip(group,self.processor.batch_decode(tokens,skip_special_tokens=True),strict=True))
            if len(group)==1:timings[group[0]]=(time.perf_counter()-start)*1000
        answers={}
        for k,v in raw.items():
            if self.spec.get('profile')=='structured_json':
                try:
                    obj=json.loads(v);answers[k]=str(obj[k]) if isinstance(obj,dict) and set(obj)=={k} else 'parse_error'
                    if answers[k] not in self.contract['domains'][domain]['questions'][k]['criteria']:answers[k]='parse_error'
                except (ValueError,KeyError,TypeError):answers[k]='parse_error'
            else:answers[k]=normalize_text(self.contract,domain,k,v)
        self.last_metrics=dict(inference_calls=len(groups),attribute_ms=timings)
        return answers,raw

class Laya(BLIP):
    def __init__(self,spec,contract,device='cpu',offline=False,**kwargs):
        sys.path.insert(0,str(ROOT/'vendor/laya-vision'))
        import torch
        import laya
        import laya.vlm as vlm
        def encode_images(self,pixel_values,pixel_attention_mask=None):
            result=self._image_features(pixel_values[None],pixel_attention_mask[None] if pixel_attention_mask is not None else None)
            return torch.stack(result) if isinstance(result,list) else result
        vlm.VLMDecisionModel.encode_images=encode_images
        self.contract=contract;self.device=device;self.spec=spec
        self.model=laya.load_vlm(spec['model'],revision=spec['revision'],device=device)
        # A different input budget is an explicit experimental configuration, not silent truncation.
        if spec.get('head_max_len'):self.model.cfg['head_max_len']=spec['head_max_len']
    def predict(self,image,domain):
        from .protocols import questions
        qs=questions(self.spec,self.contract,domain);raw={'answers':{}};timings={}
        groups=[{k:q} for k,q in qs.items()] if self.spec.get('profile')=='native_serial' else [qs]
        for group in groups:
            self.synchronize();start=time.perf_counter()
            response=self.model.predict(image,group,strict=True);self.synchronize()
            raw['answers'].update(response.get('answers',{}))
            if len(group)==1:timings[next(iter(group))]=(time.perf_counter()-start)*1000
        self.last_metrics=dict(inference_calls=len(groups),attribute_ms=timings)
        return {k:str(raw.get('answers',{}).get(k,{}).get('choice','parse_error')) for k in self.contract['domains'][domain]['questions']},raw

REGISTRY={'mock':MockAdapter,'systemone_http':SystemOneHTTP,'blip':BLIP,'laya':Laya}

def create_adapter(spec,contract,**context):
    adapter=spec['adapter']
    if adapter.startswith('python:'):
        module,cls=adapter.removeprefix('python:').rsplit(':',1);factory=getattr(importlib.import_module(module),cls)
    else:factory=REGISTRY[adapter]
    return factory(spec,contract,**context)

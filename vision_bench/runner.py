"""Serial, resumable comparison runner with frozen input and machine identity."""
from __future__ import annotations
from datetime import datetime,timezone
import gc
import importlib.metadata
import os
from pathlib import Path
import platform
import random
import re
import subprocess
import sys
import threading
import time

from PIL import Image
import psutil
from .adapters import create_adapter,endpoint,REGISTRY
from .core import ROOT,load_json,write_json,sha256,fingerprint,validate_manifest,resolve_image,finalize


def environment():
    packages={}
    for p in ['torch','torchvision','transformers','huggingface-hub','pillow','psutil','numpy','matplotlib']:
        try:packages[p]=importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:packages[p]=None
    cpu=platform.processor()
    if Path('/proc/cpuinfo').exists():cpu=next((s.split(':',1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')),cpu)
    return dict(hostname=platform.node(),platform=platform.platform(),python=platform.python_version(),cpu=cpu,
                logical_cpus=psutil.cpu_count(),physical_cpus=psutil.cpu_count(logical=False),ram_bytes=psutil.virtual_memory().total,packages=packages,
                installed_packages=dict(sorted((d.metadata['Name'],d.version) for d in importlib.metadata.distributions() if d.metadata.get('Name'))))

def validate_config(config,selected=None):
    specs=config.get('models',[])
    if not specs or len({s['id'] for s in specs})!=len(specs):raise ValueError('Config needs models with unique IDs')
    if selected:
        unknown=set(selected)-{s['id'] for s in specs}
        if unknown:raise ValueError(f'Unknown model IDs: {sorted(unknown)}')
        specs=[s for s in specs if s['id'] in selected]
    for spec in specs:
        if not re.fullmatch(r'[A-Za-z0-9_.-]+',spec['id']):raise ValueError('Model ID must be a filename-safe label')
        adapter=spec['adapter']
        if adapter not in REGISTRY and not adapter.startswith('python:'):raise ValueError(f'Unsupported adapter: {adapter}')
        for key in ['token','api_key','password','authorization']:
            if key in spec:raise ValueError('Use token_env, never literal credentials in the configuration')
        if adapter in {'blip','laya'} and not re.fullmatch(r'[a-f0-9]{40}',spec.get('revision','')):raise ValueError('Pin local Hub models to a full commit SHA')
        if spec.get('token_env') and not re.fullmatch(r'[A-Z_][A-Z0-9_]*',spec['token_env']):raise ValueError('token_env must name an environment variable')
        if spec.get('image_encoding','data_url') not in {'data_url','base64'}:raise ValueError('image_encoding must be data_url or base64')
    return specs

def preflight(config,manifest,contract,data_root,selected=None,require_dependencies=True):
    specs=validate_config(config,selected);doc=validate_manifest(manifest,data_root,contract);issues=[]
    env=environment()
    for spec in specs:
        adapter=spec['adapter']
        limit=spec.get('max_choices')
        if limit:
            for domain in contract['domains'].values():
                for key,question in domain['questions'].items():
                    if question['type']=='choice' and len(question['criteria'])>limit:
                        issues.append(f'{spec["id"]}: {key} exceeds {limit} choices; use the shared description_ollama_v1.json contract for all compared models')
        if adapter in {'blip','laya'}:
            if require_dependencies:
                for p in ['torch','torchvision','transformers','huggingface-hub']:
                    if not env['packages'].get(p):issues.append(f'{spec["id"]}: missing {p}; install requirements/cpu.txt or GPU environment')
            if adapter=='laya' and not (ROOT/'vendor/laya-vision/laya/vlm.py').is_file():issues.append('Initialize vendor/laya-vision with git submodule update --init --recursive')
        if adapter=='systemone_http':
            try:endpoint(spec)
            except ValueError as exc:issues.append(f'{spec["id"]}: {exc}')
            token=spec.get('token_env')
            if token and not os.environ.get(token):issues.append(f'{spec["id"]}: set {token}')
    return specs,doc,issues

class Resources:
    def __init__(self,server_pid=None):
        self.processes=[psutil.Process()]
        if server_pid is not None:
            proc=psutil.Process(int(server_pid));self.processes.extend([proc,*proc.children(recursive=True)])
        self.stop=threading.Event();self.peak=0;self.thread=threading.Thread(target=self.poll,daemon=True)
    def cpu(self):
        total=0
        for p in self.processes:
            try:c=p.cpu_times();total+=c.user+c.system
            except psutil.NoSuchProcess:pass
        return total
    def rss(self):
        total=0
        for p in self.processes:
            try:total+=p.memory_info().rss
            except psutil.NoSuchProcess:pass
        return total
    def poll(self):
        while not self.stop.is_set():self.peak=max(self.peak,self.rss());self.stop.wait(.05)
    def __enter__(self):self.peak=self.rss();self.before=self.cpu();self.thread.start();return self
    def __exit__(self,*args):self.stop.set();self.thread.join();self.peak=max(self.peak,self.rss());self.cpu_s=max(0,self.cpu()-self.before)

def server_metadata(spec):
    name=spec.get('server_metadata_env');path=os.environ.get(name,'') if name else ''
    if not path:return None
    metadata=load_json(path)
    if metadata.get('revision')!=spec.get('revision'):raise ValueError('Clef server metadata revision differs from model config')
    from urllib.parse import urlsplit
    target=urlsplit(endpoint(spec));observed=urlsplit(metadata.get('endpoint',''))
    if target.hostname not in {'localhost','127.0.0.1','::1'} or target.port!=observed.port or target.path!=observed.path:
        raise ValueError('Local server resource metadata does not match the requested endpoint')
    if metadata.get('pid'):
        p=psutil.Process(int(metadata['pid']))
        # Reject stale PID metadata after PID reuse.
        if abs(p.create_time()-metadata.get('process_created_s',0))>.5:raise ValueError('Server PID metadata is stale')
    return metadata

def run(args):
    contract=load_json(args.contract);config=load_json(args.config)
    specs,manifest,issues=preflight(config,args.manifest,contract,args.data_root,args.models)
    if issues:raise ValueError('\n'.join(issues))
    if args.repetitions<1 or args.threads<1:raise ValueError('threads and repetitions must be positive')
    if args.limit is not None and args.limit<1:raise ValueError('limit must be positive')
    items=manifest['items'][:args.limit] if args.limit else manifest['items']
    if args.dry_run:
        return dict(status='validated-no-inference',models=[s['id'] for s in specs],events=len(items),requests=len(items)*len(specs)*args.repetitions,
                    domains=sorted({it['domain'] for it in items}),note='No weights loaded and no HTTP inference sent. Runtime protocol support is checked by an actual smoke run.')
    if args.offline:os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
    env=environment();machine=args.machine_id or env['hostname'];code={str(p.relative_to(ROOT)):sha256(p) for p in sorted((ROOT/'vision_bench').rglob('*.py'))}
    if args.device.startswith('cuda') and any(s['adapter'] in {'blip','laya'} for s in specs):
        import torch
        env['cuda']=dict(torch_runtime=torch.version.cuda,devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
    external={s['id']:server_metadata(s) for s in specs if s['adapter']=='systemone_http'}
    frozen=dict(schema_version=1,models=specs,contract=contract,manifest=manifest,item_ids=[it['id'] for it in items],
                repetitions=args.repetitions,threads=args.threads,device=args.device,max_edge=args.max_edge,offline=args.offline,
                seed=args.seed,machine_id=machine,environment=env,code=code,
                resolved_http_endpoints={s['id']:endpoint(s) for s in specs if s['adapter']=='systemone_http'},
                external_provenance={k:{a:b for a,b in v.items() if a not in {'pid','process_created_s'}} if v else None for k,v in external.items()})
    identity=fingerprint(frozen);out=Path(args.output).resolve()
    if out.exists() and not args.resume:raise ValueError('Output exists; use a new run directory or --resume')
    if args.resume and not (out/'frozen_config.json').exists():raise ValueError('Cannot resume: frozen config missing')
    out.mkdir(parents=True,exist_ok=True)
    if args.resume:
        if fingerprint(load_json(out/'frozen_config.json'))!=identity:raise ValueError('Resume rejected: code, data, machine, dependencies or configuration changed')
    else:
        write_json(out/'frozen_config.json',frozen)
        write_json(out/'run.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),identity=identity,machine_id=machine,hardware=env,
            contract_hash=fingerprint(contract),dataset_hash=fingerprint([{k:v for k,v in it.items() if k in {'id','domain','sha256','expected','annotations','group_id'}} for it in items]),label_status=manifest.get('label_status','exploratory_unadjudicated'),
            resource_note='CPU/RSS scope is explicit per record; HTTP client resources are not server resources. Energy and monetary cost not measured.',
            simulated=any(s['adapter']=='mock' for s in specs)))
    data=out/'predictions.jsonl';done=set();measured_errors=0
    if data.exists():
        for line in data.read_text().splitlines():
            import json
            row=json.loads(line)
            if row['request_key'] in done:raise ValueError('Duplicate request key in existing results')
            done.add(row['request_key']);measured_errors+=bool(row.get('error'))
    def append(filename,row):
        import json
        with (out/filename).open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
    import importlib.util
    if any(s['adapter'] in {'laya','blip'} for s in specs):
        import torch
        torch.set_num_threads(args.threads);torch.manual_seed(args.seed)
    total=len(items)*len(specs)*args.repetitions
    for block in range(args.repetitions):
        ordered=specs[block%len(specs):]+specs[:block%len(specs)]
        if block%2:ordered=list(reversed(ordered))
        for spec in ordered:
            pending=[it for it in items if f'{block}/{spec["id"]}/{it["id"]}' not in done]
            if not pending:continue
            metadata=server_metadata(spec) if spec['adapter']=='systemone_http' else None
            pid=metadata.get('pid') if metadata else None
            scope='client_plus_local_server' if pid else ('client_only' if spec['adapter']=='systemone_http' else 'in_process_model')
            write_json(out/'status.json',dict(status='running',phase='warmup',model=spec['id'],completed=len(done),planned=total,errors=measured_errors))
            start=time.perf_counter();adapter=create_adapter(spec,contract,device=args.device,offline=args.offline)
            try:
                adapter.synchronize();load_s=time.perf_counter()-start
                warm=time.perf_counter()
                warmup_errors=[]
                for domain in sorted({it['domain'] for it in items}):
                    it=next(it for it in items if it['domain']==domain)
                    try:
                        with Image.open(resolve_image(args.data_root,it['rel_path'])) as src:image=src.convert('RGB')
                        image.thumbnail((args.max_edge,args.max_edge));adapter.predict(image,domain);adapter.synchronize()
                    except Exception:
                        warmup_errors.append(domain)
                        print(f'Warm-up failed for {spec["id"]}/{domain}; recording all measured attempts.',flush=True)
                append('loads.jsonl',dict(model=spec['id'],block=block,load_or_client_init_s=load_s,warmup_s=time.perf_counter()-warm,warmup_failed_domains=warmup_errors,server_metadata=metadata,resource_scope=scope))
                random.Random(args.seed+block).shuffle(pending)
                for it in pending:
                    key=f'{block}/{spec["id"]}/{it["id"]}'
                    gpu=args.device.startswith('cuda') and spec['adapter'] in {'blip','laya'}
                    if gpu:sys.modules['torch'].cuda.reset_peak_memory_stats(args.device)
                    adapter.synchronize();timer=time.perf_counter()
                    with Resources(pid) as resource:
                        try:
                            with Image.open(resolve_image(args.data_root,it['rel_path'])) as src:image=src.convert('RGB')
                            image.thumbnail((args.max_edge,args.max_edge))
                            answers,raw=adapter.predict(image,it['domain']);result=finalize(contract,it['domain'],answers);adapter.synchronize();error=None
                        except Exception as exc:
                            raw={};result=finalize(contract,it['domain'],{});error=f'{type(exc).__name__}: {exc}'
                            for s in specs:
                                token=os.environ.get(s.get('token_env',''),'')
                                if token:error=error.replace(token,'[REDACTED]')
                    row=dict(request_key=key,model=spec['id'],adapter=spec['adapter'],block=block,id=it['id'],domain=it['domain'],image_sha256=it['sha256'],
                             elapsed_ms=(time.perf_counter()-timer)*1000,cpu_s=resource.cpu_s,peak_rss_bytes=resource.peak,resource_scope=scope,
                             error=error,raw=raw,**result)
                    if gpu:row['gpu_peak_allocated_bytes']=sys.modules['torch'].cuda.max_memory_allocated(args.device)
                    if it.get('expected') is not None:row.update(expected=it['expected'],exploratory_exact=(not error and result['predicted']==it['expected']))
                    append('predictions.jsonl',row);done.add(key);measured_errors+=bool(error)
                    write_json(out/'status.json',dict(status='running',phase='predict',model=spec['id'],current_filename=it.get('filename',it['id']),completed=len(done),planned=total,errors=measured_errors))
                    print(f'{len(done)}/{total} {spec["id"]} {it["id"]}'+(f' ERROR: {error}' if error else ''),flush=True)
            finally:
                adapter.close();del adapter;gc.collect()
                if 'torch' in sys.modules and args.device.startswith('cuda'):sys.modules['torch'].cuda.empty_cache()
    write_json(out/'status.json',dict(status='complete',completed=len(done),planned=total,finished_utc=datetime.now(timezone.utc).isoformat()))
    return dict(status='complete',output=str(out),completed=len(done))

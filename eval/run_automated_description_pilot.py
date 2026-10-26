"""Human-independent pilot: audit, complete prompt bundles, resources, perturbations and queue.

All quality labels are historical and exploratory. No reserved confirmatory data is opened.
"""
from __future__ import annotations
import argparse
import collections
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
from io import BytesIO
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import threading
import time

# Set before importing libraries that could resolve model files.
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import psutil
from automated_description_core import normalize, render, validate, gate, PARSER_VERSION
from run_test_7_complete_benchmark import PROMPTS, QUESTIONS
from run_test_1_color_benchmark import _apply_laya_transformers_compat_patch

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'eval/datasets/test_7_complete_tags_manifest.json'
SEED=20261026
REVISIONS={'laya':'f2fe3c12cb6d04c59d8a190250bf3fb40fc828dc','blip':'787b3d35d57e49572baabd22884b3d5a05acf072'}
ALT={
 'vehicle':dict(presence='Is a vehicle visible?', make='What is the make of this vehicle?',color='What color is the vehicle?',body='What type of vehicle body is shown?'),
 'person':dict(presence='Is a person visible?',helmet='Does the person have a helmet on their head?',carrying='What bag or object is the person carrying?',clothing='What upper-body clothing is the person wearing?')
}

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write_json(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n');os.replace(tmp,path)

def select(items, positives, negatives):
    out=[]
    for domain in ['vehicle','person']:
        rows=[it for it in items if it['domain']==domain]
        out.extend([it for it in rows if it['expected']['presence']=='yes'][:positives])
        out.extend([it for it in rows if it['expected']['presence']=='no'][:negatives])
    return out

def config(model,variant,domain):
    style,mode=variant.rsplit('_',1)
    q=deepcopy(QUESTIONS[domain]);p=deepcopy(PROMPTS[domain])
    if model=='blip':
        if style=='specific':p=deepcopy(ALT[domain])
        if style=='uncertain':p={k:v+(' If not visible, answer unknown.' if k!='presence' else '') for k,v in p.items()}
    else:
        if style=='short':
            for key in q:
                q[key]['criteria']={label:label.replace('_',' ') for label in q[key]['criteria']}
        if style=='visible':
            for key in q:
                if key!='presence':q[key]['instructions']+=' Use the not-visible or unknown option when the attribute cannot be identified.'
    return q,p,mode

VARIANTS={'blip':[f'{s}_{m}' for s in ['direct','specific','uncertain'] for m in ['joint','gated']],
          'laya':[f'{s}_{m}' for s in ['descriptive','short','visible'] for m in ['joint','gated']]}
BASE={'blip':'direct_joint','laya':'descriptive_joint'}

class Monitor:
    def __init__(self):
        self.proc=psutil.Process();self.stop=threading.Event();self.peak=0;self.samples=[]
        self.thread=threading.Thread(target=self.sample,daemon=True)
    def sample(self):
        while not self.stop.is_set():
            rss=self.proc.memory_info().rss;self.peak=max(self.peak,rss)
            self.samples.append(dict(monotonic_s=time.perf_counter(),rss_bytes=rss,available_bytes=psutil.virtual_memory().available))
            self.stop.wait(.1)
    def __enter__(self):self.thread.start();return self
    def __exit__(self,*args):self.stop.set();self.thread.join()

class Engine:
    def __init__(self,name):
        self.name=name
        if name=='blip':
            from transformers import AutoModelForVisualQuestionAnswering,AutoProcessor
            self.processor=AutoProcessor.from_pretrained('Salesforce/blip-vqa-base',revision=REVISIONS[name],local_files_only=True)
            self.net=AutoModelForVisualQuestionAnswering.from_pretrained('Salesforce/blip-vqa-base',revision=REVISIONS[name],local_files_only=True).to('cpu').eval()
        else:
            _apply_laya_transformers_compat_patch()
            import laya
            self.net=laya.load_vlm('thaitea/laya-vision',revision=REVISIONS[name],device='cpu')
    def part(self,image,domain,q,p,keys):
        if self.name=='laya':
            raw=self.net.predict(image,{k:q[k] for k in keys})
            answers={k:str(raw.get('answers',{}).get(k,{}).get('choice','parse_error')) for k in keys}
            return answers,raw
        import torch
        inputs=self.processor(images=[image]*len(keys),text=[p[k] for k in keys],padding=True,return_tensors='pt')
        with torch.inference_mode():tokens=self.net.generate(**inputs,max_new_tokens=12,do_sample=False)
        raw=dict(zip(keys,self.processor.batch_decode(tokens,skip_special_tokens=True),strict=True))
        return {k:normalize(domain,k,v) for k,v in raw.items()},raw
    def infer(self,image,domain,variant):
        q,p,mode=config(self.name,variant,domain);keys=list(p)
        if mode=='joint':answers,raw=self.part(image,domain,q,p,keys);calls=1;questions=len(keys)
        else:
            answers,first=self.part(image,domain,q,p,['presence']);raw={'presence_call':first};calls=1;questions=1
            if answers['presence']=='yes':
                more,last=self.part(image,domain,q,p,[k for k in keys if k!='presence']);answers.update(more);raw['attributes_call']=last;calls+=1;questions+=3
            else:
                # Distinguish no-execution after uncertainty from a valid negative decision.
                answers.update({k:'parse_error' for k in keys if k!='presence'})
        before=dict(answers);answers=gate(domain,answers)
        return dict(raw=raw,normalized_before_gate=before,predicted=answers,valid=validate(domain,answers),
                    rendered=render(domain,answers),calls=calls,questions_executed=questions)

def transformed(image,variant):
    if variant=='original':return image
    if variant=='resize64':return image.resize((64,64),Image.Resampling.BILINEAR).resize(image.size,Image.Resampling.BILINEAR)
    if variant=='jpeg20':
        buf=BytesIO();image.save(buf,format='JPEG',quality=20);buf.seek(0)
        with Image.open(buf) as src:return src.convert('RGB')
    if variant=='dark50':return ImageEnhance.Brightness(image).enhance(.5)
    if variant=='blur2':return image.filter(ImageFilter.GaussianBlur(2))
    raise ValueError(variant)

def audit(items,out):
    seen=collections.defaultdict(list);rows=[]
    for it in items:
        path=ROOT/it['rel_path'];sha=digest(path)
        with Image.open(path) as im:
            width,height=im.size
            small=im.convert('L').resize((9,8));a=np.array(small);bits=a[:,1:]>a[:,:-1]
            dhash=sum(int(bit)<<i for i,bit in enumerate(bits.flatten()))
        assert sha==it['sha256'],it['id']
        assert set(it['expected'])==set(PROMPTS[it['domain']]),it['id']
        seen[sha].append(it['id'])
        rows.append(dict(id=it['id'],domain=it['domain'],sha256=sha,width=width,height=height,dhash=f'{dhash:016x}'))
    near=[]
    for i,a in enumerate(rows):
        for b in rows[i+1:]:
            distance=(int(a['dhash'],16)^int(b['dhash'],16)).bit_count()
            if distance<=4:near.append(dict(a=a['id'],b=b['id'],hamming=distance))
    # Track exact overlap with every historical manifest without modifying them.
    historical=collections.defaultdict(list);missing=[]
    for path in sorted((ROOT/'eval/datasets').glob('*.json')):
        try:doc=json.loads(path.read_text())
        except (ValueError,OSError):continue
        its=doc.get('items',[]) if isinstance(doc,dict) else []
        for it in its:
            rel=it.get('rel_path')
            if rel:
                source=ROOT/rel
                if source.is_file():historical[digest(source)].append(dict(manifest=path.name,id=it.get('id',it.get('filename'))))
                else:missing.append(dict(manifest=path.name,rel_path=rel))
    result=dict(status='passed',rows=rows,duplicate_groups=[v for v in seen.values() if len(v)>1],
                near_duplicates_for_review=near,overlap_with_historical_manifests={it['id']:historical[it['sha256']] for it in items},
                historical_missing_files=missing,labels='Historical agent-reviewed labels, not human-confirmed. No new labels inferred.')
    write_json(out/'audit.json',result)
    return result

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run-dir',required=True);ap.add_argument('--resume',action='store_true');ap.add_argument('--threads',type=int,default=8)
    args=ap.parse_args();out=Path(args.run_dir).resolve()
    if out.exists() and not args.resume:ap.error('run-dir exists; use a new path or --resume')
    out.mkdir(parents=True,exist_ok=True)
    import torch
    torch.set_num_threads(args.threads);torch.manual_seed(SEED);random.seed(SEED);np.random.seed(SEED)
    items=json.loads(SOURCE.read_text())['items'];samples=dict(performance=select(items,6,2),prompts=select(items,4,2),robustness=select(items,4,2),queue=select(items,2,1))
    frozen=dict(seed=SEED,threads=args.threads,parser=PARSER_VERSION,model_revisions=REVISIONS,
                source_sha256=digest(SOURCE),source_manifest=str(SOURCE.relative_to(ROOT)),
                scripts_sha256={name:digest(ROOT/'eval'/name) for name in ['run_automated_description_pilot.py','automated_description_core.py','run_test_7_complete_benchmark.py','run_test_1_color_benchmark.py','compare_laya_blip.py','run_test_3_person_benchmark.py']},
                sample_ids={k:[it['id'] for it in v] for k,v in samples.items()},
                configurations={m:{v:{d:config(m,v,d) for d in PROMPTS} for v in vs} for m,vs in VARIANTS.items()},
                perturbations=['resize64','jpeg20','dark50','blur2'],performance_blocks=5,queue_concurrency=[1,2,4],
                planned_records=160+144+96+16+36,
                labels='Exploratory only; no human-independent claims of semantic truth for photographs.',
                concurrency_policy='One serialized inference worker, multiple calling threads; measures queueing, not parallel model inference.')
    frozen_path=out/'frozen_config.json'
    if frozen_path.exists():assert json.loads(frozen_path.read_text())==json.loads(json.dumps(frozen)),'Source/config changed; create new run'
    else:write_json(frozen_path,frozen)
    if not (out/'audit.json').exists():audit(items,out)
    status_path=out/'status.json';data_path=out/'predictions.jsonl';done=set()
    if data_path.exists():
        for line in data_path.read_text().splitlines():done.add(json.loads(line)['request_key'])
    proc=psutil.Process();lock=threading.Lock();writer_lock=threading.Lock()
    def append(path,row):
        with writer_lock:
            with path.open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    def record(engine,it,stage,key,variant=None,transform='original',**extra):
        if key in done:return
        arrival=time.perf_counter()
        with lock:
            acquired=time.perf_counter();cpu=time.process_time();start=time.perf_counter()
            try:
                if 'synthetic_kind' in it:
                    kind=it['synthetic_kind']
                    if kind=='noise':image=Image.fromarray(np.random.default_rng(SEED).integers(0,256,(256,256,3),dtype=np.uint8))
                    else:image=Image.new('RGB',(256,256),{'black':(0,0,0),'gray':(127,127,127),'white':(255,255,255)}[kind])
                else:
                    with Image.open(ROOT/it['rel_path']) as im:image=im.convert('RGB')
                image=transformed(image,transform)
                result=engine.infer(image,it['domain'],variant or BASE[engine.name]);error=None
            except Exception as exc:result={};error=repr(exc)
            elapsed=time.perf_counter()-start;cpu_s=time.process_time()-cpu;end=time.perf_counter()
            row=dict(request_key=key,stage=stage,model=engine.name,id=it['id'],domain=it['domain'],variant=variant or BASE[engine.name],transform=transform,
                     elapsed_ms=elapsed*1000,cpu_s=cpu_s,queue_ms=(acquired-arrival)*1000,response_ms=(end-arrival)*1000,
                     rss_end_bytes=proc.memory_info().rss,error=error,result=result,**extra)
            if 'expected' in it:
                row['exploratory_expected']=it['expected'];row['exploratory_exact']=result.get('predicted')==it['expected'] if not error else False
            append(data_path,row);done.add(key)
            write_json(status_path,dict(status='running',completed=len(done),planned=frozen['planned_records'],last=key))
        if len(done)%12==0:print(f'{len(done)}/{frozen["planned_records"]} {key}',flush=True)
    metadata_path=out/'environment.json'
    if not metadata_path.exists():
        write_json(metadata_path,dict(started_utc=datetime.now(timezone.utc).isoformat(),platform=platform.platform(),cpu=platform.processor(),
          cpu_model=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),''),
          logical_cpus=psutil.cpu_count(),physical_cpus=psutil.cpu_count(logical=False),ram_bytes=psutil.virtual_memory().total,
          available_at_start_bytes=psutil.virtual_memory().available,
          packages={p:importlib.metadata.version(p) for p in ['torch','transformers','pillow','numpy','psutil','huggingface-hub']},
          energy_sensors=[str(p) for p in Path('/sys/class/powercap').glob('**/energy_uj')],energy_measured=False,
          timing='Decode, transform, processor, inference, parser, absence gate and render included; model loading, warm-up and JSON writes excluded.',
          concurrent_processes=[dict(name=p.info['name'],rss=p.info['memory_info'].rss) for p in psutil.process_iter(['name','memory_info']) if p.info['memory_info'] and p.info['memory_info'].rss>200*2**20]))
    loads=out/'loads.jsonl'
    def loaded(name,phase):
        gc.collect();rss_before=proc.memory_info().rss;start=time.perf_counter();engine=Engine(name);load_s=time.perf_counter()-start
        warm_start=time.perf_counter()
        for domain in PROMPTS:
            it=next(x for x in items if x['domain']==domain)
            with Image.open(ROOT/it['rel_path']) as im:engine.infer(im.convert('RGB'),domain,BASE[name])
        append(loads,dict(model=name,phase=phase,load_s=load_s,warmup_s=time.perf_counter()-warm_start,rss_before_bytes=rss_before,rss_loaded_bytes=proc.memory_info().rss,
                         provenance=getattr(engine.net,'cfg',{})))
        return engine
    with Monitor() as monitor:
        # Paired blocks alternate model order, preventing all-A-then-all-B timing confound.
        for block in range(5):
            order=['laya','blip'] if block%2==0 else ['blip','laya']
            for name in order:
                todo=[it for it in samples['performance'] if f'performance/{block}/{name}/{it["id"]}' not in done]
                if not todo:continue
                engine=loaded(name,f'performance-{block}')
                random.Random(SEED+block).shuffle(todo)
                for it in todo:record(engine,it,'performance',f'performance/{block}/{name}/{it["id"]}',block=block)
                del engine;gc.collect()
        for name in ['laya','blip']:
            if all(f'queue/4/{name}/{it["id"]}' in done for it in samples['queue']) and len(done)==frozen['planned_records']:continue
            engine=loaded(name,'experiments')
            for variant in VARIANTS[name]:
                for it in samples['prompts']:record(engine,it,'prompts',f'prompts/{name}/{variant}/{it["id"]}',variant=variant)
            for transform in frozen['perturbations']:
                for it in samples['robustness']:record(engine,it,'robustness',f'robustness/{name}/{transform}/{it["id"]}',transform=transform)
            for kind in ['black','gray','white','noise']:
                for domain in PROMPTS:
                    it=dict(id=f'synthetic_{kind}_{domain}',domain=domain,synthetic_kind=kind)
                    record(engine,it,'synthetic',f'synthetic/{name}/{kind}/{domain}',synthetic_kind=kind)
            for workers in [1,2,4]:
                todo=[it for it in samples['queue'] if f'queue/{workers}/{name}/{it["id"]}' not in done]
                if not todo:continue
                start=time.perf_counter();cpu=time.process_time()
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures=[pool.submit(record,engine,it,'queue',f'queue/{workers}/{name}/{it["id"]}',concurrency=workers) for it in todo]
                    for future in as_completed(futures):future.result()
                elapsed=time.perf_counter()-start
                append(out/'queue_batches.jsonl',dict(model=name,concurrency=workers,requests=len(todo),elapsed_s=elapsed,cpu_s=time.process_time()-cpu,throughput_per_s=len(todo)/elapsed,partial_resume=len(todo)!=len(samples['queue'])))
            del engine;gc.collect()
    for row in monitor.samples:append(out/'resource_samples.jsonl',row)
    assert len(done)==frozen['planned_records'],(len(done),frozen['planned_records'])
    write_json(status_path,dict(status='complete',completed=len(done),planned=frozen['planned_records'],process_peak_rss_bytes=monitor.peak,ended_utc=datetime.now(timezone.utc).isoformat()))
    print(f'COMPLETE {out}',flush=True)

if __name__=='__main__':main()

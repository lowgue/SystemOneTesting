"""Evaluate complete four-field bundles, with fixed prompts and no reference-dependent options."""
from __future__ import annotations
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import statistics
import time

from PIL import Image
from run_test_1_color_benchmark import COLOR_CRITERIA, load_blip_model, load_laya_model, _normalise_blip_color
from compare_laya_blip import MAKE_LABELS, BODY_LABELS, _normalise_text
from run_test_3_person_benchmark import LAYA_QUESTIONS

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT/'eval/datasets/test_7_complete_tags_manifest.json'
OUTPUT = ROOT/'eval/results/test_7_complete_tags_results.json'
PROMPTS = {
    'vehicle': {'presence':'Is there a vehicle in the image?', 'make':'What brand is the vehicle?',
                'color':'What is the predominant color of this vehicle?', 'body':'What body style is the vehicle?'},
    'person': {'presence':'Is there a person in the image?', 'helmet':'Is the person wearing a helmet?',
               'carrying':'What is the person carrying?', 'clothing':'What is the person wearing on their upper body?'}
}
PRESENCE = {'yes':'Yes','no':'No'}
QUESTIONS = {
    'vehicle': {
        'presence':dict(type='choice', instructions=PROMPTS['vehicle']['presence'], criteria=PRESENCE),
        'make':dict(type='choice',instructions=PROMPTS['vehicle']['make'],criteria={**MAKE_LABELS,'unknown':'Unknown or no car'}),
        'color':dict(type='choice',instructions=PROMPTS['vehicle']['color'],criteria=COLOR_CRITERIA),
        'body':dict(type='choice',instructions=PROMPTS['vehicle']['body'],criteria=BODY_LABELS),
    },
    'person': {
        'presence':dict(type='choice',instructions=PROMPTS['person']['presence'],criteria=PRESENCE),
        **{k:dict(type='choice',instructions=PROMPTS['person'][k],criteria=v['criteria']) for k,v in LAYA_QUESTIONS.items()}
    }
}

def normalize(domain, key, raw):
    text = re.sub(r'[^a-z0-9]+',' ',raw.lower()).strip()
    words = set(text.split())
    if key == 'presence':
        if words & {'no','not','none'}: return 'no'
        if words & {'yes', 'person' if domain=='person' else 'vehicle'}: return 'yes'
        return 'invalid'
    if key == 'color': return _normalise_blip_color(raw)
    if key in ('make','body'):
        return _normalise_text(raw, {'make':'marca','body':'tipo'}[key], MAKE_LABELS)
    if not text or any(t in text for t in ('not visible','cannot','unknown','unclear')): return 'not_visible'
    if key == 'helmet':
        if words & {'no','not','none'}: return 'no'
        if words & {'yes','helmet','hardhat'} or 'hard hat' in text: return 'yes'
        if words & {'hat','cap'}: return 'no'
    if key == 'carrying':
        if 'backpack' in text or 'rucksack' in text: return 'backpack'
        if words & {'bag','box','briefcase','suitcase','package','purse','handbag','bags'}: return 'bag_or_box'
        if words & {'nothing','none','no'}: return 'none'
    if key == 'clothing':
        if 'vest' in words or 'reflective' in words: return 'safety_vest'
        if words & {'jacket','coat','suit','blazer','sweater','sweatshirt','hoodie','uniform'}: return 'jacket_coat'
        if words & {'shirt','tshirt','top','polo','blouse'}: return 'shirt_tshirt'
    return 'invalid'

def gate(domain, answers):
    answers = dict(answers)
    if answers.get('presence') == 'no':
        for key in answers:
            if key != 'presence': answers[key] = 'unknown' if domain=='vehicle' else 'not_visible'
    return answers

def save(doc):
    temp = OUTPUT.with_suffix('.tmp')
    temp.write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n')
    os.replace(temp,OUTPUT)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repetitions',type=int,default=3)
    ap.add_argument('--threads',type=int,default=8)
    args = ap.parse_args()
    if args.repetitions < 1: ap.error('repetitions must be positive')
    import torch
    torch.set_num_threads(args.threads)
    manifest = json.loads(MANIFEST.read_text())
    for it in manifest['items']:
        assert hashlib.sha256((ROOT/it['rel_path']).read_bytes()).hexdigest()==it['sha256'], it['id']
        assert set(it['expected'])==set(PROMPTS[it['domain']])
    doc = dict(status='running', manifest=str(MANIFEST.relative_to(ROOT)),
               manifest_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
               device='cpu',threads=args.threads,repetitions=args.repetitions,
               platform=platform.platform(),packages={p:importlib.metadata.version(p) for p in ['torch','transformers','pillow','huggingface-hub']},
               prompts=PROMPTS,laya_questions=QUESTIONS,policy='All four fields executed; identical absence gate applied after inference. No thresholds.',
               models={},records=[])
    cache=Path(os.environ.get('HF_HUB_CACHE',str(Path.home()/'.cache/huggingface/hub')))
    doc['cached_model_revisions']={repo:[p.name for p in (cache/('models--'+repo.replace('/','--'))/'snapshots').glob('*')] for repo in ['thaitea/laya-vision','Salesforce/blip-vqa-base']}
    for name in ['laya','blip']:
        start=time.perf_counter()
        model=load_laya_model() if name=='laya' else load_blip_model('cpu')
        doc['models'][name]={'load_s':time.perf_counter()-start}
        def infer(image,domain):
            if name=='laya':
                out=model.predict(image,QUESTIONS[domain])
                return {k:str(out.get('answers',{}).get(k,{}).get('choice','invalid')) for k in PROMPTS[domain]}, out
            processor,net=model
            inputs=processor(images=[image]*4,text=list(PROMPTS[domain].values()),padding=True,return_tensors='pt')
            with torch.inference_mode(): tokens=net.generate(**inputs,max_new_tokens=12,do_sample=False)
            raw=dict(zip(PROMPTS[domain],processor.batch_decode(tokens,skip_special_tokens=True),strict=True))
            return {k:normalize(domain,k,v) for k,v in raw.items()},raw
        # Warm up both domain shapes; excluded from all timed records.
        for domain in PROMPTS:
            sample=next(it for it in manifest['items'] if it['domain']==domain)
            with Image.open(ROOT/sample['rel_path']) as src: infer(src.convert('RGB'),domain)
        for rep in range(args.repetitions):
            for idx,it in enumerate(manifest['items']):
                with Image.open(ROOT/it['rel_path']) as src: image=src.convert('RGB')
                start=time.perf_counter()
                try:
                    normalized,raw=infer(image,it['domain'])
                    predicted=gate(it['domain'],normalized)
                    error=None
                except Exception as exc:
                    normalized=predicted={k:'invalid' for k in PROMPTS[it['domain']]};raw={};error=repr(exc)
                ms=(time.perf_counter()-start)*1000
                doc['records'].append(dict(id=it['id'],domain=it['domain'],model=name,repetition=rep+1,
                                           expected=it['expected'],raw=raw,normalized_before_gate=normalized,predicted=predicted,
                                           exact_match=predicted==it['expected'],latency_ms=ms,error=error))
                save(doc)
                if (idx+1)%10==0 or idx+1==len(manifest['items']): print(f'{name} rep {rep+1}: {idx+1}/{len(manifest["items"])}',flush=True)
        del infer,model
        gc.collect()
    doc['status']='complete';save(doc)
    print(OUTPUT,flush=True)

if __name__=='__main__':main()

"""Shared input contract, strict parsing and verified dataset I/O."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT=ROOT/'benchmarks/contracts/description_v1.json'


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()
def fingerprint(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def load_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write_json(path,obj):
    path=Path(path);temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');temp.replace(path)

def resolve_image(root,rel):
    root=Path(root).resolve();path=(root/rel).resolve()
    if not path.is_relative_to(root):raise ValueError(f'Image escapes data root: {rel}')
    return path

def validate_manifest(path,data_root,contract):
    doc=load_json(path);items=doc.get('items',[])
    if not items:raise ValueError('Manifest has no items')
    ids=set()
    for it in items:
        if it['id'] in ids:raise ValueError(f'Duplicate event ID: {it["id"]}')
        ids.add(it['id']);domain=contract['domains'].get(it['domain'])
        if domain is None:raise ValueError(f'Unsupported domain: {it["domain"]}')
        image=resolve_image(data_root,it['rel_path'])
        if not image.is_file():raise ValueError(f'Missing image: {it["rel_path"]}; import the dataset bundle or use portable_vehicles.json')
        if not it.get('sha256') or sha256(image)!=it['sha256']:raise ValueError(f'Image checksum mismatch: {it["id"]}')
        annotations=it.get('annotations')
        if annotations is not None:
            fields=domain['questions']
            if not isinstance(annotations,dict) or not annotations or not set(annotations)<=set(fields):raise ValueError(f'Invalid annotation fields: {it["id"]}')
            if any(v not in fields[k]['criteria'] for k,v in annotations.items()):raise ValueError(f'Invalid annotation label: {it["id"]}')
        expected=it.get('expected')
        if expected is not None:
            if set(expected)!=set(domain['questions']):raise ValueError(f'Incomplete expected fields: {it["id"]}')
            if any(v not in domain['questions'][k]['criteria'] for k,v in expected.items()):raise ValueError(f'Invalid expected label: {it["id"]}')
    return doc

def normalize_text(contract,domain,key,raw):
    text=re.sub(r'[^a-z0-9]+',' ',str(raw).lower()).strip()
    if not text:return 'parse_error'
    if text in {'unknown','not visible','unclear','cannot tell','not identifiable','unidentifiable'}:
        return contract['absence'][domain] if key!='presence' else 'parse_error'
    if key=='presence':
        if text in {'yes','no'}:return text
        if re.match(r'^(no|not|there is no|there are no)\b',text):return 'no'
        if re.match(r'^(yes|there is a|there are)\b',text):return 'yes'
        return 'parse_error'
    if key=='helmet' and re.match(r'^(no|not)\b',text):return 'no'
    if key=='carrying' and re.match(r'^(no|not carrying|nothing|none)\b',text):return 'none'
    choices=contract['domains'][domain]['questions'][key]['criteria']
    clean=lambda s:re.sub(r'[^a-z0-9]+',' ',s.lower()).strip()
    aliases={k:[clean(k),clean(v)] for k,v in choices.items() if k not in {'unknown','not_visible'}}
    aliases.update(contract.get('aliases',{}).get(key,{}))
    has=lambda term:f' {clean(term)} ' in f' {text} '
    if key=='helmet' and (has('hard hat') or has('hardhat')):return 'yes'
    hits={k for k,terms in aliases.items() if any(has(term) for term in terms)}
    return next(iter(hits)) if len(hits)==1 else 'parse_error'

def finalize(contract,domain,answers):
    fields=contract['domains'][domain]['questions']
    pred={k:str(answers.get(k,'parse_error')) for k in fields}
    before=dict(pred)
    if pred.get('presence')=='no':pred.update({k:contract['absence'][domain] for k in pred if k!='presence'})
    valid=all(v in fields[k]['criteria'] for k,v in pred.items())
    facts={k:v for k,v in pred.items() if k!='presence' and v not in {'unknown','not_visible','parse_error'}} if pred.get('presence')=='yes' else {}
    labels=[]
    for k,v in facts.items():labels.append(f'{k}={v}')
    text=(('Veículo' if domain=='vehicle' else 'Pessoa')+(': '+', '.join(labels) if labels else '')+'.') if pred.get('presence')=='yes' else None
    return dict(predicted=pred,before_absence_gate=before,valid=valid,description=text,facts=facts)

def pack_dataset(manifest,data_root,output,contract):
    doc=validate_manifest(manifest,data_root,contract);output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    transformed=[];written=set()
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for item in doc['items']:
            it=dict(item);source=resolve_image(data_root,it['rel_path'])
            name='images/'+it['sha256']+source.suffix.lower();it['rel_path']=name;transformed.append(it)
            if name not in written:archive.write(source,name);written.add(name)
        copied={**doc,'items':transformed,'source_manifest_sha256':sha256(manifest)}
        archive.writestr('manifest.json',json.dumps(copied,ensure_ascii=False,indent=2)+'\n')
    return dict(events=len(transformed),unique_images=len(written),sha256=sha256(output))

def unpack_dataset(bundle,destination,contract):
    destination=Path(destination).resolve()
    if destination.exists():raise ValueError('Destination exists; use a new directory')
    with zipfile.ZipFile(bundle) as archive:
        names=archive.namelist()
        if len(names)!=len(set(names)) or 'manifest.json' not in names:raise ValueError('Invalid or duplicate archive entries')
        for member in archive.infolist():
            resolve_image(destination,member.filename)
            if (member.external_attr>>16)&0o170000==0o120000:raise ValueError('Symlinks are not allowed in dataset bundles')
        doc=json.loads(archive.read('manifest.json'))
        required={'manifest.json'}|{it['rel_path'] for it in doc['items']}
        if set(names)!=required:raise ValueError('Archive contents differ from manifest')
        # Verify every image before writing anything.
        for it in doc['items']:
            if hashlib.sha256(archive.read(it['rel_path'])).hexdigest()!=it['sha256']:raise ValueError('Dataset archive checksum mismatch')
        destination.mkdir(parents=True)
        for name in names:
            path=resolve_image(destination,name);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(archive.read(name))
    validate_manifest(destination/'manifest.json',destination,contract)
    return destination/'manifest.json'

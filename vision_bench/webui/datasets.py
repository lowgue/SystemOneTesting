"""Persistent photo datasets with optional complete or partial CSV ground truth."""
from __future__ import annotations

import base64
import csv
import hashlib
import shutil
from io import BytesIO, StringIO
from pathlib import Path
import re
import threading
import unicodedata
import uuid

from PIL import Image, UnidentifiedImageError

from ..core import load_json, normalize_text, validate_manifest, write_json


def clean(value):
    return ''.join(c for c in unicodedata.normalize('NFD', str(value).lower()) if unicodedata.category(c) != 'Mn').strip().replace(' ', '_')


HEADERS = {'arquivo':'filename','foto':'filename','presenca':'presence','marca':'make','cor':'color',
           'carroceria':'body','capacete':'helmet','objeto':'carrying','roupa':'clothing','vestimenta':'clothing'}
TRANSLATIONS = {'sim':'yes','nao':'no','desconhecido':'unknown','nao_identificado':'unknown','nao_visivel':'not_visible',
                'preto':'black','branco':'white','cinza':'gray_silver','prata':'gray_silver','vermelho':'red','azul':'blue',
                'verde':'green','amarelo':'yellow','laranja':'orange','marrom':'brown','roxo':'purple','rosa':'pink',
                'outro':'other','seda':'sedan','picape':'truck','cupe':'coupe','conversivel':'convertible','perua':'wagon',
                'moto':'motorcycle','onibus':'bus','mochila':'backpack','bolsa_ou_caixa':'bag_or_box','nenhum':'none',
                'colete_de_seguranca':'safety_vest','jaqueta_ou_casaco':'jacket_coat','camisa_ou_camiseta':'shirt_tshirt'}


def read_labels(text, items, contract, domain):
    """Blank cells stay unlabeled; invalid or ambiguous annotations are rejected."""
    if not text.strip():
        return {}
    text=text.lstrip('\ufeff')
    try:
        dialect=csv.Sniffer().sniff(text[:8192],delimiters=',;\t')
    except csv.Error:
        dialect=csv.excel
    reader=csv.DictReader(StringIO(text),dialect=dialect)
    columns=[HEADERS.get(clean(h),clean(h)) for h in (reader.fieldnames or [])]
    questions=contract['domains'][domain]['questions']
    if len(columns)!=len(set(columns)) or 'filename' not in columns:
        raise ValueError('O CSV precisa da coluna arquivo (ou filename), sem colunas duplicadas.')
    unknown=set(columns)-set(questions)-{'filename'}
    if unknown:
        raise ValueError('Colunas de gabarito não reconhecidas: '+', '.join(sorted(unknown)))
    names={it['filename'] for it in items}
    result={};seen=set()
    for number,row in enumerate(reader,2):
        if None in row:
            raise ValueError(f'Linha {number}: número de colunas diferente do cabeçalho.')
        row={HEADERS.get(clean(k),clean(k)):(v or '').strip() for k,v in row.items()}
        if not any(row.values()):
            continue
        name=row['filename']
        if name not in names:
            raise ValueError(f'Linha {number}: a foto {name!r} não está neste dataset.')
        if name in seen:
            raise ValueError(f'Gabarito duplicado para {name!r}.')
        seen.add(name);annotations={}
        for key,value in row.items():
            if key=='filename' or not value:
                continue
            choices=questions[key]['criteria']
            candidate=TRANSLATIONS.get(clean(value),clean(value))
            if candidate not in choices:
                candidate=normalize_text(contract,domain,key,value)
            if candidate not in choices:
                raise ValueError(f'Linha {number}, {key}: valor {value!r} inválido. Consulte os valores aceitos.')
            annotations[key]=candidate
        if annotations:
            result[name]=annotations
    return result


def dataset_info(key, manifest):
    annotated=complete=0
    for item in manifest['items']:
        labels=item.get('expected') or item.get('annotations') or {}
        annotated+=bool(labels)
        complete+=item.get('expected') is not None
    return dict(id=key,name='Exemplo: veículos' if key=='portable' else manifest.get('name',key),
                events=len(manifest['items']),annotated=annotated,complete_labels=complete,
                unique_images=len({it['sha256'] for it in manifest['items']}),
                label_status=manifest.get('label_status','exploratory_unadjudicated'))


class Datasets:
    def __init__(self, root):
        self.root=Path(root)
        self.lock=threading.RLock()

    def folder(self, dataset_id):
        if not isinstance(dataset_id,str) or not re.fullmatch(r'import-[a-f0-9]{32}',dataset_id):
            raise ValueError('Dataset inválido.')
        path=self.root/dataset_id
        if not (path/'draft.json').is_file():
            raise ValueError('Dataset não encontrado.')
        return path

    def manifests(self):
        return {path.parent.name:path for path in self.root.glob('import-*/manifest.json')}

    def create(self, name, domain):
        if domain not in {'vehicle','person'}:
            raise ValueError('Escolha veículo ou pessoa para o dataset.')
        if not isinstance(name,str) or not name.strip() or len(name)>120:
            raise ValueError('Informe um nome de até 120 caracteres para o dataset.')
        key='import-'+uuid.uuid4().hex
        folder=self.root/key
        folder.mkdir(parents=True)
        write_json(folder/'draft.json',dict(id=key,name=name.strip(),domain=domain,items=[]))
        return dict(id=key,name=name.strip(),events=0)

    def add_image(self, dataset_id, filename, encoded):
        with self.lock:
            folder=self.folder(dataset_id)
            if (folder/'manifest.json').exists():
                raise ValueError('Este dataset já foi finalizado. Importe outro para alterar as fotos.')
            draft=load_json(folder/'draft.json')
            if not isinstance(filename,str) or not filename or len(filename)>240 or '/' in filename or '\\' in filename:
                raise ValueError('Nome de foto inválido; use apenas o nome do arquivo.')
            existing=next((it for it in draft['items'] if it['filename']==filename),None)
            if len(draft['items'])>=2000 and not existing:
                raise ValueError('Limite de 2.000 fotos por dataset. Divida o conjunto em partes.')
            try:
                raw=base64.b64decode(encoded.split(',')[-1],validate=True)
                with Image.open(BytesIO(raw)) as source:
                    if source.width*source.height>25_000_000:
                        raise ValueError('Use fotos com até 25 megapixels.')
                    image=source.convert('RGB')
                image.thumbnail((2048,2048))
            except (ValueError,AttributeError,UnidentifiedImageError,OSError,Image.DecompressionBombError) as exc:
                raise ValueError(f'Foto {filename!r} inválida: {exc}') from exc
            buffer=BytesIO();image.save(buffer,format='PNG')
            content=buffer.getvalue();digest=hashlib.sha256(content).hexdigest()
            if existing:
                if existing['sha256']==digest:
                    return dict(id=dataset_id,events=len(draft['items']))
                raise ValueError(f'Já existe outra foto chamada {filename!r}. Renomeie arquivos com nomes repetidos.')
            item_id=f'photo-{len(draft["items"])+1:06d}'
            (folder/'images').mkdir(exist_ok=True)
            path=folder/'images'/f'{item_id}.png'
            path.write_bytes(content)
            draft['items'].append(dict(id=item_id,filename=filename,domain=draft['domain'],
                                      rel_path='images/'+path.name,sha256=digest))
            write_json(folder/'draft.json',draft)
            return dict(id=dataset_id,events=len(draft['items']))

    def discard(self, dataset_id):
        with self.lock:
            folder=self.folder(dataset_id)
            if (folder/'manifest.json').exists():
                raise ValueError('Um dataset finalizado não pode ser descartado por esta operação.')
            shutil.rmtree(folder)
            return dict(discarded=True)

    def finalize(self, dataset_id, text, contract):
        with self.lock:
            folder=self.folder(dataset_id)
            if (folder/'manifest.json').exists():
                raise ValueError('Dataset já finalizado.')
            draft=load_json(folder/'draft.json')
            if not draft['items']:
                raise ValueError('Envie pelo menos uma foto.')
            annotations=read_labels(text,draft['items'],contract,draft['domain'])
            questions=contract['domains'][draft['domain']]['questions']
            items=[]
            for item in draft['items']:
                copied=dict(item);labels=annotations.get(item['filename'])
                if labels:
                    copied['expected' if set(labels)==set(questions) else 'annotations']=labels
                items.append(copied)
            manifest=dict(schema_version=1,name=draft['name'],label_status='user_provided' if annotations else 'unlabeled',items=items)
            # Validate the complete document before publishing it to the catalog.
            write_json(folder/'validation.json',manifest)
            try:
                validate_manifest(folder/'validation.json',folder,contract)
                (folder/'validation.json').replace(folder/'manifest.json')
            finally:
                (folder/'validation.json').unlink(missing_ok=True)
            return dataset_info(dataset_id,manifest)

"""Application services shared by the ASGI API and the legacy HTTP adapter."""
from __future__ import annotations
import base64
from io import BytesIO
from pathlib import Path
import re
import sys
from urllib.parse import urlsplit
import urllib.request
import json
from PIL import Image, UnidentifiedImageError
import psutil
from ..core import ROOT, load_json, resolve_image
from ..runner import validate_config
from .jobs import Jobs
from .datasets import Datasets, dataset_info

class Studio:
    def __init__(self, config_dir=None, jobs_root=None):
        self.config_dir = Path(config_dir or ROOT / 'benchmarks/configs')
        self.jobs = Jobs(jobs_root)
        self.datasets = Datasets(self.jobs.root.parent / "datasets")
        self.contracts = {
            'ollama': ROOT / 'benchmarks/contracts/description_ollama_v1.json',
            'original': ROOT / 'benchmarks/contracts/description_v1.json',
            'historical': ROOT / 'benchmarks/contracts/historical_tasks_v1.json',
        }
        self.manifests = {'portable': ROOT / 'benchmarks/datasets/portable_vehicles.json'}
        self.manifests.update({'historical-'+p.stem:p for p in sorted((ROOT/'benchmarks/datasets/historical').glob('*.json'))})
        self.sources = {
            'studio': self.jobs.root,
            'reports': ROOT / '.artifacts/reports',
            'scientific': ROOT / 'eval/figures/scientific_20261026_final',
        }

    def configs(self):
        result = {}
        for path in sorted(self.config_dir.glob('*.json')):
            try:
                doc = load_json(path)
                validate_config(doc)
                result[path.stem] = path
            except (ValueError, KeyError):
                pass
        return result

    def dataset_paths(self):
        return {**self.manifests, **self.datasets.manifests()}

    def catalog(self):
        from ..protocols import PROFILES
        return dict(protocols=PROFILES,configs=[dict(id=key, models=[dict(id=s['id'], adapter=s['adapter'],
            simulated=s['adapter']=='mock') for s in load_json(path)['models']]) for key,path in self.configs().items()],
            contracts=[dict(id=k, name=load_json(v)['name'], domains=list(load_json(v)['domains'])) for k,v in self.contracts.items()],
            datasets=[dict(dataset_info(k,load_json(v)),available=all(resolve_image(ROOT if k in self.manifests else v.parent,it['rel_path']).is_file() for it in load_json(v)['items'])) for k,v in self.dataset_paths().items()],
            current_job=self.jobs.current)

    def prepare(self, body):
        if not isinstance(body, dict):
            raise ValueError('Envie um objeto JSON.')
        config = self.configs().get(body.get('config'))
        contract_id=body.get('contract','ollama')
        if body.get('kind')=='benchmark':
            manifest_path=self.dataset_paths().get(body.get('dataset','portable'))
            if manifest_path:
                contract_id=load_json(manifest_path).get('required_contract') or contract_id
        contract = self.contracts.get(contract_id)
        if not config or not contract:
            raise ValueError('Selecione uma configuração e um contrato da lista.')
        models = body.get('models')
        if not isinstance(models, list) or not models or not all(isinstance(m,str) for m in models):
            raise ValueError('Selecione pelo menos um modelo.')
        specs = validate_config(load_json(config), models)
        doc = load_json(contract)
        for spec in specs:
            if spec.get('max_choices') and any(len(q.get('criteria',{})) > spec['max_choices']
                for d in doc['domains'].values() for q in d['questions'].values()):
                raise ValueError('Este modelo requer o contrato compartilhado Ollama (até 26 alternativas).')
        device = body.get('device', 'cpu')
        if not isinstance(device,str) or not re.fullmatch(r'cpu|cuda(?::\d+)?',device):
            raise ValueError('Dispositivo deve ser cpu, cuda ou cuda:N.')
        threads = body.get('threads', 8)
        if type(threads) is not int or not 1 <= threads <= 256:
            raise ValueError('Threads deve estar entre 1 e 256.')
        request = dict(kind=body.get('kind', 'live'), config=str(config), contract=str(contract),
                       models=[s['id'] for s in specs], device=device, threads=threads)
        image = None
        if request['kind'] == 'live':
            domain = body.get('domain', 'vehicle')
            if domain not in doc['domains']:
                raise ValueError('Domínio inválido.')
            request['domain'] = domain
            encoded = body.get('image_data', '')
            if not isinstance(encoded,str) or not encoded:
                raise ValueError('Escolha uma imagem antes de iniciar.')
            try:
                data = base64.b64decode(encoded.split(',')[-1], validate=True)
                with Image.open(BytesIO(data)) as source:
                    if source.width * source.height > 25_000_000:
                        raise ValueError('Use uma imagem com até 25 megapixels.')
                    image = source.convert('RGB')
                image.thumbnail((2048,2048))
            except (ValueError, UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
                raise ValueError(f'Imagem inválida: {exc}') from exc
        elif request['kind'] == 'benchmark':
            from ..protocols import MODES
            mode=body.get('comparison','baseline')
            if mode not in MODES:raise ValueError('Comparação inválida.')
            dataset_id=body.get('dataset','portable')
            manifest = self.dataset_paths().get(dataset_id)
            reps = body.get('repetitions',3)
            limit = body.get('limit', 0)
            if not manifest or type(reps) is not int or not 1 <= reps <= 100 or type(limit) is not int or not 0 <= limit <= 100000:
                raise ValueError('Dataset, repetições ou limite inválidos.')
            request['comparison']=mode
            request.update(manifest=str(manifest), dataset=dataset_id, data_root=str(ROOT if dataset_id in self.manifests else manifest.parent), repetitions=reps, limit=limit)
        else:
            raise ValueError('Tipo de tarefa inválido.')
        return request, image

    def artifacts(self):
        files = []
        for source, root in self.sources.items():
            if not root.exists():
                continue
            for path in sorted(root.rglob('*')):
                rel = path.relative_to(root).as_posix()
                if path.is_file() and path.suffix in {'.png','.pdf','.csv','.md','.json','.jsonl'}:
                    if source == 'studio' and not ('/report/' in rel or '/run/' in rel):
                        continue
                    files.append(dict(source=source, name=rel, url=f'/files/{source}/{rel}'))
        return files

    def diagnostics(self):
        memory = psutil.virtual_memory()
        result = dict(python=sys.version.split()[0], ram_total_gb=round(memory.total/2**30,1),
                      ram_available_gb=round(memory.available/2**30,1), inference_tested=False)
        # Probe only the configured Ollama origin; never load weights or run inference.
        import os
        value = os.environ.get('OLLAMA_SYSTEMONE_URL','http://127.0.0.1:11434/v1/systemone')
        parsed = urlsplit(value)
        if parsed.hostname not in {'127.0.0.1','localhost','::1','ollama','host.docker.internal'} or parsed.scheme != 'http' or parsed.username or parsed.password:
            result.update(ollama='not-probed',error='Diagnóstico automático disponível apenas para Ollama HTTP local.')
            return result
        origin = f'{parsed.scheme}://{parsed.netloc}'
        try:
            for key, route in [('version','version'),('models','tags'),('loaded','ps')]:
                with urllib.request.urlopen(origin+'/api/'+route,timeout=3) as response:
                    data = json.load(response)
                result[key] = data.get('models',[]) if key != 'version' else data.get('version')
            result['ollama'] = 'online'
            result['clef_installed'] = any(m.get('name') == 'clef-flash:9b' for m in result['models'])
        except (OSError,ValueError) as exc:
            result.update(ollama='unavailable',error=str(exc))
        return result

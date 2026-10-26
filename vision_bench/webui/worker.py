"""Isolated model loading and serial prediction; never runs in HTTP threads."""
from __future__ import annotations

from datetime import datetime, timezone
import gc
import sys
import time
from pathlib import Path

from PIL import Image

from ..adapters import create_adapter
from ..core import fingerprint, finalize, load_json, sha256, write_json


def run(folder):
    request = load_json(folder / 'request.json')
    status = load_json(folder / 'status.json')
    status.update(state='running', started_utc=datetime.now(timezone.utc).isoformat())
    write_json(folder / 'status.json', status)
    try:
        if request['kind'] == 'benchmark':
            from ..cli import main
            run_path = folder / 'run'
            arguments = ['run', '--config', request['config'], '--contract', request['contract'],
                         '--manifest', request['manifest'], '--models', *request['models'],
                         '--output', str(run_path), '--repetitions', str(request['repetitions']),
                         '--threads', str(request['threads']), '--device', request['device']]
            if request.get('limit'):
                arguments += ['--limit', str(request['limit'])]
            if main(arguments):
                raise RuntimeError('O benchmark falhou. Consulte o registro abaixo; os arquivos parciais foram preservados.')
            if main(['report', str(run_path), '--output', str(folder / 'report')]):
                raise RuntimeError('A rodada terminou, mas a geração do relatório falhou. Consulte o registro.')
            status['errors'] = sum(row['errors'] for row in load_json(folder / 'report/metrics.json'))
        else:
            contract = load_json(request['contract'])
            specs = load_json(request['config'])['models']
            specs = [s for s in specs if s['id'] in request['models']]
            with Image.open(folder / 'input.png') as source:
                image = source.convert('RGB')
            image.thumbnail((1024, 1024))
            status.update(contract_sha256=fingerprint(contract), image_sha256=sha256(folder / 'input.png'),
                          domain=request['domain'], device=request['device'],
                          note='Teste interativo sem aquecimento; tempo de carregamento separado. Sem gabarito ou estimativa de acurácia.')
            for spec in specs:
                adapter = None
                result = dict(model=spec['id'], simulated=spec['adapter'] == 'mock')
                status.update(current_model=spec['id'], phase='Carregando modelo ou preparando cliente HTTP')
                write_json(folder / 'status.json', status)
                try:
                    if spec['adapter'] in {'blip', 'laya'}:
                        import torch
                        torch.set_num_threads(request['threads'])
                    timer = time.perf_counter()
                    adapter = create_adapter(spec, contract, device=request['device'], offline=False, diagnostics=True)
                    adapter.synchronize()
                    result['load_ms'] = (time.perf_counter() - timer) * 1000
                    status['phase'] = 'Avaliando imagem'
                    write_json(folder / 'status.json', status)
                    timer = time.perf_counter()
                    answers, raw = adapter.predict(image, request['domain'])
                    adapter.synchronize()
                    result.update(elapsed_ms=(time.perf_counter() - timer) * 1000,
                                  raw=raw, **finalize(contract, request['domain'], answers))
                except Exception as exc:
                    result['error'] = f'{type(exc).__name__}: {exc}'
                finally:
                    if adapter is not None:
                        adapter.close()
                        del adapter
                    gc.collect()
                    if 'torch' in sys.modules and request['device'].startswith('cuda'):
                        sys.modules['torch'].cuda.empty_cache()
                status['results'].append(result)
                write_json(folder / 'status.json', status)
            status.pop('current_model', None)
        errors=sum(bool(r.get('error')) for r in status.get('results',[])) if request['kind']=='live' else status.get('errors',0)
        status.update(state='complete', errors=errors, phase='Concluído com falhas' if errors else 'Concluído', finished_utc=datetime.now(timezone.utc).isoformat())
    except Exception as exc:
        status.update(state='failed', error=f'{type(exc).__name__}: {exc}')
    write_json(folder / 'status.json', status)


if __name__ == '__main__':
    run(Path(sys.argv[1]).resolve())

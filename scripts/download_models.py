"""Automated download, caching and verification for vision benchmark models.

Supports:
- laya-vision (thaitea/laya-vision + SmolVLM-256M backbone)
- d1-omni-600M (LiquidAI/d1-omni-600M)
- clef-flash GGUF (ggml-org/Clef-Flash-GGUF for llama-server)
- clef-flash Ollama (clef-flash:9b via Ollama service)
- blip-vqa (Salesforce/blip-vqa-base)

Ensures all weights, backbones and tokenizers are pre-cached for 100% offline runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request
import urllib.error

ROOT = Path(__file__).resolve().parents[1]

MODEL_SPECS = {
    'laya': {
        'name': 'Laya Vision (thaitea/laya-vision)',
        'description': 'Decision model with SmolVLM backbone for structured System One questions',
        'repo_id': 'thaitea/laya-vision',
        'revision': 'f2fe3c12cb6d04c59d8a190250bf3fb40fc828dc',
        'backbone_repo': 'HuggingFaceTB/SmolVLM-256M-Instruct',
        'estimated_size_mb': 1300,
        'requires_submodule': True,
    },
    'd1-omni': {
        'name': 'Liquid AI d1-omni-600M (LiquidAI/d1-omni-600M)',
        'description': 'Multimodal decision model (~587M params, zero generated tokens) via transformers',
        'repo_id': 'LiquidAI/d1-omni-600M',
        'revision': '414f8d6438174f5b2133a9c21a478fc42625e308',
        'estimated_size_mb': 2350,
        'requires_submodule': False,
    },
    'clef-gguf': {
        'name': 'Clef-Flash GGUF Q4_K_M (ggml-org/Clef-Flash-GGUF)',
        'description': 'Quantized GGUF model + mmproj projector for local llama-server',
        'weights_file': 'benchmarks/configs/clef_weights.json',
        'estimated_size_mb': 7350,
        'requires_submodule': False,
    },
    'clef-ollama': {
        'name': 'Clef-Flash Ollama (clef-flash:9b)',
        'description': 'Clef-Flash deployed via Ollama local API service',
        'ollama_tag': 'clef-flash:9b',
        'estimated_size_mb': 5500,
        'requires_submodule': False,
    },
    'blip': {
        'name': 'BLIP VQA Base (Salesforce/blip-vqa-base)',
        'description': 'Generative baseline visual question answering model',
        'repo_id': 'Salesforce/blip-vqa-base',
        'revision': '787b3d35d57e49572baabd22884b3d5a05acf072',
        'estimated_size_mb': 1900,
        'requires_submodule': False,
    },
}

PRESETS = {
    'vision-all': ['laya', 'd1-omni', 'clef-gguf', 'clef-ollama'],
    'vision-local': ['laya', 'd1-omni', 'clef-gguf'],
    'vision-ollama': ['laya', 'd1-omni', 'clef-ollama'],
    'all': ['laya', 'd1-omni', 'clef-gguf', 'clef-ollama', 'blip'],
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def ensure_laya_submodule() -> bool:
    vlm_path = ROOT / 'vendor/laya-vision/laya/vlm.py'
    if vlm_path.is_file():
        return True
    print('--> Inicializando submódulo vendor/laya-vision via git...')
    try:
        subprocess.run(['git', 'submodule', 'update', '--init', '--recursive'], cwd=ROOT, check=True)
        return vlm_path.is_file()
    except Exception as exc:
        print(f'AVISO: Falha ao inicializar submódulo laya-vision: {exc}', file=sys.stderr)
        return False


def download_hf_repo(repo_id: str, revision: str | None = None, token: str | None = None) -> str:
    from huggingface_hub import snapshot_download
    print(f'--> Baixando repositório Hugging Face: {repo_id} (revisão: {revision or "latest"})...')
    start = time.perf_counter()
    path = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        token=token,
        resume_download=True,
    )
    elapsed = time.perf_counter() - start
    print(f'    ✓ {repo_id} em cache ({elapsed:.1f}s): {path}')
    return path


def download_clef_gguf(token: str | None = None) -> dict[str, str]:
    from huggingface_hub import hf_hub_download
    weights_path = ROOT / MODEL_SPECS['clef-gguf']['weights_file']
    weights = json.loads(weights_path.read_text(encoding='utf-8'))
    repo = weights['repository']
    rev = weights['revision']
    result = {}
    print(f'--> Baixando pesos GGUF do Clef ({repo}, rev: {rev})...')
    for file_info in weights['files']:
        fname = file_info['filename']
        expected_sha = file_info['sha256']
        expected_size = file_info['size']
        print(f'    Arquivo: {fname} (~{expected_size / 1024 / 1024:.0f} MB)...')
        start = time.perf_counter()
        downloaded = hf_hub_download(
            repo_id=repo,
            filename=fname,
            revision=rev,
            token=token,
            resume_download=True,
        )
        p = Path(downloaded)
        elapsed = time.perf_counter() - start
        print(f'    Verificando SHA-256 de {fname}...')
        observed_sha = sha256_file(p)
        if observed_sha != expected_sha:
            raise ValueError(f'Corrupção no arquivo {fname}! SHA esperado {expected_sha}, observado {observed_sha}')
        print(f'    ✓ {fname} verificado com sucesso ({elapsed:.1f}s).')
        result[fname] = str(p)
    return result


def download_clef_ollama(ollama_url: str = 'http://127.0.0.1:11434', tag: str = 'clef-flash:9b') -> bool:
    print(f'--> Solicitando pull de {tag} no Ollama...')
    # Check if CLI exists first
    ollama_cli = shutil.which('ollama')
    if ollama_cli:
        print(f'    Executando comando: ollama pull {tag}...')
        proc = subprocess.run([ollama_cli, 'pull', tag])
        if proc.returncode == 0:
            print(f'    ✓ Modelo {tag} disponível no Ollama local.')
            return True

    # Try HTTP API
    endpoint = f'{ollama_url.rstrip("/")}/api/pull'
    print(f'    Enviando requisição HTTP para {endpoint}...')
    try:
        req = urllib.request.Request(
            endpoint,
            data=json.dumps({'model': tag, 'stream': False}).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        with urllib.request.urlopen(req, timeout=1800) as resp:
            data = json.load(resp)
            if data.get('status') == 'success':
                print(f'    ✓ {tag} baixado com sucesso via API do Ollama.')
                return True
    except urllib.error.URLError as exc:
        print(f'    AVISO: Serviço Ollama não acessível em {ollama_url} ({exc}).')
        print(f'    Para usar Clef no Ollama, inicie "ollama serve" e execute "ollama pull {tag}".')
        return False
    return False


def verify_model(key: str, device: str = 'cpu') -> bool:
    print(f'--> Verificando carregamento do modelo [{key}]...')
    try:
        if key == 'd1-omni':
            import torch
            from transformers import AutoModel
            spec = MODEL_SPECS['d1-omni']
            dtype = torch.float16 if device.startswith('cuda') else torch.float32
            model = AutoModel.from_pretrained(
                spec['repo_id'],
                revision=spec['revision'],
                trust_remote_code=True,
                torch_dtype=dtype,
                local_files_only=True
            ).to(device).eval()
            _ = model.tokenizer
            del model
            print('    ✓ d1-omni-600M carregado com sucesso (pesos + tokenizador).')
            return True

        if key == 'laya':
            ensure_laya_submodule()
            sys.path.insert(0, str(ROOT / 'vendor/laya-vision'))
            import laya
            spec = MODEL_SPECS['laya']
            model = laya.load_vlm(spec['repo_id'], revision=spec['revision'], device=device)
            del model
            print('    ✓ Laya Vision carregado com sucesso (pesos + backbone SmolVLM).')
            return True

        if key == 'blip':
            from transformers import AutoProcessor, AutoModelForVisualQuestionAnswering
            spec = MODEL_SPECS['blip']
            proc = AutoProcessor.from_pretrained(spec['repo_id'], revision=spec['revision'], local_files_only=True)
            model = AutoModelForVisualQuestionAnswering.from_pretrained(spec['repo_id'], revision=spec['revision'], local_files_only=True).to(device).eval()
            del proc, model
            print('    ✓ BLIP VQA carregado com sucesso.')
            return True

        if key == 'clef-gguf':
            weights_path = ROOT / MODEL_SPECS['clef-gguf']['weights_file']
            weights = json.loads(weights_path.read_text(encoding='utf-8'))
            from huggingface_hub import hf_hub_download
            for f in weights['files']:
                p = Path(hf_hub_download(repo_id=weights['repository'], filename=f['filename'], revision=weights['revision'], local_files_only=True))
                if not p.is_file():
                    raise FileNotFoundError(f'GGUF ausente: {f["filename"]}')
            print('    ✓ Clef GGUF e mmproj presentes e íntegros em cache.')
            return True

        if key == 'clef-ollama':
            tag = MODEL_SPECS['clef-ollama']['ollama_tag']
            cli = shutil.which('ollama')
            if cli:
                out = subprocess.run([cli, 'list'], capture_output=True, text=True)
                if tag.split(':')[0] in out.stdout:
                    print(f'    ✓ {tag} confirmado no catálogo do Ollama.')
                    return True
            print(f'    ? Verifique se "{tag}" está visível com "ollama list".')
            return True

    except Exception as exc:
        print(f'    ✗ ERRO na verificação de [{key}]: {exc}', file=sys.stderr)
        return False
    return True


def parse_selected_models(choices: list[str]) -> list[str]:
    selected = set()
    for item in choices:
        item = item.strip().lower()
        if item in PRESETS:
            selected.update(PRESETS[item])
        elif item in MODEL_SPECS:
            selected.add(item)
        else:
            raise ValueError(f'Modelo ou preset desconhecido: "{item}". Opções válidas: {sorted(MODEL_SPECS)} ou presets {sorted(PRESETS)}')
    return [m for m in ['laya', 'd1-omni', 'clef-gguf', 'clef-ollama', 'blip'] if m in selected]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--models',
        nargs='+',
        default=['laya', 'd1-omni', 'clef-gguf'],
        help='Modelos para instalar: laya, d1-omni, clef-gguf, clef-ollama, blip, vision-local, vision-ollama, vision-all, all'
    )
    parser.add_argument('--verify', action='store_true', default=True, help='Testa o carregamento de cada modelo após download (default: ativado)')
    parser.add_argument('--no-verify', dest='verify', action='store_false', help='Desativa a verificação imediata')
    parser.add_argument('--device', default='cpu', help='Dispositivo para teste de carregamento: cpu ou cuda')
    parser.add_argument('--dry-run', action='store_true', help='Apenas exibe os modelos, requisitos e espaço em disco sem baixar')
    parser.add_argument('--hf-token', default=os.environ.get('HF_TOKEN'), help='Token Hugging Face opcional (ou variável de ambiente HF_TOKEN)')
    parser.add_argument('--ollama-url', default='http://127.0.0.1:11434', help='URL base da API Ollama local (default: http://127.0.0.1:11434)')

    args = parser.parse_args()
    selected = parse_selected_models(args.models)

    total_mb = sum(MODEL_SPECS[m]['estimated_size_mb'] for m in selected)
    print('======================================================================')
    print(' Instalador Automático de Modelos Visuais para Benchmarking')
    print('======================================================================')
    print(f'Modelos selecionados ({len(selected)}):')
    for m in selected:
        info = MODEL_SPECS[m]
        print(f'  • {m:12} : {info["name"]} (~{info["estimated_size_mb"]} MB)')
    print(f'Espaço em disco estimado total: ~{total_mb / 1024:.2f} GB')
    print('======================================================================\n')

    if args.dry_run:
        print('Modo --dry-run: nenhum arquivo foi baixado.')
        return 0

    token = args.hf_token
    failed = []

    for key in selected:
        print(f'\n>>> Processando [{key}]...')
        try:
            if key == 'laya':
                ensure_laya_submodule()
                spec = MODEL_SPECS['laya']
                download_hf_repo(spec['repo_id'], revision=spec['revision'], token=token)
                print('    Baixando backbone SmolVLM para garantir operação offline...')
                download_hf_repo(spec['backbone_repo'], token=token)

            elif key == 'd1-omni':
                spec = MODEL_SPECS['d1-omni']
                download_hf_repo(spec['repo_id'], revision=spec['revision'], token=token)

            elif key == 'clef-gguf':
                download_clef_gguf(token=token)

            elif key == 'clef-ollama':
                tag = MODEL_SPECS['clef-ollama']['ollama_tag']
                download_clef_ollama(ollama_url=args.ollama_url, tag=tag)

            elif key == 'blip':
                spec = MODEL_SPECS['blip']
                download_hf_repo(spec['repo_id'], revision=spec['revision'], token=token)

            if args.verify:
                ok = verify_model(key, device=args.device)
                if not ok:
                    failed.append(key)
        except Exception as exc:
            print(f'ERRO ao instalar [{key}]: {exc}', file=sys.stderr)
            failed.append(key)

    print('\n======================================================================')
    if failed:
        print(f'✗ Concluído com falhas em: {failed}')
        print('Verifique a conexão de rede ou dependências.')
        return 1
    else:
        print('✓ Todos os modelos selecionados foram instalados e verificados com sucesso!')
        print('Execute o diagnóstico geral para confirmar o ambiente:')
        print('    .venv/bin/python -m vision_bench doctor')
        return 0


if __name__ == '__main__':
    sys.exit(main())

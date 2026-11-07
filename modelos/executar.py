#!/usr/bin/env python3
"""
CLI Unificado para Execução dos Testes Padronizados System One.

Permite executar qualquer teste específico por modelo ou rodar o mesmo teste
em todos os modelos para comparação imediata.

Exemplos de uso:
    python modelos/executar.py --listar
    python modelos/executar.py --modelo blip --teste test_01_cor_veiculo.py
    python modelos/executar.py --modelo clef --teste 01 --mock
    python modelos/executar.py --modelo todos --teste 01 --limit 5
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
MODELOS_DISPONIVEIS = ["blip", "clef", "laya", "d1_omni"]


def listar_testes():
    """Lista todos os testes padronizados disponíveis."""
    print("\n" + "=" * 80)
    print(" TESTES PADRONIZADOS DISPONÍVEIS PARA TODOS OS SYSTEM ONES")
    print("=" * 80)

    dir_ref = ROOT / "modelos" / "blip"
    print("\n📁 Testes de Datasets (modelos/<modelo>/testes_datasets/):")
    for f in sorted((dir_ref / "testes_datasets").glob("test_*.py")):
        print(f"   • {f.name}")

    print("\n📁 Testes de Prompts (modelos/<modelo>/testes_prompts/):")
    for f in sorted((dir_ref / "testes_prompts").glob("test_*.py")):
        print(f"   • {f.name}")

    print("\n🤖 Modelos Suportados:")
    for m in MODELOS_DISPONIVEIS:
        print(f"   • {m}")
    print("=" * 80 + "\n")


def resolver_caminho_teste(modelo: str, teste_arg: str) -> Path | None:
    """Encontra o arquivo de teste correspondente ao argumento."""
    dir_modelo = ROOT / "modelos" / modelo

    # 1. Busca exata
    candidatos = list(dir_modelo.rglob(teste_arg))
    if candidatos:
        return candidatos[0]

    # 2. Busca por prefixo/número (ex: '01', 'cor', 'formatos')
    padrao = f"*{teste_arg}*.py"
    candidatos = list(dir_modelo.rglob(padrao))
    if candidatos:
        return candidatos[0]

    return None


def main():
    """Run main."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modelo", "-m", choices=MODELOS_DISPONIVEIS + ["todos"], help="Modelo a ser testado (ou 'todos')")
    parser.add_argument("--teste", "-t", type=str, help="Nome ou número do teste (ex: '01', 'cor_veiculo', 'test_01_formatos_prompt.py')")
    parser.add_argument("--listar", "-l", action="store_true", help="Lista todos os testes e modelos disponíveis")
    parser.add_argument("--limit", "-n", type=int, default=None, help="Limita imagens")
    parser.add_argument("--mock", action="store_true", help="Executa em modo simulado/mock")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Dispositivo (cpu/cuda)")
    parser.add_argument("--url", type=str, default=None, help="URL para modelo Clef HTTP")
    args, extras = parser.parse_known_args()

    if args.listar or not (args.modelo and args.teste):
        listar_testes()
        if not (args.modelo and args.teste):
            print("Use --modelo <nome> --teste <numero/nome> para executar um teste.\n")
            return

    modelos_alvo = MODELOS_DISPONIVEIS if args.modelo == "todos" else [args.modelo]

    for mod in modelos_alvo:
        caminho_teste = resolver_caminho_teste(mod, args.teste)
        if not caminho_teste or not caminho_teste.is_file():
            print(f"[ERRO] Teste '{args.teste}' não encontrado para o modelo '{mod}'.")
            continue

        cmd = [sys.executable, str(caminho_teste)]
        if args.limit:
            cmd.extend(["--limit", str(args.limit)])
        if args.mock:
            cmd.append("--mock")
        if args.device:
            cmd.extend(["--device", args.device])
        if args.url:
            cmd.extend(["--url", args.url])
        cmd.extend(extras)

        print(f"\n>>> Executando: {' '.join(cmd)}")
        subprocess.run(cmd)


if __name__ == "__main__":
    main()

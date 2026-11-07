#!/usr/bin/env python3
"""
Teste de Prompts: 02 - Variação do Texto de Instruções
Modelo: Laya Vision (thaitea) (laya)

Execute diretamente:
    python test_02_variacao_instrucoes.py [--limit 5] [--mock] [--device cpu]
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modelos.comum.motor import executar_teste_prompt

# ==============================================================================
# CONFIGURAÇÃO DO TESTE (EDITE AQUI FACILMENTE)
# ==============================================================================
CONFIG = {
    "nome": "02 - Variação Descritiva das Opções",
    "campo_alvo": "color",
    "dominio": "vehicle",
    "pergunta_base": "Describe only visible attributes of the target vehicle.",
    "formatos": ['native_short', 'native_descriptive'],
    "criterios": {
        "black": "Black",
        "white": "White",
        "gray_silver": "Gray or silver",
        "red": "Red",
        "blue": "Blue",
        "unknown": "Unknown / not visible",
    },
    "dataset": "benchmarks/datasets/vehicles/portable.json",
}

if __name__ == "__main__":
    executar_teste_prompt(modelo="laya", config=CONFIG)

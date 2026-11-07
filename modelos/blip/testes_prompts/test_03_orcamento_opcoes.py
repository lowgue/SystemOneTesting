#!/usr/bin/env python3
"""
Teste de Prompts: 03 - Orçamento de Opções (Tamanho da Taxonomia)
Modelo: BLIP VQA (Salesforce) (blip)

Execute diretamente:
    python test_03_orcamento_opcoes.py [--limit 5] [--mock] [--device cpu]
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
    "nome": "03 - Variação Descritiva das Opções",
    "campo_alvo": "color",
    "dominio": "vehicle",
    "pergunta_base": "What is the color of this vehicle?",
    "formatos": ['short_options', 'long_descriptions'],
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
    executar_teste_prompt(modelo="blip", config=CONFIG)

#!/usr/bin/env python3
"""
Teste de Prompts: 01 - Comparação de Formatos de Prompt
Modelo: LiquidAI d1-omni-600M (d1_omni)

Execute diretamente:
    python test_01_formatos_prompt.py [--limit 5] [--mock] [--device cpu]
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.common.motor import executar_teste_prompt

# ==============================================================================
# CONFIGURAÇÃO DO TESTE (EDITE AQUI FACILMENTE)
# ==============================================================================
CONFIG = {
    "nome": "01 - Comparação de Formatos de Prompt",
    "campo_alvo": "color",
    "dominio": "vehicle",
    "pergunta_base": "What is the predominant color of this vehicle?",
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
    executar_teste_prompt(modelo="d1_omni", config=CONFIG)

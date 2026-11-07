#!/usr/bin/env python3
"""
Teste: 07 - Robustez a Degradações
Modelo: LiquidAI d1-omni-600M (d1_omni)

Execute diretamente:
    python test_07_robustez_imagem.py [--limit 5] [--mock] [--device cpu]
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modelos.comum.motor import executar_teste_dataset

# ==============================================================================
# CONFIGURAÇÃO DO TESTE (EDITE AQUI FACILMENTE)
# ==============================================================================
CONFIG = {
    "nome": "07 - Robustez a Degradações",
    "campo_alvo": "color",
    "dominio": "vehicle",
    "pergunta": "What is the color under image degradations?",
    "criterios": {
        "black": "Black",
        "white": "White",
        "gray_silver": "Gray or silver",
        "red": "Red",
        "blue": "Blue",
        "green": "Green",
        "yellow": "Yellow",
        "unknown": "Not identifiable",
    },
    "dataset": "benchmarks/datasets/vehicles/portable.json",
    "repeticoes": 1,
}

if __name__ == "__main__":
    executar_teste_dataset(modelo="d1_omni", config=CONFIG)

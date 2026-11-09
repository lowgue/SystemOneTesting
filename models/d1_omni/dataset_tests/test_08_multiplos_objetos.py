#!/usr/bin/env python3
"""
Teste: 08 - Múltiplos Objetos e Associação
Modelo: LiquidAI d1-omni-600M (d1_omni)

Execute diretamente:
    python test_08_multiplos_objetos.py [--limit 5] [--mock] [--device cpu]
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.common.motor import executar_teste_dataset

# ==============================================================================
# CONFIGURAÇÃO DO TESTE (EDITE AQUI FACILMENTE)
# ==============================================================================
CONFIG = {
    "nome": "08 - Múltiplos Objetos e Associação",
    "campo_alvo": "color",
    "dominio": "vehicle",
    "pergunta": "What is the predominant color of the primary vehicle in the scene?",
    "criterios": {
        "black": "Black",
        "white": "White",
        "gray_silver": "Gray or silver",
        "red": "Red",
        "blue": "Blue",
        "unknown": "Unknown",
    },
    "dataset": "benchmarks/datasets/vehicles/portable.json",
    "repeticoes": 1,
}

if __name__ == "__main__":
    executar_teste_dataset(modelo="d1_omni", config=CONFIG)

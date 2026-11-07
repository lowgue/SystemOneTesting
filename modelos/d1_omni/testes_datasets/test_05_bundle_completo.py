#!/usr/bin/env python3
"""
Teste: 05 - Bundle Completo Multi-atributo
Modelo: LiquidAI d1-omni-600M (d1_omni)

Execute diretamente:
    python test_05_bundle_completo.py [--limit 5] [--mock] [--device cpu]
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
    "nome": "05 - Bundle Completo Multi-atributo",
    "campo_alvo": "bundle_completo",
    "dominio": "vehicle",
    "pergunta": "Describe only the visible attributes of the vehicle.",
    "criterios": {
        "presence": "Vehicle presence",
        "make": "Vehicle make",
        "color": "Vehicle color",
        "body": "Vehicle body style",
    },
    "dataset": "benchmarks/datasets/vehicles/portable.json",
    "repeticoes": 1,
}

if __name__ == "__main__":
    executar_teste_dataset(modelo="d1_omni", config=CONFIG)

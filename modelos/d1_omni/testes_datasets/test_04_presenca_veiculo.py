#!/usr/bin/env python3
"""
Teste: 04 - Presença de Veículo
Modelo: LiquidAI d1-omni-600M (d1_omni)

Execute diretamente:
    python test_04_presenca_veiculo.py [--limit 5] [--mock] [--device cpu]
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
    "nome": "04 - Presença de Veículo",
    "campo_alvo": "presence",
    "dominio": "vehicle",
    "pergunta": "Is there a vehicle in the image?",
    "criterios": {
        "yes": "Yes, there is a vehicle",
        "no": "No vehicle present",
    },
    "dataset": "benchmarks/datasets/vehicles/portable.json",
    "repeticoes": 1,
}

if __name__ == "__main__":
    executar_teste_dataset(modelo="d1_omni", config=CONFIG)

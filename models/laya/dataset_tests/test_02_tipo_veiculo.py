#!/usr/bin/env python3
"""
Teste: 02 - Tipo de Veículo (Carroceria)
Modelo: Laya Vision (thaitea) (laya)

Execute diretamente:
    python test_02_tipo_veiculo.py [--limit 5] [--mock] [--device cpu]
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
    "nome": "02 - Tipo de Veículo (Carroceria)",
    "campo_alvo": "body",
    "dominio": "vehicle",
    "pergunta": "What body style is the vehicle?",
    "criterios": {
        "sedan": "Sedan",
        "suv": "SUV",
        "hatch": "Hatchback",
        "truck": "Pickup truck",
        "van": "Van or minivan",
        "coupe": "Coupe",
        "convertible": "Convertible",
        "wagon": "Wagon",
        "other": "Another body style",
        "unknown": "No car or not identifiable",
    },
    "dataset": "benchmarks/datasets/vehicles/portable.json",
    "repeticoes": 1,
}

if __name__ == "__main__":
    executar_teste_dataset(modelo="laya", config=CONFIG)

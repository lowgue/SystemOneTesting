#!/usr/bin/env python3
"""
Teste: 03 - Marca do Veículo
Modelo: LiquidAI d1-omni-600M (d1_omni)

Execute diretamente:
    python test_03_marca_veiculo.py [--limit 5] [--mock] [--device cpu]
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
    "nome": "03 - Marca do Veículo",
    "campo_alvo": "make",
    "dominio": "vehicle",
    "pergunta": "What brand is the vehicle?",
    "criterios": {
        "audi": "Audi",
        "bmw": "BMW",
        "chevrolet": "Chevrolet",
        "dodge": "Dodge",
        "ford": "Ford",
        "honda": "Honda",
        "hyundai": "Hyundai",
        "jeep": "Jeep",
        "land_rover": "Land Rover",
        "toyota": "Toyota",
        "volkswagen": "Volkswagen",
        "unknown": "Unknown or no car",
    },
    "dataset": "benchmarks/datasets/vehicles/portable.json",
    "repeticoes": 1,
}

if __name__ == "__main__":
    executar_teste_dataset(modelo="d1_omni", config=CONFIG)

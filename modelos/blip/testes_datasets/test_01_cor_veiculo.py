#!/usr/bin/env python3
"""
Teste: 01 - Cor do Veículo
Modelo: BLIP VQA (Salesforce) (blip)

Execute diretamente:
    python test_01_cor_veiculo.py [--limit 5] [--mock] [--device cpu]
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
    "nome": "01 - Cor do Veículo",
    "campo_alvo": "color",
    "dominio": "vehicle",
    "pergunta": "What is the predominant color of this vehicle?",
    "criterios": {
        "black": "Black",
        "white": "White",
        "gray_silver": "Gray or silver",
        "red": "Red",
        "blue": "Blue",
        "green": "Green",
        "yellow": "Yellow",
        "orange": "Orange",
        "brown": "Brown",
        "purple": "Purple",
        "pink": "Pink",
        "unknown": "No car or color not identifiable / not visible",
    },
    "dataset": "benchmarks/datasets/vehicles/portable.json",
    "repeticoes": 1,
}

if __name__ == "__main__":
    executar_teste_dataset(modelo="blip", config=CONFIG)

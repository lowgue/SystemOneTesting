#!/usr/bin/env python3
"""
Teste: 06 - Atributos de Pessoa
Modelo: Laya Vision (thaitea) (laya)

Execute diretamente:
    python test_06_atributos_pessoa.py [--limit 5] [--mock] [--device cpu]
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
    "nome": "06 - Atributos de Pessoa",
    "campo_alvo": "helmet",
    "dominio": "person",
    "pergunta": "Is the person wearing a helmet?",
    "criterios": {
        "yes": "Yes, wearing helmet or hard hat",
        "no": "No, bare head or regular cap/hat",
        "not_visible": "Head not visible or no person",
    },
    "dataset": "benchmarks/datasets/historical/people/manifests/test_3_person_attributes_manifest.json",
    "repeticoes": 1,
}

if __name__ == "__main__":
    executar_teste_dataset(modelo="laya", config=CONFIG)

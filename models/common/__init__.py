"""Pacote compartilhado de utilitários e motor de testes para os 4 modelos System One."""

from .motor import executar_teste_dataset, executar_teste_prompt
from .model_specs import SPECS_MODELOS, obter_spec_modelo
from .metricas import intervalo_wilson_95, resumir_latencias, imprimir_tabela_resumo

__all__ = [
    "executar_teste_dataset",
    "executar_teste_prompt",
    "SPECS_MODELOS",
    "obter_spec_modelo",
    "intervalo_wilson_95",
    "resumir_latencias",
    "imprimir_tabela_resumo",
]

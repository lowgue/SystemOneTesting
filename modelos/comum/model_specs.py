"""Definição padronizada dos 4 modelos System One suportados."""

from __future__ import annotations
import os
from typing import Any, Dict

SPECS_MODELOS: Dict[str, Dict[str, Any]] = {
    "blip": {
        "id": "blip-vqa",
        "nome": "Salesforce BLIP VQA (Base)",
        "adapter": "blip",
        "model": "Salesforce/blip-vqa-base",
        "revision": "787b3d35d57e49572baabd22884b3d5a05acf072",
        "tipo": "generativo_vqa",
        "perfil_padrao": "short_open",
    },
    "clef": {
        "id": "clef-flash-local-q4",
        "nome": "Clef-Flash Q4_K_M (System One API)",
        "adapter": "systemone_http",
        "url_env": "CLEF_LOCAL_SYSTEMONE_URL",
        "url": os.environ.get("CLEF_LOCAL_SYSTEMONE_URL", "http://127.0.0.1:11437/v1/systemone"),
        "timeout_s": 300,
        "model": "ggml-org/Clef-Flash-GGUF",
        "quantization": "Q4_K_M",
        "tipo": "http_api",
        "perfil_padrao": "native_short",
    },
    "laya": {
        "id": "laya-vision",
        "nome": "Laya Vision (thaitea/laya-vision)",
        "adapter": "laya",
        "model": "thaitea/laya-vision",
        "revision": "f2fe3c12cb6d04c59d8a190250bf3fb40fc828dc",
        "head_max_len": 512,
        "tipo": "decisao_estruturada",
        "perfil_padrao": "native_short",
    },
    "d1_omni": {
        "id": "d1-omni-600M",
        "nome": "LiquidAI d1-omni-600M",
        "adapter": "d1_omni",
        "model": "LiquidAI/d1-omni-600M",
        "revision": "414f8d6438174f5b2133a9c21a478fc42625e308",
        "tipo": "decisao_multimodal",
        "perfil_padrao": "native_short",
    },
}


def obter_spec_modelo(nome_ou_id: str, usar_mock: bool = False, url_custom: str | None = None) -> Dict[str, Any]:
    """Retorna a especificação configurada do modelo ou um mock adapter."""
    chave = nome_ou_id.lower().replace("-", "_")
    if chave not in SPECS_MODELOS:
        chaves_validas = ", ".join(SPECS_MODELOS.keys())
        raise ValueError(f"Modelo '{nome_ou_id}' desconhecido. Modelos válidos: {chaves_validas}")

    spec = dict(SPECS_MODELOS[chave])
    if usar_mock:
        spec["adapter"] = "mock"
        spec["nome"] += " [SIMULADO/MOCK]"

    if url_custom and spec["adapter"] == "systemone_http":
        spec["url"] = url_custom

    return spec

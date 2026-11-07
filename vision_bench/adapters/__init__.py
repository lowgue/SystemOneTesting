"""Public adapter registry and factory for model-specific request implementations."""

from __future__ import annotations

import importlib
from typing import Any

from .base import Adapter, MockAdapter
from .d1_omni import D1Omni
from .http import SystemOneHTTP, endpoint
from .laya import Laya
from .transformers import BLIP

REGISTRY: dict[str, type[MockAdapter]] = {
    "mock": MockAdapter,
    "systemone_http": SystemOneHTTP,
    "blip": BLIP,
    "laya": Laya,
    "d1_omni": D1Omni,
}


def create_adapter(
    spec: dict[str, Any],
    contract: dict[str, Any],
    **context: Any,
) -> Adapter:
    """Construct the configured built-in adapter or import a custom adapter."""
    adapter_name = spec["adapter"]
    if adapter_name.startswith("python:"):
        module_name, class_name = adapter_name.removeprefix("python:").rsplit(":", 1)
        adapter_class = getattr(importlib.import_module(module_name), class_name)
    else:
        try:
            adapter_class = REGISTRY[adapter_name]
        except KeyError as error:
            raise ValueError(f"Unknown adapter: {adapter_name}") from error
    return adapter_class(spec, contract, **context)


__all__ = [
    "Adapter",
    "BLIP",
    "D1Omni",
    "Laya",
    "MockAdapter",
    "REGISTRY",
    "SystemOneHTTP",
    "create_adapter",
    "endpoint",
]

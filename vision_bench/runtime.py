"""Hardware-aware device and CPU thread selection for local inference."""

from __future__ import annotations

import importlib

import psutil


def resolve_device(requested: str, has_local_torch_models: bool) -> str:
    """Resolve auto or validate a requested CUDA device for local Torch models."""
    explicit_cuda = requested == "cuda" or (
        requested.startswith("cuda:")
        and requested.split(":", 1)[1].isdigit()
    )
    if requested not in {"auto", "cpu"} and not explicit_cuda:
        raise ValueError(f"Dispositivo inválido: {requested}")
    if not has_local_torch_models:
        if requested.startswith("cuda"):
            raise ValueError("CUDA é controlado pelo servidor para adapters HTTP.")
        return "cpu"
    if requested == "cpu":
        return "cpu"

    try:
        torch = importlib.import_module("torch")
    except ImportError:
        if requested == "auto":
            return "cpu"
        raise RuntimeError("PyTorch não está instalado; instale o backend CUDA.") from None

    if not torch.cuda.is_available():
        if requested == "auto":
            return "cpu"
        raise RuntimeError(
            "CUDA foi solicitado, mas PyTorch não detectou uma GPU compatível. "
            "Confira o driver NVIDIA e instale a build CUDA do PyTorch."
        )

    device_index = (
        0
        if requested == "auto" or requested == "cuda"
        else int(requested.split(":", 1)[1])
    )
    if device_index >= torch.cuda.device_count():
        raise ValueError(f"GPU CUDA {device_index} não existe nesta máquina.")
    return f"cuda:{device_index}"


def resolve_thread_count(requested: str | int) -> int:
    """Use one Torch thread per physical CPU core when the count is automatic."""
    if requested == "auto":
        physical_cores = psutil.cpu_count(logical=False)
        logical_cores = psutil.cpu_count(logical=True)
        return max(1, physical_cores or logical_cores or 1)

    try:
        thread_count = int(requested)
    except (TypeError, ValueError):
        raise ValueError("Threads deve ser 'auto' ou um inteiro positivo.") from None
    if thread_count < 1:
        raise ValueError("Threads deve ser um inteiro positivo.")
    return thread_count

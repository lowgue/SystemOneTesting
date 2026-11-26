"""GPU memory probing and automatic vLLM batch sizing."""

from __future__ import annotations

import shutil
import subprocess

AUTO_CONCURRENCY_FALLBACK = 4
AUTO_CONCURRENCY_CAP = 64
MIN_SEQUENCE_MEMORY_MIB = 512
RESERVED_MEMORY_MIB = 512


def free_gpu_memory_mib(device: int = 0) -> int | None:
    """Return free VRAM in MiB for one NVIDIA device, or None if unavailable.

    The query uses ``nvidia-smi`` so the result reflects memory visible to the
    vLLM server process. A missing driver, a parsing failure, or a timed-out
    query returns None instead of guessing a hardware capacity.
    """
    executable = shutil.which("nvidia-smi")
    if not executable:
        return None
    command = [
        executable,
        f"--id={device}",
        "--query-gpu=memory.total,memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    lines = result.stdout.strip().splitlines()
    if not lines:
        return None
    try:
        total_mib, used_mib = (
            float(value.strip()) for value in lines[0].split(",", maxsplit=1)
        )
    except (TypeError, ValueError):
        return None
    return max(0, int(total_mib - used_mib))


def resolve_concurrency(
    requested: str | int,
    max_concurrency: int = AUTO_CONCURRENCY_CAP,
    device: int = 0,
) -> int:
    """Resolve an explicit batch size or size it from free GPU memory.

    ``auto`` divides the free VRAM, minus a small reserve, by a conservative
    per-sequence budget and caps the result at ``max_concurrency``. The cap
    bounds client threads and vLLM's ``--max-num-seqs``. When ``nvidia-smi``
    cannot report memory, the function falls back to a small safe batch.
    """
    if max_concurrency < 1:
        raise ValueError("O limite de concorrência deve ser positivo.")
    if requested == "auto":
        free_mib = free_gpu_memory_mib(device)
        if free_mib is None:
            return min(AUTO_CONCURRENCY_FALLBACK, max_concurrency)
        usable_mib = max(0, free_mib - RESERVED_MEMORY_MIB)
        return max(1, min(max_concurrency, usable_mib // MIN_SEQUENCE_MEMORY_MIB))
    try:
        concurrency = int(requested)
    except (TypeError, ValueError):
        raise ValueError("Concorrência deve ser 'auto' ou um inteiro positivo.") from None
    if concurrency < 1:
        raise ValueError("Concorrência deve ser 'auto' ou um inteiro positivo.")
    return concurrency

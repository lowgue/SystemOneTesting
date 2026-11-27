"""GPU memory probing and automatic vLLM batch sizing."""

from __future__ import annotations

import re
import shutil
import subprocess
import urllib.error
import urllib.request

AUTO_CONCURRENCY_FALLBACK = 4
AUTO_CONCURRENCY_CAP = 64
MIN_SEQUENCE_MEMORY_MIB = 512
RESERVED_MEMORY_MIB = 512
_METRIC_LINE = re.compile(
    r"^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)(?:\{[^}]*\})?\s+"
    r"(?P<value>[-+0-9.eE]+)",
    re.MULTILINE,
)


def gpu_memory_usage_mib(device: int = 0) -> tuple[int, int] | None:
    """Return total and used VRAM in MiB for one NVIDIA device, or None.

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
    return int(total_mib), max(0, int(used_mib))


def free_gpu_memory_mib(device: int = 0) -> int | None:
    """Return free VRAM in MiB for one NVIDIA device, or None if unavailable."""
    usage = gpu_memory_usage_mib(device)
    if usage is None:
        return None
    total_mib, used_mib = usage
    return max(0, total_mib - used_mib)


def vllm_cache_metrics(metrics_url: str) -> dict[str, float] | None:
    """Read vLLM Prometheus gauges that describe live KV-cache pressure.

    ``metrics_url`` normally points at ``/metrics`` of the running vLLM API
    server. The function returns the latest ``num_requests_running`` and
    ``kv_cache_usage_perc`` values, or None when the endpoint is unreachable or
    the gauges are absent. An idle server reports zero for both values, in
    which case caller-side server limits provide the starting capacity.
    """
    try:
        with urllib.request.urlopen(metrics_url, timeout=5) as response:
            text = response.read().decode("utf-8", errors="replace")
    except (OSError, urllib.error.URLError, TimeoutError):
        return None

    def latest(name: str) -> float | None:
        """Return the last sample for one gauge, ignoring label variations."""
        values = [
            float(match.group("value"))
            for match in _METRIC_LINE.finditer(text)
            if match.group("name") == name
        ]
        return values[-1] if values else None

    running = latest("vllm:num_requests_running")
    usage = latest("vllm:kv_cache_usage_perc")
    if running is None and usage is None:
        return None
    return dict(
        num_requests_running=running if running is not None else 0.0,
        kv_cache_usage_perc=usage if usage is not None else 0.0,
    )


def resolve_concurrency(
    requested: str | int,
    max_concurrency: int = AUTO_CONCURRENCY_CAP,
    device: int = 0,
    *,
    metrics_url: str | None = None,
    server_max_num_seqs: int | None = None,
) -> int:
    """Resolve an explicit batch size or size it from live vLLM/GPU capacity.

    ``auto`` first honors the server's configured ``--max-num-seqs`` when it is
    supplied, which is the reliable cap for an already warmed external server.
    If the vLLM ``/metrics`` endpoint reports active KV-cache usage, the
    available capacity is derived from ``running / usage`` with one slot kept
    in reserve. Without vLLM metrics, the function keeps the legacy free-VRAM
    estimate; without ``nvidia-smi`` it falls back to a small safe batch.
    """
    if max_concurrency < 1:
        raise ValueError("O limite de concorrência deve ser positivo.")
    if server_max_num_seqs is not None and server_max_num_seqs < 1:
        raise ValueError("O limite de sequências do servidor deve ser positivo.")

    if requested == "auto":
        cap = max_concurrency
        if server_max_num_seqs is not None:
            cap = min(cap, server_max_num_seqs)
        if metrics_url:
            metrics = vllm_cache_metrics(metrics_url)
            if metrics:
                running = int(metrics["num_requests_running"])
                usage = metrics["kv_cache_usage_perc"]
                if running > 0 and usage > 0:
                    capacity = running / usage
                    available = int(capacity) - running - 1
                    return max(1, min(cap, available))
        if server_max_num_seqs is not None:
            return max(1, cap)
        free_mib = free_gpu_memory_mib(device)
        if free_mib is None:
            return min(AUTO_CONCURRENCY_FALLBACK, cap)
        usable_mib = max(0, free_mib - RESERVED_MEMORY_MIB)
        return max(1, min(cap, usable_mib // MIN_SEQUENCE_MEMORY_MIB))

    try:
        concurrency = int(requested)
    except (TypeError, ValueError):
        raise ValueError("Concorrência deve ser 'auto' ou um inteiro positivo.") from None
    if concurrency < 1:
        raise ValueError("Concorrência deve ser 'auto' ou um inteiro positivo.")
    return concurrency

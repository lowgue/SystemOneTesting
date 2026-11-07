"""Download pinned Clef weights and manage a compatible local server process."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import psutil

from .core import load_json, sha256, write_json


DEFAULT_OLLAMA_SERVER = Path("/usr/local/lib/ollama/llama-server")
LOCAL_SERVER_HOSTS = {"127.0.0.1", "localhost", "::1"}


def serve(args: Any) -> dict[str, Any]:
    """Launch the configured server and record its process and pinned weights.

    Dry-run mode validates settings and returns the command without downloading
    model files or starting a process. Normal mode blocks until the server exits
    or the user interrupts it.
    """
    weights = load_json(args.weights)
    executable = _resolve_server_executable(args.llama_server)
    _validate_server_settings(args)
    command = _build_server_command(
        executable,
        args.host,
        args.port,
        args.context,
        args.threads,
        args.gpu_layers,
        model="<model.gguf>",
        projector="<projector.gguf>",
    )
    if args.dry_run:
        return {
            "command": command,
            "weights": weights,
            "note": "No weights downloaded and no server started. Requires Clef/System One support.",
        }

    hf_executable = _resolve_hf_executable()
    weight_files = _download_weights(hf_executable, weights)
    model_path, projector_path = _model_paths(weight_files)
    command = _build_server_command(
        executable,
        args.host,
        args.port,
        args.context,
        args.threads,
        args.gpu_layers,
        model=model_path,
        projector=projector_path,
    )
    runtime_version = _server_version(executable)
    metadata_path = Path(args.metadata).resolve()
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(command)

    try:
        _write_process_metadata(
            process,
            metadata_path,
            args,
            weights,
            weight_files,
            command,
            runtime_version,
        )
        print(
            f"Server PID: {process.pid}. In the benchmark shell, "
            f"set CLEF_SERVER_METADATA={metadata_path}",
            flush=True,
        )
        exit_code = process.wait()
        if exit_code:
            raise RuntimeError(
                f"llama-server exited with code {exit_code}; "
                "check Clef support and memory"
            )
    except KeyboardInterrupt:
        _stop_process(process)
    finally:
        if process.poll() is None:
            _stop_process(process)

    return {"status": "server-stopped", "metadata": str(metadata_path)}


def _resolve_server_executable(configured_path: str | None) -> str:
    """Find a server executable from the CLI, environment, or known local path."""
    candidate = configured_path or os.environ.get("LLAMA_SERVER") or "llama-server"
    executable = shutil.which(candidate) or candidate
    if not Path(executable).is_file() and not configured_path:
        if DEFAULT_OLLAMA_SERVER.is_file():
            executable = str(DEFAULT_OLLAMA_SERVER)
    if not Path(executable).is_file():
        raise ValueError(
            "Set --llama-server or LLAMA_SERVER to a Clef-capable executable"
        )
    return str(Path(executable).resolve())


def _validate_server_settings(args: Any) -> None:
    """Reject invalid ports and resource settings before downloading weights."""
    if args.context < 1 or args.threads < 1 or not 1 <= args.port <= 65_535:
        raise ValueError("Invalid server context, thread count or port")


def _build_server_command(
    executable: str,
    host: str,
    port: int,
    context: int,
    threads: int,
    gpu_layers: int,
    model: str,
    projector: str,
) -> list[str]:
    """Build a one-slot llama-server command for the pinned Clef model."""
    return [
        executable,
        "--model",
        model,
        "--mmproj",
        projector,
        "--host",
        host,
        "--port",
        str(port),
        "-c",
        str(context),
        "-np",
        "1",
        "-t",
        str(threads),
        "-tb",
        str(threads),
        "-ngl",
        str(gpu_layers),
    ]


def _resolve_hf_executable() -> str:
    """Find the Hugging Face CLI beside Python or on the active PATH."""
    executable = shutil.which("hf")
    adjacent = Path(sys.executable).parent / ("hf.exe" if os.name == "nt" else "hf")
    if adjacent.is_file():
        executable = str(adjacent)
    if not executable:
        raise ValueError("Install requirements/models.txt to provide the hf CLI")
    return executable


def _download_weights(
    hf_executable: str,
    weights: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Download every pinned file and verify its SHA-256 before launch."""
    files = {}
    for info in weights["files"]:
        process = subprocess.run(
            [
                hf_executable,
                "download",
                weights["repository"],
                info["filename"],
                "--revision",
                weights["revision"],
                "--quiet",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        downloaded_path = Path(process.stdout.strip().splitlines()[-1])
        observed_hash = sha256(downloaded_path)
        if observed_hash != info["sha256"]:
            raise ValueError(f"Weight checksum mismatch: {info['filename']}")
        files[info["filename"]] = {
            "path": str(downloaded_path),
            "sha256": observed_hash,
            "size": downloaded_path.stat().st_size,
        }
    return files


def _model_paths(files: dict[str, dict[str, Any]]) -> tuple[str, str]:
    """Return the model and projector paths from downloaded file metadata."""
    model = next(value["path"] for name, value in files.items() if not name.startswith("mmproj"))
    projector = next(value["path"] for name, value in files.items() if name.startswith("mmproj"))
    return model, projector


def _server_version(executable: str) -> str:
    """Read the server's reported version for reproducibility metadata."""
    result = subprocess.run(
        [executable, "--version"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    return (result.stdout + result.stderr).strip()


def _write_process_metadata(
    process: subprocess.Popen[Any],
    metadata_path: Path,
    args: Any,
    weights: dict[str, Any],
    files: dict[str, dict[str, Any]],
    command: list[str],
    runtime_version: str,
) -> None:
    """Persist process identity, weight hashes, and the effective launch command."""
    process_created = psutil.Process(process.pid).create_time()
    metadata = {
        "pid": process.pid,
        "process_created_s": process_created,
        "revision": weights["revision"],
        "quantization": "Q4_K_M",
        "weights": files,
        "command": command,
        "runtime_version": runtime_version,
        "endpoint": f"http://{args.host}:{args.port}/v1/systemone",
    }
    write_json(metadata_path, metadata)


def _stop_process(process: subprocess.Popen[Any]) -> None:
    """Ask the server to stop and force termination after a short grace period."""
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()

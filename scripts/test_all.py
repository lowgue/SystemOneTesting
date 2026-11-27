"""Run unit tests and batch GPU benchmarks for every configured vLLM model."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import venv
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv"
PYTHON = VENV / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
BUSY_GPU_THRESHOLD_MIB = 1024
sys.path.insert(0, str(ROOT))

from vision_bench.gpu import (  # noqa: E402
    gpu_memory_usage_mib,
    resolve_concurrency,
)


def run(command: list[str]) -> None:
    """Run a command from the repository root and stop when it fails."""
    subprocess.run(command, cwd=ROOT, check=True)


def check_gpu_availability(allow_busy: bool = False) -> None:
    """Abort when another service already holds meaningful GPU memory.

    Raising early avoids four consecutive vLLM startup failures when a service
    such as Ollama is occupying VRAM. ``allow_busy`` keeps the old behavior for
    operators who intentionally share the GPU.
    """
    usage = gpu_memory_usage_mib(0)
    if usage is None or allow_busy or usage[1] <= BUSY_GPU_THRESHOLD_MIB:
        return
    print(
        f"ERRO: a GPU já está com {usage[1]} MiB em uso. "
        "Encerre outros serviços antes do benchmark, por exemplo com "
        "`sudo systemctl stop ollama` ou `ollama stop <modelo>`. "
        "Para verificar: "
        "`nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv`. "
        "Se quiser continuar mesmo assim, use --allow-busy-gpu.",
        file=sys.stderr,
        flush=True,
    )
    raise SystemExit(2)


def find_free_port() -> int:
    """Reserve an available local TCP port number for a temporary vLLM server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def metrics_url_from_endpoint(url: str) -> str:
    """Derive the vLLM Prometheus ``/metrics`` URL from a served endpoint."""
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"Endpoint vLLM inválido para métricas: {url}")
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, "/metrics", "", ""))


def load_vllm_manifest(path: Path) -> dict:
    """Validate the manifest that describes models served by temporary vLLM processes."""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError(f"Versão de schema não suportada em {path}")
    models = manifest.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError(f"Nenhum modelo definido em {path}")
    identifiers = set()
    for entry in models:
        for field in (
            "id",
            "adapter",
            "model",
            "served_model",
            "route",
            "contract",
            "url_env",
        ):
            if not entry.get(field):
                raise ValueError(f"Modelo vLLM sem {field}: {entry}")
        if entry["id"] in identifiers:
            raise ValueError(f"ID de modelo vLLM duplicado: {entry['id']}")
        if entry["route"] not in {"systemone", "chat/completions"}:
            raise ValueError(f"Rota vLLM desconhecida: {entry['route']}")
        identifiers.add(entry["id"])
    return manifest


def start_vllm_server(
    executable: Path,
    entry: dict,
    manifest: dict,
    concurrency: int,
    port: int,
    log_stream: Any,
) -> subprocess.Popen:
    """Start one vLLM model server and wait for its model API to answer."""
    command = [
        str(executable),
        "serve",
        entry["served_model"],
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--max-model-len",
        str(manifest.get("max_model_len", 4096)),
        "--gpu-memory-utilization",
        str(manifest.get("gpu_memory_utilization", 0.9)),
        "--max-num-seqs",
        str(concurrency),
        "--trust-remote-code",
    ]
    if manifest.get("kv_cache_dtype"):
        command.extend(["--kv-cache-dtype", str(manifest["kv_cache_dtype"])])
    if manifest.get("enable_prefix_caching"):
        command.append("--enable-prefix-caching")
    if manifest.get("per_request_metrics"):
        command.append("--enable-per-request-metrics")
    if entry.get("logprobs_mode"):
        command.extend(["--logprobs-mode", str(entry["logprobs_mode"])])
    if entry.get("revision"):
        command.extend(["--revision", entry["revision"]])
    server = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=log_stream,
        stderr=log_stream,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 1800
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError("vLLM saiu antes da API ficar pronta")
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/v1/models", timeout=3
                ):
                    return server
            except (urllib.error.URLError, TimeoutError):
                time.sleep(2)
        raise TimeoutError("Tempo esgotado esperando o servidor vLLM")
    except BaseException:
        stop_server(server)
        raise


def _signal_process_group(process_id: int, signal_number: int) -> None:
    """Send a signal to a process group, ignoring groups that already exited."""
    try:
        os.killpg(process_id, signal_number)
    except (ProcessLookupError, PermissionError):
        pass


def _process_group_exists(process_id: int) -> bool:
    """Report whether a process group created by this script still exists."""
    try:
        os.killpg(process_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def stop_process_tree(process: subprocess.Popen, timeout: int = 30) -> None:
    """Terminate a child process and every worker it spawned in its own group."""
    _signal_process_group(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        pass
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and _process_group_exists(process.pid):
        time.sleep(0.2)
    if _process_group_exists(process.pid):
        _signal_process_group(process.pid, signal.SIGKILL)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def stop_server(server: subprocess.Popen) -> None:
    """Stop the vLLM server and its EngineCore workers after a benchmark."""
    stop_process_tree(server)


def start_gpu_monitor(csv_path: Path) -> subprocess.Popen | None:
    """Start nvidia-smi telemetry writing one CSV row per second."""
    if not shutil.which("nvidia-smi"):
        return None
    stream = csv_path.open("w", encoding="utf-8")
    stream.write(
        "timestamp,utilization_gpu_pct,memory_used_MiB,memory_total_MiB,power_W\n"
    )
    stream.flush()
    return subprocess.Popen(
        [
            "nvidia-smi",
            "--id=0",
            "--query-gpu=timestamp,utilization.gpu,memory.used,"
            "memory.total,power.draw",
            "--format=csv,noheader,nounits",
            "--loop=1",
        ],
        stdout=stream,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def stop_gpu_monitor(monitor: subprocess.Popen | None) -> None:
    """Stop an nvidia-smi monitor and close its CSV output."""
    if monitor is None:
        return
    stop_process_tree(monitor, timeout=10)
    if monitor.stdout is not None:
        monitor.stdout.close()


def write_model_config(path: Path, entry: dict) -> None:
    """Write the temporary single-model configuration consumed by the runner."""
    serving_fields = {"served_model", "route", "contract"}
    spec = {
        key: value for key, value in entry.items() if key not in serving_fields
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"schema_version": 1, "models": [spec]}, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )


def benchmark_model(
    args: argparse.Namespace,
    entry: dict,
    manifest: dict,
    concurrency: int,
    run_root: Path,
    external: bool = False,
) -> Path:
    """Benchmark one model, using a warmed external server when requested."""
    config_path = run_root / "configs" / f"{entry['id']}.json"
    write_model_config(config_path, entry)
    log_path = run_root / f"{entry['id']}-vllm.log"
    with log_path.open("w", encoding="utf-8") as log_stream:
        if external:
            print(
                f"Reutilizando servidor vLLM externo: {os.environ.get(entry['url_env'], '')}",
                file=log_stream,
                flush=True,
            )
            server = None
        else:
            port = find_free_port()
            os.environ[entry["url_env"]] = (
                f"http://127.0.0.1:{port}/v1/{entry['route']}"
            )
            server = start_vllm_server(
                args.vllm_executable, entry, manifest, concurrency, port, log_stream
            )
        monitor = None
        try:
            monitor = start_gpu_monitor(
                run_root / f"{entry['id']}-gpu.csv"
            )
            output = run_root / entry["id"]
            run(
                [
                    str(PYTHON),
                    "-m",
                    "vision_bench",
                    "run",
                    "--config",
                    str(config_path),
                    "--contract",
                    str(
                        ROOT
                        / "benchmarks"
                        / "contracts"
                        / entry["contract"]
                    ),
                    "--models",
                    entry["id"],
                    "--device",
                    "cpu",
                    "--threads",
                    "auto",
                    "--repetitions",
                    str(args.repetitions),
                    "--concurrency",
                    str(concurrency),
                    "--output",
                    str(output),
                ]
            )
            status = json.loads((output / "status.json").read_text(encoding="utf-8"))
            if status.get("status") != "complete":
                raise RuntimeError(f"Execução incompleta: {output}")
            return output
        finally:
            stop_gpu_monitor(monitor)
            if server is not None:
                stop_server(server)


def main() -> int:
    """Install dependencies, run tests, and batch benchmark every selected model."""
    parser = argparse.ArgumentParser(
        description="Prepara o ambiente, executa os testes e gera gráficos de benchmark."
    )
    parser.add_argument(
        "--no-install",
        action="store_true",
        help="Usa o ambiente .venv existente sem instalar ou atualizar dependências.",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Roda todos os modelos do manifesto vLLM com requisições em batch.",
    )
    parser.add_argument(
        "--include-local-benchmarks",
        action="store_true",
        help="Também roda BLIP, Laya e D1 em processo como baseline serial.",
    )
    parser.add_argument(
        "--torch-index-url",
        default="https://download.pytorch.org/whl/cu128",
        help="Índice do PyTorch CUDA para --include-local-benchmarks (padrão: cu128).",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=3,
        help="Repetições por imagem no benchmark GPU (padrão: 3).",
    )
    parser.add_argument(
        "--concurrency",
        default="auto",
        help="Imagens simultâneas no vLLM: inteiro positivo ou auto (padrão: auto).",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=64,
        help="Teto da concorrência automática calculada pela GPU (padrão: 64).",
    )
    parser.add_argument(
        "--allow-busy-gpu",
        action="store_true",
        help="Continua mesmo se outro processo já ocupar memória da GPU.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="IDs do manifesto vLLM; sem este argumento, roda todos os modelos.",
    )
    parser.add_argument(
        "--vllm-models-config",
        type=Path,
        default=ROOT / "benchmarks" / "configs" / "vllm-models.json",
        help="Manifesto com os modelos servidos pelo vLLM.",
    )
    parser.add_argument(
        "--vllm-executable",
        type=Path,
        default=Path.home() / "venvs" / "systemone-vllm" / "bin" / "vllm",
        help="Executável vLLM instalado no ambiente dedicado.",
    )
    parser.add_argument(
        "--external-model",
        help="ID do manifesto já servido por um vLLM externo aquecido; não inicia nem encerra servidor.",
    )
    parser.add_argument(
        "--external-url",
        help="URL do servidor externo; preenche a variável url_env do modelo selecionado.",
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Roda só a suíte unitária; não inicia inferências.",
    )
    args = parser.parse_args()

    if args.repetitions < 1:
        parser.error("--repetitions precisa ser positivo.")
    if args.max_concurrency < 1:
        parser.error("--max-concurrency precisa ser positivo.")
    if args.external_model and args.models:
        parser.error("--external-model não pode ser combinado com --models.")
    if sys.version_info < (3, 12):
        parser.error("Python 3.12 ou superior é necessário.")
    manifest = load_vllm_manifest(args.vllm_models_config)
    known_models = {entry["id"] for entry in manifest["models"]}
    selected = set(args.models or [])
    unknown_models = selected - known_models
    if unknown_models:
        parser.error(
            f"Modelos desconhecidos: {', '.join(sorted(unknown_models))}. "
            f"Disponíveis: {', '.join(sorted(known_models))}"
        )

    external = bool(args.external_model)
    metrics_url = None
    server_max_num_seqs = manifest.get("max_num_seqs")
    if external:
        if args.external_model not in known_models:
            parser.error(
                f"Modelo externo desconhecido: {args.external_model}. "
                f"Disponíveis: {', '.join(sorted(known_models))}"
            )
        selected = {args.external_model}
        external_entry = next(
            entry for entry in manifest["models"] if entry["id"] == args.external_model
        )
        if args.external_url:
            os.environ[external_entry["url_env"]] = args.external_url
        endpoint = os.environ.get(external_entry["url_env"], "")
        if not endpoint:
            parser.error(
                f"Defina {external_entry['url_env']} ou use --external-url para o servidor externo."
            )
        metrics_url = metrics_url_from_endpoint(endpoint)
    elif args.gpu and not args.skip_benchmark:
        check_gpu_availability(args.allow_busy_gpu)

    try:
        concurrency = resolve_concurrency(
            args.concurrency,
            args.max_concurrency,
            metrics_url=metrics_url,
            server_max_num_seqs=server_max_num_seqs,
        )
    except ValueError as error:
        parser.error(str(error))
    if (
        args.gpu
        and not args.skip_benchmark
        and not external
        and not args.vllm_executable.is_file()
    ):
        parser.error(
            f"vLLM não encontrado em {args.vllm_executable}; "
            "instale-o no ambiente dedicado ou informe --vllm-executable."
        )
    venv_usable = PYTHON.is_file()
    if venv_usable:
        try:
            subprocess.run(
                [str(PYTHON), "-c", "import encodings"],
                cwd=ROOT,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.CalledProcessError):
            venv_usable = False
    if args.no_install and not venv_usable:
        parser.error(
            "O ambiente .venv está ausente ou quebrado; remova .venv e rode "
            "novamente sem --no-install para recriá-lo."
        )
    if not venv_usable:
        shutil.rmtree(VENV, ignore_errors=True)
        venv.EnvBuilder(with_pip=True).create(VENV)
    if not args.no_install:
        if args.gpu and args.include_local_benchmarks:
            run(["git", "submodule", "update", "--init", "--recursive"])
            run(
                [
                    str(PYTHON),
                    "-m",
                    "pip",
                    "install",
                    "torch",
                    "torchvision",
                    "--index-url",
                    args.torch_index_url,
                ]
            )
            run(
                [
                    str(PYTHON),
                    "-m",
                    "pip",
                    "install",
                    "-r",
                    str(ROOT / "requirements" / "test.txt"),
                    "-r",
                    str(ROOT / "requirements" / "models.txt"),
                    "-e",
                    str(ROOT),
                    "-e",
                    str(ROOT / "models" / "laya" / "laya_vision"),
                ]
            )
        else:
            run(
                [
                    str(PYTHON),
                    "-m",
                    "pip",
                    "install",
                    "-r",
                    str(ROOT / "requirements" / "test.txt"),
                    "-e",
                    str(ROOT),
                ]
            )
    if args.gpu and args.include_local_benchmarks:
        run(
            [
                str(PYTHON),
                "-c",
                "import torch; assert torch.cuda.is_available(), "
                "'PyTorch não encontrou CUDA; confira driver e instalação.'; "
                "print('GPU:', torch.cuda.get_device_name(0), "
                "'CUDA runtime:', torch.version.cuda)",
            ]
        )
    elif args.gpu and not args.skip_benchmark and not external:
        vllm_python = args.vllm_executable.parent / "python"
        run(
            [
                str(vllm_python),
                "-c",
                "import torch; assert torch.cuda.is_available(), "
                "'PyTorch do vLLM não encontrou CUDA; confira driver e instalação.'; "
                "print('GPU:', torch.cuda.get_device_name(0), "
                "'CUDA runtime:', torch.version.cuda)",
            ]
        )
    run([str(PYTHON), "-m", "unittest", "discover", "-s", "tests", "-v"])
    if args.gpu and not args.skip_benchmark:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_root = ROOT / ".artifacts" / "runs" / f"gpu-{timestamp}"
        run_root.mkdir(parents=True, exist_ok=True)
        print(
            f"Batch automático: {concurrency} imagens simultâneas "
            f"(limite configurado: {args.max_concurrency}; "
            f"servidor: {'externo aquecido' if external else 'gerenciado'}).",
            flush=True,
        )
        completed_runs = []
        contract_path = (
            ROOT / "benchmarks" / "contracts" / "description_ollama_v1.json"
        )
        if args.include_local_benchmarks:
            local_output = run_root / "local-models"
            try:
                run(
                    [
                        str(PYTHON),
                        "-m",
                        "vision_bench",
                        "run",
                        "--config",
                        str(ROOT / "benchmarks" / "configs" / "local.json"),
                        "--contract",
                        str(contract_path),
                        "--models",
                        "blip-vqa",
                        "laya-vision-head512",
                        "d1-omni-600M",
                        "--device",
                        "auto",
                        "--threads",
                        "auto",
                        "--repetitions",
                        str(args.repetitions),
                        "--output",
                        str(local_output),
                    ]
                )
                completed_runs.append(local_output)
            except subprocess.CalledProcessError as error:
                print(f"Baseline local falhou: {error}", file=sys.stderr)
        for entry in manifest["models"]:
            if selected and entry["id"] not in selected:
                continue
            print(
                f"{'Reutilizando vLLM externo' if external else 'Iniciando vLLM'} "
                f"para {entry['id']}...",
                flush=True,
            )
            try:
                completed_runs.append(
                    benchmark_model(
                        args, entry, manifest, concurrency, run_root, external
                    )
                )
            except (
                OSError,
                RuntimeError,
                subprocess.SubprocessError,
                TimeoutError,
                ValueError,
                KeyError,
            ) as error:
                print(
                    f"Benchmark {entry['id']} falhou: {error}; "
                    f"detalhes em {run_root / (entry['id'] + '-vllm.log')}",
                    file=sys.stderr,
                )
        if completed_runs:
            run(
                [
                    str(PYTHON),
                    str(ROOT / "scripts" / "report_runs.py"),
                    *(str(path) for path in completed_runs),
                ]
            )
        else:
            print("Nenhuma execução GPU terminou; não há gráficos para gerar.")
    else:
        run([str(PYTHON), str(ROOT / "scripts" / "report_runs.py")])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run unit tests and GPU benchmarks, including batched Clef through vLLM."""

import argparse
from datetime import datetime
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import venv


ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv"
PYTHON = VENV / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def run(command: list[str]) -> None:
    """Run a command from the repository root and stop when it fails."""
    subprocess.run(command, cwd=ROOT, check=True)


def start_clef_server(
    executable: Path, model: str, concurrency: int
) -> subprocess.Popen:
    """Start the local vLLM Clef server and wait for its model API."""
    command = [
        str(executable),
        "serve",
        model,
        "--max-model-len",
        "4096",
        "--gpu-memory-utilization",
        "0.90",
        "--max-num-seqs",
        str(concurrency),
    ]
    try:
        with socket.create_connection(("127.0.0.1", 8000), timeout=0.2):
            raise RuntimeError(
                "A server already uses port 8000; stop it before test_all starts vLLM"
            )
    except OSError:
        pass
    server = subprocess.Popen(command, cwd=ROOT)
    try:
        deadline = time.monotonic() + 1800
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError("vLLM exited before the Clef API became ready")
            try:
                with urllib.request.urlopen(
                    "http://127.0.0.1:8000/v1/models", timeout=3
                ):
                    return server
            except (urllib.error.URLError, TimeoutError):
                time.sleep(2)
        raise TimeoutError("Timed out waiting for the vLLM Clef server")
    except BaseException:
        stop_server(server)
        raise


def stop_server(server: subprocess.Popen) -> None:
    """Stop the vLLM server started for the benchmark and wait for cleanup."""
    if server.poll() is not None:
        return
    server.terminate()
    try:
        server.wait(timeout=30)
    except subprocess.TimeoutExpired:
        server.kill()
        server.wait()


def main() -> int:
    """Install dependencies, run tests, and benchmark local and vLLM models."""
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
        help="Instala PyTorch CUDA, roda os modelos locais na GPU e gera gráficos.",
    )
    parser.add_argument(
        "--torch-index-url",
        default="https://download.pytorch.org/whl/cu128",
        help="Índice oficial do PyTorch para CUDA (padrão: cu128).",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=3,
        help="Repetições por imagem no benchmark GPU (padrão: 3).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Imagens simultâneas enviadas ao vLLM para batching (padrão: 4).",
    )
    parser.add_argument(
        "--vllm-executable",
        type=Path,
        default=Path.home() / "venvs" / "systemone-vllm" / "bin" / "vllm",
        help="Executável vLLM instalado no ambiente dedicado.",
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Com --gpu, prepara CUDA e roda testes unitários sem inferência.",
    )
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions precisa ser positivo.")
    if args.concurrency < 1:
        parser.error("--concurrency precisa ser positivo.")
    if sys.version_info < (3, 12):
        parser.error("Python 3.12 ou superior é necessário.")
    if (
        args.gpu
        and not args.skip_benchmark
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
        if args.gpu:
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
    if args.gpu:
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
    run([str(PYTHON), "-m", "unittest", "discover", "-s", "tests", "-v"])
    if args.gpu and not args.skip_benchmark:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_path = ROOT / ".artifacts" / "runs" / f"gpu-{timestamp}"
        run(
            [
                str(PYTHON),
                "-m",
                "vision_bench",
                "run",
                "--config",
                str(ROOT / "benchmarks" / "configs" / "local.json"),
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
                str(run_path),
            ]
        )
        run([str(PYTHON), str(ROOT / "scripts" / "report_runs.py"), str(run_path)])
        server = start_clef_server(
            args.vllm_executable, "Cloudflare/clef-flash", args.concurrency
        )
        os.environ["VLLM_SYSTEMONE_URL"] = "http://127.0.0.1:8000/v1/systemone"
        gpu_log = None
        monitor = None
        try:
            clef_run = run_path / "clef-vllm"
            if shutil.which("nvidia-smi"):
                gpu_log = (run_path / "clef-gpu.csv").open("w", encoding="utf-8")
                gpu_log.write(
                    "timestamp,utilization_gpu_pct,memory_used_MiB,"
                    "memory_total_MiB,power_W\n"
                )
                monitor = subprocess.Popen(
                    [
                        "nvidia-smi",
                        "--id=0",
                        "--query-gpu=timestamp,utilization.gpu,memory.used,"
                        "memory.total,power.draw",
                        "--format=csv,noheader,nounits",
                        "--loop=1",
                    ],
                    stdout=gpu_log,
                    stderr=subprocess.DEVNULL,
                )
            run(
                [
                    str(PYTHON),
                    "-m",
                    "vision_bench",
                    "run",
                    "--config",
                    str(ROOT / "benchmarks" / "configs" / "vllm-systemone.json"),
                    "--contract",
                    str(
                        ROOT
                        / "benchmarks"
                        / "contracts"
                        / "description_ollama_v1.json"
                    ),
                    "--models",
                    "clef-flash-vllm",
                    "--device",
                    "cpu",
                    "--threads",
                    "auto",
                    "--repetitions",
                    str(args.repetitions),
                    "--concurrency",
                    str(args.concurrency),
                    "--output",
                    str(clef_run),
                ]
            )
            run(
                [
                    str(PYTHON),
                    str(ROOT / "scripts" / "report_runs.py"),
                    str(clef_run),
                ]
            )
        finally:
            if monitor is not None:
                monitor.terminate()
                try:
                    monitor.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    monitor.kill()
                    monitor.wait()
            if gpu_log is not None:
                gpu_log.close()
            stop_server(server)
    else:
        run([str(PYTHON), str(ROOT / "scripts" / "report_runs.py")])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

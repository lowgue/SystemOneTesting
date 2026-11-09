"""Set up the lightweight test environment, run every test, and save available charts."""

import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import venv


ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv"
PYTHON = VENV / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def run(command: list[str]) -> None:
    """Run a command from the repository root and stop when it fails."""
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    """Install dependencies, run tests, and optionally benchmark local CUDA models."""
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
        "--skip-benchmark",
        action="store_true",
        help="Com --gpu, prepara CUDA e roda testes unitários sem inferência.",
    )
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions precisa ser positivo.")
    if sys.version_info < (3, 12):
        parser.error("Python 3.12 ou superior é necessário.")
    if args.no_install and not PYTHON.is_file():
        parser.error("O ambiente .venv não existe; rode sem --no-install para criá-lo.")
    if not PYTHON.is_file():
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
    else:
        run([str(PYTHON), str(ROOT / "scripts" / "report_runs.py")])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

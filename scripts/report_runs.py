"""Create charts in graficos/relatorios from completed benchmark runs."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = ROOT / ".artifacts" / "runs"
REPORTS_ROOT = ROOT / "graficos" / "relatorios"


def completed_runs(run_paths: list[Path]) -> list[Path]:
    """Return runs with complete status, rejecting explicitly selected incomplete runs."""
    complete = []
    for run_path in run_paths:
        status_path = run_path / "status.json"
        if not status_path.is_file():
            raise FileNotFoundError(f"Não encontrei status.json em {run_path}")
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") != "complete":
            raise ValueError(f"A execução ainda não terminou: {run_path}")
        complete.append(run_path)
    return complete


def discover_completed_runs() -> list[Path]:
    """Find completed benchmark runs saved under .artifacts/runs."""
    candidates = sorted(RUNS_ROOT.glob("*/status.json"))
    complete = []
    for status_path in candidates:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") == "complete":
            complete.append(status_path.parent)
    return complete


def create_report(run_paths: list[Path], python: str = sys.executable) -> Path:
    """Create performance and dataset charts under graficos/relatorios."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output = REPORTS_ROOT / f"execucao-{timestamp}"
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    command = [
        python,
        "-m",
        "vision_bench",
        "report",
        *(str(path) for path in run_paths),
        "--output",
        str(output),
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    from vision_bench.reporting.dataset_metrics import generate

    for run_path in run_paths:
        evaluation_output = output / "avaliacoes" / run_path.name
        evaluation_output.mkdir(parents=True, exist_ok=True)
        generate(run_path, evaluation_output)
    return output


def main() -> int:
    """Generate charts for provided runs or discover completed local runs."""
    parser = argparse.ArgumentParser(
        description="Gera relatórios e gráficos em graficos/relatorios/."
    )
    parser.add_argument(
        "runs",
        nargs="*",
        type=Path,
        help="Pastas de execução; sem argumentos, procura em .artifacts/runs/.",
    )
    args = parser.parse_args()
    run_paths = (
        completed_runs([path.resolve() for path in args.runs])
        if args.runs
        else discover_completed_runs()
    )
    if not run_paths:
        print("Nenhuma execução completa encontrada; ainda não há gráficos para gerar.")
        return 0
    report_path = create_report(run_paths)
    print(f"Relatórios e gráficos salvos em: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

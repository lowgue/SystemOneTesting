"""Reports support any number of model IDs and keep machines/runs separate."""

import csv
import json
from pathlib import Path
import statistics
from ..core import load_json, write_json
from .charts.style import model_color


def generate(runs, output):
    """Create separate comparison reports for completed runs without pooling machines."""
    import numpy as np

    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    summary = []
    lines = [
        "# Comparação portátil de modelos visuais",
        "",
        "Cada execução e máquina é apresentada separadamente. Não há ranking agregado entre hardwares. A validade de esquema não é acurácia; rótulos exploratórios não comprovam qualidade final. CPU/RSS de um cliente HTTP não medem o servidor remoto.",
        "",
    ]
    plot_inputs = []
    for run in map(Path, runs):
        meta = load_json(run / "run.json")
        status = load_json(run / "status.json")
        frozen = load_json(run / "frozen_config.json")
        if status["status"] != "complete":
            raise ValueError(f"Run incomplete: {run}")
        rows = [
            json.loads(s) for s in (run / "predictions.jsonl").read_text().splitlines()
        ]
        if len(rows) != status["planned"] or len(
            {r["request_key"] for r in rows}
        ) != len(rows):
            raise ValueError("Missing or duplicate records")
        label = f'{meta["machine_id"]} / {run.name}'
        lines += [
            f"## {label}",
            "",
            f'Device: `{frozen["device"]}`; threads: {frozen["threads"]}; rótulos: `{meta["label_status"]}`.',
            f'Contrato: `{meta["contract_hash"][:12]}`; dataset: `{meta["dataset_hash"][:12]}`. CPU: {meta["hardware"]["cpu"]}.',
            "",
        ]
        if meta["simulated"]:
            lines += [
                "**SIMULAÇÃO: este run contém adapter mock e serve apenas para validar infraestrutura.**",
                "",
            ]
        lines += [
            "| Modelo | Eventos / requisições | p50 / p95 (s) | CPU média (s) | RSS pico amostrado (MiB) | Escopo CPU/RSS | Esquema válido (1ª repetição) | Positivos completos corretos* | Erros |",
            "|---|---:|---:|---:|---:|---|---:|---:|---:|",
        ]
        for spec in frozen["models"]:
            model = spec["id"]
            rs = [r for r in rows if r["model"] == model]
            first = [r for r in rs if r["block"] == 0]
            ok = [r for r in rs if not r["error"]]
            times = [r["elapsed_ms"] / 1000 for r in ok]
            positive = [
                r for r in first if r.get("expected", {}).get("presence") == "yes"
            ]
            correct = sum(r.get("exploratory_exact", False) for r in positive)
            med = float(np.median(times)) if times else None
            p95 = float(np.quantile(times, 0.95)) if times else None
            scopes = sorted({r["resource_scope"] for r in rs})
            row = dict(
                run=run.name,
                machine=meta["machine_id"],
                model=model,
                device=frozen["device"],
                events=len(first),
                requests=len(rs),
                p50_s=med,
                p95_s=p95,
                mean_cpu_s=statistics.mean(r["cpu_s"] for r in rs),
                peak_rss_mib=max(r["peak_rss_bytes"] for r in rs) / 2**20,
                resource_scope=",".join(scopes),
                first_valid=sum(r["valid"] for r in first),
                positive_n=len(positive),
                exploratory_positive_exact=correct,
                errors=sum(bool(r["error"]) for r in rs),
            )
            summary.append(row)
            timing = f"{med:.3f} / {p95:.3f}" if times else "sem sucesso"
            lines.append(
                f'| {model} | {len(first)} / {len(rs)} | {timing} | {row["mean_cpu_s"]:.3f} | {row["peak_rss_mib"]:.0f} | {row["resource_scope"]} | {row["first_valid"]}/{len(first)} | {correct}/{len(positive)} | {row["errors"]} |'
            )
        lines += [
            "",
            "*Métrica exploratória, somente onde existe gabarito. Todos os positivos, inclusive falhas e rejeições, permanecem no denominador. As repetições não aumentam o número de imagens independentes.",
            "",
        ]
        plot_inputs.append(
            dict(
                label=label,
                rows=rows,
                models=[s["id"] for s in frozen["models"]],
                simulated=meta["simulated"],
            )
        )
    from .charts.comparison import generate_performance_charts

    generate_performance_charts(plot_inputs, lines, out)
    write_json(out / "metrics.json", summary)
    with (out / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(out / "README.md")

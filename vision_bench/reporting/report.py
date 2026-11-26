"""Reports support any number of model IDs and keep machines/runs separate."""

import csv
import json
from pathlib import Path
import statistics
from ..core import load_json, write_json


def _format_range(stats: dict, digits: int = 2) -> str:
    """Format mean, minimum, and maximum for a report table cell."""
    if stats["mean"] is None:
        return "—"
    return " / ".join(f'{stats[key]:.{digits}f}' for key in ("mean", "minimum", "maximum"))


def _read_gpu_samples(run: Path) -> list[dict[str, str]]:
    """Read nvidia-smi samples stored beside or inside a completed run."""
    candidates = [
        run.parent / f"{run.name}-gpu.csv",
        run.parent / "clef-gpu.csv",
        run / "gpu.csv",
    ]
    for csv_path in candidates:
        if csv_path.is_file():
            with csv_path.open(newline="", encoding="utf-8") as stream:
                return list(csv.DictReader(stream))
    return []


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
    quality_summaries = []
    comparable_runs = []
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
        gpu_samples = _read_gpu_samples(run)
        from .dataset_metrics import evaluate

        evaluation = evaluate(frozen, rows)
        evaluation_by_model = {item["model"]: item for item in evaluation["models"]}
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
            "| Modelo | Eventos / requisições | Latência média / mín / máx (ms) | Throughput médio / mín / máx (req/s) | CPU média / mín / máx (s) | RSS média / mín / máx (MiB) | VRAM média / mín / máx (MiB) | Esquema válido | Positivos corretos* | Erros |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
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
            batch_rows = {}
            for record in rs:
                if record.get("batch_id"):
                    batch_rows.setdefault(record["batch_id"], record)
            measured_ms = (
                sum(record["batch_elapsed_ms"] for record in batch_rows.values())
                if batch_rows
                else sum(record["elapsed_ms"] for record in rs)
            )
            throughput = len(ok) / (measured_ms / 1000) if measured_ms else None
            stats = evaluation_by_model[model]["performance_stats"]
            latency_stats = stats["latency_ms"]
            throughput_stats = stats["throughput_requests_s"]
            cpu_stats = stats["cpu_s"]
            rss_stats = stats["peak_rss_mib"]
            vram_stats = stats["gpu_peak_allocated_mib"]
            row = dict(
                run=run.name,
                machine=meta["machine_id"],
                model=model,
                device=frozen["device"],
                events=len(first),
                requests=len(rs),
                p50_s=med,
                p95_s=p95,
                throughput_requests_s=throughput,
                throughput_mode=(
                    "concurrent_batches" if batch_rows else "serial_requests"
                ),
                measured_s=measured_ms / 1000,
                mean_cpu_s=statistics.mean(r["cpu_s"] for r in rs),
                peak_rss_mib=max(r["peak_rss_bytes"] for r in rs) / 2**20,
                mean_latency_ms=latency_stats["mean"],
                min_latency_ms=latency_stats["minimum"],
                max_latency_ms=latency_stats["maximum"],
                mean_throughput_requests_s=throughput_stats["mean"],
                min_throughput_requests_s=throughput_stats["minimum"],
                max_throughput_requests_s=throughput_stats["maximum"],
                mean_cpu_sample_s=cpu_stats["mean"],
                min_cpu_sample_s=cpu_stats["minimum"],
                max_cpu_sample_s=cpu_stats["maximum"],
                mean_rss_mib=rss_stats["mean"],
                min_rss_mib=rss_stats["minimum"],
                max_rss_mib=rss_stats["maximum"],
                mean_vram_mib=vram_stats["mean"],
                min_vram_mib=vram_stats["minimum"],
                max_vram_mib=vram_stats["maximum"],
                resource_scope=",".join(scopes),
                first_valid=sum(r["valid"] for r in first),
                positive_n=len(positive),
                exploratory_positive_exact=correct,
                errors=sum(bool(r["error"]) for r in rs),
            )
            summary.append(row)
            lines.append(
                f'| {model} | {len(first)} / {len(rs)} | {_format_range(latency_stats)} | {_format_range(throughput_stats)} | {_format_range(cpu_stats, 3)} | {_format_range(rss_stats, 1)} | {_format_range(vram_stats, 1)} | {row["first_valid"]}/{len(first)} | {correct}/{len(positive)} | {row["errors"]} |'
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
                concurrency=frozen.get("concurrency"),
                gpu_samples=gpu_samples,
            )
        )
        quality_summaries.append((label, evaluation))
        comparable_runs.append(
            dict(
                machine=meta["machine_id"],
                contract=meta["contract_hash"],
                dataset=meta["dataset_hash"],
                simulated=meta["simulated"],
                frozen=frozen,
                models=[s["id"] for s in frozen["models"]],
                rows=rows,
                gpu_samples=gpu_samples,
            )
        )
    if len(comparable_runs) > 1:
        reference = comparable_runs[0]
        comparable = [
            item
            for item in comparable_runs
            if (
                item["machine"] == reference["machine"]
                and item["contract"] == reference["contract"]
                and item["dataset"] == reference["dataset"]
            )
        ]
        if len(comparable) > 1:
            combined_frozen = dict(reference["frozen"])
            combined_frozen["models"] = [
                spec
                for item in comparable
                for spec in item["frozen"]["models"]
            ]
            combined_rows = [row for item in comparable for row in item["rows"]]
            combined_models = [
                model for item in comparable for model in item["models"]
            ]
            plot_inputs.append(
                dict(
                    label="Todos os modelos · mesmo hardware",
                    rows=combined_rows,
                    models=combined_models,
                    simulated=any(item["simulated"] for item in comparable),
                    concurrency=max(
                        int(item["frozen"].get("concurrency", 0))
                        for item in comparable
                    ),
                    gpu_samples=[
                        sample
                        for item in comparable
                        for sample in item["gpu_samples"]
                    ],
                )
            )
            from .dataset_metrics import evaluate
            quality_summaries.append(
                (
                    "Todos os modelos · mesmo hardware",
                    evaluate(combined_frozen, combined_rows),
                )
            )
            lines += [
                "## Todos os modelos · mesmo hardware",
                "",
                "Gráficos agregados apenas entre execuções com o mesmo contrato, dataset e máquina.",
                "",
            ]
    from .charts.comparison import generate_performance_charts
    from .charts.quality import generate_quality_charts

    for index, (label, quality_summary) in enumerate(quality_summaries, 1):
        chart = generate_quality_charts(quality_summary, out, index)
        if chart:
            lines.extend(
                [f'![Acurácia — {label}](charts/accuracy/accuracy_{index}.png)', ""]
            )
    generate_performance_charts(plot_inputs, lines, out)
    write_json(out / "metrics.json", summary)
    with (out / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(out / "README.md")

"""Generate scientific latency, RAM, and GPU performance figures."""

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .paths import chart_directory
from .style import model_color, model_label


def _sample_values(
    rows: list[dict[str, Any]], model: str, key: str
) -> list[float]:
    """Return one median measurement per image for an available row metric."""
    by_image: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["model"] != model or row.get("error") or row.get(key) is None:
            continue
        by_image[row["id"]].append(float(row[key]))
    return [float(np.median(values)) for values in by_image.values()]


def _resource_rows(rows: list[dict[str, Any]], model: str) -> list[dict[str, Any]]:
    """Deduplicate shared batch resource samples before plotting them."""
    selected = [row for row in rows if row["model"] == model and not row.get("error")]
    unique: dict[str, dict[str, Any]] = {}
    for row in selected:
        key = row.get("batch_id", row["request_key"])
        unique.setdefault(key, row)
    return list(unique.values())


def _plot_distributions(
    axis: Any,
    models: list[str],
    values_by_model: dict[str, list[float]],
    colors: dict[str, str],
    unit: str,
) -> None:
    """Draw a box plot and repeatable jittered observations for each model."""
    rng = np.random.default_rng(20261123)
    positions = []
    values = []
    labels = []
    for position, model in enumerate(models, 1):
        sample = values_by_model[model]
        if not sample:
            continue
        axis.boxplot(
            [sample],
            positions=[position],
            widths=0.5,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.5},
        )
        axis.scatter(
            position + rng.uniform(-0.08, 0.08, len(sample)),
            sample,
            color=colors[model],
            edgecolor="white",
            linewidth=0.4,
            s=22,
            zorder=3,
        )
        positions.append(position)
        values.append(sample)
        labels.append(f"{model_label(model)}\n(n={len(sample)})")
    axis.set_xticks(positions, labels, rotation=15, ha="right")
    axis.set_ylabel(unit)
    axis.set_ylim(bottom=0)
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)


def _save_figure(figure: Any, output: Path, stem: str) -> None:
    """Save one publication-ready figure as PNG, PDF, and SVG."""
    for extension in ("png", "pdf", "svg"):
        figure.savefig(
            output / f"{stem}.{extension}",
            dpi=300,
            bbox_inches="tight",
        )


def generate_performance_charts(
    plot_inputs: list[dict[str, Any]], lines: list[str], output: Path
) -> None:
    """Write separate latency, RAM, and measured GPU performance charts."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for index, entry in enumerate(plot_inputs, 1):
        rows = entry["rows"]
        models = entry["models"]
        colors = {model: model_color(model) for model in models}
        figures = []

        latency_output = chart_directory(output, "latency")
        latency_values = {
            model: _sample_values(rows, model, "elapsed_ms") for model in models
        }
        if any(latency_values.values()):
            figure, axis = plt.subplots(figsize=(max(7, len(models) * 1.8), 5))
            _plot_distributions(axis, models, latency_values, colors, "Latência (ms)")
            axis.set_title(f'{entry["label"]} · latência por imagem')
            figure.text(
                0.01,
                0.01,
                "Pontos: medianas por imagem entre repetições; caixas: quartis. "
                "Amostras sem erro; carga e warm-up excluídos.",
                fontsize=8,
            )
            figure.subplots_adjust(bottom=0.25, top=0.88)
            _save_figure(figure, latency_output, f"latency_{index}")
            plt.close(figure)
            figures.append(
                f'![Latência — {entry["label"]}](charts/latency/latency_{index}.png)'
            )

        memory_output = chart_directory(output, "memory")
        memory_rows = {
            model: _resource_rows(rows, model) for model in models
        }
        memory_values = {
            model: [
                float(row["peak_rss_bytes"]) / 2**20
                for row in memory_rows[model]
                if row.get("peak_rss_bytes") is not None
            ]
            for model in models
        }
        if any(memory_values.values()):
            figure, axis = plt.subplots(figsize=(max(7, len(models) * 1.8), 5))
            _plot_distributions(axis, models, memory_values, colors, "Pico de RAM (MiB)")
            axis.set_title(f'{entry["label"]} · memória RAM')
            figure.text(
                0.01,
                0.01,
                "RSS amostrado por imagem; lotes concorrentes contam uma medição por lote. "
                "Servidores HTTP externos não estão incluídos.",
                fontsize=8,
            )
            figure.subplots_adjust(bottom=0.25, top=0.88)
            _save_figure(figure, memory_output, f"ram_{index}")
            plt.close(figure)
            figures.append(
                f'![Memória RAM — {entry["label"]}](charts/memory/ram_{index}.png)'
            )

        gpu_memory_values = {
            model: [
                float(row["gpu_peak_allocated_bytes"]) / 2**20
                for row in _resource_rows(rows, model)
                if row.get("gpu_peak_allocated_bytes")
            ]
            for model in models
        }
        batch_values: dict[str, list[float]] = {model: [] for model in models}
        seen_batches = set()
        for row in rows:
            batch_id = row.get("batch_id")
            if not batch_id or batch_id in seen_batches:
                continue
            seen_batches.add(batch_id)
            elapsed_ms = row.get("batch_elapsed_ms")
            if elapsed_ms and row.get("batch_size"):
                batch_values[row["model"]].append(
                    row["batch_size"] * 1000 / elapsed_ms
                )

        has_gpu_data = any(gpu_memory_values.values()) or any(batch_values.values())
        if has_gpu_data:
            gpu_output = chart_directory(output, "gpu")
            figure, axes = plt.subplots(
                1,
                2,
                figsize=(max(10, len(models) * 2.4), 5),
                squeeze=False,
            )
            _plot_distributions(
                axes[0, 0], models, gpu_memory_values, colors, "VRAM alocada (MiB)"
            )
            axes[0, 0].set_title("Pico de VRAM do processo medido")
            _plot_distributions(
                axes[0, 1], models, batch_values, colors, "Requisições por segundo"
            )
            axes[0, 1].set_title("Throughput observado por lote")
            figure.suptitle(entry["label"])
            figure.text(
                0.01,
                0.01,
                "VRAM é registrada somente para adapters PyTorch locais. "
                "Throughput é medido pelo cliente; GPU de servidor externo não é amostrada.",
                fontsize=8,
            )
            figure.subplots_adjust(bottom=0.27, top=0.82, wspace=0.3)
            _save_figure(figure, gpu_output, f"gpu_{index}")
            plt.close(figure)
            figures.append(
                f'![GPU — {entry["label"]}](charts/gpu/gpu_{index}.png)'
            )

        lines.extend(figures)
        lines.append("")

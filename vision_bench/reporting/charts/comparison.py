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


def _plot_min_mean_max(
    axis: Any,
    models: list[str],
    samples: dict[str, list[float]],
    colors: dict[str, str],
    unit: str,
) -> None:
    """Compare each model's observed minimum, mean, and maximum."""
    positions = []
    labels = []
    for position, model in enumerate(models, 1):
        values = samples[model]
        if not values:
            continue
        low, mean, high = min(values), float(np.mean(values)), max(values)
        axis.plot([low, high], [position, position], color=colors[model], linewidth=3)
        axis.scatter([low, high], [position, position], color=colors[model], s=24)
        axis.scatter([mean], [position], color="black", marker="D", s=28, zorder=3)
        positions.append(position)
        labels.append(model_label(model))
        axis.annotate(
            f"mín {low:.2f} · média {mean:.2f} · máx {high:.2f}",
            (high, position),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
        )
    axis.set_yticks(positions, labels)
    axis.set_xlabel(unit)
    axis.grid(axis="x", alpha=0.25)
    axis.set_axisbelow(True)


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

            figure, axes = plt.subplots(1, 2, figsize=(12, 5))
            _plot_min_mean_max(
                axes[0], models, latency_values, colors, "Latência (ms)"
            )
            axes[0].set_title("Mínimo, média e máximo")
            for model in models:
                values = latency_values[model]
                if values:
                    axes[1].hist(
                        values,
                        bins="auto",
                        alpha=0.45,
                        label=model_label(model),
                        color=colors[model],
                    )
            axes[1].set(
                title="Distribuição da latência",
                xlabel="Latência (ms)",
                ylabel="Observações",
            )
            axes[1].legend()
            axes[1].grid(axis="y", alpha=0.25)
            figure.suptitle(f'{entry["label"]} · estatísticas e histograma')
            figure.tight_layout()
            _save_figure(figure, latency_output, f"latency_summary_{index}")
            plt.close(figure)
            figures.append(
                f'![Resumo e histograma de latência — {entry["label"]}]'
                f'(charts/latency/latency_summary_{index}.png)'
            )

            figure, axis = plt.subplots(figsize=(8, 5))
            for model in models:
                values = latency_values[model]
                if values:
                    axis.plot(
                        range(1, len(values) + 1),
                        values,
                        marker="o",
                        linewidth=1,
                        label=model_label(model),
                        color=colors[model],
                    )
            axis.set(
                title="Latência cartesiana por observação",
                xlabel="Observação bem-sucedida",
                ylabel="Latência (ms)",
            )
            axis.grid(alpha=0.25)
            axis.legend()
            figure.tight_layout()
            _save_figure(figure, latency_output, f"latency_series_{index}")
            plt.close(figure)
            figures.append(
                f'![Série de latência — {entry["label"]}]'
                f'(charts/latency/latency_series_{index}.png)'
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
            figure, axis = plt.subplots(figsize=(8, 4.5))
            _plot_min_mean_max(axis, models, memory_values, colors, "RSS (MiB)")
            axis.set_title("RAM: mínimo, média e máximo")
            figure.tight_layout()
            _save_figure(figure, memory_output, f"ram_summary_{index}")
            plt.close(figure)
            figures.append(
                f'![Resumo de RAM — {entry["label"]}]'
                f'(charts/memory/ram_summary_{index}.png)'
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
        batch_latency_values: dict[str, list[float]] = {model: [] for model in models}
        seen_batches: set[tuple[str, str]] = set()
        for row in rows:
            batch_id = row.get("batch_id")
            batch_key = (row["model"], batch_id)
            if not batch_id or batch_key in seen_batches:
                continue
            seen_batches.add(batch_key)
            elapsed_ms = row.get("batch_elapsed_ms")
            if elapsed_ms and row.get("batch_size"):
                batch_values[row["model"]].append(
                    row["batch_size"] * 1000 / elapsed_ms
                )
                batch_latency_values[row["model"]].append(
                    elapsed_ms / row["batch_size"]
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

            figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
            _plot_min_mean_max(axes[0], models, batch_values, colors, "req/s")
            axes[0].set_title("Throughput: mínimo, média e máximo")
            _plot_min_mean_max(axes[1], models, gpu_memory_values, colors, "MiB")
            axes[1].set_title("VRAM: mínimo, média e máximo")
            figure.tight_layout()
            _save_figure(figure, gpu_output, f"gpu_summary_{index}")
            plt.close(figure)
            figures.append(
                f'![Resumo de throughput e VRAM — {entry["label"]}]'
                f'(charts/gpu/gpu_summary_{index}.png)'
            )

            figure, axis = plt.subplots(figsize=(8, 5))
            for model in models:
                x = batch_latency_values[model]
                y = batch_values[model]
                if x and y:
                    axis.scatter(
                        x,
                        y,
                        color=colors[model],
                        label=model_label(model),
                        alpha=0.8,
                    )
            axis.set(
                title="Latência × throughput",
                xlabel="Latência média do lote por imagem (ms)",
                ylabel="Throughput por lote (req/s)",
            )
            axis.grid(alpha=0.25)
            axis.legend()
            figure.tight_layout()
            _save_figure(figure, gpu_output, f"latency_throughput_{index}")
            plt.close(figure)
            figures.append(
                f'![Latência e throughput — {entry["label"]}]'
                f'(charts/gpu/latency_throughput_{index}.png)'
            )

        gpu_samples = entry.get("gpu_samples", [])
        utilization = []
        gpu_memory = []
        power = []
        timestamps = []
        for sample in gpu_samples:
            try:
                utilization_value = float(sample["utilization_gpu_pct"])
                memory_value = float(sample["memory_used_MiB"])
                power_value = float(sample["power_W"])
                timestamp = sample["timestamp"]
            except (KeyError, TypeError, ValueError):
                continue
            utilization.append(utilization_value)
            gpu_memory.append(memory_value)
            power.append(power_value)
            timestamps.append(timestamp)
        if timestamps:
            gpu_output = chart_directory(output, "gpu")
            figure, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            x = range(len(timestamps))
            axes[0].plot(x, utilization, color="#2b6cb0", marker=".")
            axes[0].axhline(
                np.mean(utilization),
                color="#2b6cb0",
                linestyle="--",
                label=f"Média: {np.mean(utilization):.1f}%",
            )
            axes[0].set(ylabel="Utilização GPU (%)", ylim=(0, 100))
            axes[0].legend()
            axes[1].plot(x, gpu_memory, color="#805ad5", marker=".")
            axes[1].axhline(
                np.mean(gpu_memory),
                color="#805ad5",
                linestyle="--",
                label=f"Média: {np.mean(gpu_memory):.1f} MiB",
            )
            axes[1].set(ylabel="Memória GPU (MiB)")
            axes[1].legend()
            axes[2].plot(x, power, color="#dd6b20", marker=".")
            axes[2].axhline(
                np.mean(power),
                color="#dd6b20",
                linestyle="--",
                label=f"Média: {np.mean(power):.1f} W",
            )
            axes[2].set(ylabel="Potência (W)", xlabel="Amostra nvidia-smi")
            axes[2].legend()
            tick_step = max(1, len(timestamps) // 8)
            axes[2].set_xticks(
                list(x)[::tick_step], timestamps[::tick_step], rotation=20, ha="right"
            )
            for axis in axes:
                axis.grid(alpha=0.25)
            batch = entry.get("concurrency")
            batch_text = f" · batch {batch}" if batch else ""
            figure.suptitle(f'{entry["label"]} · telemetria da GPU{batch_text}')
            figure.text(
                0.01,
                0.005,
                "Amostras sequenciais de nvidia-smi; em execuções com vários modelos, "
                "cada bloco corresponde a um modelo servido por vez.",
                fontsize=8,
            )
            figure.tight_layout(rect=(0, 0.03, 1, 1))
            _save_figure(figure, gpu_output, f"gpu_telemetry_{index}")
            plt.close(figure)
            figures.append(
                f'![Telemetria da GPU — {entry["label"]}]'
                f'(charts/gpu/gpu_telemetry_{index}.png)'
            )

        lines.extend(figures)
        lines.append("")

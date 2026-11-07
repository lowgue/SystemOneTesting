"""Performance distribution charts for comparison reports."""

from collections import defaultdict
import statistics

import numpy as np

from .paths import chart_directory
from .style import model_color


def generate_performance_charts(plot_inputs, lines, output):
    """Write latency and resource comparison charts for each run."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chart_output = chart_directory(output, "performance")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    for index, entry in enumerate(plot_inputs, 1):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(
            ("SIMULAÇÃO — " if entry["simulated"] else "") + entry["label"],
            x=0.06,
            ha="left",
            fontsize=13,
            fontweight="bold",
        )
        palettes = plt.get_cmap("tab10")
        rng = np.random.default_rng(20261026)
        for pos, m in enumerate(entry["models"], 1):
            groups = defaultdict(list)
            for r in entry["rows"]:
                if r["model"] == m and not r["error"]:
                    groups[r["id"]].append(r)
            for ax, key, scale in [
                (axes[0], "elapsed_ms", 1000),
                (axes[1], "cpu_s", 1),
            ]:
                values = [
                    statistics.median(r[key] / scale for r in rs)
                    for rs in groups.values()
                ]
                if values:
                    ax.boxplot(
                        [values],
                        positions=[pos],
                        widths=0.4,
                        manage_ticks=False,
                        showfliers=False,
                        medianprops={"color": "black"},
                    )
                    ax.scatter(
                        pos + rng.uniform(-0.1, 0.1, len(values)),
                        values,
                        color=model_color(m),
                        s=20,
                        alpha=0.7,
                    )
        for ax in axes:
            ax.set_xticks(
                range(1, len(entry["models"]) + 1),
                entry["models"],
                rotation=18,
                ha="right",
            )
            ax.set_ylim(bottom=0)
            ax.grid(axis="y", alpha=0.2)
        axes[0].set_ylabel("Mediana intraimagem da latência (s)")
        axes[1].set_ylabel("Mediana intraimagem de CPU-s (escopo por modelo)")
        axes[0].set_title("Evento completo, com pré e pós-processamento")
        axes[1].set_title("HTTP sem PID mede somente o cliente")
        fig.text(
            0.06,
            0.025,
            "Pontos: medianas por evento, não repetições independentes. Caixas: quartis; sem IC populacional para esta amostra.\nCarga e warm-up excluídos. Recursos remotos, energia e custo monetário não estão incluídos nos recursos do cliente.",
            fontsize=8,
        )
        fig.subplots_adjust(left=0.09, right=0.97, bottom=0.30, top=0.82, wspace=0.35)
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(chart_output / f"comparison_{index}.{ext}", dpi=300)
        plt.close(fig)
        lines += [
            f'![{entry["label"]}](charts/performance/comparison_{index}.png)',
            "",
            f"[PDF](charts/performance/comparison_{index}.pdf) · [SVG](charts/performance/comparison_{index}.svg)",
            "",
        ]

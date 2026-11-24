"""Accuracy charts with sample counts and Wilson confidence intervals."""

from math import sqrt
from pathlib import Path
from typing import Any

from .paths import chart_directory, chart_reference
from .style import model_color, model_label


def _wilson_interval(correct: int, sample_size: int) -> tuple[float, float]:
    """Return a two-sided 95% Wilson interval for a binomial proportion."""
    if sample_size == 0:
        return 0.0, 0.0

    z = 1.959963984540054
    proportion = correct / sample_size
    denominator = 1 + z**2 / sample_size
    center = (proportion + z**2 / (2 * sample_size)) / denominator
    margin = (
        z
        * sqrt(
            proportion * (1 - proportion) / sample_size
            + z**2 / (4 * sample_size**2)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def generate_quality_charts(
    summary: dict[str, Any], output: Path, index: int = 1
) -> Path | None:
    """Write per-field accuracy points with Wilson intervals and sample counts."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    observations = []
    for model in summary["models"]:
        for field in model["fields"]:
            sample_size = field["n"]
            if not sample_size:
                continue
            accuracy = field["accuracy"]
            lower, upper = _wilson_interval(field["correct"], sample_size)
            observations.append(
                {
                    "label": (
                        f'{field["domain"]} / {field["field"]} — '
                        f'{model_label(model)}'
                    ),
                    "model": model,
                    "correct": field["correct"],
                    "sample_size": sample_size,
                    "accuracy": accuracy,
                    "lower_error": accuracy - lower,
                    "upper_error": upper - accuracy,
                }
            )

    summary["charts"] = []
    if not observations:
        return None

    chart_output = chart_directory(output, "accuracy")
    figure_height = max(4, 0.38 * len(observations) + 1.8)
    figure, axis = plt.subplots(figsize=(10, figure_height))
    y_positions = list(range(len(observations)))
    for y_position, observation in zip(y_positions, observations):
        axis.errorbar(
            observation["accuracy"],
            y_position,
            xerr=[
                [observation["lower_error"]],
                [observation["upper_error"]],
            ],
            fmt="o",
            color=model_color(observation["model"]),
            capsize=3,
            markersize=5,
            linewidth=1.2,
        )
        axis.text(
            1.025,
            y_position,
            f'{observation["correct"]}/{observation["sample_size"]}',
            va="center",
            fontsize=8,
            transform=axis.get_yaxis_transform(),
        )

    axis.set_yticks(y_positions, [row["label"] for row in observations])
    axis.invert_yaxis()
    axis.set_xlim(0, 1)
    axis.xaxis.set_major_formatter(PercentFormatter(1))
    axis.set_xlabel("Acurácia")
    axis.set_title(
        ("SIMULAÇÃO — " if summary["simulated"] else "")
        + f'Acurácia por atributo · {summary["dataset_name"]}'
    )
    axis.grid(axis="x", alpha=0.25)
    axis.set_axisbelow(True)
    figure.text(
        0.01,
        0.01,
        "Primeira repetição por imagem; barras: IC 95% de Wilson; rótulos: acertos/N. "
        "Falhas contam como erros.",
        fontsize=8,
    )
    figure.subplots_adjust(left=0.29, right=0.88, bottom=0.12, top=0.92)

    stem = f"accuracy_{index}"
    chart_path = chart_output / stem
    for extension in ("png", "pdf", "svg"):
        figure.savefig(
            chart_path.with_suffix(f".{extension}"),
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)

    summary["charts"].append(
        {
            "title": "Acurácia por atributo com IC 95% de Wilson",
            "file": chart_reference("accuracy", f"{stem}.png"),
        }
    )
    return chart_path.with_suffix(".png")

"""Quality, per-attribute and confusion-matrix chart generation."""

import numpy as np

from .paths import chart_directory, chart_reference


def generate_quality_charts(summary, output):
    """Generate quality, attribute, class, and confusion-matrix charts."""
    _quality_chart(summary, output)
    _attribute_charts(summary, output)


def _quality_chart(summary, output):
    """Write the overview chart for first-repetition field accuracy."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .style import baseline_models, model_color, model_label

    models = baseline_models(summary["models"])
    if not any(model["fields"] for model in models):
        return
    quality_output = chart_directory(output, "quality")
    keys = sorted(
        {(f["domain"], f["field"]) for model in models for f in model["fields"]}
    )
    fig, ax = plt.subplots(figsize=(max(8, len(keys) * 1.2), 5))
    width = 0.8 / len(models)
    x = np.arange(len(keys))
    for index, model in enumerate(models):
        values = {(f["domain"], f["field"]): f["accuracy"] for f in model["fields"]}
        ax.bar(
            x - 0.4 + width / 2 + index * width,
            [100 * values.get(key, 0) for key in keys],
            width,
            label=model_label(model),
            color=model_color(model),
        )
    ax.set_xticks(x, [f"{domain}\n{field}" for domain, field in keys])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Acurácia por atributo (%)")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    ax.set_title(
        ("SIMULAÇÃO — " if summary["simulated"] else "") + summary["dataset_name"]
    )
    fig.text(
        0.02,
        0.015,
        "Primeira repetição por foto. Falhas contam como erros. Gabaritos ausentes são excluídos.\nRepetições de tempo não aumentam a amostra de qualidade.",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    for suffix in ["png", "pdf"]:
        fig.savefig(quality_output / f"accuracy.{suffix}", dpi=300)
    plt.close(fig)


def _attribute_charts(summary, output):
    """Write per-field accuracy, format, latency, class, and confusion charts."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    summary["charts"] = []
    attribute_output = chart_directory(output, "attributes")
    class_output = chart_directory(output, "class_metrics")
    confusion_output = chart_directory(output, "confusion_matrices")
    keys = sorted(
        {(f["domain"], f["field"]) for m in summary["models"] for f in m["fields"]}
    )
    for domain, key in keys:
        models = [
            m
            for m in summary["models"]
            if any(f["domain"] == domain and f["field"] == key for f in m["fields"])
        ]
        fields = [
            next(f for f in m["fields"] if f["domain"] == domain and f["field"] == key)
            for m in models
        ]
        from .style import model_color, model_label

        names = [model_label(m) for m in models]
        x = np.arange(len(models))
        colors = [model_color(m) for m in models]
        fig, axes = plt.subplots(2, 2, figsize=(max(11, len(models) * 1.1), 10))
        ax = axes[0, 0]
        ax.barh(x, [f["accuracy"] for f in fields], color=colors)
        ax.set_yticks(x, names, fontsize=7)
        ax.set_xlim(0, 1.05)
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.set_title("Acurácia — primeira repetição")
        for i, f in enumerate(fields):
            ax.text(
                min(f["accuracy"] + 0.01, 0.92),
                i,
                f"{f['correct']}/{f['n']}",
                va="center",
                fontsize=8,
            )
        ax = axes[0, 1]
        for i, (metric, label) in enumerate(
            [
                ("macro_precision", "Precisão macro"),
                ("macro_recall", "Recall macro"),
                ("macro_f1", "F1 macro"),
            ]
        ):
            ax.bar(
                x + (i - 1) * 0.25,
                [f[metric] for f in fields],
                0.25,
                label=label,
                color=["#64748B", "#A1AAB8", "#D3DAE4"][i],
                hatch=["", "//", ".."][i],
            )
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.legend(fontsize=8)
        ax.set_title("Qualidade com peso igual por classe")
        ax = axes[1, 0]
        for i, (metric, label) in enumerate(
            [
                ("raw_adherence", "Aderência bruta"),
                ("output_format_valid", "Formato válido"),
                ("ambiguity_rate", "Ambiguidade"),
            ]
        ):
            ax.bar(
                x + (i - 1) * 0.25,
                [np.nan if f.get(metric) is None else f[metric] for f in fields],
                0.25,
                label=label,
                color=["#64748B", "#A1AAB8", "#D3DAE4"][i],
                hatch=["", "//", ".."][i],
            )
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.legend(fontsize=8)
        ax.set_title("Saída antes da normalização (menor ambiguidade é melhor)")
        ax = axes[1, 1]
        measured = [f["attribute_latency"]["p50_ms"] for f in fields]
        ax.bar(x, [np.nan if v is None else v for v in measured], color=colors)
        ax.set_ylabel("ms")
        ax.set_title("Latência p50 em chamada isolada por atributo")
        if not any(v is not None for v in measured):
            ax.text(
                0.5,
                0.5,
                "Requests conjuntos: tempo por atributo não é isolável.",
                ha="center",
                va="center",
                transform=ax.transAxes,
                wrap=True,
            )
        for ax in [axes[0, 1], axes[1, 0], axes[1, 1]]:
            ax.set_xticks(x, names, rotation=45, ha="right", fontsize=7)
        for ax in axes.flat:
            ax.grid(axis="y", alpha=0.18)
            ax.set_axisbelow(True)
        fig.suptitle(
            ("SIMULAÇÃO — " if summary["simulated"] else "")
            + f'{domain} / {key} · {summary["dataset_name"]}',
            fontsize=13,
        )
        fig.text(
            0.02,
            0.012,
            "Qualidade: uma previsão por evento. Repetições medem tempo e não ampliam a amostra. Gabaritos históricos são exploratórios.\nBarras ausentes indicam métrica não medida. Tempos de requests conjuntos não foram divididos artificialmente.",
            fontsize=8,
        )
        fig.tight_layout(rect=(0, 0.05, 1, 0.95))
        stem = f"attribute_{domain}_{key}"
        for ext in ["png", "pdf"]:
            fig.savefig(attribute_output / f"{stem}.{ext}", dpi=300)
        plt.close(fig)
        summary["charts"].append(
            dict(title=f"{domain}: {key} — qualidade, formato e tempo", file=chart_reference("attributes", stem))
        )
        labels = sorted({c["label"] for f in fields for c in f["classes"]})
        matrix = np.full((len(models), len(labels)), np.nan)
        for i, f in enumerate(fields):
            for c in f["classes"]:
                matrix[i, labels.index(c["label"])] = c["f1"]
        fig, ax = plt.subplots(
            figsize=(max(8, len(labels) * 0.65), max(3, len(models) * 0.6 + 2))
        )
        im = ax.imshow(matrix, vmin=0, vmax=1, cmap="Greys", aspect="auto")
        fig.colorbar(im, ax=ax, label="F1 por classe")
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(models)), names, fontsize=8)
        for i in range(len(models)):
            for j in range(len(labels)):
                if np.isfinite(matrix[i, j]):
                    ax.text(
                        j,
                        i,
                        f"{matrix[i,j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if matrix[i, j] > 0.6 else "black",
                    )
        ax.set_title(
            f"{domain} / {key}: F1 de cada classe — classes com suporte no gabarito"
        )
        fig.tight_layout()
        stem = f"classes_{domain}_{key}"
        for ext in ["png", "pdf"]:
            fig.savefig(class_output / f"{stem}.{ext}", dpi=300)
        plt.close(fig)
        summary["charts"].append(
            dict(title=f"{domain}: {key} — F1 por classe", file=chart_reference("class_metrics", stem))
        )
        for m, f in zip(models, fields):
            labels = sorted(
                set(f["confusion"])
                | {p for row in f["confusion"].values() for p in row}
            )
            matrix = np.array(
                [[f["confusion"].get(a, {}).get(b, 0) for b in labels] for a in labels]
            )
            fig, ax = plt.subplots(
                figsize=(max(6, len(labels) * 0.5), max(5, len(labels) * 0.45))
            )
            im = ax.imshow(
                matrix,
                cmap=(
                    "Oranges"
                    if model_color(m) == "#F78C25"
                    else "Blues" if model_color(m) == "#3378FF" else "Greys"
                ),
            )
            fig.colorbar(im, ax=ax, label="Eventos")
            ax.set_xticks(
                range(len(labels)), labels, rotation=60, ha="right", fontsize=7
            )
            ax.set_yticks(range(len(labels)), labels, fontsize=7)
            ax.set_xlabel("Previsão")
            ax.set_ylabel("Gabarito")
            ax.set_title(f'{m["model"]}\n{domain} / {key}')
            if len(labels) < 18:
                for i in range(len(labels)):
                    for j in range(len(labels)):
                        if matrix[i, j]:
                            ax.text(
                                j,
                                i,
                                str(matrix[i, j]),
                                ha="center",
                                va="center",
                                fontsize=8,
                            )
            fig.tight_layout()
            stem = f'confusion_{m["model"]}_{domain}_{key}'
            for ext in ["png", "pdf"]:
                fig.savefig(confusion_output / f"{stem}.{ext}", dpi=300)
            plt.close(fig)
            summary["charts"].append(
                dict(title=f'{m["model"]}: matriz {domain}/{key}', file=chart_reference("confusion_matrices", stem))
            )

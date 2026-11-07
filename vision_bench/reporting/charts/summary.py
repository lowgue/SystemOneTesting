"""Readable horizontal comparisons; no pooling of prompt variants or runs."""

import numpy as np
from .paths import chart_directory, chart_reference
from .style import baseline_models, label, model_color, model_label


def generate_summary_charts(summary, output):
    """Generate compact accuracy, latency, and per-class comparison charts."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    models = baseline_models(summary["models"])
    charts = []

    def bars(stem, title, subtitle, categories, series, percent=True):
        """Render one horizontal comparison chart and register its report path."""
        if stem == "simple_overview":
            chart_type = "quality"
        elif stem == "simple_latency":
            chart_type = "performance"
        else:
            chart_type = "attributes"
        chart_output = chart_directory(output, chart_type)
        fig, ax = plt.subplots(figsize=(11, max(3.5, len(categories) * 0.72 + 1.8)))
        fig.patch.set_facecolor("#F6F8FC")
        ax.set_facecolor("#F6F8FC")
        y = np.arange(len(categories))
        height = 0.65 / max(1, len(series))
        maximum = (
            1
            if percent
            else max(
                (v for _, values, _ in series for v in values if v is not None),
                default=1,
            )
            * 1.2
            or 1
        )
        for i, (m, values, texts) in enumerate(series):
            positions = y - 0.325 + height / 2 + i * height
            ax.barh(
                positions,
                [np.nan if v is None else v for v in values],
                height * 0.85,
                color=model_color(m),
                label=model_label(m),
                zorder=3,
            )
            for pos, v, text in zip(positions, values, texts):
                ax.text(
                    (v if v is not None else 0) + maximum * 0.012,
                    pos,
                    text,
                    va="center",
                    fontsize=9,
                    color="#18324F",
                )
        ax.set_yticks(y, categories, fontweight="normal")
        ax.invert_yaxis()
        ax.set_xlim(0, maximum * 1.16)
        ax.grid(axis="x", color="#E0E7F1", alpha=0.9, zorder=0)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(axis="both", length=0, labelcolor="#36516E", pad=9)
        if percent:
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
            ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        else:
            ax.set_xlabel("Milissegundos · menor é mais rápido")
        fig.suptitle(
            ("SIMULAÇÃO — " if summary["simulated"] else "") + title,
            x=0.03,
            y=0.99,
            ha="left",
            fontsize=15,
            fontweight="bold",
            color="#102A43",
        )
        fig.text(0.03, 0.88, subtitle, fontsize=9, color="#526B87")
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(0, 1.12),
            frameon=False,
            ncol=2,
            fontsize=9,
        )
        fig.text(
            0.03,
            0.02,
            "Mesmas imagens · qualidade na primeira repetição · gabaritos exploratórios quando indicado no relatório.",
            fontsize=8,
            color="#526B87",
        )
        fig.subplots_adjust(left=0.22, right=0.91, bottom=0.14, top=0.70)
        for ext in ["png", "pdf"]:
            fig.savefig(chart_output / f"{stem}.{ext}", dpi=300)
        plt.close(fig)
        charts.append(
            dict(
                title=title,
                file=chart_reference(chart_type, stem),
                level="simple",
            )
        )

    bars(
        "simple_overview",
        "Acurácia dos modelos",
        "Percentual de atributos corretos · protocolos padrão, sem combinar variantes",
        ["Atributos corretos"],
        [
            (
                m,
                [m["field_accuracy"]],
                [
                    (
                        f"{m['field_correct']}/{m['field_n']} ({m['field_accuracy']:.0%})"
                        if m["field_accuracy"] is not None
                        else "Sem gabarito"
                    )
                ],
            )
            for m in models
        ],
    )
    bars(
        "simple_latency",
        "Tempo de avaliação por foto",
        "Mediana das chamadas medidas · carga e aquecimento excluídos",
        ["Tempo mediano"],
        [
            (
                m,
                [m["latency"]["p50_ms"]],
                [
                    (
                        f"{m['latency']['p50_ms']:,.0f} ms"
                        if m["latency"]["p50_ms"] is not None
                        else "Sem sucesso"
                    )
                ],
            )
            for m in models
        ],
        False,
    )
    keys = sorted({(f["domain"], f["field"]) for m in models for f in m["fields"]})
    for domain, key in keys:
        fields = [
            (
                m,
                next(
                    (
                        f
                        for f in m["fields"]
                        if f["domain"] == domain and f["field"] == key
                    ),
                    None,
                ),
            )
            for m in models
        ]
        classes = sorted({c["label"] for _, f in fields if f for c in f["classes"]})
        series = []
        for m, f in fields:
            lookup = {c["label"]: c for c in f["classes"]} if f else {}
            values = []
            texts = []
            for value in classes:
                c = lookup.get(value)
                values.append(c["recall"] if c else None)
                texts.append(
                    f"{round(c['recall']*c['support'])}/{c['support']} ({c['recall']:.0%})"
                    if c
                    else "Sem gabarito"
                )
            series.append((m, values, texts))
        bars(
            f"simple_{domain}_{key}",
            f"Acurácia por {label(key).lower()}",
            f"{domain} · acertos / exemplos de cada classe · protocolos identificados na legenda",
            [label(c) for c in classes],
            series,
        )
    return charts


generate = generate_summary_charts

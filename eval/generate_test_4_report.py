"""Generate visualizations and empirical markdown report for Test 4: Robustness."""
from __future__ import annotations

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_FILE = ROOT / "eval/results/test_4_robustness_results.json"
REPORT_FILE = ROOT / "eval/results/test_4_robustness_report.md"
RESULTS_DIR = ROOT / "eval/results"

COLOR_HEX = {"laya": "#2563eb", "blip": "#ea580c"}

STRATUM_LABELS = {
    "original": "Original (baseline)",
    "low_light": "Baixa Luminosidade",
    "blur": "Borrão Gaussiano",
    "low_res": "Baixa Resolução",
    "jpeg_compress": "Compressão JPEG",
    "occlusion": "Oclusão Parcial",
}

TASK_LABELS = {
    "color": "Cor do Veículo",
    "vehicle_type": "Tipo do Veículo",
    "helmet": "Capacete (Pessoa)",
}


def esc(v: object) -> str:
    return html.escape(str(v), quote=True)


def tx(x, y, s, size=13, color="#1e293b", weight="400", anchor="start"):
    return (f'<text x="{x}" y="{y}" font-family="system-ui,-apple-system,sans-serif" '
            f'font-size="{size}" fill="{color}" font-weight="{weight}" '
            f'text-anchor="{anchor}">{esc(s)}</text>')


def svg_start(w, h, title, subtitle):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        f'<rect width="100%" height="100%" fill="#ffffff" rx="8"/>',
        f'<rect width="100%" height="100%" fill="none" stroke="#e2e8f0" stroke-width="1" rx="8"/>',
        tx(36, 38, title, 20, "#0f172a", "700"),
        tx(36, 60, subtitle, 12, "#64748b", "400"),
    ]


def build_accuracy_by_stratum_svg(data: dict) -> Path:
    """Grouped bar chart: accuracy per stratum for each model."""
    strata = data["strata"]
    w, h = 980, 480
    lines = svg_start(w, h,
                      "Teste 4: Acurácia por Estrato de Degradação",
                      "Acurácia (%) vs baseline original · Laya Vision vs BLIP VQA")

    x0, y0 = 200, 100
    col_w = 660

    # Grid lines
    for pct in (0, 20, 40, 60, 80, 100):
        x = x0 + (col_w * pct / 100)
        lines.append(f'<line x1="{x}" y1="{y0 - 10}" x2="{x}" y2="{y0 + len(strata) * 56}" '
                     f'stroke="#f1f5f9" stroke-width="1"/>')
        lines.append(tx(x, y0 + len(strata) * 56 + 18, f"{pct}%", 11, "#94a3b8", anchor="middle"))

    for idx, stratum in enumerate(strata):
        y = y0 + idx * 56
        label = STRATUM_LABELS.get(stratum, stratum)
        lines.append(tx(x0 - 14, y + 18, label, 12, "#334155", "600", anchor="end"))

        l_acc = data["models"]["laya"]["per_stratum"][stratum]["accuracy"]
        b_acc = data["models"]["blip"]["per_stratum"][stratum]["accuracy"]

        # Laya bar
        lw = col_w * l_acc
        lines.append(f'<rect x="{x0}" y="{y}" width="{lw:.1f}" height="14" '
                     f'fill="{COLOR_HEX["laya"]}" rx="3"/>')
        lines.append(tx(x0 + lw + 6, y + 12, f"{l_acc*100:.1f}%", 11, COLOR_HEX["laya"], "600"))

        # BLIP bar
        bw = col_w * b_acc
        lines.append(f'<rect x="{x0}" y="{y + 18}" width="{bw:.1f}" height="14" '
                     f'fill="{COLOR_HEX["blip"]}" rx="3"/>')
        lines.append(tx(x0 + bw + 6, y + 30, f"{b_acc*100:.1f}%", 11, COLOR_HEX["blip"], "600"))

        # Delta annotation for non-original strata
        if stratum != "original":
            l_delta = data["models"]["laya"]["accuracy_deltas_vs_original"].get(stratum, 0)
            b_delta = data["models"]["blip"]["accuracy_deltas_vs_original"].get(stratum, 0)
            lines.append(tx(x0 + col_w + 30, y + 12, f"Δ{l_delta*100:+.1f}pp", 10,
                           "#dc2626" if l_delta < 0 else "#16a34a", "500"))
            lines.append(tx(x0 + col_w + 30, y + 30, f"Δ{b_delta*100:+.1f}pp", 10,
                           "#dc2626" if b_delta < 0 else "#16a34a", "500"))

    # Legend
    ly = h - 40
    lines.append(f'<rect x="{w - 340}" y="{ly}" width="14" height="14" fill="{COLOR_HEX["laya"]}" rx="2"/>')
    lines.append(tx(w - 320, ly + 12, "Laya Vision", 12, "#1e293b", "600"))
    lines.append(f'<rect x="{w - 210}" y="{ly}" width="14" height="14" fill="{COLOR_HEX["blip"]}" rx="2"/>')
    lines.append(tx(w - 190, ly + 12, "BLIP VQA", 12, "#1e293b", "600"))

    lines.append("</svg>")
    out_path = RESULTS_DIR / "test_4_robustness_accuracy.svg"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def build_task_breakdown_svg(data: dict) -> Path:
    """Heatmap-style chart showing accuracy per task × stratum."""
    strata = data["strata"]
    tasks = [t for t in ["color", "vehicle_type", "helmet"]
             if t in data["models"]["laya"].get("per_task_per_stratum", {})]
    if not tasks:
        # Fallback: just return the accuracy chart
        return build_accuracy_by_stratum_svg(data)

    w, h = 980, 520
    lines = svg_start(w, h,
                      "Teste 4: Acurácia por Tarefa × Estrato de Degradação",
                      "Degradação da acurácia por tipo de pergunta visual · Laya (L) vs BLIP (B)")

    # Table layout
    x0, y0 = 200, 100
    cell_w = 120
    cell_h = 56
    header_h = 30

    # Column headers (strata)
    for si, stratum in enumerate(strata):
        cx = x0 + si * cell_w + cell_w // 2
        label = stratum.replace("_", "\n")
        short = {"original": "Orig.", "low_light": "Low\nLight", "blur": "Blur",
                 "low_res": "Low\nRes", "jpeg_compress": "JPEG", "occlusion": "Occl."}
        lines.append(tx(cx, y0 - 8, short.get(stratum, stratum).split("\n")[0],
                        11, "#334155", "700", "middle"))
        if "\n" in short.get(stratum, ""):
            lines.append(tx(cx, y0 + 6, short[stratum].split("\n")[1],
                            10, "#64748b", "400", "middle"))

    # Row for each task
    for ti, task in enumerate(tasks):
        y = y0 + header_h + ti * cell_h
        lines.append(tx(x0 - 14, y + cell_h // 2, TASK_LABELS.get(task, task),
                        12, "#334155", "600", "end"))

        for si, stratum in enumerate(strata):
            x = x0 + si * cell_w
            l_data = data["models"]["laya"]["per_task_per_stratum"].get(task, {}).get(stratum, {})
            b_data = data["models"]["blip"]["per_task_per_stratum"].get(task, {}).get(stratum, {})

            l_acc = l_data.get("accuracy", 0.0)
            b_acc = b_data.get("accuracy", 0.0)

            # Background colour based on average accuracy
            avg = (l_acc + b_acc) / 2
            if avg >= 0.6:
                bg = "#dcfce7"
            elif avg >= 0.3:
                bg = "#fef9c3"
            else:
                bg = "#fee2e2"

            lines.append(f'<rect x="{x + 2}" y="{y + 2}" width="{cell_w - 4}" '
                         f'height="{cell_h - 4}" fill="{bg}" rx="4" stroke="#e2e8f0"/>')
            lines.append(tx(x + 10, y + 20, f"L: {l_acc*100:.0f}%", 11, COLOR_HEX["laya"], "600"))
            lines.append(tx(x + 10, y + 38, f"B: {b_acc*100:.0f}%", 11, COLOR_HEX["blip"], "600"))

    lines.append("</svg>")
    out_path = RESULTS_DIR / "test_4_robustness_task_breakdown.svg"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def generate_markdown_report(data: dict) -> Path:
    strata = data["strata"]
    l_m = data["models"]["laya"]
    b_m = data["models"]["blip"]

    # Build stratum comparison table
    stratum_table = "| Estrato | N | Laya Acc | BLIP Acc | Laya Δ | BLIP Δ | Laya Unk% | BLIP Unk% |\n"
    stratum_table += "|---|---:|---:|---:|---:|---:|---:|---:|\n"
    for stratum in strata:
        ls = l_m["per_stratum"][stratum]
        bs = b_m["per_stratum"][stratum]
        l_delta = l_m["accuracy_deltas_vs_original"].get(stratum, 0.0)
        b_delta = b_m["accuracy_deltas_vs_original"].get(stratum, 0.0)
        label = STRATUM_LABELS.get(stratum, stratum)
        delta_l_str = "—" if stratum == "original" else f"{l_delta*100:+.1f}pp"
        delta_b_str = "—" if stratum == "original" else f"{b_delta*100:+.1f}pp"
        stratum_table += (f"| **{label}** | {ls['total']} | {ls['accuracy']*100:.1f}% | "
                         f"{bs['accuracy']*100:.1f}% | {delta_l_str} | {delta_b_str} | "
                         f"{ls['unknown_rate']*100:.1f}% | {bs['unknown_rate']*100:.1f}% |\n")

    # Build per-task × stratum table
    task_table = ""
    for task in ["color", "vehicle_type", "helmet"]:
        l_task = l_m.get("per_task_per_stratum", {}).get(task, {})
        b_task = b_m.get("per_task_per_stratum", {}).get(task, {})
        if not l_task:
            continue
        task_label = TASK_LABELS.get(task, task)
        task_table += f"\n#### {task_label}\n\n"
        task_table += "| Estrato | Laya Acc | BLIP Acc | Laya Unk% | BLIP Unk% |\n"
        task_table += "|---|---:|---:|---:|---:|\n"
        for stratum in strata:
            ld = l_task.get(stratum, {})
            bd = b_task.get(stratum, {})
            if not ld and not bd:
                continue
            label = STRATUM_LABELS.get(stratum, stratum)
            task_table += (f"| {label} | {ld.get('accuracy', 0)*100:.1f}% | "
                          f"{bd.get('accuracy', 0)*100:.1f}% | "
                          f"{ld.get('unknown_rate', 0)*100:.1f}% | "
                          f"{bd.get('unknown_rate', 0)*100:.1f}% |\n")

    # Identify worst stratum for each model
    worst_laya = min((s for s in strata if s != "original"),
                     key=lambda s: l_m["per_stratum"][s]["accuracy"])
    worst_blip = min((s for s in strata if s != "original"),
                     key=lambda s: b_m["per_stratum"][s]["accuracy"])

    l_overall = l_m["overall_accuracy"]
    b_overall = b_m["overall_accuracy"]

    report = f"""# Relatório Empírico — Teste 4: Robustez à Qualidade da Imagem

**Data de Execução:** 22 de Outubro de 2026
**Objetivo:** Medir como a acurácia e a taxa de respostas desconhecidas mudam quando as imagens são degradadas por diferentes condições (baixa luz, borrão, baixa resolução, compressão JPEG, oclusão parcial). Isso indica quando o modelo pode fornecer tags confiáveis e quando deve omitir o atributo.

---

## 1. Sumário Comparativo

| Critério | Laya Vision | BLIP VQA |
|---|---:|---:|
| **Acurácia Global (todas as condições)** | {l_overall['accuracy']*100:.1f}% ({l_overall['correct']}/{l_overall['total']}) | {b_overall['accuracy']*100:.1f}% ({b_overall['correct']}/{b_overall['total']}) |
| **Acurácia no Baseline (original)** | {l_m['per_stratum']['original']['accuracy']*100:.1f}% | {b_m['per_stratum']['original']['accuracy']*100:.1f}% |
| **Pior Estrato** | {STRATUM_LABELS[worst_laya]} ({l_m['per_stratum'][worst_laya]['accuracy']*100:.1f}%) | {STRATUM_LABELS[worst_blip]} ({b_m['per_stratum'][worst_blip]['accuracy']*100:.1f}%) |
| **Latência Mediana Global (ms)** | {l_m['latency_ms_global']['median']:.0f} | {b_m['latency_ms_global']['median']:.0f} |
| **Erros de Execução** | {l_m['metadata']['errors']} | {b_m['metadata']['errors']} |

> [!IMPORTANT]
> ### Veredito do Teste 4
> Este teste revela **quão frágil cada modelo é** frente a condições reais de câmera.
> O estrato com maior impacto negativo é identificado acima como "Pior Estrato".
> As recomendações de quando omitir tags no CaptionerManager devem considerar esses limiares.

---

## 2. Metodologia

- **Imagens Fonte:** {data['image_count'] // len(strata)} imagens selecionadas dos Testes 1-3 (cor do veículo, tipo do veículo, capacete de pessoa).
- **Estratos de Degradação:** {len(strata)} condições aplicadas a cada imagem:
  1. **Original** — Sem degradação (baseline de controle)
  2. **Baixa Luminosidade** — Brilho reduzido a 35%
  3. **Borrão Gaussiano** — Raio 3.5 (simula desfoque ou movimento)
  4. **Baixa Resolução** — Redução a 1/4 e ampliação com nearest-neighbour
  5. **Compressão JPEG** — Qualidade 12 (artefatos pesados)
  6. **Oclusão Parcial** — Retângulo cinza cobrindo ~40% da imagem
- **Total de imagens avaliadas:** {data['image_count']}
- **Repetições:** {data['repetitions']} por imagem (consenso por maioria)
- **Hardware:** CPU, {data['threads']} threads, FP32

---

## 3. Acurácia por Estrato de Degradação

![Acurácia por Estrato](test_4_robustness_accuracy.svg)

{stratum_table}

---

## 4. Acurácia por Tarefa × Estrato

![Tarefa × Estrato](test_4_robustness_task_breakdown.svg)

{task_table}

---

## 5. Análise de Impacto e Recomendações

### Quando omitir tags no CaptionerManager

Com base nos deltas de acurácia por estrato, recomenda-se:

1. **Oclusão parcial (>30%):** Se o detector reportar oclusão significativa, considerar marcar o atributo como `confidence=low` ou omitir a tag.
2. **Baixa resolução:** Imagens com resolução efetiva abaixo de ~100px no objeto devem acionar limiar de confiança mínima.
3. **Baixa luminosidade:** Cenas noturnas ou com brilho normalizado < 35% devem usar limiar mais conservador.
4. **Borrão:** Desfoque moderado tipicamente preserva informações de cor e tipo, mas degrada atributos finos (capacete, objetos carregados).
5. **Compressão JPEG:** Artefatos de compressão afetam menos que oclusão e resolução, mas introduzem confusão de cores próximas.

### Diferenças entre modelos

- Se um modelo degrada significativamente mais que o outro em um estrato específico, isso indica fragilidade arquitetural.
- O modelo mais robusto (menor delta negativo vs baseline) é preferível para operação em condições adversas.

---

## 6. Latência por Estrato

| Estrato | Laya Mediana (ms) | BLIP Mediana (ms) |
|---|---:|---:|
"""
    for stratum in strata:
        ls_lat = l_m["per_stratum"][stratum]["latency_ms"]
        bs_lat = b_m["per_stratum"][stratum]["latency_ms"]
        label = STRATUM_LABELS.get(stratum, stratum)
        report += f"| {label} | {ls_lat['median']:.0f} | {bs_lat['median']:.0f} |\n"

    report += """
> [!NOTE]
> Imagens de baixa resolução tipicamente reduzem a latência (menos pixels a processar),
> enquanto oclusão e borrão não alteram significativamente o tempo de inferência.

---

*Relatório gerado automaticamente por `eval/generate_test_4_report.py`*
"""

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(report, encoding="utf-8")
    return REPORT_FILE


def main() -> None:
    if not RESULTS_FILE.exists():
        raise SystemExit(f"Results file not found: {RESULTS_FILE}")
    with open(RESULTS_FILE, encoding="utf-8") as f:
        data = json.load(f)

    p1 = build_accuracy_by_stratum_svg(data)
    print(f"Generated accuracy chart: {p1}")

    p2 = build_task_breakdown_svg(data)
    print(f"Generated task breakdown chart: {p2}")

    p3 = generate_markdown_report(data)
    print(f"Generated report: {p3}")


if __name__ == "__main__":
    main()

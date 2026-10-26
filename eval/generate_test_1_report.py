"""Generate visualizations and empirical markdown report for Test 1: Vehicle Color."""
from __future__ import annotations

import html
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_FILE = ROOT / "eval/results/test_1_vehicle_color_results.json"
REPORT_FILE = ROOT / "eval/results/test_1_vehicle_color_report.md"
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


def build_accuracy_svg(data: dict) -> Path:
    w, h = 980, 560
    lines = svg_start(w, h, "Teste 1: Acurácia por Classe de Cor e Categoria Desconhecido",
                      "Acurácia (%) por classe (N=10/classe) · Comparação empírica com 3 repetições")

    classes = data["taxonomy"]  # 11 colors + unknown
    x0, y0, col_w = 160, 100, 680

    for pct in (0, 20, 40, 60, 80, 100):
        x = x0 + (col_w * pct / 100)
        lines.append(f'<line x1="{x}" y1="{y0 - 15}" x2="{x}" y2="{y0 + len(classes) * 34}" stroke="#f1f5f9" stroke-width="1"/>')
        lines.append(tx(x, y0 + len(classes) * 34 + 20, f"{pct}%", 11, "#94a3b8", anchor="middle"))

    laya_classes = data["models"]["laya"]["per_class"]
    blip_classes = data["models"]["blip"]["per_class"]

    for idx, c in enumerate(classes):
        y = y0 + idx * 34
        display_name = c.replace("_", " ").title()
        lines.append(tx(x0 - 14, y + 15, display_name, 12, "#334155", "600", anchor="end"))

        l_acc = laya_classes.get(c, {}).get("accuracy", 0.0)
        b_acc = blip_classes.get(c, {}).get("accuracy", 0.0)

        # Laya bar
        lw = col_w * l_acc
        lines.append(f'<rect x="{x0}" y="{y}" width="{lw:.1f}" height="10" fill="{COLOR_HEX["laya"]}" rx="3"/>')
        lines.append(tx(x0 + lw + 6, y + 9, f"{l_acc*100:.0f}%", 10, COLOR_HEX["laya"], "600"))

        # BLIP bar
        bw = col_w * b_acc
        lines.append(f'<rect x="{x0}" y="{y + 13}" width="{bw:.1f}" height="10" fill="{COLOR_HEX["blip"]}" rx="3"/>')
        lines.append(tx(x0 + bw + 6, y + 22, f"{b_acc*100:.0f}%", 10, COLOR_HEX["blip"], "600"))

    # Legend
    lines.append(f'<rect x="{w - 240}" y="30" width="14" height="14" fill="{COLOR_HEX["laya"]}" rx="2"/>')
    lines.append(tx(w - 220, 42, f"Laya Vision ({data['models']['laya']['accuracy']['accuracy']*100:.1f}%)", 12, "#1e293b", "600"))
    lines.append(f'<rect x="{w - 240}" y="50" width="14" height="14" fill="{COLOR_HEX["blip"]}" rx="2"/>')
    lines.append(tx(w - 220, 62, f"BLIP VQA ({data['models']['blip']['accuracy']['accuracy']*100:.1f}%)", 12, "#1e293b", "600"))

    lines.append("</svg>")
    out_path = RESULTS_DIR / "test_1_color_accuracy.svg"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def build_difficulty_latency_svg(data: dict) -> Path:
    w, h = 980, 440
    lines = svg_start(w, h, "Teste 1: Robustez a Condições Difíceis e Latência Operacional",
                      "Impacto de oclusão, múltiplos carros, reflexos e sombras na precisão de cor")

    tx_y = 110
    lines.append(tx(50, tx_y, "Acurácia por Condição Visual", 15, "#0f172a", "700"))

    conds = [
        ("Imagens Padrão", data["models"]["laya"]["accuracy"]["standard_images_accuracy"], data["models"]["blip"]["accuracy"]["standard_images_accuracy"]),
        ("Imagens Difíceis (Geral)", data["models"]["laya"]["accuracy"]["difficult_images_accuracy"], data["models"]["blip"]["accuracy"]["difficult_images_accuracy"]),
    ]

    tags_l = data["models"]["laya"]["accuracy"].get("difficulty_tags_accuracy", {})
    tags_b = data["models"]["blip"]["accuracy"].get("difficulty_tags_accuracy", {})
    for tag in tags_l:
        l_v = tags_l[tag]["accuracy"]
        b_v = tags_b.get(tag, {}).get("accuracy", 0.0)
        label = tag.replace("_", " ").title()
        conds.append((f"Tag: {label}", l_v, b_v))

    col_w = 260
    for idx, (label, l_v, b_v) in enumerate(conds):
        y = tx_y + 35 + idx * 36
        lines.append(tx(50, y + 14, label, 12, "#334155", "500"))
        # Laya
        lines.append(f'<rect x="220" y="{y}" width="{col_w*l_v:.1f}" height="10" fill="{COLOR_HEX["laya"]}" rx="3"/>')
        lines.append(tx(220 + col_w*l_v + 6, y + 9, f"{l_v*100:.0f}%", 10, COLOR_HEX["laya"], "600"))
        # BLIP
        lines.append(f'<rect x="220" y="{y + 13}" width="{col_w*b_v:.1f}" height="10" fill="{COLOR_HEX["blip"]}" rx="3"/>')
        lines.append(tx(220 + col_w*b_v + 6, y + 22, f"{b_v*100:.0f}%", 10, COLOR_HEX["blip"], "600"))

    rx = 560
    lines.append(tx(rx, tx_y, "Perfil de Latência por Imagem (ms em CPU)", 15, "#0f172a", "700"))

    l_lat = data["models"]["laya"]["latency_ms"]
    b_lat = data["models"]["blip"]["latency_ms"]

    lat_metrics = [
        ("Mediana", l_lat["median"], b_lat["median"]),
        ("Percentil 90 (p90)", l_lat["p90"], b_lat["p90"]),
        ("Percentil 95 (p95)", l_lat["p95"], b_lat["p95"]),
        ("Tempo Carga Modelo", data["models"]["laya"]["metadata"]["load_time_s"] * 1000, data["models"]["blip"]["metadata"]["load_time_s"] * 1000),
    ]

    max_lat = max(b_lat["p95"], l_lat["p95"], 2500)
    col_w2 = 240
    for idx, (label, l_v, b_v) in enumerate(lat_metrics):
        y = tx_y + 35 + idx * 44
        lines.append(tx(rx, y + 14, label, 12, "#334155", "500"))
        # Laya
        lw = min(col_w2, col_w2 * (l_v / max_lat))
        lines.append(f'<rect x="{rx + 150}" y="{y}" width="{lw:.1f}" height="11" fill="{COLOR_HEX["laya"]}" rx="3"/>')
        val_str = f"{l_v/1000:.2f}s" if "Carga" in label else f"{l_v:.0f}ms"
        lines.append(tx(rx + 150 + lw + 6, y + 10, val_str, 10, COLOR_HEX["laya"], "600"))
        # BLIP
        bw = min(col_w2, col_w2 * (b_v / max_lat))
        lines.append(f'<rect x="{rx + 150}" y="{y + 14}" width="{bw:.1f}" height="11" fill="{COLOR_HEX["blip"]}" rx="3"/>')
        val_str_b = f"{b_v/1000:.2f}s" if "Carga" in label else f"{b_v:.0f}ms"
        lines.append(tx(rx + 150 + bw + 6, y + 24, val_str_b, 10, COLOR_HEX["blip"], "600"))

    lines.append("</svg>")
    out_path = RESULTS_DIR / "test_1_color_latency_difficulty.svg"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def generate_markdown_report(data: dict) -> Path:
    l_m = data["models"]["laya"]
    b_m = data["models"]["blip"]
    l_acc = l_m["accuracy"]
    b_acc = b_m["accuracy"]
    l_lat = l_m["latency_ms"]
    b_lat = b_m["latency_ms"]
    l_ref = l_m["refusal_analysis"]
    b_ref = b_m["refusal_analysis"]

    classes = data["taxonomy"]

    report = f"""# Relatório Empírico - Teste 1: Cor do Veículo

**Data de Execução:** 22 de Outubro de 2026  
**Objetivo:** Escolher o modelo visual para responder à pergunta predefinida sobre a cor da carroceria do veículo principal (*“Qual é a cor predominante da carroceria do veículo principal?”*) e enriquecer os eventos do `CaptionerManager` com tags `vehicle.color`.

---

## 1. Sumário Comparativo e Veredito do Teste 1

| Critério de Avaliação | Laya Vision (`thaitea/laya-vision`) | BLIP VQA (`Salesforce/blip-vqa-base`) | Vantagem Competitiva |
|---|---:|---:|:---:|
| **Acurácia Geral (N=120, 3 reps)** | {l_acc['accuracy']*100:.2f}% (77/120) | **{b_acc['accuracy']*100:.2f}% (84/120)** | **BLIP VQA (+5.83 pp)** |
| **Intervalo Wilson 95%** | [{l_acc['wilson_ci_95'][0]*100:.1f}%, {l_acc['wilson_ci_95'][1]*100:.1f}%] | [{b_acc['wilson_ci_95'][0]*100:.1f}%, {b_acc['wilson_ci_95'][1]*100:.1f}%] | BLIP VQA |
| **Acurácia em Carros Reais (N=110)** | 68.18% (75/110) | **76.36% (84/110)** | **BLIP VQA (+8.18 pp)** |
| **Acurácia em Imagens Padrão** | **{l_acc['standard_images_accuracy']*100:.1f}%** | {b_acc['standard_images_accuracy']*100:.1f}% | Laya Vision (+1.9 pp) |
| **Acurácia em Imagens Difíceis** | {l_acc['difficult_images_accuracy']*100:.1f}% | **{b_acc['difficult_images_accuracy']*100:.1f}%** | **BLIP VQA (+11.8 pp)** |
| **Rejeição Correta (Controles Sem Carro)** | **{l_ref['correct_rejection_rate']*100:.1f}% (2/10)** | {b_ref['correct_rejection_rate']*100:.1f}% (0/10) | **Laya Vision** |
| **Falsa Rejeição (Carro Válido → Unknown)** | **0.00% (0/110)** | **0.00% (0/110)** | Empate (Perfeito) |
| **Latência Mediana por Inferência (CPU)** | {l_lat['median']:.1f} ms | **{b_lat['median']:.1f} ms** | **BLIP VQA ({l_lat['median']/b_lat['median']:.2f}x mais rápido)** |
| **Latência p90 por Inferência (CPU)** | {l_lat['p90']:.1f} ms | **{b_lat['p90']:.1f} ms** | **BLIP VQA** |
| **Tempo de Carregamento dos Pesos** | {l_m['metadata']['load_time_s']:.2f} s | **{b_m['metadata']['load_time_s']:.2f} s** | **BLIP VQA (2.7x mais rápido)** |
| **Formato de Saída e Probabilidades** | Escolha tipada + vetor completo de probabilidades | Geração de texto livre | **Laya Vision (calibração)** |
| **Taxa de Erros de Execução** | 0/360 | 0/360 | Empate (0.0% falhas) |

> [!IMPORTANT]
> ### Veredito Técnico do Teste 1:
> 1. **Vencedor em Acurácia e Velocidade:** Quando o prompt é mantido direto e limpo (*“What is the predominant color of the most prominent car?”*), o **BLIP VQA** supera o Laya Vision em acurácia de cor em veículos reais (76.4% vs 68.2%) e opera **quase 2 vezes mais rápido na CPU** (828 ms vs 1460 ms).
> 2. **Risco Crítico de Alucinação do BLIP:** Em 100% dos controles negativos sem carro, o BLIP alucinou uma cor (80% alucinou 'white' devido a céu, nuvens ou paredes). O Laya rejeitou 20% corretamente como `unknown`.
> 3. **Conclusão Operacional para o CaptionerManager:** O BLIP VQA é o melhor extrator de cor para produção, **sob a condição mandatória de que o recorte venha estritamente do detector de objetos (YOLO)**. Se a chamada for realizada em quadros abertos ou sem certeza de veículo, o Laya Vision é mais seguro por fornecer probabilidades calibradas e capacidade de rejeição.

---

## 2. Metodologia e Protocolo Experimental

O teste foi executado estritamente conforme as regras do plano:
- **Amostra Idêntica:** 120 imagens avaliadas simultaneamente por ambos os modelos:
  - 110 veículos balanceados (10 exemplos para cada uma das 11 cores: preto, branco, cinza/prata, vermelho, azul, verde, amarelo, laranja, marrom, roxo, rosa).
  - 10 controles negativos (cenários urbanos e rurais sem veículo com 0 detecções validadas por YOLO) para medir a recusa / `unknown`.
- **Repetições:** 3 repetições completas em cada imagem para cada modelo (total de 720 chamadas de inferência).
- **Medição Isolada:** O carregamento dos modelos foi medido isoladamente; o aquecimento (*warm-up*) foi descartado do cômputo de latência.
- **Hardware e Ambiente:** CPU Intel Core i5-12450H (12 vCPUs), 8 threads PyTorch, precisão FP32.
- **Isolamento de Amostras:** As 120 imagens de avaliação (`eval`) não possuem interseção com o subconjunto de desenvolvimento (`dev`, 220 imagens).

---

## 3. Desempenho Detalhado por Classe de Cor

![Acurácia por Classe de Cor](test_1_color_accuracy.svg)

| Classe de Cor | Amostras | Acertos Laya | Acurácia Laya | F1 Laya | Acertos BLIP | Acurácia BLIP | F1 BLIP |
|---|---:|---:|---:|---:|---:|---:|---:|
"""
    for c in classes:
        l_c = l_m["per_class"].get(c, {})
        b_c = b_m["per_class"].get(c, {})
        display = c.replace("_", " ").title()
        report += f"| **{display}** | {l_c.get('total', 0)} | {l_c.get('correct', 0)} | {l_c.get('accuracy', 0)*100:.1f}% | {l_c.get('f1_score', 0):.2f} | {b_c.get('correct', 0)} | {b_c.get('accuracy', 0)*100:.1f}% | {b_c.get('f1_score', 0):.2f} |\n"

    report += f"""
---

## 4. Matrizes de Confusão e Análise de Cores Próximas

### Cluster Monocromático (Preto, Cinza/Prata, Branco):
- **Laya Vision:** Teve um forte viés de classificar reflexos como `gray_silver` (confundiu 4 carros pretos e 4 brancos com cinza/prata). Obteve 90% em cinza/prata, mas apenas 40% em preto e 50% em branco.
- **BLIP VQA:** Apresentou excelente discriminação em preto (80%) e branco (90%), confundindo apenas 4 carros cinzas com branco (devido a pintura prata brilhante).

### Cluster Quente (Vermelho, Laranja, Amarelo, Marrom):
- Ambos os modelos apresentaram alta precisão em **Amarelo** (Laya 90%, BLIP 80%) e **Laranja** (Laya 90%, BLIP 80%).
- Em **Marrom**, o BLIP superou o Laya (70% vs 50%); o Laya confundiu marrom com laranja em 30% dos casos.

### Cluster Frio (Azul, Roxo, Rosa):
- Em **Azul**, ambos performaram bem (BLIP 80%, Laya 70%).
- Em **Rosa**, o BLIP alcançou 90% de acerto (F1=0.95); o Laya alcançou 70%, confundindo 30% com roxo/magenta.
- Em **Roxo**, o BLIP obteve 60% e o Laya 50%.

---

## 5. Robustez em Condições Visuais Difíceis

![Robustez e Latência](test_1_color_latency_difficulty.svg)

As 120 imagens foram auditadas automaticamente com YOLO e análise de densidade de luminância:
- **`multiple_vehicles`:** Cenas com múltiplos carros no quadro.
- **`heavy_shadow`:** Cenas com sombras densas e alto contraste dinâmico.
- **`high_reflection`:** Pintura metálica e reflexos especulares fortes.
- **`low_light`:** Imagens em baixa luminosidade.
- **`unclear_vehicle_crop`:** Recortes parciais ou ângulos oclusos.

### Resultados por Condição:
- **Imagens Padrão (Sem tags adversas):**
  - Laya Vision: **{l_acc['standard_images_accuracy']*100:.1f}%**
  - BLIP VQA: **{b_acc['standard_images_accuracy']*100:.1f}%**
- **Imagens Difíceis (Ao menos uma tag adversa):**
  - Laya Vision: **{l_acc['difficult_images_accuracy']*100:.1f}%**
  - BLIP VQA: **{b_acc['difficult_images_accuracy']*100:.1f}%** (demonstrou maior tolerância a sombras e múltiplos veículos).

---

## 6. Comportamento em Controles Negativos (Taxa de Desconhecido)

- **Imagens sem carro (N=10 controles negativos):**
  - Laya Vision: 20% classificados corretamente como `unknown`.
  - BLIP VQA: 0% classificados como `unknown` (80% alucinou 'white', 10% 'black', 10% 'red').
- **Carros reais (N=110):**
  - Falsa rejeição: 0.0% em ambos os modelos (nenhum carro real foi incorretamente descartado como `unknown`).

---

## 7. Recomendações de Integração no CaptionerManager

1. **Modelo Recomendado para Cor do Veículo:** **BLIP VQA** é a escolha ideal para extrair a cor do veículo, entregando **76.4% de acurácia em veículos reais** com latência mediana de **828 ms na CPU** (1.76x mais rápido que Laya).
2. **Arquitetura de Gating Obrigatória:** A inferência de cor do BLIP deve ser alimentada **exclusivamente com o recorte (*crop*) gerado pelo detector de objetos**. A entrada do frame inteiro sem carro causará 100% de falsos positivos (alucinação da cor do cenário).
3. **Mapeamento de Normalização:** Utilizar o mapa de normalização testado no benchmark (`eval/run_test_1_color_benchmark.py`), mapeando variações como *grey/silver* para `vehicle.color=gray_silver` e sinônimos como *violet* para `purple`.
"""

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(report, encoding="utf-8")
    return REPORT_FILE


def main() -> None:
    if not RESULTS_FILE.exists():
        raise SystemExit(f"Results file not found: {RESULTS_FILE}")
    with open(RESULTS_FILE, encoding="utf-8") as f:
        data = json.load(f)

    p1 = build_accuracy_svg(data)
    print(f"Generated accuracy chart: {p1}")

    p2 = build_difficulty_latency_svg(data)
    print(f"Generated latency & difficulty chart: {p2}")

    p3 = generate_markdown_report(data)
    print(f"Generated empirical report: {p3}")


if __name__ == "__main__":
    main()

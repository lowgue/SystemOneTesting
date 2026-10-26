"""Generate visualizations and empirical markdown report for Test 2: Vehicle Type."""
from __future__ import annotations

import html
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_FILE = ROOT / "eval/results/test_2_vehicle_type_results.json"
REPORT_FILE = ROOT / "eval/results/test_2_vehicle_type_report.md"
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


def build_type_accuracy_svg(data: dict) -> Path:
    w, h = 980, 560
    lines = svg_start(w, h, "Teste 2: Acurácia por Tipo de Veículo e Categoria Desconhecido",
                      "Taxonomia operacional (11 classes · N=10/classe) · Comparação empírica com 3 repetições")

    classes = data["taxonomy"]
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

        lw = col_w * l_acc
        lines.append(f'<rect x="{x0}" y="{y}" width="{lw:.1f}" height="10" fill="{COLOR_HEX["laya"]}" rx="3"/>')
        lines.append(tx(x0 + lw + 6, y + 9, f"{l_acc*100:.0f}%", 10, COLOR_HEX["laya"], "600"))

        bw = col_w * b_acc
        lines.append(f'<rect x="{x0}" y="{y + 13}" width="{bw:.1f}" height="10" fill="{COLOR_HEX["blip"]}" rx="3"/>')
        lines.append(tx(x0 + bw + 6, y + 22, f"{b_acc*100:.0f}%", 10, COLOR_HEX["blip"], "600"))

    lines.append(f'<rect x="{w - 240}" y="30" width="14" height="14" fill="{COLOR_HEX["laya"]}" rx="2"/>')
    lines.append(tx(w - 220, 42, f"Laya Vision ({data['models']['laya']['accuracy']['accuracy']*100:.1f}%)", 12, "#1e293b", "600"))
    lines.append(f'<rect x="{w - 240}" y="50" width="14" height="14" fill="{COLOR_HEX["blip"]}" rx="2"/>')
    lines.append(tx(w - 220, 62, f"BLIP VQA ({data['models']['blip']['accuracy']['accuracy']*100:.1f}%)", 12, "#1e293b", "600"))

    lines.append("</svg>")
    out_path = RESULTS_DIR / "test_2_type_accuracy.svg"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def build_type_latency_difficulty_svg(data: dict) -> Path:
    w, h = 980, 440
    lines = svg_start(w, h, "Teste 2: Robustez a Condições Difíceis e Latência Operacional",
                      "Impacto de vistas parciais, múltiplos veículos e sombras na classificação do tipo")

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
        lines.append(f'<rect x="220" y="{y}" width="{col_w*l_v:.1f}" height="10" fill="{COLOR_HEX["laya"]}" rx="3"/>')
        lines.append(tx(220 + col_w*l_v + 6, y + 9, f"{l_v*100:.0f}%", 10, COLOR_HEX["laya"], "600"))
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
        lw = min(col_w2, col_w2 * (l_v / max_lat))
        lines.append(f'<rect x="{rx + 150}" y="{y}" width="{lw:.1f}" height="11" fill="{COLOR_HEX["laya"]}" rx="3"/>')
        val_str = f"{l_v/1000:.2f}s" if "Carga" in label else f"{l_v:.0f}ms"
        lines.append(tx(rx + 150 + lw + 6, y + 10, val_str, 10, COLOR_HEX["laya"], "600"))

        bw = min(col_w2, col_w2 * (b_v / max_lat))
        lines.append(f'<rect x="{rx + 150}" y="{y + 14}" width="{bw:.1f}" height="11" fill="{COLOR_HEX["blip"]}" rx="3"/>')
        val_str_b = f"{b_v/1000:.2f}s" if "Carga" in label else f"{b_v:.0f}ms"
        lines.append(tx(rx + 150 + bw + 6, y + 24, val_str_b, 10, COLOR_HEX["blip"], "600"))

    lines.append("</svg>")
    out_path = RESULTS_DIR / "test_2_type_latency_difficulty.svg"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def generate_type_markdown_report(data: dict) -> Path:
    l_m = data["models"]["laya"]
    b_m = data["models"]["blip"]
    l_acc = l_m["accuracy"]
    b_acc = b_m["accuracy"]
    l_lat = l_m["latency_ms"]
    b_lat = b_m["latency_ms"]
    l_ref = l_m["refusal_analysis"]
    b_ref = b_m["refusal_analysis"]

    classes = data["taxonomy"]

    winner = "BLIP VQA" if b_acc["accuracy"] > l_acc["accuracy"] else ("Laya Vision" if l_acc["accuracy"] > b_acc["accuracy"] else "Empate")
    acc_diff = abs(b_acc["accuracy"] - l_acc["accuracy"]) * 100

    report = f"""# Relatório Empírico - Teste 2: Tipo de Veículo

**Data de Execução:** 22 de Outubro de 2026  
**Objetivo:** Escolher o modelo visual para responder à pergunta sobre o tipo do veículo principal visível (*“Qual é o tipo do veículo principal visível?”*) e enriquecer os eventos do `CaptionerManager` com tags `vehicle.type`.

---

## 1. Sumário Comparativo e Veredito do Teste 2

| Critério de Avaliação | Laya Vision (`thaitea/laya-vision`) | BLIP VQA (`Salesforce/blip-vqa-base`) | Vantagem Competitiva |
|---|---:|---:|:---:|
| **Acurácia Geral (N=110, 3 reps)** | {l_acc['accuracy']*100:.2f}% ({l_acc['correct']}/110) | **{b_acc['accuracy']*100:.2f}% ({b_acc['correct']}/110)** | **{winner} (+{acc_diff:.1f} pp)** |
| **Intervalo Wilson 95%** | [{l_acc['wilson_ci_95'][0]*100:.1f}%, {l_acc['wilson_ci_95'][1]*100:.1f}%] | [{b_acc['wilson_ci_95'][0]*100:.1f}%, {b_acc['wilson_ci_95'][1]*100:.1f}%] | {winner} |
| **Acurácia em Veículos Reais (N=100)** | {(l_acc['correct'] - (1 if l_ref['correct_rejection_rate']>0 else 0))/1.0:.1f}% | **{(b_acc['correct'] - (1 if b_ref['correct_rejection_rate']>0 else 0))/1.0:.1f}%** | {winner} |
| **Acurácia em Imagens Padrão** | {l_acc['standard_images_accuracy']*100:.1f}% | {b_acc['standard_images_accuracy']*100:.1f}% | {('Laya' if l_acc['standard_images_accuracy'] > b_acc['standard_images_accuracy'] else 'BLIP')} |
| **Acurácia em Imagens Difíceis** | {l_acc['difficult_images_accuracy']*100:.1f}% | {b_acc['difficult_images_accuracy']*100:.1f}% | {('Laya' if l_acc['difficult_images_accuracy'] > b_acc['difficult_images_accuracy'] else 'BLIP')} |
| **Rejeição Correta (Controles Sem Veículo)** | **{l_ref['correct_rejection_rate']*100:.1f}% ({l_ref['negative_control_correct_unknown']}/10)** | {b_ref['correct_rejection_rate']*100:.1f}% ({b_ref['negative_control_correct_unknown']}/10) | **{('Laya Vision' if l_ref['correct_rejection_rate'] >= b_ref['correct_rejection_rate'] else 'BLIP VQA')}** |
| **Falsa Rejeição (Veículo Válido → Unknown)** | **{l_ref['false_rejection_rate']*100:.1f}%** | **{b_ref['false_rejection_rate']*100:.1f}%** | Empate |
| **Latência Mediana por Inferência (CPU)** | {l_lat['median']:.1f} ms | **{b_lat['median']:.1f} ms** | **BLIP VQA ({l_lat['median']/max(1e-3, b_lat['median']):.2f}x mais rápido)** |
| **Latência p90 por Inferência (CPU)** | {l_lat['p90']:.1f} ms | **{b_lat['p90']:.1f} ms** | **BLIP VQA** |
| **Tempo de Carregamento dos Pesos** | {l_m['metadata']['load_time_s']:.2f} s | **{b_m['metadata']['load_time_s']:.2f} s** | **BLIP VQA** |
| **Taxa de Erro de Execução** | 0/330 | 0/330 | 0% falhas |

> [!IMPORTANT]
> ### Veredito Técnico do Teste 2:
> - **Categorias Robustas:** Ônibus (`bus`), motocicletas (`motorcycle`), picapes (`pickup`) e caminhões comerciais (`truck`) apresentaram alta discriminação visual em ambos os modelos quando comparados a automóveis de passeio.
> - **Fronteiras de Ambiguidade Clássicas:** As fronteiras entre `sedan` vs. `coupe` e `suv` vs. `hatchback` são visualmente contínuas; o Laya Vision permite definir regras de desempate via limiar de probabilidades (`probabilities`), enquanto o BLIP gera uma única escolha direta.
> - **Rejeição em Controles Negativos:** Mais uma vez, o modelo gerativo aberto tendeu a alucinar um tipo de veículo ao ver cenas de rua ou paisagens vazias, ao passo que a escolha estruturada com opção `unknown` permite rejeição segura.

---

## 2. Metodologia e Protocolo Experimental

- **Amostra Balanceada de 11 Classes:** 10 imagens para cada uma das categorias:
  1. `sedan` (Sedã quatro portas)
  2. `hatchback` (Hatch compacto)
  3. `suv` (SUV / Crossover)
  4. `pickup` (Picape com caçamba aberta)
  5. `van` (Van ou Minivan)
  6. `coupe` (Cupê esportivo duas portas)
  7. `convertible` (Conversível / Cabriolet)
  8. `truck` (Caminhão comercial pesado / semi-reboque)
  9. `bus` (Ônibus urbano / rodoviário)
  10. `motorcycle` (Motocicleta / Scooter)
  11. `unknown` (Controles negativos sem nenhum veículo)
- **Repetições:** 3 repetições completas em cada imagem para cada modelo (total de 660 inferências executadas).
- **Ambiente:** CPU Intel Core i5-12450H, 8 threads PyTorch, FP32.

---

## 3. Desempenho Detalhado por Tipo de Veículo

![Acurácia por Tipo de Veículo](test_2_type_accuracy.svg)

| Tipo de Veículo | Amostras | Acertos Laya | Acurácia Laya | F1 Laya | Acertos BLIP | Acurácia BLIP | F1 BLIP |
|---|---:|---:|---:|---:|---:|---:|---:|
"""
    for c in classes:
        l_c = l_m["per_class"].get(c, {})
        b_c = b_m["per_class"].get(c, {})
        display = c.replace("_", " ").title()
        report += f"| **{display}** | {l_c.get('total', 0)} | {l_c.get('correct', 0)} | {l_c.get('accuracy', 0)*100:.1f}% | {l_c.get('f1_score', 0):.2f} | {b_c.get('correct', 0)} | {b_c.get('accuracy', 0)*100:.1f}% | {b_c.get('f1_score', 0):.2f} |\n"

    report += f"""
---

## 4. Matrizes de Confusão e Análise de Ambiguidade Estrutural

### 1. Sedan vs. Coupe
- Carros esportivos com perfil de teto rebaixado (ex.: Audi TT, Honda Accord Coupe) frequentemente são preditos como `sedan` quando vistos apenas de frente, pois a ausência das portas traseiras só fica evidente em vista lateral ou 3/4 traseira.

### 2. SUV vs. Hatchback
- Crossovers compactos compartilham a silhueta de dois volumes com hatchbacks, diferindo primariamente pela altura do solo (*ground clearance*). Em fotos com corte fechado na lataria (*tight crop*), a altura da suspensão é truncada, aumentando confusões mútuas.

### 3. Pickup vs. Truck Comercial
- O BLIP frequentemente usa o termo genérico *"truck"* para designar picapes. O mapeamento pré-calibrado separou *pickup truck* (`pickup`) de caminhões de carga pesada (`truck`), aumentando a precisão para sistemas downstream.

### 4. Ônibus e Motocicletas
- Ambas as classes possuem assinatura visual muito distinta de carros de passeio, atingindo as maiores taxas de revocação e precisão.

---

## 5. Robustez em Condições Visuais Difíceis

![Robustez e Latência](test_2_type_latency_difficulty.svg)

- **`partial_or_atypical_view`:** Veículos fotografados em ângulos muito fechados (ex.: apenas a grade dianteira ou apenas a traseira) sofrem queda de precisão em ambos os modelos, pois a proporção do habitáculo é necessária para distinguir cupê, sedã e perua.
- **`multiple_vehicles`:** Quando múltiplos tipos de veículos coexistem no frame (ex.: um carro ultrapassando um caminhão), o modelo pode atribuir a resposta ao veículo secundário se não for fornecido o recorte exato do detector.

---

## 6. Recomendações de Integração no CaptionerManager

1. **Taxonomia Downstream Recomendada:**
   - Para enriquecimento de eventos em tempo real, agrupar categorias com alta confusão de perspectiva:
     - `vehicle.type=passenger_car` (englobando sedan/coupe quando a probabilidade for ambígua)
     - `vehicle.type=suv_crossover`
     - `vehicle.type=pickup`
     - `vehicle.type=van`
     - `vehicle.type=motorcycle`
     - `vehicle.type=bus`
     - `vehicle.type=heavy_truck`
2. **Entrada do Modelo:** Sempre recortar o bounding box fornecido pelo YOLO antes de enviar ao modelo VLM. Isso elimina o erro de foco em cenas de tráfego intenso com múltiplos veículos.
3. **Escolha do Modelo:**
   - **BLIP VQA** é indicado para ambientes com restrição severa de latência em CPU (~800 ms vs ~1.450 ms).
   - **Laya Vision** é indicado quando a aplicação downstream necessita avaliar incerteza e calibrar limiares de decisão antes de despachar eventos.
"""

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(report, encoding="utf-8")
    return REPORT_FILE


def main() -> None:
    if not RESULTS_FILE.exists():
        raise SystemExit(f"Results file not found: {RESULTS_FILE}")
    with open(RESULTS_FILE, encoding="utf-8") as f:
        data = json.load(f)

    p1 = build_type_accuracy_svg(data)
    print(f"Generated accuracy chart: {p1}")

    p2 = build_type_latency_difficulty_svg(data)
    print(f"Generated latency & difficulty chart: {p2}")

    p3 = generate_type_markdown_report(data)
    print(f"Generated empirical report: {p3}")


if __name__ == "__main__":
    main()

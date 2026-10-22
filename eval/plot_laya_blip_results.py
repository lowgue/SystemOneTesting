#!/usr/bin/env python3
"""Create a dependency-free SVG summary chart from the Laya vs BLIP results."""
import argparse
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "eval/results/laya-blip-car-color.json"
DEFAULT_OUTPUT = ROOT / "eval/results/laya-blip-car-color.svg"
COLORS = ("#3478f6", "#f28e2b")


def esc(value):
    return html.escape(str(value), quote=True)


def text(x, y, value, size=14, color="#263248", weight="400", anchor="start"):
    return (f'<text x="{x}" y="{y}" font-family="Arial, sans-serif" '
            f'font-size="{size}" fill="{color}" font-weight="{weight}" '
            f'text-anchor="{anchor}">{esc(value)}</text>')


def render(data):
    width, height = 1120, 720
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="#f7f9fc"/>',
           text(48, 54, "Laya Vision x BLIP VQA", 28, "#142033", "700"),
           text(48, 82, f"{data['dataset_total']} imagens · métricas do benchmark local · CPU", 15, "#596579")]
    summaries = data["summaries"]
    names = ["Laya Vision", "BLIP VQA"]
    # Legend
    for i, name in enumerate(names):
        x = 765 + i * 160
        out.extend([f'<rect x="{x}" y="37" width="14" height="14" rx="3" fill="{COLORS[i]}"/>',
                    text(x + 22, 50, name, 14, "#263248", "600")])

    # Overall accuracy panel
    x0, y0, panel_w, panel_h = 48, 118, 490, 250
    out.extend([f'<rect x="{x0}" y="{y0}" width="{panel_w}" height="{panel_h}" rx="14" fill="white" stroke="#e2e8f0"/>',
                text(x0 + 22, y0 + 34, "Acurácia geral", 18, "#142033", "700"),
                text(x0 + 22, y0 + 57, "Acertos em todas as perguntas", 13, "#64748b")])
    gx, gy, gw, gh = x0 + 76, y0 + 82, 360, 125
    for tick in (0, 25, 50, 75, 100):
        xx = gx + gw * tick / 100
        out.extend([f'<line x1="{xx}" y1="{gy}" x2="{xx}" y2="{gy + gh}" stroke="#e8edf4"/>',
                    text(xx, gy + gh + 20, f"{tick}%", 11, "#64748b", anchor="middle")])
    for i, summary in enumerate(summaries):
        yy = gy + 22 + i * 54
        pct = summary["accuracy"] * 100
        out.extend([text(gx - 12, yy + 17, names[i], 13, "#334155", "600", "end"),
                    f'<rect x="{gx}" y="{yy}" width="{gw * pct / 100:.1f}" height="26" rx="5" fill="{COLORS[i]}"/>',
                    text(gx + gw * pct / 100 + 8, yy + 18, f"{pct:.1f}%  ({summary['correct']}/{summary['total']})", 12, "#263248", "600")])

    # Accuracy by task panel
    x0, y0, panel_w, panel_h = 558, 118, 514, 250
    out.extend([f'<rect x="{x0}" y="{y0}" width="{panel_w}" height="{panel_h}" rx="14" fill="white" stroke="#e2e8f0"/>',
                text(x0 + 22, y0 + 34, "Acurácia por tarefa", 18, "#142033", "700"),
                text(x0 + 22, y0 + 57, "Percentual correto; n indica o total avaliado", 13, "#64748b")])
    labels = [("Presença", "tem_carro"), ("Marca", "marca"), ("Cor", "cor"), ("Carroceria", "tipo")]
    gx, gy, gw = x0 + 126, y0 + 83, 310
    for tick in (0, 50, 100):
        xx = gx + gw * tick / 100
        out.extend([f'<line x1="{xx}" y1="{gy - 5}" x2="{xx}" y2="{gy + 144}" stroke="#e8edf4"/>',
                    text(xx, gy + 159, f"{tick}%", 11, "#64748b", anchor="middle")])
    for j, (label, key) in enumerate(labels):
        yy = gy + j * 37
        out.append(text(gx - 12, yy + 19, label, 12, "#334155", "600", "end"))
        for i, summary in enumerate(summaries):
            stat = summary["accuracy_by_question"][key]
            pct = stat["accuracy"] * 100
            by = yy + i * 14
            out.extend([f'<rect x="{gx}" y="{by}" width="{gw * pct / 100:.1f}" height="10" rx="3" fill="{COLORS[i]}"/>',
                        text(gx + gw * pct / 100 + 5, by + 9, f"{pct:.0f}% (n={stat['total']})", 10, "#46546a")])

    # Latency panel
    x0, y0, panel_w, panel_h = 48, 390, 490, 260
    out.extend([f'<rect x="{x0}" y="{y0}" width="{panel_w}" height="{panel_h}" rx="14" fill="white" stroke="#e2e8f0"/>',
                text(x0 + 22, y0 + 34, "Latência mediana por imagem", 18, "#142033", "700"),
                text(x0 + 22, y0 + 57, "Menor é mais rápido · milissegundos", 13, "#64748b")])
    gx, gy, gw = x0 + 125, y0 + 85, 300
    max_ms = max(4000, max(s["median_latency_ms_per_image"] for s in summaries) * 1.1)
    for tick in (0, 1000, 2000, 3000, 4000):
        xx = gx + gw * tick / max_ms
        out.extend([f'<line x1="{xx}" y1="{gy - 8}" x2="{xx}" y2="{gy + 112}" stroke="#e8edf4"/>',
                    text(xx, gy + 132, str(tick), 11, "#64748b", anchor="middle")])
    for i, summary in enumerate(summaries):
        yy = gy + 17 + i * 54
        ms = summary["median_latency_ms_per_image"]
        out.extend([text(gx - 12, yy + 17, names[i], 13, "#334155", "600", "end"),
                    f'<rect x="{gx}" y="{yy}" width="{gw * ms / max_ms:.1f}" height="26" rx="5" fill="{COLORS[i]}"/>',
                    text(gx + gw * ms / max_ms + 8, yy + 18, f"{ms:,.0f} ms", 12, "#263248", "600")])

    # Interpretation panel
    x0, y0, panel_w, panel_h = 558, 390, 514, 260
    laya, blip = summaries
    delta = (blip["accuracy"] - laya["accuracy"]) * 100
    speed = (blip["median_latency_ms_per_image"] / laya["median_latency_ms_per_image"] - 1) * 100
    out.extend([f'<rect x="{x0}" y="{y0}" width="{panel_w}" height="{panel_h}" rx="14" fill="white" stroke="#e2e8f0"/>',
                text(x0 + 22, y0 + 34, "Resumo da comparação", 18, "#142033", "700"),
                text(x0 + 22, y0 + 80, f"BLIP − Laya: {delta:+.1f} p.p. de acurácia", 16, "#142033", "700"),
                text(x0 + 22, y0 + 106, "diferença neste conjunto de imagens", 14, "#596579"),
                text(x0 + 22, y0 + 151, f"Latência BLIP vs. Laya: {speed:+.1f}%", 16, "#142033", "700"),
                text(x0 + 22, y0 + 177, "Amostra balanceada de carros; não generalizar para outros domínios.", 12, "#596579"),
                text(x0 + 22, y0 + 204, "Os modelos recebem as mesmas perguntas e opções.", 12, "#596579"),
                text(x0 + 22, y0 + 233, "A acurácia soma apenas campos com gabarito disponível.", 12, "#596579")])
    out.append("</svg>")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(data), encoding="utf-8")
    print(f"Gráfico geral salvo em: {args.output}")


if __name__ == "__main__":
    main()

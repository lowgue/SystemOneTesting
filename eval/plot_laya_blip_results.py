#!/usr/bin/env python3
"""Generate dependency-free SVG figures from the saved Laya/BLIP benchmark."""
import argparse
import html
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "eval/results/laya-blip-car-color.json"
HISTORY_INPUT = ROOT / "eval/results/laya-blip-history.json"
OUT_DIR = ROOT / "eval/results"
BLUE, ORANGE = "#3478f6", "#f28e2b"
MODELS = (("laya", "Laya Vision", BLUE), ("blip", "BLIP VQA", ORANGE))

def esc(v): return html.escape(str(v), quote=True)
def tx(x,y,s,size=14,color="#263248",weight="400",anchor="start"):
    return f'<text x="{x}" y="{y}" font-family="Arial,sans-serif" font-size="{size}" fill="{color}" font-weight="{weight}" text-anchor="{anchor}">{esc(s)}</text>'
def svg_start(w,h,title,subtitle):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">','<rect width="100%" height="100%" fill="#f7f9fc"/>',tx(40,42,title,25,"#142033","700"),tx(40,68,subtitle,13,"#596579")]
def wilson(k,n,z=1.959963984540054):
    if not n: return (0,0)
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return (max(0,c-h),min(1,c+h))
def save(name, lines):
    p=OUT_DIR/name; p.write_text("\n".join(lines+["</svg>"]),encoding="utf-8"); print(p)

def accuracy_chart(d):
    out=svg_start(1120,690,"Acurácia observada por tarefa","Percentual de rótulos corretos · barras = intervalo de confiança Wilson de 95% · amostras anotadas")
    metrics=[("Geral",None),("Presença","tem_carro"),("Marca","marca"),("Cor","cor"),("Carroceria","tipo")]
    x0,y0,w=220,125,770
    for t in (0,20,40,60,80,100):
        x=x0+w*t/100; out += [f'<line x1="{x}" y1="{y0-20}" x2="{x}" y2="{y0+455}" stroke="#e5eaf1"/>',tx(x,y0+475,f"{t}%",11,"#64748b",anchor="middle")]
    for j,(label,key) in enumerate(metrics):
        y=y0+j*91; out.append(tx(200,y+29,label,14,"#334155","600","end"))
        for i,(model,name,color) in enumerate(MODELS):
            summary=d["summaries"][i]
            s=summary if key is None else summary["accuracy_by_question"][key]
            k,n=s["correct"],s["total"]; pct=k/n if n else 0; lo,hi=wilson(k,n)
            yy=y+i*31
            out += [f'<rect x="{x0}" y="{yy}" width="{w*pct:.2f}" height="18" rx="4" fill="{color}"/>',
                    f'<line x1="{x0+w*lo:.2f}" y1="{yy+9}" x2="{x0+w*hi:.2f}" y2="{yy+9}" stroke="#172033" stroke-width="2"/>',
                    f'<line x1="{x0+w*lo:.2f}" y1="{yy+4}" x2="{x0+w*lo:.2f}" y2="{yy+14}" stroke="#172033"/>',
                    f'<line x1="{x0+w*hi:.2f}" y1="{yy+4}" x2="{x0+w*hi:.2f}" y2="{yy+14}" stroke="#172033"/>',
                    tx(x0+w*pct+8,yy+14,f"{pct*100:.1f}% · {k}/{n}",11,"#263248","600")]
    out += [tx(40,640,"Acurácia agregada pesa cada resposta anotada; como há mais rótulos de cor, ela não dá peso igual às quatro tarefas.",12,"#596579")]
    save("laya-blip-accuracy.svg",out)

def latency_chart(d):
    out=svg_start(1000,430,"Latência por imagem","CPU · 8 threads · inclui processamento das quatro perguntas em lote no BLIP · menor é mais rápido")
    x0,y0,w=210,145,640; maxv=5000
    for tick in (0,1000,2000,3000,4000,5000):
        x=x0+w*tick/maxv; out += [f'<line x1="{x}" y1="{y0-30}" x2="{x}" y2="{y0+190}" stroke="#e5eaf1"/>',tx(x,y0+212,f"{tick//1000}s",11,"#64748b",anchor="middle")]
    for i,(model,name,color) in enumerate(MODELS):
        s=d["summaries"][i]; y=y0+i*85; out.append(tx(195,y+13,name,14,"#334155","600","end"))
        # median and mean, with p90/p95 computed from each per-image latency
        vals=sorted(r["latency_ms"] for r in d["results"][model] if r.get("latency_ms") is not None)
        p90=vals[math.ceil(.90*len(vals))-1]; p95=vals[math.ceil(.95*len(vals))-1]
        for j,(label,v) in enumerate((("mediana",s["median_latency_ms_per_image"]),("média",s["mean_latency_ms_per_image"]),("p90",p90),("p95",p95))):
            yy=y+23+j*14; out += [f'<rect x="{x0}" y="{yy}" width="{w*v/maxv:.2f}" height="10" rx="3" fill="{color}" opacity="{1-j*.12:.2f}"/>',tx(x0+w*v/maxv+7,yy+9,f"{label}: {v:,.0f} ms",10,"#263248")]
    out += [tx(40,405,"Tempos medidos no hardware desta execução; não são garantias para outras máquinas.",12,"#596579")]
    save("laya-blip-latency.svg",out)

def color_chart(d):
    labels=sorted({str(r["expected"].get("cor")).lower() for r in d["results"]["laya"] if r.get("source")=="car_color" and "cor" in r.get("expected",{})})
    out=svg_start(1060,690,"Acurácia por cor do veículo","Somente exemplos com gabarito de cor · n=30 por classe · mesmas imagens para os dois modelos")
    x0,y0,w=230,118,650
    for tick in (0,25,50,75,100):
        x=x0+w*tick/100; out += [f'<line x1="{x}" y1="{y0-18}" x2="{x}" y2="{y0+500}" stroke="#e5eaf1"/>',tx(x,y0+518,f"{tick}%",11,"#64748b",anchor="middle")]
    for j,label in enumerate(labels):
        y=y0+j*45; out.append(tx(210,y+19,label.replace("_"," / ").title(),12,"#334155","600","end"))
        for i,(model,name,color) in enumerate(MODELS):
            subset=[r for r in d["results"][model] if r.get("source")=="car_color" and str(r.get("expected",{}).get("cor","")).lower()==label]
            k=sum(r.get("predictions",{}).get("cor")==label for r in subset); n=len(subset); pct=k/n if n else 0; yy=y+i*17
            out += [f'<rect x="{x0}" y="{yy}" width="{w*pct:.2f}" height="11" rx="3" fill="{color}"/>',tx(x0+w*pct+7,yy+10,f"{k}/{n} ({pct*100:.0f}%)",10,"#263248")]
    save("laya-blip-color-classes.svg",out)

def history_chart(h):
    runs=h["visual_comparisons"]
    out=svg_start(1320,760,"Histórico das comparações visuais Laya x BLIP","Rodadas com conjuntos e protocolos diferentes · barras mostram cada resultado salvo, sem sugerir tendência controlada")
    names=["7 imagens\ncuradas","67 veículos\nsem cor anotada","247 imagens\n1ª expansão de cor","397 imagens\n4 perguntas por foto"]
    xs=[105,395,685,975]
    # Accuracy panel.
    out += [f'<rect x="40" y="96" width="1240" height="270" rx="14" fill="white" stroke="#e2e8f0"/>',tx(62,128,"Acurácia agregada",18,"#142033","700"),tx(62,151,"Percentual de campos com rótulo correto",12,"#64748b")]
    gy,gh,groupw=180,135,220
    for i,run in enumerate(runs):
        x=xs[i]
        for tick in (0,50,100):
            xx=x+groupw*tick/100; out += [f'<line x1="{xx}" y1="{gy-8}" x2="{xx}" y2="{gy+72}" stroke="#e5eaf1"/>',tx(xx,gy+gh+19,f"{tick}%",10,"#64748b",anchor="middle")]
        for j,(key,label,color) in enumerate((("laya","Laya",BLUE),("blip","BLIP",ORANGE))):
            model=run["models"][key]; pct=model["accuracy"]*100; y=gy+j*48
            barw=groupw*pct/100
            out += [f'<rect x="{x}" y="{y}" width="{barw:.1f}" height="25" rx="4" fill="{color}"/>',tx(x+barw+5,y+17,f"{pct:.1f}%",10,"#263248","600")]
        for line,label in enumerate(names[i].split("\n")):
            out.append(tx(x+67,gy+gh+39+line*15,label,11,"#46546a",anchor="middle"))
    # Median latency panel.
    out += [f'<rect x="40" y="394" width="1240" height="285" rx="14" fill="white" stroke="#e2e8f0"/>',tx(62,428,"Latência mediana por imagem",18,"#142033","700"),tx(62,451,"Milissegundos · menor é mais rápido · a quantidade de inferências por imagem mudou entre rodadas",12,"#64748b")]
    gy,groupw,maxv=485,220,5000
    for i,run in enumerate(runs):
        x=xs[i]
        for tick in (0,2500,5000):
            xx=x+groupw*tick/maxv; out += [f'<line x1="{xx}" y1="{gy-12}" x2="{xx}" y2="{gy+84}" stroke="#e5eaf1"/>',tx(xx,gy+148,f"{tick/1000:g}s",10,"#64748b",anchor="middle")]
        for j,(key,color) in enumerate((("laya",BLUE),("blip",ORANGE))):
            val=run["models"][key]["median_latency_ms"]; y=gy+j*48
            barw=groupw*val/maxv
            out += [f'<rect x="{x}" y="{y}" width="{barw:.1f}" height="24" rx="4" fill="{color}"/>',tx(x+barw+5,y+17,f"{val:,.0f} ms",10,"#263248","600")]
        for line,label in enumerate(names[i].split("\n")):
            out.append(tx(x+67,gy+170+line*15,label,11,"#46546a",anchor="middle"))
    out += [tx(42,720,"A mudança entre 247 e 397 imagens inclui passar de uma pergunta nas imagens de cor para quatro perguntas em cada imagem.",12,"#596579")]
    save("laya-blip-history.svg",out)

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--input",type=Path,default=DEFAULT_INPUT); p.add_argument("--history",type=Path,default=HISTORY_INPUT); args=p.parse_args(); d=json.loads(args.input.read_text(encoding="utf-8")); h=json.loads(args.history.read_text(encoding="utf-8"))
    accuracy_chart(d); latency_chart(d); color_chart(d); history_chart(h)
if __name__=="__main__": main()

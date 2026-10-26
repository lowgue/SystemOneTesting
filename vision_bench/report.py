"""Reports support any number of model IDs and keep machines/runs separate."""
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics
from .core import load_json,write_json


def generate(runs,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    summary=[];lines=['# Comparação portátil de modelos visuais','',
      'Cada execução e máquina é apresentada separadamente. Não há ranking agregado entre hardwares. A validade de esquema não é acurácia; rótulos exploratórios não comprovam qualidade final. CPU/RSS de um cliente HTTP não medem o servidor remoto.','']
    plot_inputs=[]
    for run in map(Path,runs):
        meta=load_json(run/'run.json');status=load_json(run/'status.json');frozen=load_json(run/'frozen_config.json')
        if status['status']!='complete':raise ValueError(f'Run incomplete: {run}')
        rows=[json.loads(s) for s in (run/'predictions.jsonl').read_text().splitlines()]
        if len(rows)!=status['planned'] or len({r['request_key'] for r in rows})!=len(rows):raise ValueError('Missing or duplicate records')
        label=f'{meta["machine_id"]} / {run.name}'
        lines += [f'## {label}','',f'Device: `{frozen["device"]}`; threads: {frozen["threads"]}; rótulos: `{meta["label_status"]}`.',
                  f'Contrato: `{meta["contract_hash"][:12]}`; dataset: `{meta["dataset_hash"][:12]}`. CPU: {meta["hardware"]["cpu"]}.','']
        if meta['simulated']:lines += ['**SIMULAÇÃO: este run contém adapter mock e serve apenas para validar infraestrutura.**','']
        lines += ['| Modelo | Eventos / requisições | p50 / p95 (s) | CPU média (s) | RSS pico amostrado (MiB) | Escopo CPU/RSS | Esquema válido (1ª repetição) | Positivos completos corretos* | Erros |',
                  '|---|---:|---:|---:|---:|---|---:|---:|---:|']
        for spec in frozen['models']:
            model=spec['id'];rs=[r for r in rows if r['model']==model];first=[r for r in rs if r['block']==0]
            ok=[r for r in rs if not r['error']];times=[r['elapsed_ms']/1000 for r in ok]
            positive=[r for r in first if r.get('expected',{}).get('presence')=='yes']
            correct=sum(r.get('exploratory_exact',False) for r in positive)
            med=float(np.median(times)) if times else None;p95=float(np.quantile(times,.95)) if times else None
            scopes=sorted({r['resource_scope'] for r in rs})
            row=dict(run=run.name,machine=meta['machine_id'],model=model,device=frozen['device'],events=len(first),requests=len(rs),
              p50_s=med,p95_s=p95,mean_cpu_s=statistics.mean(r['cpu_s'] for r in rs),peak_rss_mib=max(r['peak_rss_bytes'] for r in rs)/2**20,
              resource_scope=','.join(scopes),first_valid=sum(r['valid'] for r in first),positive_n=len(positive),exploratory_positive_exact=correct,errors=sum(bool(r['error']) for r in rs))
            summary.append(row)
            timing=f'{med:.3f} / {p95:.3f}' if times else 'sem sucesso'
            lines.append(f'| {model} | {len(first)} / {len(rs)} | {timing} | {row["mean_cpu_s"]:.3f} | {row["peak_rss_mib"]:.0f} | {row["resource_scope"]} | {row["first_valid"]}/{len(first)} | {correct}/{len(positive)} | {row["errors"]} |')
        lines += ['','*Métrica exploratória, somente onde existe gabarito. Todos os positivos, inclusive falhas e rejeições, permanecem no denominador. As repetições não aumentam o número de imagens independentes.','']
        plot_inputs.append(dict(label=label,rows=rows,models=[s['id'] for s in frozen['models']],simulated=meta['simulated']))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    for index,entry in enumerate(plot_inputs,1):
        fig,axes=plt.subplots(1,2,figsize=(12,6));fig.suptitle(('SIMULAÇÃO — ' if entry['simulated'] else '')+entry['label'],x=.06,ha='left',fontsize=13,fontweight='bold')
        palettes=plt.get_cmap('tab10');rng=np.random.default_rng(20261026)
        for pos,m in enumerate(entry['models'],1):
            groups=defaultdict(list)
            for r in entry['rows']:
                if r['model']==m and not r['error']:groups[r['id']].append(r)
            for ax,key,scale in [(axes[0],'elapsed_ms',1000),(axes[1],'cpu_s',1)]:
                values=[statistics.median(r[key]/scale for r in rs) for rs in groups.values()]
                if values:
                    ax.boxplot([values],positions=[pos],widths=.4,manage_ticks=False,showfliers=False,medianprops={'color':'black'})
                    ax.scatter(pos+rng.uniform(-.1,.1,len(values)),values,color=palettes((pos-1)%10),s=20,alpha=.7)
        for ax in axes:
            ax.set_xticks(range(1,len(entry['models'])+1),entry['models'],rotation=18,ha='right');ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.2)
        axes[0].set_ylabel('Mediana intraimagem da latência (s)');axes[1].set_ylabel('Mediana intraimagem de CPU-s (escopo por modelo)')
        axes[0].set_title('Evento completo, com pré e pós-processamento');axes[1].set_title('HTTP sem PID mede somente o cliente')
        fig.text(.06,.025,'Pontos: medianas por evento, não repetições independentes. Caixas: quartis; sem IC populacional para esta amostra.\nCarga e warm-up excluídos. Recursos remotos, energia e custo monetário não estão incluídos nos recursos do cliente.',fontsize=8)
        fig.subplots_adjust(left=.09,right=.97,bottom=.30,top=.82,wspace=.35)
        for ext in ['png','svg','pdf']:fig.savefig(out/f'comparison_{index}.{ext}',dpi=300)
        plt.close(fig);lines += [f'![{entry["label"]}](comparison_{index}.png)','',f'[PDF](comparison_{index}.pdf) · [SVG](comparison_{index}.svg)','']
    write_json(out/'metrics.json',summary)
    with (out/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    (out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');return str(out/'README.md')

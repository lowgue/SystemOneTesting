"""Publication-style figures from frozen empirical pilot evidence; no new model calls."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
MODELS=['laya','blip'];COLORS={'laya':'#0072B2','blip':'#D55E00'}
NAMES={'laya':'Laya Vision','blip':'BLIP VQA'}
DOMAINS=['vehicle','person'];DNAMES={'vehicle':'Veículos','person':'Pessoas'}
TRANS=['resize64','jpeg20','dark50','blur2']
TNAMES=['Resolução 64×64','JPEG, qualidade 20','Brilho ×0,5','Blur, raio 2']
STYLES={'direct':'Direto','specific':'Específico','uncertain':'Com incerteza','descriptive':'Descritivo','short':'Curto','visible':'Com visibilidade'}
SEED=20261026


def readl(path):return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
def med(vals):return float(np.median(vals))
def wilson(k,n,z=1.959963984540054):
    p=k/n;den=1+z*z/n;center=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return max(0,center-half),min(1,center+half)

def variant_name(v):
    style,mode=v.rsplit('_',1)
    return STYLES[style]+(' · conjunta' if mode=='joint' else ' · presença primeiro')

def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,'axes.labelsize':10,
       'figure.titlesize':15,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#64748B',
       'axes.labelcolor':'#1E293B','text.color':'#1E293B','xtick.color':'#475569','ytick.color':'#475569',
       'grid.color':'#E2E8F0','grid.linewidth':.6,'axes.axisbelow':True,'legend.frameon':False,
       'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})

def panel(ax,label,title):
    ax.set_title(f'{label}  {title}',loc='left',pad=12,fontweight='bold');ax.grid(axis='y',alpha=.75)

def note(fig,text):
    fig.text(.06,.025,text,ha='left',va='bottom',fontsize=8,color='#475569',linespacing=1.5)

def layout(fig,title,subtitle='',bottom=.17,top=.78):
    fig.suptitle(title,x=.06,y=.97,ha='left',fontweight='bold')
    if subtitle:fig.text(.06,.895,subtitle,ha='left',va='top',fontsize=9,color='#475569')
    fig.subplots_adjust(left=.08,right=.97,bottom=bottom,top=top,wspace=.36,hspace=.6)

def boxpoints(ax,groups,labels,colors,ylim=None):
    rng=np.random.default_rng(SEED)
    for x,(values,color) in enumerate(zip(groups,colors),1):
        vals=np.array(values)
        ax.boxplot([vals],positions=[x],widths=.45,patch_artist=True,showfliers=False,manage_ticks=False,
          boxprops=dict(facecolor=color,alpha=.18,edgecolor=color),medianprops=dict(color='#0F172A',linewidth=1.6),
          whiskerprops=dict(color=color),capprops=dict(color=color))
        ax.scatter(x+rng.uniform(-.14,.14,len(vals)),vals,s=26,color=color,alpha=.8,edgecolor='white',linewidth=.4,zorder=3)
    ax.set_xticks(range(1,len(labels)+1),labels)
    ax.set_xlim(.4,len(labels)+.6)
    if ylim:ax.set_ylim(*ylim)


def boot_median(rows,value='elapsed_ms',scale=1000,resamples=4000):
    # One row per image for prompt variants. Stratify by domain.
    rng=np.random.default_rng(SEED)
    strata=[np.array([r[value]/scale for r in rows if r['domain']==d]) for d in DOMAINS]
    strata=[a for a in strata if len(a)]
    sims=np.array([np.median(np.concatenate([rng.choice(a,len(a),replace=True) for a in strata])) for _ in range(resamples)])
    return float(np.median([r[value]/scale for r in rows])),*np.quantile(sims,[.025,.975]).tolist()


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--pilot',default='eval/experiments/automated_pilot_20261026')
    ap.add_argument('--output',default='eval/figures/scientific_20261026');args=ap.parse_args()
    src=(ROOT/args.pilot).resolve();out=(ROOT/args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    style();rows=readl(src/'predictions.jsonl');cfg=json.loads((src/'frozen_config.json').read_text());metrics=json.loads((src/'metrics.json').read_text())
    assert json.loads((src/'status.json').read_text())['status']=='complete'
    perf=[r for r in rows if r['stage']=='performance'];prompt=[r for r in rows if r['stage']=='prompts'];robust=[r for r in rows if r['stage']=='robustness']
    budget_src=src.parent/'laya_option_budget_20261026';budget=json.loads((budget_src/'budget_audit.json').read_text());bp=readl(budget_src/'predictions.jsonl')
    power_src=src.parent/'paired_power_20261026';power=json.loads((power_src/'power.json').read_text())
    ids={d:sorted({r['id'] for r in perf if r['domain']==d}) for d in DOMAINS}
    # Avoid implying independent scenes if identical images appear under different IDs.
    audit=json.loads((src/'audit.json').read_text());hashes={r['id']:r['sha256'] for r in audit['rows']}
    perf_ids=set().union(*map(set,ids.values()));assert len({hashes[i] for i in perf_ids})==len(perf_ids)
    per_image={(m,i):{k:med([r[k] for r in perf if r['model']==m and r['id']==i]) for k in ['elapsed_ms','cpu_s']} for m in MODELS for i in perf_ids}
    tables=[];figures=[]
    pdf=PdfPages(out/'relatorio_estatistico.pdf',metadata={'Title':'Avaliação científica de descrições visuais — piloto automatizado','Author':'SystemOneTesting','Subject':'Desempenho, integridade e robustez; sem avaliação confirmatória de acurácia'})
    def save(fig,name,title,method):
        for ext in ['png','svg','pdf']:fig.savefig(out/f'{name}.{ext}',dpi=300)
        pdf.savefig(fig)
        figures.append(dict(file=name,title=title,method=method));plt.close(fig)
    def addtable(figure,**kwargs):tables.append(dict(figure=figure,**kwargs))

    # 01: independent image units, not 80 repetitions counted as 80 images.
    fig,axes=plt.subplots(1,2,figsize=(12.8,5.8));layout(fig,'01  Desempenho por imagem e domínio','16 imagens distintas por modelo; cinco repetições por imagem. Cada ponto é a mediana da imagem.',bottom=.2)
    labels=[];latgroups=[];cpugroups=[];colors=[]
    for d in DOMAINS:
        for m in MODELS:
            labels.append(DNAMES[d]+'\n'+NAMES[m]);colors.append(COLORS[m])
            latgroups.append([per_image[m,i]['elapsed_ms']/1000 for i in ids[d]])
            cpugroups.append([per_image[m,i]['cpu_s'] for i in ids[d]])
    boxpoints(axes[0],latgroups,labels,colors,(0,5));boxpoints(axes[1],cpugroups,labels,colors,(0,40))
    for ax in axes:ax.tick_params(axis='x',labelsize=8)
    axes[0].set_ylabel('Latência mediana por imagem (s)');axes[1].set_ylabel('CPU mediana por imagem (CPU-s)')
    panel(axes[0],'A','Tempo de parede');panel(axes[1],'B','Trabalho computacional')
    note(fig,'Caixa: quartis; linha: mediana; hastes: até 1,5×IQR; pontos incluem todos os valores. CPU-s soma trabalho das threads.\nLaya: configuração padrão trunca a pergunta de marca. Menor custo observado não prova equivalência da entrada nem qualidade.')
    save(fig,'01_desempenho_distribuicoes','Desempenho por domínio','Pontos: medianas de cinco repetições; oito imagens por domínio e modelo; caixas descritivas.')

    # 02: paired model comparison and cluster/block uncertainty.
    fig,axes=plt.subplots(1,2,figsize=(12.8,5.8));layout(fig,'02  Comparação pareada da latência','Mesmas imagens nos dois modelos; pareamento preservado nas reamostragens.',bottom=.2)
    for d,marker in zip(DOMAINS,['o','s']):
        x=[per_image['laya',i]['elapsed_ms']/1000 for i in ids[d]];y=[per_image['blip',i]['elapsed_ms']/1000 for i in ids[d]]
        axes[0].scatter(x,y,label=DNAMES[d],marker=marker,s=45,edgecolor='white',linewidth=.5,color='#0072B2' if d=='vehicle' else '#009E73')
    axes[0].plot([0,5],[0,5],color='#64748B',ls='--',lw=1,label='Tempos iguais');axes[0].set(xlim=(0,5),ylim=(0,5),xlabel='Laya: mediana da imagem (s)',ylabel='BLIP: mediana da imagem (s)');axes[0].legend(fontsize=8,loc='lower right')
    axes[0].set_aspect('equal',adjustable='box');panel(axes[0],'A','Cada ponto é uma imagem')
    rng=np.random.default_rng(SEED);bootstats={};blocks=sorted({r['block'] for r in perf});index={(r['model'],r['id'],r['block']):r['elapsed_ms'] for r in perf}
    for pos,d in enumerate(['vehicle','person','all']):
        selected=ids[d] if d!='all' else ids['vehicle']+ids['person'];mat={m:np.array([[index[m,i,b] for b in blocks] for i in selected]) for m in MODELS}
        strata=[[k for k,i in enumerate(selected) if i in ids[s]] for s in DOMAINS];strata=[s for s in strata if s];rat=[]
        for _ in range(4000):
            ix=np.concatenate([rng.choice(s,len(s),replace=True) for s in strata]);bc=rng.choice(len(blocks),len(blocks),replace=True)
            rat.append(np.median(mat['blip'][np.ix_(ix,bc)])/np.median(mat['laya'][np.ix_(ix,bc)]))
        value=np.median(mat['blip'])/np.median(mat['laya']);lo,hi=np.quantile(rat,[.025,.975]);bootstats[d]=dict(ratio=float(value),lower=float(lo),upper=float(hi))
        axes[1].errorbar(value,pos,xerr=[[value-lo],[hi-value]],fmt='o',color='#6A3D9A',capsize=4,lw=1.5)
        axes[1].text(2.43,pos,f'{value:.2f} [{lo:.2f}; {hi:.2f}]',va='center',ha='right',fontsize=8)
        addtable('02_pareado',domain=d,measure='BLIP/Laya median ratio',estimate=value,lower95=lo,upper95=hi,n_images=len(selected))
    axes[1].set(yticks=[0,1,2],yticklabels=['Veículos','Pessoas','Conjunto'],xlabel='Razão das medianas: BLIP / Laya',xlim=(.8,2.5),ylim=(2.6,-.6));axes[1].axvline(1,color='#64748B',ls='--',lw=1);axes[1].grid(axis='x');panel(axes[1],'B','Razão com IC pareado de 95%')
    note(fig,'Bootstrap: 4.000 reamostragens de imagens dentro do domínio e dos cinco blocos; pares mantidos e seed fixa.\nIntervalos condicionais a esta amostra de conveniência. Marca no Laya sofre truncamento; não é comparação de qualidade.')
    save(fig,'02_latencia_pareada','Comparação pareada e intervalos','Bootstrap estratificado por domínio e pareado por imagem e bloco, 4.000 reamostragens.')

    # 03: timing blocks and deterministic reproducibility.
    fig,axes=plt.subplots(1,2,figsize=(12.8,5.8));layout(fig,'03  Repetibilidade e variação entre blocos','A ordem de execução alternou entre Laya→BLIP e BLIP→Laya.',bottom=.2)
    for m in MODELS:
        for i in perf_ids:
            rs=sorted([r for r in perf if r['model']==m and r['id']==i],key=lambda r:r['block'])
            axes[0].plot(range(1,6),[r['elapsed_ms']/1000 for r in rs],color=COLORS[m],alpha=.12,lw=.7)
        medians=[med([r['elapsed_ms']/1000 for r in perf if r['model']==m and r['block']==b]) for b in blocks]
        axes[0].plot(range(1,6),medians,'o-',color=COLORS[m],label=NAMES[m],lw=2)
        cvs=[]
        for i in sorted(perf_ids):
            vals=[r['elapsed_ms'] for r in perf if r['model']==m and r['id']==i];cvs.append(100*np.std(vals,ddof=1)/np.mean(vals))
        x=MODELS.index(m)+1;axes[1].scatter(x+np.random.default_rng(SEED).uniform(-.13,.13,len(cvs)),cvs,color=COLORS[m],s=24)
        axes[1].plot([x-.22,x+.22],[med(cvs)]*2,color='#0F172A',lw=2)
    axes[0].set(xlabel='Bloco de medição',ylabel='Latência (s)',xticks=range(1,6),ylim=(0,5));axes[0].legend(fontsize=8);panel(axes[0],'A','Linhas finas: imagens; grossas: mediana do bloco')
    axes[1].set(xticks=[1,2],xticklabels=[NAMES[m] for m in MODELS],ylabel='Coeficiente de variação intraimagem (%)',xlim=(.5,2.5),ylim=(0,None));panel(axes[1],'B','Variação temporal entre cinco repetições')
    note(fig,'O coeficiente de variação usa desvio-padrão amostral / média. Cada ponto corresponde a uma imagem.\nSaídas categóricas foram idênticas em 16/16 imagens por modelo nas cinco repetições; estabilidade não demonstra correção.')
    save(fig,'03_repetibilidade','Repetibilidade temporal e de saídas','Cinco blocos descritivos; CV por imagem; sem pseudo-replicação do tamanho da amostra.')

    # 04–05: all complete configurations, no single-attribute winner.
    for m in MODELS:
        fig,axes=plt.subplots(1,3,figsize=(15.5,6.7));layout(fig,f'{4+MODELS.index(m):02d}  {NAMES[m]} — configurações completas de prompt','As mesmas 12 imagens em cada configuração. Esquema válido não significa descrição correta.',bottom=.2,top=.78)
        variants=list(cfg['configurations'][m]);ys=np.arange(6);labels=[variant_name(v) for v in variants]
        for y,v in enumerate(variants):
            rs=[r for r in prompt if r['model']==m and r['variant']==v];value,lo,hi=boot_median(rs)
            axes[0].errorbar(value,y,xerr=[[value-lo],[hi-value]],fmt='o',color=COLORS[m],capsize=3)
            k=sum(r['result']['valid'] for r in rs);a,b=wilson(k,len(rs));p=k/len(rs)
            axes[1].errorbar(100*p,y,xerr=[[100*(p-a)],[100*(b-p)]],fmt='o',color=COLORS[m],capsize=3)
            axes[1].text(104,y,f'{k}/{len(rs)}',va='center',fontsize=8)
            cuts=metrics['prompts'][m][v]['truncated_events'];axes[2].barh(y,cuts,color='#CC79A7',height=.55,alpha=.85);axes[2].text(cuts+.15,y,f'{cuts}/12',va='center',fontsize=8)
            addtable(f'0{4+MODELS.index(m)}_prompts_{m}',model=m,variant=v,measure='median latency s',estimate=value,lower95=lo,upper95=hi,n_images=len(rs),valid=k,valid_lower95=a,valid_upper95=b,truncated=cuts)
        for a in axes:a.set_ylim(5.7,-.7);a.set_yticks(ys);a.grid(axis='x');a.grid(axis='y',visible=False)
        axes[0].set_yticklabels(labels,fontsize=8);axes[1].set_yticklabels([]);axes[2].set_yticklabels([])
        axes[0].set(xlabel='Latência mediana (s), IC 95%',xlim=(0,5.5));axes[1].set(xlabel='Eventos com esquema válido (%)',xlim=(0,116));axes[2].set(xlabel='Eventos com entrada truncada',xlim=(0,13),xticks=[0,3,6,9,12])
        panel(axes[0],'A','Tempo por evento completo');panel(axes[1],'B','Conformidade de esquema');panel(axes[2],'C','Integridade da entrada')
        fig.subplots_adjust(left=.20,wspace=.27)
        note(fig,'Latência: bootstrap de 4.000 amostras de imagens, estratificado por domínio; validade: intervalo Wilson de 95% (n=12).\nICs são exploratórios, sem correção para múltiplos prompts. Truncamento: contagem mecânica. Presença primeiro pode omitir atributos nos negativos.')
        save(fig,f'0{4+MODELS.index(m)}_prompts_{m}',f'Prompts — {NAMES[m]}','Conjuntos completos; bootstrap por imagem para latência e Wilson para validade; n=12 imagens por configuração.')

    # 06: response changes, never labeled as accuracy loss.
    fig,axes=plt.subplots(1,2,figsize=(12.8,5.8));layout(fig,'06  Sensibilidade a degradações da imagem','Métrica: conjunto de respostas diferente do original. Uma alteração pode ser erro ou correção.',bottom=.2)
    for ax,m in zip(axes,MODELS):
        for y,t in enumerate(TRANS):
            rs=[r for r in robust if r['model']==m and r['transform']==t];base='descriptive_joint' if m=='laya' else 'direct_joint'
            orig={r['id']:r['result']['predicted'] for r in prompt if r['model']==m and r['variant']==base};k=sum(r['result']['predicted']!=orig[r['id']] for r in rs);n=len(rs);lo,hi=wilson(k,n);p=k/n
            ax.errorbar(100*p,y,xerr=[[100*(p-lo)],[100*(hi-p)]],fmt='o',color=COLORS[m],capsize=4)
            ax.text(101,y,f'{k}/{n}',va='center',fontsize=8);addtable('06_robustez',model=m,transform=t,measure='changed answer bundles',estimate=p,lower95=lo,upper95=hi,n_images=n,count=k)
        ax.set(xlim=(0,113),ylim=(3.7,-.7),yticks=range(4),yticklabels=TNAMES,xlabel='Conjuntos de respostas alterados (%)');ax.grid(axis='x');ax.grid(axis='y',visible=False);panel(ax,'A' if m=='laya' else 'B',NAMES[m])
    fig.subplots_adjust(left=.16,wspace=.43)
    note(fig,'Pontos: contagens / 12 imagens; hastes: Wilson 95%, sem ajuste por multiplicidade. Os mesmos originais aparecem em todas as condições.\nTransformações não são cenas novas. Baixa sensibilidade também pode significar repetição de uma resposta errada; não é um ranking de acurácia.')
    save(fig,'06_robustez','Sensibilidade a degradações','Mudança pareada em relação ao original; Wilson exploratório por transformação; não mede perda de acurácia.')

    # 07: designed controls, no manufactured population confidence interval.
    synth=[r for r in rows if r['stage']=='synthetic'];kinds=['black','gray','white','noise'];matrix=np.zeros((4,4),int)
    fig,ax=plt.subplots(figsize=(11.5,5.7));layout(fig,'07  Presença indevida em controles sintéticos','Quatro imagens geradas, consultadas para dois domínios em cada modelo. Cada célula é uma observação.',bottom=.22)
    labs=[]
    for y,(m,d) in enumerate([(m,d) for m in MODELS for d in DOMAINS]):
        labs.append(NAMES[m]+' · '+DNAMES[d])
        for x,kind in enumerate(kinds):
            r=next(r for r in synth if r['model']==m and r['domain']==d and r['synthetic_kind']==kind)
            p=r['result']['predicted']['presence'];matrix[y,x]=1 if p=='yes' else (0 if p=='no' else 2)
    ax.imshow(matrix,cmap=ListedColormap(['#E6F2F7','#D55E00','#CC79A7']),vmin=0,vmax=2,aspect='auto')
    for y in range(4):
        for x in range(4):ax.text(x,y,'Presença indevida' if matrix[y,x]==1 else ('Ausência' if matrix[y,x]==0 else 'Inválido'),ha='center',va='center',fontsize=9,color='white' if matrix[y,x]==1 else '#1E293B')
    ax.set(xticks=range(4),xticklabels=['Preto uniforme','Cinza uniforme','Branco uniforme','Ruído'],yticks=range(4),yticklabels=labs);ax.tick_params(length=0);fig.subplots_adjust(left=.25)
    note(fig,'Presença real conhecida por construção: não há pessoa nem veículo. A mesma imagem é reutilizada entre tarefas e modelos.\nNão há intervalo de população: são controles planejados, sem amostragem aleatória de cenas reais. Não estimar falso positivo em produção a partir desta figura.')
    save(fig,'07_controles_sinteticos','Controles sintéticos sem objetos','Matriz de observações exatas; nenhuma inferência populacional sobre quatro controles programáticos.')

    # 08: token-budget integrity and actual paired inference cost.
    fig,axes=plt.subplots(1,2,figsize=(14,6.7));layout(fig,'08  Laya — orçamento de tokens e truncamento','Auditoria mecânica da entrada; seis imagens no piloto de inferência. Mantidas todas as 48 marcas + desconhecida.',bottom=.22)
    scenarios=[(s,h) for s in ['descriptive','short','visible'] for h in [256,384,512]];ns=[8,12,16,24,32,48];a=np.zeros((9,6));ann=[]
    for y,(s,h) in enumerate(scenarios):
        for x,n in enumerate(ns):
            r=next(r for r in budget['rows'] if r['style']==s and r['head_max_len']==h and r['brands']==n);cut=r['truncation'] or {};a[y,x]=1 if cut.get('instructions') else (.5 if r['truncated'] else 0)
            axes[0].text(x,y,'I+O' if a[y,x]==1 else ('O' if a[y,x]==.5 else 'Íntegra'),ha='center',va='center',fontsize=7,color='white' if a[y,x]==1 else '#1E293B',zorder=3)
    axes[0].imshow(a,cmap=ListedColormap(['#E6F2F7','#F0E442','#D55E00']),vmin=0,vmax=1,aspect='auto')
    axes[0].set(xticks=range(6),xticklabels=[f'{n}+1' for n in ns],yticks=range(9),yticklabels=[f'{STYLES[s]} · {h}' for s,h in scenarios],xlabel='Marcas + opção desconhecida');axes[0].tick_params(labelsize=8,length=0);panel(axes[0],'A','Truncamento por critérios / orçamento');axes[0].grid(False)
    bid=sorted({r['id'] for r in bp})
    for i in bid:
        vals=[next(r['elapsed_ms']/1000 for r in bp if r['id']==i and r['head_max_len']==h) for h in [256,512]]
        axes[1].plot([0,1],vals,'o-',color='#64748B',alpha=.65,ms=5,lw=1)
    for x,h in enumerate([256,512]):
        vals=[r['elapsed_ms']/1000 for r in bp if r['head_max_len']==h];v=med(vals);axes[1].plot([x-.18,x+.18],[v]*2,color='#0072B2',lw=3);axes[1].text(x,v+.16,f'Mediana: {v:.2f} s',ha='center',fontsize=9)
    axes[1].set(xticks=[0,1],xticklabels=['256 tokens\n6/6 truncadas','512 tokens\n0/6 truncadas'],ylabel='Latência do evento completo (s)',ylim=(0,5),xlim=(-.45,1.45));panel(axes[1],'B','Mesmas seis imagens, dois orçamentos')
    fig.subplots_adjust(left=.18,wspace=.45)
    note(fig,'I+O: corte de instrução e opções; O: corte de opções. Cada célula da auditoria é determinística, sem IC.\nLinhas: pares de imagens; marcas grossas: medianas. Ordem 256→512 não contrabalanceada; sem IC causal. Entrada íntegra não comprova melhora semântica.')
    save(fig,'08_orcamento_laya','Orçamento de tokens e efeito na latência','54 construções de entrada; 12 inferências em seis imagens; comparação pareada descritiva sem adjudicação de qualidade.')

    # 09: queueing is not parallel model execution.
    fig,axes=plt.subplots(1,2,figsize=(12.8,5.8));layout(fig,'09  Fila com múltiplos clientes e um worker','Um único worker de inferência serializado. Seis requisições por nível e modelo.',bottom=.2)
    queue=[r for r in rows if r['stage']=='queue'];batches=readl(src/'queue_batches.jsonl')
    for m,offset in [('laya',-.06),('blip',.06)]:
        medians=[]
        for x,c in enumerate([1,2,4]):
            vals=[r['response_ms']/1000 for r in queue if r['model']==m and r['concurrency']==c];xx=x+offset
            axes[0].scatter(np.full(len(vals),xx)+np.random.default_rng(SEED).uniform(-.035,.035,len(vals)),vals,color=COLORS[m],s=23,alpha=.7)
            medians.append(med(vals))
        axes[0].plot(np.arange(3)+offset,medians,'o-',color=COLORS[m],label=NAMES[m])
        through=[next(b['throughput_per_s'] for b in batches if b['model']==m and b['concurrency']==c) for c in [1,2,4]];axes[1].plot(range(3),through,'o-',color=COLORS[m],label=NAMES[m])
    for ax in axes:ax.set_xticks(range(3),['1','2','4']);ax.set_xlabel('Clientes simultâneos');ax.set_ylim(0,None);ax.legend(fontsize=8)
    axes[0].set_ylabel('Resposta incluindo espera na fila (s)');axes[1].set_ylabel('Throughput observado (eventos/s)');panel(axes[0],'A','Pontos: requisições; linhas: medianas');panel(axes[1],'B','Um lote observado por condição')
    note(fig,'Sem barras de confiança: apenas um lote por condição; requisições da mesma fila não são réplicas independentes.\nEste smoke test verifica fila e integridade. Não demonstra escalabilidade de inferência paralela, capacidade sustentada ou atendimento a SLA.')
    save(fig,'09_fila','Fila e throughput','Seis requisições e um lote por nível; análise descritiva sem IC de throughput.')

    # 10: sizing simulations independent of weak photographic labels.
    fig,axes=plt.subplots(1,2,figsize=(13.2,6));layout(fig,'10  Planejamento estatístico da próxima coleta','Simulação sob hipóteses explícitas; nenhum resultado de qualidade dos modelos é usado para estimar o poder.',bottom=.21)
    palette=['#0072B2','#009E73','#D55E00','#CC79A7']
    for q,c,marker in zip([.1,.2,.3,.5],palette,['o','s','^','D']):
        rs=sorted([r for r in power['rows'] if r['alpha']==.025 and r['discordance']==q],key=lambda r:r['n_pairs'])
        axes[0].plot([r['n_pairs'] for r in rs],[100*r['power'] for r in rs],marker=marker,color=c,label=f'Discordância {100*q:.0f}%',ms=4,lw=1.5)
    axes[0].axhline(80,color='#64748B',ls='--',lw=1);axes[0].axvline(600,color='#94A3B8',ls=':',lw=1);axes[0].set(xlabel='Pares independentes por domínio',ylabel='Poder simulado (%)',ylim=(0,102),xlim=(200,2500));axes[0].legend(fontsize=8,loc='lower right');panel(axes[0],'A','Detectar +5 pontos contra diferença zero')
    n=np.arange(50,1001)
    for family,c in zip([1,2,6],palette):
        alpha=.05/family;upper=100*(1-alpha**(1/n));axes[1].plot(n,upper,color=c,label=f'{family} '+('comparações' if family>1 else 'comparação'))
        required=math.ceil(math.log(alpha)/math.log(.99));axes[1].scatter([required],[100*(1-alpha**(1/required))],s=27,color=c);axes[1].annotate(str(required),(required,1),xytext=(3,8),textcoords='offset points',fontsize=8,color=c)
    axes[1].axhline(1,color='#64748B',ls='--',lw=1);axes[1].set(xlabel='Negativos independentes, com zero falhas',ylabel='Limite superior unilateral (%)',ylim=(0,6),xlim=(50,1000));axes[1].legend(fontsize=8);panel(axes[1],'B','Limite binomial exato com multiplicidade')
    note(fig,'A: teste pareado exato bilateral, α=0,025 por domínio; 20.000 simulações por ponto. Discordância é hipotética; linhas só conectam a grade.\nB: limite 1−α^(1/n), com Bonferroni ilustrativo para 1, 2 ou 6 comparações. Poder contra zero não comprova vantagem maior que +5 pontos.')
    save(fig,'10_planejamento_amostral','Poder e tamanho da amostra','48 cenários de simulação; curva binomial exata determinística; pressupõe independência dos eventos.')

    # 11: global process memory trace, no fabricated attribution to a model.
    resource=readl(src/'resource_samples.jsonl');t=np.array([r['monotonic_s']-resource[0]['monotonic_s'] for r in resource])/60
    rss=np.array([r['rss_bytes'] for r in resource])/2**30;available=np.array([r['available_bytes'] for r in resource])/2**30
    fig,axes=plt.subplots(2,1,figsize=(12.8,7),sharex=True);layout(fig,'11  Memória ao longo do processo de avaliação','Amostragem a cada 100 ms; ambos os modelos foram carregados em momentos diferentes dentro do mesmo processo.',bottom=.16,top=.78)
    axes[0].plot(t,rss,color='#0072B2',lw=.9);axes[0].set(ylabel='RSS do processo (GiB)',ylim=(0,None));panel(axes[0],'A','Memória residente do processo, não apenas dos pesos')
    axes[1].plot(t,available,color='#009E73',lw=.9);axes[1].set(ylabel='RAM disponível na máquina (GiB)',xlabel='Tempo desde o início da monitoração (min)',ylim=(0,None));panel(axes[1],'B','Contexto de memória do sistema')
    note(fig,'O processo inclui Python, tensores, buffers e memória retida pelo alocador. O traço não é segmentado por modelo porque os logs não têm\nmarcadores temporais de etapa suficientes para essa atribuição. RAM disponível também depende dos outros processos da máquina.')
    save(fig,'11_memoria_processo','Memória do processo e do sistema','Série temporal de RSS e RAM disponível, 100 ms; sem atribuição de picos a modelos específicos.')

    # Dedicated per-model panels with identical axis limits.
    for index,m in enumerate(MODELS,12):
        fig,axes=plt.subplots(2,2,figsize=(12.8,9));layout(fig,f'{index:02d}  Perfil estatístico — {NAMES[m]}','Cada painel identifica a unidade observada. Desempenho e consistência não substituem acurácia factual.',bottom=.15,top=.83)
        gs=[[per_image[m,i]['elapsed_ms']/1000 for i in ids[d]] for d in DOMAINS]
        boxpoints(axes[0,0],gs,[DNAMES[d] for d in DOMAINS],[COLORS[m]]*2,(0,5));axes[0,0].set_ylabel('Latência mediana por imagem (s)');panel(axes[0,0],'A','Oito imagens por domínio; cinco repetições')
        cs=[[per_image[m,i]['cpu_s'] for i in ids[d]] for d in DOMAINS]
        boxpoints(axes[0,1],cs,[DNAMES[d] for d in DOMAINS],[COLORS[m]]*2,(0,40));axes[0,1].set_ylabel('CPU mediana por imagem (CPU-s)');panel(axes[0,1],'B','Trabalho computacional por domínio')
        variants=list(cfg['configurations'][m])
        for y,v in enumerate(variants):
            rs=[r for r in prompt if r['model']==m and r['variant']==v];k=sum(r['result']['valid'] for r in rs);lo,hi=wilson(k,len(rs));p=k/len(rs)
            axes[1,0].errorbar(100*p,y,xerr=[[100*(p-lo)],[100*(hi-p)]],fmt='o',color=COLORS[m],capsize=3)
        axes[1,0].set(yticks=range(6),yticklabels=[variant_name(v) for v in variants],ylim=(5.7,-.7),xlim=(0,104),xlabel='Esquema válido (%), Wilson 95%');axes[1,0].tick_params(axis='y',labelsize=8);axes[1,0].grid(axis='x');panel(axes[1,0],'C','Formato em 12 imagens por configuração')
        for y,tr in enumerate(TRANS):
            a=metrics['robustness'][m][tr];k=a['changed_bundles'];n=a['images'];lo,hi=wilson(k,n);p=k/n
            axes[1,1].errorbar(100*p,y,xerr=[[100*(p-lo)],[100*(hi-p)]],fmt='o',color=COLORS[m],capsize=3)
        axes[1,1].set(yticks=range(4),yticklabels=TNAMES,ylim=(3.7,-.7),xlim=(0,104),xlabel='Respostas alteradas (%), Wilson 95%');axes[1,1].tick_params(axis='y',labelsize=8);axes[1,1].grid(axis='x');panel(axes[1,1],'D','Sensibilidade em 12 imagens')
        fig.subplots_adjust(left=.18,wspace=.46,hspace=.8)
        note(fig,'Eixos idênticos nos dois perfis. ICs de C e D são exploratórios; imagens de conveniência, sem correção por múltiplas comparações.\nLaya: pergunta de marca truncada na configuração padrão. BLIP: parser estrito pode rejeitar respostas fora da taxonomia; esquema ≠ correção.')
        save(fig,f'{index:02d}_perfil_{m}',f'Perfil individual — {NAMES[m]}','Resumo por modelo; mesmos limites de eixo para comparabilidade; unidades e incerteza especificadas por painel.')

    pdf.close()
    # Figure data exported in tidy form, without discarding raw observations.
    columns=sorted({key for row in tables for key in row})
    with (out/'estatisticas_dos_graficos.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(tables)
    with (out/'observacoes_desempenho.csv').open('w',newline='') as f:
        keys=['model','domain','id','block','elapsed_ms','cpu_s','rss_end_bytes'];w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:r[k] for k in keys} for r in perf)
    def export_csv(filename, records):
        columns=sorted({k for r in records for k in r})
        with (out/filename).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();writer.writerows(records)
    export_csv('observacoes_experimentos.csv',[dict(model=r['model'],domain=r['domain'],image_id=r['id'],stage=r['stage'],variant=r['variant'],transform=r['transform'],latency_ms=r['elapsed_ms'],cpu_s=r['cpu_s'],queue_ms=r['queue_ms'],response_ms=r['response_ms'],schema_valid=r['result'].get('valid'),**{f'prediction_{k}':v for k,v in r['result'].get('predicted',{}).items()}) for r in rows])
    export_csv('auditoria_orcamento.csv',[dict(style=r['style'],brands=r['brands'],head_tokens=r['head_max_len'],input_tokens=r['input_tokens'],truncated=r['truncated'],instruction_cut=bool((r['truncation'] or {}).get('instructions')),options_cut=len((r['truncation'] or {}).get('options',[]))) for r in budget['rows']])
    export_csv('inferencias_orcamento.csv',[dict(image_id=r['id'],head_tokens=r['head_max_len'],latency_ms=r['elapsed_ms'],truncated=any('truncated' in a for a in r['result']['raw']['answers'].values())) for r in bp])
    export_csv('simulacao_poder.csv',power['rows'])
    provenance=dict(seed=SEED,python_packages={'matplotlib':matplotlib.__version__,'numpy':np.__version__},
       input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [src/'predictions.jsonl',src/'metrics.json',src/'resource_samples.jsonl',budget_src/'budget_audit.json',budget_src/'predictions.jsonl',power_src/'power.json']},
       plotting_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),figures=figures,paired_bootstrap=bootstats)
    (out/'proveniencia.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
    lines=['# Atlas estatístico — Laya Vision e BLIP VQA','',
      '**13 figuras**, com exportações PNG (300 dpi), SVG e PDF vetorial. [PDF completo](relatorio_estatistico.pdf).', '',
      'Amostra exploratória. Não há nova anotação humana ou avaliação confirmatória da verdade das descrições. A configuração padrão do Laya trunca a pergunta de marca; os gráficos de custo precisam ser lidos com essa limitação.', '',
      '## Convenções científicas','',
      '- Desempenho: 16 imagens distintas, oito por domínio, com cinco repetições; cada ponto de distribuição é a mediana por imagem. As 80 requisições por modelo não são 80 cenas independentes.',
      '- Comparação de latência: bootstrap pareado por imagem e bloco, estratificado por domínio, 4.000 reamostragens; seed fixa. A figura 02 usa seu próprio cálculo, registrado na proveniência.',
      '- Prompts: bootstrap estratificado por domínio para a mediana; Wilson 95% para proporções de esquema válido, com 12 imagens por configuração. Os limites são exploratórios e condicionais ao modelo de amostragem; a coleta de conveniência limita generalização.',
      '- Robustez: mudança de respostas em relação ao original, não queda de acurácia. Wilson 95% exploratório por transformação; atributos e variantes da mesma imagem não são amostras independentes.',
      '- Não há ajuste para múltiplos prompts ou degradações. Não interpretar uma pequena diferença entre barras como significância ou superioridade de qualidade.',
      '- Controles sintéticos e auditoria de tokens: contagens determinísticas, sem fabricar intervalos populacionais. Fila: um lote por condição; não estimar IC de throughput a partir de seis clientes dependentes.',
      '- Custo: tempo de parede, CPU-s e memória do processo. Não há medição de energia nem custo em reais; não converter uma redução temporal em economia elétrica.',
      '- Cores acessíveis, escalas iguais entre modelos, unidades explícitas, eixos iniciados em zero quando representam magnitudes; razões usam referência de igualdade em 1.', '',
      '## Figuras','']
    for item in figures:
        n=item['file'];lines += [f'### {item["title"]}','',f'![{item["title"]}]({n}.png)','',item['method'],'',f'[SVG]({n}.svg) · [PDF]({n}.pdf) · [PNG 300 dpi]({n}.png)','']
    lines += ['## Dados e reprodução','', '[Estatísticas dos gráficos](estatisticas_dos_graficos.csv) · [Observações de desempenho](observacoes_desempenho.csv) · [Todos os experimentos](observacoes_experimentos.csv) · [Auditoria de tokens](auditoria_orcamento.csv) · [Inferências por orçamento](inferencias_orcamento.csv) · [Poder simulado](simulacao_poder.csv) · [Proveniência e hashes](proveniencia.json)','',
      'Reprodução em diretório novo: `.venv/bin/python eval/plot_scientific_description_results.py --output eval/figures/nova_execucao`.', '',
      '[Relatório empírico de origem](../../experiments/automated_pilot_20261026/report.md). Nenhum resultado anterior foi sobrescrito.']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    print(out/'README.md');print(out/'relatorio_estatistico.pdf');print(f'{len(figures)} figures, 3 formats each, plus combined PDF.')

if __name__=='__main__':main()

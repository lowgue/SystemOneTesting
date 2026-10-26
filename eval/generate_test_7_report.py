"""Generate complete-bundle metrics without treating repeated runs as independent images."""
import collections
import json
from pathlib import Path
import statistics
from xml.sax.saxutils import escape

ROOT=Path(__file__).resolve().parents[1]
RESULT=ROOT/'eval/results/test_7_complete_tags_results.json'
REPORT=RESULT.with_name('test_7_complete_tags_report.md')

def pct(k,n): return f'{100*k/n:.1f}%' if n else '—'
def quantile(vals,p):
    vals=sorted(vals)
    pos=(len(vals)-1)*p
    low=int(pos);high=min(low+1,len(vals)-1)
    return vals[low]+(vals[high]-vals[low])*(pos-low)

def main():
    doc=json.loads(RESULT.read_text())
    assert doc['status']=='complete', 'Benchmark incomplete'
    manifest=json.loads((ROOT/doc['manifest']).read_text())
    items=manifest['items']; records=doc['records']
    assert len(records)==len(items)*2*doc['repetitions']
    first=[r for r in records if r['repetition']==1]
    lines=['# Teste 7 — Entrega completa de atributos por imagem','',
           f'Amostra revisada: **{len(items)} imagens**, quatro campos por imagem, {doc["repetitions"]} repetições por modelo. CPU, {doc["threads"]} threads.', '',
           'Acurácia principal: todos os quatro campos corretos simultaneamente na primeira repetição. As repetições medem latência e estabilidade; não aumentam o número de imagens independentes.', '',
           '## Resultado por evento completo','',
           '| Domínio | Modelo | Eventos completos corretos | Positivos completos corretos | Campos corretos | Latência mediana / p90 (ms) |',
           '|---|---|---:|---:|---:|---:|']
    metrics={}
    for domain in ['vehicle','person','all']:
        metrics[domain]={}
        for model in ['laya','blip']:
            rows=[r for r in first if r['model']==model and (domain=='all' or r['domain']==domain)]
            pos=[r for r in rows if r['expected']['presence']=='yes']
            timed=[r['latency_ms'] for r in records if r['model']==model and not r['error'] and (domain=='all' or r['domain']==domain)]
            correct=sum(r['exact_match'] for r in rows)
            cp=sum(r['exact_match'] for r in pos)
            fields=sum(sum(r['predicted'].get(k)==v for k,v in r['expected'].items()) for r in rows)
            metrics[domain][model]=dict(images=len(rows),exact=correct,positive_images=len(pos),positive_exact=cp,fields_correct=fields,median_ms=statistics.median(timed),p90_ms=quantile(timed,.9))
            lines.append(f'| {domain} | {model} | {correct}/{len(rows)} ({pct(correct,len(rows))}) | {cp}/{len(pos)} ({pct(cp,len(pos))}) | {fields}/{len(rows)*4} ({pct(fields,len(rows)*4)}) | {statistics.median(timed):.1f} / {quantile(timed,.9):.1f} |')
    a=metrics['all']['laya'];b=metrics['all']['blip']
    winner='Empate' if a['positive_exact']==b['positive_exact'] else ('Laya' if a['positive_exact']>b['positive_exact'] else 'BLIP')
    lines += ['',f'**Melhor entrega completa nos casos positivos desta amostra: {winner}.** Compare também os domínios separadamente. Uma saída com quatro campos válidos pode conter quatro erros; formato e correção são medidos separadamente.', '',
              f'Nos {a["positive_images"]} casos com objeto presente, o conjunto inteiro ficou correto em {a["positive_exact"]} imagens para Laya e {b["positive_exact"]} para BLIP. A diferença é exploratória e não sustenta uma escolha definitiva. Nenhum dos dois demonstrou entrega completa confiável nesta configuração; revisão ou tratamento explícito de falhas continua necessário.', '', '![Acerto do conjunto completo](test_7_complete_tags_accuracy.svg)', '',
              '## Atributos, abstenções e erros','',
              '| Domínio | Campo | Modelo | Acertos (todos) | Acertos com valor observável |', '|---|---|---|---:|---:|']
    for domain in ['vehicle','person']:
        keys=list(next(it['expected'] for it in items if it['domain']==domain))
        for key in keys:
            for model in ['laya','blip']:
                rows=[r for r in first if r['domain']==domain and r['model']==model]
                obs=[r for r in rows if r['expected'][key] not in ('unknown','not_visible')]
                k=sum(r['predicted'][key]==r['expected'][key] for r in rows)
                ko=sum(r['predicted'][key]==r['expected'][key] for r in obs)
                lines.append(f'| {domain} | {key} | {model} | {k}/{len(rows)} ({pct(k,len(rows))}) | {ko}/{len(obs)} ({pct(ko,len(obs))}) |')
    lines+=['','Os valores `unknown` e `not_visible` são abstenções explícitas. No normalizador de pessoas, respostas não reconhecidas viram `invalid`. Para veículos, o normalizador histórico pode retornar `unknown` ou `other` para respostas genéricas; esses casos ficam visíveis nos dados brutos. A validade de formato abaixo verifica a lista de opções de cada campo e não implica correção visual.','',
            '| Modelo | Eventos com campos inválidos (1ª execução) | Erros de execução (todas) | Imagens com variação entre repetições | Tempo de carga (s) |',
            '|---|---:|---:|---:|---:|']
    for model in ['laya','blip']:
        rows=[r for r in first if r['model']==model]
        varying=sum(len({json.dumps(r['predicted'],sort_keys=True) for r in records if r['id']==it['id'] and r['model']==model})>1 for it in items)
        invalid=sum(any(v not in doc['laya_questions'][r['domain']][k]['criteria'] for k,v in r['predicted'].items()) for r in rows)
        errors=sum(bool(r['error']) for r in records if r['model']==model)
        lines.append(f'| {model} | {invalid}/{len(rows)} | {errors} | {varying}/{len(items)} | {doc["models"][model]["load_s"]:.2f} |')
    lines+=['','## Controles sem objeto','', '| Domínio | Modelo | Presença falsamente positiva | Atributos indevidos antes do filtro | Atributos indevidos após filtro |','|---|---|---:|---:|---:|']
    for domain in ['vehicle','person']:
        sentinel='unknown' if domain=='vehicle' else 'not_visible'
        for model in ['laya','blip']:
            rows=[r for r in first if r['domain']==domain and r['model']==model and r['expected']['presence']=='no']
            false=sum(r['predicted']['presence']=='yes' for r in rows)
            counts=[sum(any(v!=sentinel for k,v in r[field].items() if k!='presence') for r in rows) for field in ['normalized_before_gate','predicted']]
            lines.append(f'| {domain} | {model} | {false}/{len(rows)} | {counts[0]}/{len(rows)} | {counts[1]}/{len(rows)} |')
    lines+=['','## Método e limites','',
            '- Veículos: presença, marca, cor e carroceria. Pessoas: presença, capacete, transporte de objeto e roupa superior.',
            '- BLIP: quatro perguntas curtas abertas, processadas em batch; normalização lexical local. Laya: quatro decisões tipadas em uma chamada, com opções fixas para todas as imagens. Nenhuma opção depende do gabarito; todas as 48 marcas conhecidas do comparador estão disponíveis.',
            '- Ambos executam os quatro campos. Quando a presença prevista é `no`, o mesmo filtro substitui os outros três campos por ausência. Respostas anteriores ao filtro são preservadas.',
            '- Latência inclui preparação dos tensores, inferência dos quatro campos, normalização e filtro. Leitura de arquivo, carga do modelo e warm-up ficam fora. Não é latência de uma pergunta isolada.',
            '- Rótulos revisados visualmente pelo agente antes da inferência; sem adjudicação humana independente. As cores foram revisadas nas 20 imagens de veículos; a marca de `real_ford_2.jpg` foi corrigida para Land Rover neste manifesto.',
            '- O dataset anterior de pessoas tinha rótulos derivados de categorias, imagens sem pessoa em categorias de capacetes e pessoas em controles negativos. Usamos um subconjunto visualmente revisado; os gabaritos antigos não foram alterados.',
            '- Imagens reutilizadas e amostra pequena, sem balanceamento por marca ou condição. Configurações selecionadas nos testes anteriores são candidatas; não se demonstra que sejam ótimas para todos os atributos.',
            '- Não há implementação do CaptionerManager neste repositório. Este teste avalia o contrato explícito de quatro atributos por domínio, sem validar integração, detecção ou recorte.',
            '- Probabilidades do Laya ficam nos dados brutos e não são presumidas calibradas. BLIP não recebe confiança inventada. Não houve ajuste de limiar no conjunto de teste.',
            '- As revisões disponíveis no cache e versões dos pacotes constam no JSON; execução local offline.', '',
            '## Respostas por imagem (primeira repetição)','', '| Imagem | Gabarito | Laya | BLIP |', '|---|---|---|---|']
    for it in items:
        def fmt(d): return ', '.join(f'{k}={v}' for k,v in d.items())
        rows={r['model']:r for r in first if r['id']==it['id']}
        lines.append(f'| [{it["id"]}](../../{it["rel_path"]}) | {fmt(it["expected"])} | {fmt(rows["laya"]["predicted"])} | {fmt(rows["blip"]["predicted"])} |')
    lines+=['','[Resultados brutos](test_7_complete_tags_results.json) · [Manifesto com hashes e anotações](../datasets/test_7_complete_tags_manifest.json)','',
            'Reprodução: `.venv/bin/python eval/run_test_7_complete_benchmark.py --threads 8 --repetitions 3` e `.venv/bin/python eval/generate_test_7_report.py`.']
    REPORT.write_text('\n'.join(lines)+'\n')
    RESULT.with_name('test_7_complete_tags_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="800" height="340" viewBox="0 0 800 340">','<rect width="800" height="340" fill="white"/>','<text x="24" y="30" font-family="sans-serif" font-size="20">Quatro atributos corretos por imagem (1ª repetição)</text>']
    for i,domain in enumerate(['vehicle','person','all']):
        for j,model in enumerate(['laya','blip']):
            m=metrics[domain][model];y=60+i*85+j*30;value=100*m['exact']/m['images'];color='#2563eb' if model=='laya' else '#d97706'
            svg += [f'<text x="20" y="{y+17}" font-family="sans-serif" font-size="14">{domain} / {model}</text>',f'<rect x="160" y="{y}" width="{value*5}" height="23" fill="{color}"/>', f'<text x="{170+value*5}" y="{y+17}" font-family="sans-serif" font-size="14">{value:.1f}% ({m["exact"]}/{m["images"]})</text>']
    svg.append('</svg>');RESULT.with_name('test_7_complete_tags_accuracy.svg').write_text('\n'.join(svg)+'\n')
    print(REPORT)
    print(json.dumps(metrics,indent=2))

if __name__=='__main__':main()

"""Summarize the automated pilot; keep mechanical findings separate from semantic quality."""
import argparse
import collections
import json
import math
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET

import numpy as np


def read_jsonl(path):return [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
def percent(a,b):return round(100*a/b,2) if b else None
def ratio_text(a,b):return f'{a}/{b} ({100*a/b:.1f}%)' if b else '—'
def latency(rows):
    vals=[r['elapsed_ms'] for r in rows if not r['error']]
    return dict(requests=len(rows),errors=sum(bool(r['error']) for r in rows),
                p50_ms=float(np.median(vals)),p90_ms=float(np.quantile(vals,.90)),p95_ms=float(np.quantile(vals,.95)),
                mean_ms=statistics.mean(vals),mean_cpu_s=statistics.mean(r['cpu_s'] for r in rows),
                process_rss_end_max_mib=max(r['rss_end_bytes'] for r in rows)/2**20,
                valid=sum(r.get('result',{}).get('valid',False) for r in rows))

def truncations(raw):
    hits=[]
    if isinstance(raw,dict):
        for k,v in raw.get('answers',{}).items():
            if 'truncated' in v:hits.append((k,v['truncated']))
        for k in ['presence_call','attributes_call']:
            if k in raw:hits.extend(truncations(raw[k]))
    return hits

def typed_answers(raw):
    if isinstance(raw,dict):
        yield from raw.get('answers',{}).items()
        for key in ['presence_call','attributes_call']:
            if key in raw:yield from typed_answers(raw[key])

def bootstrap(perf):
    indexed={(r['model'],r['id'],r['block']):r for r in perf}
    ids=sorted({r['id'] for r in perf});blocks=sorted({r['block'] for r in perf})
    domains={r['id']:r['domain'] for r in perf}
    matrices={m:np.array([[indexed[m,i,b]['elapsed_ms'] for b in blocks] for i in ids]) for m in ['laya','blip']}
    rng=np.random.default_rng(20261026);diff=[];ratios=[]
    strata=[[k for k,i in enumerate(ids) if domains[i]==d] for d in ['vehicle','person']]
    for _ in range(3000):
        ix=np.concatenate([rng.choice(s,len(s),replace=True) for s in strata]);cols=rng.choice(len(blocks),len(blocks),replace=True)
        med={m:float(np.median(v[np.ix_(ix,cols)])) for m,v in matrices.items()}
        diff.append(med['blip']-med['laya']);ratios.append(med['blip']/med['laya'])
    med={m:float(np.median(v)) for m,v in matrices.items()}
    return dict(method='3000 paired resamples of image IDs within domain and measurement blocks; convenience sample, no population generalization.',
                blip_minus_laya_median_ms=med['blip']-med['laya'],difference_ci95_ms=np.quantile(diff,[.025,.975]).tolist(),
                blip_over_laya_median_ratio=med['blip']/med['laya'],ratio_ci95=np.quantile(ratios,[.025,.975]).tolist())

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('run_dir');args=ap.parse_args();out=Path(args.run_dir).resolve()
    status=json.loads((out/'status.json').read_text());assert status['status']=='complete','Run incomplete'
    config=json.loads((out/'frozen_config.json').read_text());rows=read_jsonl(out/'predictions.jsonl')
    assert len(rows)==config['planned_records']==len({r['request_key'] for r in rows})
    audit=json.loads((out/'audit.json').read_text());env=json.loads((out/'environment.json').read_text())
    perf=[r for r in rows if r['stage']=='performance'];metrics={'performance':{},'prompts':{},'robustness':{},'synthetic':{},'integrity':{}}
    lines=['# Piloto automatizado — descrições visuais','',
           f'Execução concluída: **{len(rows)} eventos processados**, com respostas completas por domínio. Os eventos são reutilizações e variações de imagens históricas; não são {len(rows)} cenas independentes.', '',
           '**Escopo:** desempenho, estabilidade, formato, truncamento de entrada, sensibilidade a perturbações e fila. A qualidade semântica baseada nos gabaritos antigos permanece exploratória; não houve anotação humana nova.', '',
           '**Limitação da comparação de custo:** as medições principais do Laya usam o orçamento padrão, que trunca a pergunta de marca. Portanto, menor tempo não demonstra menor custo para receber as mesmas instruções completas. A auditoria adicional com 512 tokens removeu o truncamento no lote de seis imagens, mas precisa de validação de qualidade e medição maior antes de substituir essa configuração.', '',
           '## Integridade e amostra','',
           f'- {len(audit["rows"])} entradas históricas auditadas; {len({r["sha256"] for r in audit["rows"]})} imagens únicas por hash.',
           f'- {len(audit["duplicate_groups"])} grupos de duplicatas exatas entre tarefas: '+str(audit['duplicate_groups'])+'.',
           f'- {len(audit["historical_missing_files"])} arquivos ausentes encontrados nos manifestos históricos com caminho explícito.',
           '- Divisão entre desenvolvimento e teste final não foi criada a partir dessas imagens. Todo este piloto pertence à exploração; futuras cenas confirmatórias permanecem por coletar.',
           f'- Máquina: {env["cpu_model"]}; {env["physical_cpus"]} núcleos físicos, {env["logical_cpus"]} lógicos, {env["ram_bytes"]/2**30:.1f} GiB de RAM; PyTorch com {config["threads"]} threads.',
           '- Pesos carregados offline e fixados por commit; configurações e hashes de código congelados antes das inferências. Nenhum resultado dos testes anteriores foi sobrescrito.', '',
           '## Tempo e recursos — cinco blocos alternando ordem dos modelos','',
           '| Domínio | Modelo | Requisições | p50 / p90 / p95 (ms) | CPU média (s/evento) | RSS máximo ao fim da requisição (MiB) |',
           '|---|---|---:|---:|---:|---:|']
    for domain in ['vehicle','person','all']:
        metrics['performance'][domain]={}
        for model in ['laya','blip']:
            rs=[r for r in perf if r['model']==model and (domain=='all' or r['domain']==domain)]
            m=latency(rs);metrics['performance'][domain][model]=m
            lines.append(f'| {domain} | {model} | {len(rs)} | {m["p50_ms"]:.1f} / {m["p90_ms"]:.1f} / {m["p95_ms"]:.1f} | {m["mean_cpu_s"]:.2f} | {m["process_rss_end_max_mib"]:.0f} |')
    pair=bootstrap(perf);metrics['paired_latency']=pair
    lines += ['',f'Razão de medianas BLIP/Laya: **{pair["blip_over_laya_median_ratio"]:.2f}×**, intervalo pareado de 95% [{pair["ratio_ci95"][0]:.2f}, {pair["ratio_ci95"][1]:.2f}]. Reamostragem estratificada de imagens e blocos, mantendo os pares; não é uma estimativa universal de hardware ou tráfego.', '',
              f'Pico de RSS do processo amostrado a cada 100 ms durante toda a execução: **{status["process_peak_rss_bytes"]/2**20:.0f} MiB**. A coluna por modelo mede RSS nos finais das requisições; inclui Python e memória retida pelo alocador, não é memória exclusiva dos pesos nem pico interno de cada inferência.', '',
              'O tempo inclui abertura da imagem, processamento, todos os campos, normalização, filtro e renderização. Carga, warm-up e escrita do JSON ficam fora. A máquina manteve seus demais processos ativos; o ambiente inicial registra os maiores consumidores. p99 não é usado para uma conclusão de cauda com esta amostra.', '',
              '| Modelo | Tempo projetado por 1.000 eventos | CPU projetada por 1.000 eventos |', '|---|---:|---:|']
    for model in ['laya','blip']:
        m=metrics['performance']['all'][model]
        lines.append(f'| {model} | {m["mean_ms"]/60:.1f} min | {m["mean_cpu_s"]*1000/60:.1f} CPU-min |')
    loads=read_jsonl(out/'loads.jsonl')
    metrics['loads']={m:dict(observations=len([r for r in loads if r['model']==m]),median_load_s=statistics.median(r['load_s'] for r in loads if r['model']==m),median_warmup_s=statistics.median(r['warmup_s'] for r in loads if r['model']==m)) for m in ['laya','blip']}
    lines += ['', 'Projeções usam a média observada para esta mistura de imagens. Tempo de CPU soma trabalho de múltiplas threads e pode exceder o tempo de parede. Energia e custo em reais: **não medidos**, pois não há sensor acessível nem tarifa/hardware amortizado definidos.', '',
              'Carga observada com cache do sistema não controlado (não é carga fria): '+ '; '.join(f'{m}: {v["median_load_s"]:.2f} s de mediana em {v["observations"]} carregamentos' for m,v in metrics['loads'].items())+'. Warm-up é registrado à parte e excluído das latências.', '',
              '## Estabilidade de saídas','', '| Modelo | Imagens avaliadas | Mesma saída nas cinco repetições | Erros de execução |', '|---|---:|---:|---:|']
    for model in ['laya','blip']:
        rs=[r for r in perf if r['model']==model];ids=sorted({r['id'] for r in rs})
        stable=sum(len({json.dumps(r['result'].get('predicted'),sort_keys=True) for r in rs if r['id']==i})==1 for i in ids)
        metrics.setdefault('stability',{})[model]=dict(images=len(ids),stable=stable)
        lines.append(f'| {model} | {len(ids)} | {ratio_text(stable,len(ids))} | {sum(bool(r["error"]) for r in rs)} |')
    lines += ['', 'Estabilidade determinística não comprova correção.', '',
              '## Seis configurações completas por modelo','',
              'Doze imagens por configuração, incluindo positivos e controles. Não selecionar vencedor de qualidade com este gabarito exploratório. As linhas abaixo mostram custo/formato; a taxa de acerto nominal fica no JSON de métricas e não é confirmatória.', '',
              '| Modelo | Configuração | Eventos com esquema válido | p50 (ms) | Perguntas médias executadas | Eventos com entrada truncada |', '|---|---|---:|---:|---:|---:|']
    for model in ['laya','blip']:
        metrics['prompts'][model]={}
        for variant in config['configurations'][model]:
            rs=[r for r in rows if r['stage']=='prompts' and r['model']==model and r['variant']==variant]
            m=latency(rs);m['mean_questions']=statistics.mean(r['result'].get('questions_executed',0) for r in rs)
            m['truncated_events']=sum(bool(truncations(r['result'].get('raw'))) for r in rs)
            positives=[r for r in rs if r['exploratory_expected']['presence']=='yes']
            m['exploratory_positive_exact']=sum(r['exploratory_exact'] for r in positives);m['exploratory_positive_n']=len(positives)
            metrics['prompts'][model][variant]=m
            lines.append(f'| {model} | {variant} | {ratio_text(m["valid"],len(rs))} | {m["p50_ms"]:.1f} | {m["mean_questions"]:.2f} | {m["truncated_events"]}/{len(rs)} |')
    lines += ['', '`joint` executa todos os campos juntos. `gated` testa presença antes dos atributos; uma ausência prevista evita executar os outros campos. Se a presença é falsa, o evento positivo inteiro falha. O ganho depende da prevalência real de negativos, não da proporção artificial deste lote.', '',
              '### Truncamento no Laya','']
    trunc_count=collections.Counter();instruction_count=collections.Counter(); examples=[]
    for r in rows:
        for key,cut in truncations(r.get('result',{}).get('raw')):
            trunc_count[key]+=1
            if cut.get('instructions'):instruction_count[key]+=1
            if len(examples)<3:examples.append(dict(request_key=r['request_key'],field=key,truncation=cut))
    metrics['truncation']=dict(occurrences_by_field=dict(trunc_count),instruction_cut_by_field=dict(instruction_count),examples=examples)
    lines += [f'Campos com avisos de truncamento (inclui repetições): `{dict(trunc_count)}`. Instrução cortada: `{dict(instruction_count)}`.', '',
              'Os resultados brutos preservam quais opções e instruções foram cortadas. Quando a pergunta de marca é truncada, a comparação não representa a instrução integral pretendida. Isso limita a interpretação de acurácia anterior e exige uma configuração que caiba no orçamento antes de um estudo confirmatório. Não foram removidas marcas conforme o gabarito.', '',
              '## Sensibilidade a degradações','',
              'Comparação de cada variante com a inferência original da mesma imagem e configuração-base. Uma mudança pode ser erro ou correção; sem nova revisão de visibilidade, não chamamos isso de queda de acurácia.', '',
              '| Modelo | Transformação | Conjuntos alterados | Campos alterados | Eventos com esquema válido |', '|---|---|---:|---:|---:|']
    for model in ['laya','blip']:
        metrics['robustness'][model]={}
        base='descriptive_joint' if model=='laya' else 'direct_joint'
        orig={r['id']:r for r in rows if r['stage']=='prompts' and r['model']==model and r['variant']==base}
        for transform in config['perturbations']:
            rs=[r for r in rows if r['stage']=='robustness' and r['model']==model and r['transform']==transform]
            bundle=sum(r['result'].get('predicted')!=orig[r['id']]['result'].get('predicted') for r in rs)
            fields=sum(sum(v!=orig[r['id']]['result'].get('predicted',{}).get(k) for k,v in r['result'].get('predicted',{}).items()) for r in rs)
            m=dict(images=len(rs),changed_bundles=bundle,changed_fields=fields,valid=sum(r['result'].get('valid',False) for r in rs));metrics['robustness'][model][transform]=m
            lines.append(f'| {model} | {transform} | {ratio_text(bundle,len(rs))} | {fields}/{len(rs)*4} | {m["valid"]}/{len(rs)} |')
    lines += ['', 'Transformações: redução a 64×64 e retorno ao tamanho original; JPEG qualidade 20; brilho em 50%; blur gaussiano de raio 2. Nenhuma variante conta como cena independente.', '',
              '## Controles sintéticos sem objetos','', '| Modelo | Casos | Presença positiva indevida | Sem presença válida (parse/error) |', '|---|---:|---:|---:|']
    for model in ['laya','blip']:
        rs=[r for r in rows if r['stage']=='synthetic' and r['model']==model]
        yes=sum(r['result'].get('predicted',{}).get('presence')=='yes' for r in rs)
        invalid=sum(r['result'].get('predicted',{}).get('presence') not in {'yes','no'} for r in rs)
        metrics['synthetic'][model]=dict(cases=len(rs),false_yes=yes,invalid=invalid)
        lines.append(f'| {model} | {len(rs)} | {yes}/{len(rs)} | {invalid}/{len(rs)} |')
    lines += ['', 'Fundos uniformes preto/cinza/branco e ruído pseudoaleatório conhecido, consultados para ambos os domínios. Esses controles têm origem programática; não representam a diversidade de negativos reais. Desempenho neles não demonstra uma taxa operacional de falsos positivos.', '',
              '## Fila com 1, 2 e 4 clientes','',
              '**Um único worker de inferência serializado**, compartilhado por até quatro threads chamadoras. Este ensaio mede espera/fila e valida integridade sob chamadas concorrentes; não mede inferência paralela nem escalabilidade de múltiplos workers.', '',
              '| Modelo | Clientes | Requisições | Throughput (eventos/s) | p50 resposta incluindo fila (ms) | Espera média (ms) |', '|---|---:|---:|---:|---:|---:|']
    batches=read_jsonl(out/'queue_batches.jsonl');metrics['queue']=batches
    for batch in batches:
        rs=[r for r in rows if r['stage']=='queue' and r['model']==batch['model'] and r['concurrency']==batch['concurrency']]
        lines.append(f'| {batch["model"]} | {batch["concurrency"]} | {len(rs)} | {batch["throughput_per_s"]:.3f} | {np.median([r["response_ms"] for r in rs]):.1f} | {statistics.mean(r["queue_ms"] for r in rs):.1f} |')
    lines += ['', 'Cada nível tem apenas seis requisições. Trata-se de smoke test da fila; capacidade sustentada e SLA requerem um teste de carga maior. A espera não deve ser confundida com tempo interno do modelo.', '',
              '## Parser e renderização','']
    prior=Path(__file__).resolve().parents[1]/'eval/results/test_7_complete_tags_results.json'
    if prior.exists():
        from automated_description_core import normalize
        previous=json.loads(prior.read_text());changed=[]
        for r in previous['records']:
            if r['model']=='blip' and r['repetition']==1:
                for key,raw in r['raw'].items():
                    old=r['normalized_before_gate'][key];new=normalize(r['domain'],key,raw)
                    if old!=new:changed.append(dict(id=r['id'],field=key,raw=raw,old=old,new=new))
        metrics['parser_historical_differences']=changed
        lines += [f'O parser estrito foi congelado antes das novas inferências. Reprocessando as saídas históricas do BLIP, {len(changed)} campos mudam de interpretação; detalhes no JSON. Resposta não interpretável permanece `parse_error`, em vez de ser aceita silenciosamente como uma abstenção válida.', '']
    render_violations=[];queue_mismatches=[];probability_violations=[];typed_count=0
    for r in rows:
        result=r.get('result',{});pred=result.get('predicted',{});rendered=result.get('rendered',{})
        if any(pred.get(k)!=v for k,v in rendered.get('facts',{}).items()):render_violations.append(r['request_key'])
        if pred.get('presence')!='yes' and rendered.get('text') is not None:render_violations.append(r['request_key'])
        if r['model']=='laya':
            for field,answer in typed_answers(result.get('raw',{})):
                typed_count+=1
                probs=answer.get('probabilities',{})
                values=list(probs.values())
                expected_keys=set(config['configurations']['laya'][r['variant']][r['domain']][0][field]['criteria'])
                valid_numbers=bool(values) and all(isinstance(v,(int,float)) and math.isfinite(v) and 0<=v<=1 for v in values)
                if not valid_numbers or set(probs)!=expected_keys:
                    probability_violations.append(dict(request=r['request_key'],field=field,reason='invalid probability schema'))
                elif abs(sum(values)-1)>len(values)*.000051+.000001:
                    probability_violations.append(dict(request=r['request_key'],field=field,reason='probability sum',total=sum(values)))
                elif answer.get('choice') not in probs or probs[answer['choice']]+.000101<max(values):
                    probability_violations.append(dict(request=r['request_key'],field=field,reason='choice differs from maximum probability'))
    for model in ['laya','blip']:
        byid=collections.defaultdict(list)
        for r in rows:
            if r['stage']=='queue' and r['model']==model:byid[r['id']].append(r)
        for id,rs in byid.items():
            if len({json.dumps(r.get('result',{}).get('predicted'),sort_keys=True) for r in rs})>1:queue_mismatches.append(dict(model=model,id=id))
    metrics['integrity']=dict(records=len(rows),unique_request_keys=len({r['request_key'] for r in rows}),errors=sum(bool(r['error']) for r in rows),render_violations=render_violations,queue_prediction_mismatches=queue_mismatches,laya_typed_answers=typed_count,probability_violations=probability_violations)
    lines += [f'- Erros de execução: {metrics["integrity"]["errors"]}/{len(rows)}.',
              f'- Violações mecânicas do renderizador: {len(render_violations)}. Isso comprova consistência com os atributos previstos, não verdade dos atributos na imagem.',
              f'- Imagens com saída diferente entre níveis da fila: {len(queue_mismatches)}.', '',
              f'- Respostas tipadas do Laya verificadas: {typed_count}; violações de chaves, faixa, soma ou escolha de maior probabilidade: {len(probability_violations)}. Tolerância da soma respeita arredondamento a quatro casas. Validade numérica não comprova calibração.', '',
              '## O que fica pendente','',
              '- Anotação humana independente, seleção confirmatória de prompts e avaliação das descrições factualmente corretas.',
              '- Dados novos e reservados, quotas de classe/condição e teste de transferência de domínio.',
              '- Calibração de confiança e custo por descrição correta, pois dependem de gabarito confiável e política de aceitação.',
              '- Energia/custo monetário, detector/recorte real e integração com CaptionerManager.',
              '- Teste de carga sustentada e inferência realmente paralela com controle de memória e número de threads.', '',
              '## Artefatos','',
              '- [Configuração congelada](frozen_config.json)', '- [Auditoria](audit.json)', '- [Ambiente](environment.json)',
              '- [Predições brutas](predictions.jsonl)', '- [Recursos amostrados](resource_samples.jsonl)', '- [Métricas](metrics.json)',
              '- [Orçamento de opções do Laya](../laya_option_budget_20261026/report.md)', '',
              '- [Simulação de poder e tamanho da amostra](../paired_power_20261026/report.md)', '',
              '![Latência por conjunto completo](summary.svg)']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    (out/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n')
    svg=ET.Element('svg',xmlns='http://www.w3.org/2000/svg',width='850',height='230',viewBox='0 0 850 230')
    ET.SubElement(svg,'rect',width='850',height='230',fill='white')
    title=ET.SubElement(svg,'text',x='20',y='30',fill='#111827',**{'font-family':'sans-serif','font-size':'19'});title.text='Tempo mediano por conjunto completo — piloto CPU'
    maximum=max(metrics['performance']['all'][m]['p50_ms'] for m in ['laya','blip'])
    for i,(model,color) in enumerate([('laya','#2563eb'),('blip','#d97706')]):
        y=65+i*65;value=metrics['performance']['all'][model]['p50_ms'];width=500*value/maximum
        ET.SubElement(svg,'text',x='20',y=str(y+23),**{'font-family':'sans-serif','font-size':'18'}).text=model
        ET.SubElement(svg,'rect',x='100',y=str(y),width=str(width),height='33',fill=color)
        ET.SubElement(svg,'text',x=str(115+width),y=str(y+23),**{'font-family':'sans-serif','font-size':'16'}).text=f'{value:.0f} ms'
    ET.SubElement(svg,'text',x='20',y='215',**{'font-family':'sans-serif','font-size':'13'}).text='16 eventos históricos por modelo × 5 blocos; não mede qualidade factual.'
    ET.ElementTree(svg).write(out/'summary.svg',encoding='unicode',xml_declaration=False)
    print(out/'report.md');print(json.dumps(metrics['performance']['all'],indent=2));print(json.dumps(metrics['integrity']))

if __name__=='__main__':main()

"""Dataset evaluation: quality on first repetition, performance on measured calls."""
from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path
import statistics
from itertools import combinations

import numpy as np

from .core import load_json, write_json


def ratio(correct, total):
    return correct/total if total else None


def field_scores(pairs):
    """Invalid predictions remain errors. Macro F1 uses classes with truth support."""
    classes=sorted({truth for truth,_ in pairs})
    matrix={truth:dict(Counter(pred for expected,pred in pairs if expected==truth)) for truth in classes}
    per_class=[]
    for label in classes:
        tp=sum(a==label and b==label for a,b in pairs)
        fp=sum(a!=label and b==label for a,b in pairs)
        fn=sum(a==label and b!=label for a,b in pairs)
        precision=tp/(tp+fp) if tp+fp else 0
        recall=tp/(tp+fn) if tp+fn else 0
        f1=2*precision*recall/(precision+recall) if precision+recall else 0
        per_class.append(dict(label=label,support=tp+fn,precision=precision,recall=recall,f1=f1))
    correct=sum(a==b for a,b in pairs)
    return dict(n=len(pairs),correct=correct,accuracy=ratio(correct,len(pairs)),
                macro_precision=statistics.mean(c['precision'] for c in per_class) if per_class else None,
                macro_recall=statistics.mean(c['recall'] for c in per_class) if per_class else None,
                macro_f1=statistics.mean(c['f1'] for c in per_class) if per_class else None,
                classes=per_class,confusion=matrix)


def evaluate(frozen, rows, loads=None):
    ids=frozen['item_ids'];lookup={it['id']:it for it in frozen['manifest']['items'] if it['id'] in set(ids)}
    if len({r['request_key'] for r in rows})!=len(rows):
        raise ValueError('Duplicate predictions cannot be used for evaluation')
    summary=dict(schema_version=1,dataset_name=frozen['manifest'].get('name','Dataset de exemplo'),
                 label_status=frozen['manifest'].get('label_status','exploratory_unadjudicated'),
                 simulated=any(s['adapter']=='mock' for s in frozen['models']),
                 photos=len(ids),unique_images=len({it['sha256'] for it in lookup.values()}),
                 repetitions=frozen['repetitions'],models=[],
                 methodology=dict(quality='Primeira repetição por foto; falhas contam como erros. Campos sem gabarito são excluídos.',
                                  exact='Todos os atributos corretos na mesma foto, somente em fotos com gabarito completo.',
                                  latency='Chamadas medidas com sucesso, incluindo pré/pós-processamento; carga e aquecimento excluídos.',
                                  macro_f1='Média por classe com suporte no gabarito; divisão por zero = 0.',
                                  resources='HTTP sem PID mede o cliente, não o consumo do servidor Ollama.'))
    for spec in frozen['models']:
        model=spec['id'];rs=[r for r in rows if r['model']==model];first={r['id']:r for r in rs if r['block']==0}
        times=[r['elapsed_ms'] for r in rs if not r.get('error')]
        photos=[];fields={};exact_n=exact_correct=0
        for item_id in ids:
            item=lookup[item_id];row=first.get(item_id,{})
            expected=item.get('expected') or item.get('annotations') or {}
            predicted=row.get('predicted',{})
            error=row.get('error') or (None if row else 'Previsão ausente')
            correct={key:bool(not error and predicted.get(key)==value) for key,value in expected.items()}
            for key,value in expected.items():
                fields.setdefault((item['domain'],key),[]).append((value,'__error__' if error else predicted.get(key,'parse_error')))
            complete=set(expected)==set(frozen['contract']['domains'][item['domain']]['questions'])
            exact=None
            if complete:
                exact_n+=1;exact=all(correct.values());exact_correct+=exact
            photos.append(dict(id=item_id,filename=item.get('filename',Path(item['rel_path']).name),domain=item['domain'],
                               expected=expected,predicted=predicted,correct=correct,exact=exact,error=error,
                               elapsed_ms=row.get('elapsed_ms'),valid=bool(not error and row.get('valid'))))
        field_results=[dict(domain=domain,field=key,**field_scores(pairs)) for (domain,key),pairs in fields.items()]
        for field in field_results:
            key=field['field'];domain=field['domain']
            measured=[r for r in rs if r.get('domain',lookup[r['id']]['domain'])==domain]
            times_field=[r['protocol_metrics']['attribute_ms'][key] for r in measured if not r.get('error') and key in r.get('protocol_metrics',{}).get('attribute_ms',{})]
            diagnostics=[r['output_diagnostics'][key] for r in first.values() if r.get('domain',lookup[r['id']]['domain'])==domain and key in r.get('output_diagnostics',{})]
            field.update(attribute_latency=dict(n=len(times_field),mean_ms=statistics.mean(times_field) if times_field else None,p50_ms=float(np.median(times_field)) if times_field else None,p95_ms=float(np.quantile(times_field,.95)) if times_field else None),
                         raw_adherence=ratio(sum(d['adherent'] for d in diagnostics),len(diagnostics)),
                         output_format_valid=ratio(sum(d['format_valid'] for d in diagnostics),len(diagnostics)),
                         ambiguity_rate=ratio(sum(d['ambiguous'] for d in diagnostics),len(diagnostics)))
        field_n=sum(f['n'] for f in field_results);field_correct=sum(f['correct'] for f in field_results)
        total_ms=sum(r['elapsed_ms'] for r in rs)
        successful=sum(not bool(r.get('error')) for r in rs)
        resources=sorted({r.get('resource_scope','unknown') for r in rs})
        model_loads=[row for row in (loads or []) if row['model']==model]
        protocols=[]
        for row in rs:
            if row.get('request') and row['request'] not in protocols:protocols.append(row['request'])
        result=dict(model=model,base_model=spec.get('base_model',model),adapter=spec['adapter'],profile=spec.get('profile','short_open' if spec['adapter']=='blip' else 'native_short' if spec['adapter']=='laya' else 'baseline'),protocols=protocols,
                    inference_calls=sum(r.get('protocol_metrics',{}).get('inference_calls',0) for r in rs),
                    simulated=spec['adapter']=='mock',photos=len(ids),requests=len(rs),
                    annotated_photos=sum(bool(p['expected']) for p in photos),complete_labeled_photos=exact_n,
                    exact_correct=exact_correct,exact_accuracy=ratio(exact_correct,exact_n),
                    field_correct=field_correct,field_n=field_n,field_accuracy=ratio(field_correct,field_n),
                    fields=field_results,photo_results=photos,errors=len(rs)-successful,
                    success_rate=ratio(successful,len(rs)),schema_valid_rate=ratio(sum(bool(not r.get('error') and r.get('valid')) for r in rs),len(rs)),
                    latency=dict(mean_ms=statistics.mean(times) if times else None,p50_ms=float(np.median(times)) if times else None,
                                 p95_ms=float(np.quantile(times,.95)) if times else None,min_ms=min(times) if times else None,
                                 max_ms=max(times) if times else None,std_ms=statistics.stdev(times) if len(times)>1 else (0 if times else None)),
                    measured_s=total_ms/1000,successful_calls_s=successful/(total_ms/1000) if total_ms else None,
                    mean_cpu_s=statistics.mean(r.get('cpu_s',0) for r in rs) if rs else None,
                    peak_rss_mib=max((r.get('peak_rss_bytes',0) for r in rs),default=0)/2**20,resource_scope=resources,
                    load_s=sum(r.get('load_or_client_init_s',0) for r in model_loads),warmup_s=sum(r.get('warmup_s',0) for r in model_loads),
                    warmup_failures=sum(len(r.get('warmup_failed_domains',[])) for r in model_loads))
        summary['models'].append(result)
    summary['comparisons']=paired_comparisons(summary,lookup)
    summary['methodology'].update(protocol='BLIP gera texto; Laya escolhe alternativas. Aderência bruta é medida antes da normalização e do filtro de ausência. JSON exige objeto exato e classe válida.',
        attribute_latency='Tempo por atributo é medido somente quando há chamada isolada. O tempo de um batch ou request conjunto não é dividido entre atributos.',
        comparisons='Diferenças pareadas usam as mesmas fotos. IC 95% por bootstrap de grupos de origem (2.000 reamostragens, seed fixa); não equivale a adjudicação dos gabaritos. Não há teste de significância nem seleção automática de vencedor.',
        cost='Custo monetário e energia não foram medidos. CPU, RAM e tempo local não equivalem a preço de API.')
    return summary


def paired_comparisons(summary,lookup):
    result=[]
    for a,b in combinations(summary['models'],2):
        left={p['id']:p for p in a['photo_results']};right={p['id']:p for p in b['photo_results']}
        fields=sorted({(p['domain'],key) for p in left.values() for key in p['correct']})
        for domain,key in fields:
            pairs=[(pid,p['correct'][key],right[pid]['correct'][key]) for pid,p in left.items() if p['domain']==domain and key in p['correct'] and pid in right and key in right[pid]['correct']]
            if not pairs:continue
            groups={}
            for pid,x,y in pairs:
                item=lookup[pid];g=item.get('group_id',item['sha256']);bucket=groups.setdefault(g,[0,0]);bucket[0]+=int(y)-int(x);bucket[1]+=1
            values=np.asarray(list(groups.values()),dtype=float);interval=None
            if len(values)>1:
                rng=np.random.default_rng(20261027);deltas=[]
                batch=max(1,min(200,1_000_000//len(values)))
                for start in range(0,2000,batch):
                    draws=rng.integers(0,len(values),(min(batch,2000-start),len(values)))
                    samples=values[draws].sum(axis=1);deltas.extend(samples[:,0]/samples[:,1])
                interval=[float(v) for v in np.quantile(deltas,[.025,.975])]
            result.append(dict(a=a['model'],b=b['model'],domain=domain,field=key,n=len(pairs),groups=len(groups),
                delta_b_minus_a=sum(int(y)-int(x) for _,x,y in pairs)/len(pairs),ci95=interval,
                b_only_correct=sum(not x and y for _,x,y in pairs),a_only_correct=sum(x and not y for _,x,y in pairs)))
    return result


def generate(run, output):
    run=Path(run);output=Path(output)
    frozen=load_json(run/'frozen_config.json')
    rows=[json.loads(line) for line in (run/'predictions.jsonl').read_text().splitlines()]
    status=load_json(run/'status.json')
    if status['status']!='complete' or len(rows)!=status['planned']:
        raise ValueError('Only complete runs can produce a final dataset evaluation')
    loads=[json.loads(line) for line in (run/'loads.jsonl').read_text().splitlines()] if (run/'loads.jsonl').exists() else []
    summary=evaluate(frozen,rows,loads)
    metadata=load_json(run/'run.json')
    summary.update(machine=metadata['machine_id'],contract_hash=metadata['contract_hash'],dataset_hash=metadata['dataset_hash'])
    write_json(output/'protocols.json',[dict(model=m['model'],protocols=m['protocols']) for m in summary['models']])
    columns=['model','base_model','adapter','profile','inference_calls','simulated','photos','annotated_photos','complete_labeled_photos','requests','repetitions',
             'exact_correct','exact_accuracy','field_correct','field_n','field_accuracy','errors','success_rate',
             'schema_valid_rate','mean_ms','p50_ms','p95_ms','min_ms','max_ms','std_ms','successful_calls_s',
             'measured_s','load_s','warmup_s','warmup_failures','mean_cpu_s','peak_rss_mib','resource_scope']
    with (output/'models.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        for model in summary['models']:
            row={key:model['latency'].get(key,model.get(key)) for key in columns}
            row.update(repetitions=summary['repetitions'],resource_scope=','.join(model['resource_scope']))
            writer.writerow(row)
    with (output/'photos.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['model','filename','domain','elapsed_ms','valid','exact','error','expected','predicted'])
        writer.writeheader()
        for model in summary['models']:
            for photo in model['photo_results']:
                row={key:photo.get(key) for key in writer.fieldnames};row['model']=model['model']
                for key in ['expected','predicted']:row[key]=json.dumps(row[key],ensure_ascii=False)
                writer.writerow(row)
    with (output/'attributes.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['model','domain','field','n','correct','accuracy','macro_precision','macro_recall','macro_f1','raw_adherence','output_format_valid','ambiguity_rate','isolated_calls','attribute_p50_ms','attribute_p95_ms'])
        writer.writeheader()
        for model in summary['models']:
            for field in model['fields']:
                row={key:field.get(key) for key in writer.fieldnames}
                row.update(model=model['model'],isolated_calls=field['attribute_latency']['n'],attribute_p50_ms=field['attribute_latency']['p50_ms'],attribute_p95_ms=field['attribute_latency']['p95_ms'])
                writer.writerow(row)
    _quality_chart(summary,output)
    _attribute_charts(summary,output)
    write_json(output/'evaluation.json',summary)
    with (output/'comparisons.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        columns=['a','b','domain','field','n','groups','delta_b_minus_a','ci95','b_only_correct','a_only_correct']
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();writer.writerows(summary['comparisons'])
    return summary


def _quality_chart(summary,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    models=summary['models']
    if not any(model['fields'] for model in models):
        return
    keys=sorted({(f['domain'],f['field']) for model in models for f in model['fields']})
    fig,ax=plt.subplots(figsize=(max(8,len(keys)*1.2),5))
    width=.8/len(models);x=np.arange(len(keys))
    for index,model in enumerate(models):
        values={(f['domain'],f['field']):f['accuracy'] for f in model['fields']}
        ax.bar(x-.4+width/2+index*width,[100*values.get(key,0) for key in keys],width,label=model['model'])
    ax.set_xticks(x,[f'{domain}\n{field}' for domain,field in keys]);ax.set_ylim(0,105)
    ax.set_ylabel('Acurácia por atributo (%)');ax.grid(axis='y',alpha=.2);ax.legend()
    ax.set_title(('SIMULAÇÃO — ' if summary['simulated'] else '')+summary['dataset_name'])
    fig.text(.02,.015,'Primeira repetição por foto. Falhas contam como erros. Gabaritos ausentes são excluídos.\nRepetições de tempo não aumentam a amostra de qualidade.',fontsize=8)
    fig.tight_layout(rect=(0,.07,1,1))
    for suffix in ['png','pdf']:fig.savefig(output/f'accuracy.{suffix}',dpi=300)
    plt.close(fig)


def _attribute_charts(summary,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    summary['charts']=[]
    keys=sorted({(f['domain'],f['field']) for m in summary['models'] for f in m['fields']})
    for domain,key in keys:
        models=[m for m in summary['models'] if any(f['domain']==domain and f['field']==key for f in m['fields'])]
        fields=[next(f for f in m['fields'] if f['domain']==domain and f['field']==key) for m in models]
        names=[m['model'] for m in models];x=np.arange(len(models));colors=plt.get_cmap('tab10')(np.arange(len(models))%10)
        fig,axes=plt.subplots(2,2,figsize=(max(11,len(models)*1.1),10))
        ax=axes[0,0];ax.barh(x,[f['accuracy'] for f in fields],color=colors);ax.set_yticks(x,names,fontsize=7);ax.set_xlim(0,1.05);ax.xaxis.set_major_formatter(PercentFormatter(1));ax.set_title('Acurácia — primeira repetição')
        for i,f in enumerate(fields):ax.text(min(f['accuracy']+.01,.92),i,f"{f['correct']}/{f['n']}",va='center',fontsize=8)
        ax=axes[0,1]
        for i,(metric,label) in enumerate([('macro_precision','Precisão macro'),('macro_recall','Recall macro'),('macro_f1','F1 macro')]):ax.bar(x+(i-1)*.25,[f[metric] for f in fields],.25,label=label)
        ax.set_ylim(0,1.05);ax.yaxis.set_major_formatter(PercentFormatter(1));ax.legend(fontsize=8);ax.set_title('Qualidade com peso igual por classe')
        ax=axes[1,0]
        for i,(metric,label) in enumerate([('raw_adherence','Aderência bruta'),('output_format_valid','Formato válido'),('ambiguity_rate','Ambiguidade')]):ax.bar(x+(i-1)*.25,[np.nan if f.get(metric) is None else f[metric] for f in fields],.25,label=label)
        ax.set_ylim(0,1.05);ax.yaxis.set_major_formatter(PercentFormatter(1));ax.legend(fontsize=8);ax.set_title('Saída antes da normalização (menor ambiguidade é melhor)')
        ax=axes[1,1];measured=[f['attribute_latency']['p50_ms'] for f in fields]
        ax.bar(x,[np.nan if v is None else v for v in measured],color=colors);ax.set_ylabel('ms');ax.set_title('Latência p50 em chamada isolada por atributo')
        if not any(v is not None for v in measured):ax.text(.5,.5,'Requests conjuntos: tempo por atributo não é isolável.',ha='center',va='center',transform=ax.transAxes,wrap=True)
        for ax in [axes[0,1],axes[1,0],axes[1,1]]:ax.set_xticks(x,names,rotation=45,ha='right',fontsize=7)
        for ax in axes.flat:ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
        fig.suptitle(('SIMULAÇÃO — ' if summary['simulated'] else '')+f'{domain} / {key} · {summary["dataset_name"]}',fontsize=13)
        fig.text(.02,.012,'Qualidade: uma previsão por evento. Repetições medem tempo e não ampliam a amostra. Gabaritos históricos são exploratórios.\nBarras ausentes indicam métrica não medida. Tempos de requests conjuntos não foram divididos artificialmente.',fontsize=8)
        fig.tight_layout(rect=(0,.05,1,.95));stem=f'attribute_{domain}_{key}'
        for ext in ['png','pdf']:fig.savefig(output/f'{stem}.{ext}',dpi=300)
        plt.close(fig);summary['charts'].append(dict(title=f'{domain}: {key} — qualidade, formato e tempo',file=stem))
        labels=sorted({c['label'] for f in fields for c in f['classes']})
        matrix=np.full((len(models),len(labels)),np.nan)
        for i,f in enumerate(fields):
            for c in f['classes']:matrix[i,labels.index(c['label'])]=c['f1']
        fig,ax=plt.subplots(figsize=(max(8,len(labels)*.65),max(3,len(models)*.6+2)))
        im=ax.imshow(matrix,vmin=0,vmax=1,cmap='Blues',aspect='auto');fig.colorbar(im,ax=ax,label='F1 por classe')
        ax.set_xticks(range(len(labels)),labels,rotation=45,ha='right',fontsize=8);ax.set_yticks(range(len(models)),names,fontsize=8)
        for i in range(len(models)):
            for j in range(len(labels)):
                if np.isfinite(matrix[i,j]):ax.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',fontsize=7,color='white' if matrix[i,j]>.6 else 'black')
        ax.set_title(f'{domain} / {key}: F1 de cada classe — classes com suporte no gabarito');fig.tight_layout()
        stem=f'classes_{domain}_{key}'
        for ext in ['png','pdf']:fig.savefig(output/f'{stem}.{ext}',dpi=300)
        plt.close(fig);summary['charts'].append(dict(title=f'{domain}: {key} — F1 por classe',file=stem))
        for m,f in zip(models,fields):
            labels=sorted(set(f['confusion'])|{p for row in f['confusion'].values() for p in row})
            matrix=np.array([[f['confusion'].get(a,{}).get(b,0) for b in labels] for a in labels])
            fig,ax=plt.subplots(figsize=(max(6,len(labels)*.5),max(5,len(labels)*.45)))
            im=ax.imshow(matrix,cmap='Blues');fig.colorbar(im,ax=ax,label='Eventos');ax.set_xticks(range(len(labels)),labels,rotation=60,ha='right',fontsize=7);ax.set_yticks(range(len(labels)),labels,fontsize=7)
            ax.set_xlabel('Previsão');ax.set_ylabel('Gabarito');ax.set_title(f'{m["model"]}\n{domain} / {key}')
            if len(labels)<18:
                for i in range(len(labels)):
                    for j in range(len(labels)):
                        if matrix[i,j]:ax.text(j,i,str(matrix[i,j]),ha='center',va='center',fontsize=8)
            fig.tight_layout();stem=f'confusion_{m["model"]}_{domain}_{key}'
            for ext in ['png','pdf']:fig.savefig(output/f'{stem}.{ext}',dpi=300)
            plt.close(fig);summary['charts'].append(dict(title=f'{m["model"]}: matriz {domain}/{key}',file=stem))

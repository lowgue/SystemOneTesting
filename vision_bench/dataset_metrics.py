"""Dataset evaluation: quality on first repetition, performance on measured calls."""
from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path
import statistics

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
        field_n=sum(f['n'] for f in field_results);field_correct=sum(f['correct'] for f in field_results)
        total_ms=sum(r['elapsed_ms'] for r in rs)
        successful=sum(not bool(r.get('error')) for r in rs)
        resources=sorted({r.get('resource_scope','unknown') for r in rs})
        model_loads=[row for row in (loads or []) if row['model']==model]
        result=dict(model=model,simulated=spec['adapter']=='mock',photos=len(ids),requests=len(rs),
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
    return summary


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
    write_json(output/'evaluation.json',summary)
    columns=['model','simulated','photos','annotated_photos','complete_labeled_photos','requests','repetitions',
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
        writer=csv.DictWriter(stream,fieldnames=['model','domain','field','n','correct','accuracy','macro_f1'])
        writer.writeheader()
        for model in summary['models']:
            for field in model['fields']:writer.writerow({key:model['model'] if key=='model' else field[key] for key in writer.fieldnames})
    _quality_chart(summary,output)
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

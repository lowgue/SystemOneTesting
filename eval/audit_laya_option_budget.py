"""Audit token budgets without scoring accuracy or excluding labels based on image content."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

from PIL import Image
from run_automated_description_pilot import Engine, REVISIONS, config, SOURCE, ROOT, select


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run-dir',required=True);args=ap.parse_args();out=Path(args.run_dir).resolve()
    out.mkdir(parents=True,exist_ok=False)
    import torch
    torch.set_num_threads(8)
    engine=Engine('laya');agent=engine.net
    from laya.vlm import build_vlm_inputs,vlm_prefix,truncation_answer
    item=next(it for it in json.loads(SOURCE.read_text())['items'] if it['domain']=='vehicle')
    with Image.open(ROOT/item['rel_path']) as src:image=src.convert('RGB')
    prefix=vlm_prefix(agent.processor,[image],agent.prep)
    cfg=agent.cfg;full=config('laya','descriptive_joint','vehicle')[0]['make']['criteria'];rows=[]
    # Strictly mechanical test: no predictions or image labels are used to choose options.
    for style in ['descriptive','short','visible']:
        qbase=config('laya',style+'_joint','vehicle')[0]['make']
        labels=sorted(k for k in qbase['criteria'] if k!='unknown')
        for n in [8,12,16,24,32,48]:
            for budget in sorted(set([cfg.get('head_max_len',256),384,512])):
                q=deepcopy(qbase);keys=labels[:n]+['unknown'];q['criteria']={k:q['criteria'][k] for k in keys}
                internal=agent._to_internal(q)
                built=build_vlm_inputs(agent.processor,image,internal,max_len=cfg.get('max_len',1024),head_max_len=budget,prefix=prefix)
                cut=truncation_answer(built['truncation'],internal)
                rows.append(dict(style=style,brands=n,options=n+1,head_max_len=budget,default_budget=budget==cfg.get('head_max_len',256),
                                 input_tokens=len(built['ids']),truncated=bool(cut),truncation=cut,
                                 textual_tail=agent.processor.tokenizer.decode(built['ids'][-budget:]),labels=keys))
    default_budget=cfg.get('head_max_len',256)
    examples=select(json.loads(SOURCE.read_text())['items'],4,2)
    examples=[it for it in examples if it['domain']=='vehicle']
    experiment=dict(model_revision=REVISIONS['laya'],budgets=[default_budget,512],ids=[it['id'] for it in examples],
                    configuration=config('laya','descriptive_joint','vehicle'),purpose='Input integrity, latency and answer changes only; no semantic correctness scoring.')
    (out/'inference_config.json').write_text(json.dumps(experiment,ensure_ascii=False,indent=2)+'\n')
    predictions=[]
    try:
        for budget in experiment['budgets']:
            agent.cfg['head_max_len']=budget
            for it in examples:
                with Image.open(ROOT/it['rel_path']) as src:im=src.convert('RGB')
                start=time.perf_counter()
                answer=engine.infer(im,'vehicle','descriptive_joint')
                predictions.append(dict(id=it['id'],sha256=it['sha256'],head_max_len=budget,elapsed_ms=(time.perf_counter()-start)*1000,result=answer))
    finally:
        agent.cfg['head_max_len']=default_budget
    (out/'predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in predictions))
    result=dict(model_revision=REVISIONS['laya'],checkpoint_head_max_len=default_budget,checkpoint_max_len=cfg.get('max_len',1024),
                full_criteria=full,rows=rows,inference_events=len(predictions),limitation='Tokenization and six-image input-integrity pilot. Alternative budgets and label subsets have NOT been validated for semantic prediction quality.')
    (out/'budget_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    lines=['# Auditoria automática do orçamento de opções — Laya','',
           f'Orçamento padrão do checkpoint: **{result["checkpoint_head_max_len"]} tokens de cabeça**, sequência máxima {result["checkpoint_max_len"]}.', '',
           'Esta auditoria constrói entradas usando o tokenizer e as regras do checkpoint. Não usa gabarito nem executa avaliação de acurácia. Há um piloto adicional em seis imagens de veículos comparando a entrada padrão e a cabeça de 512 tokens, mantendo todas as marcas. As listas menores da tabela são prefixos alfabéticos fixos; não são propostas de excluir a marca correta de uma tarefa real.', '',
           '| Critérios | Marcas + desconhecida | Orçamento | Instrução cortada | Opções cortadas | Truncamento total |', '|---|---:|---:|---|---:|---|']
    for r in rows:
        t=r['truncation'] or {}
        lines.append(f'| {r["style"]} | {r["options"]} | {r["head_max_len"]} | {bool(t.get("instructions"))} | {len(t.get("options",[]))} | {r["truncated"]} |')
    lines+=['','## Inferência com a taxonomia completa','', '| Orçamento de cabeça | Eventos | Eventos com truncamento | Mediana (ms) |', '|---:|---:|---:|---:|']
    import statistics
    for budget in experiment['budgets']:
        rs=[r for r in predictions if r['head_max_len']==budget]
        cut=sum(any('truncated' in a for a in r['result']['raw']['answers'].values()) for r in rs)
        lines.append(f'| {budget} | {len(rs)} | {cut}/{len(rs)} | {statistics.median(r["elapsed_ms"] for r in rs):.1f} |')
    bykey={(r['head_max_len'],r['id']):r for r in predictions}
    changed=sum(bykey[default_budget,it['id']]['result']['predicted']!=bykey[512,it['id']]['result']['predicted'] for it in examples)
    lines+=['',f'Conjuntos de respostas alterados entre os dois orçamentos: {changed}/{len(examples)}. Mudança não significa melhora. As medições deste pequeno lote não substituem os blocos alternados do ensaio de desempenho.', '',
            '**Implicação:** validar o orçamento antes de medir qualidade. Uma decisão tipada pode ser perfeitamente válida em JSON e ainda ter recebido uma instrução ou opções cortadas.', '',
            'Aumentar o orçamento no construtor não demonstra que o modelo mantenha qualidade fora da configuração treinada. Reduzir a taxonomia também muda a tarefa. Qualquer decomposição em grupos deve preservar todas as opções necessárias e ser avaliada posteriormente com gabarito confiável.', '',
            '[Entradas, textos efetivos e detalhes dos cortes](budget_audit.json)']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print(out/'report.md')

if __name__=='__main__':main()

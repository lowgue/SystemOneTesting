import React, { useState, useEffect } from "react";
import { api } from "../api.js";
import JobState from "./JobState.jsx";
import DatasetImport from "./DatasetImport.jsx";
import Evaluation from "./Evaluation.jsx";

export default function Benchmark({
  catalog,
  job,
  busy,
  onRun,
  onCancel,
  contract,
  onCatalogRefresh,
  settings, setSettings,
}) {
  const [dataset, setDataset] = useState("portable"),
    [repetitions, setRepetitions] = useState(3),
    [limit, setLimit] = useState(0),
    [importing, setImporting] = useState(false);
  const [comparison,setComparison]=useState("baseline"),[preview,setPreview]=useState(null),[previewError,setPreviewError]=useState("");
  const selected = catalog.datasets.find((d) => d.id === dataset);
  const effectiveContract=selected?.required_contract || contract;
  useEffect(()=>{
    if(!busy && selected?.required_contract && settings?.contract!==selected.required_contract)
      setSettings(s=>({...s,contract:selected.required_contract}));
  },[selected?.required_contract,settings?.contract,busy,setSettings]);
  useEffect(()=>{
    if(job?.experiment){setDataset(job.experiment.dataset);setRepetitions(job.experiment.repetitions);setLimit(['queued','running'].includes(job.state) ? job.experiment.limit : 0);setComparison(job.experiment.comparison);}
    else if(job?.dataset)setDataset(job.dataset);
  },[job?.id]);
  useEffect(()=>{
    let active=true;
    setPreview(null);setPreviewError("");
    if(!settings?.models.length)return;
    const timer=setTimeout(()=>api('/api/protocol-preview',{...settings,contract:effectiveContract,dataset,comparison,repetitions,limit}).then(d=>{if(active)setPreview(d)}).catch(e=>{if(active)setPreviewError(e.message)}),250);
    return()=>{active=false;clearTimeout(timer)};
  },[settings,effectiveContract,dataset,comparison,repetitions,limit]);
  async function imported(result) {
    await onCatalogRefresh();
    setDataset(result.id);
    setLimit(0);
    setImporting(false);
  }
  return (
    <>
      <div className="benchmark-layout">
        <section>
          <h2>Avaliar várias fotos</h2>
          <p>
            Escolha um dataset, confira o gabarito e inicie. Todas as fotos
            serão avaliadas automaticamente.
          </p>
          <label htmlFor="dataset">Dataset</label>
          <select
            id="dataset"
            disabled={busy}
            value={dataset}
            onChange={(e) => {setDataset(e.target.value);setLimit(0);const d=catalog.datasets.find(d=>d.id===e.target.value);if(d?.required_contract)setSettings(s=>({...s,contract:d.required_contract}));}}
          >
            {catalog.datasets.map((d) => (
              <option key={d.id} value={d.id}>
                {d.name} — {d.events} fotos{d.available===false ? ' (imagens não instaladas)' : ''}
              </option>
            ))}
          </select>
          {selected && (
            <div className="dataset-summary"><p>
              {selected.events} fotos · {selected.annotated} com gabarito ·{" "}
              {selected.complete_labels} com todos os atributos anotados
            </p><p>{selected.unique_images} imagens únicas · {selected.groups ?? selected.unique_images} grupos de origem</p>
            {selected.note && <p className="hint">{selected.note}</p>}
            {selected.required_contract && <p>Contrato {selected.required_contract==='historical'?'Histórico':selected.required_contract} aplicado automaticamente para preservar as classes deste dataset.</p>}
            {selected.available!==false && <p><a href={`/api/datasets/${selected.id}/bundle`}>Baixar dataset com imagens e gabarito (ZIP)</a></p>}
            {selected.available===false && <p className="error">As imagens deste dataset precisam ser instaladas neste computador. Consulte docs/DATASETS.md.</p>}
            </div>
          )}
          <button
            className="quiet"
            disabled={busy}
            onClick={() => setImporting(!importing)}
          >
            {importing ? "Fechar importação" : "Importar minhas fotos"}
          </button>
          {importing && (
            <DatasetImport
              contract={contract}
              busy={busy}
              onImported={imported}
            />
          )}
          {!importing && (
            <>
              <details className="method" open>
                <summary>Opções de medição</summary>
                <label htmlFor="repetitions">
                  Repetições por foto e modelo
                </label>
                <input
                  id="repetitions"
                  disabled={busy}
                  type="number"
                  min="1"
                  max="100"
                  value={repetitions}
                  onChange={(e) => setRepetitions(Number(e.target.value))}
                />
                <p className="hint">
                  Usamos 3 repetições para medir tempo. A acurácia usa a
                  primeira previsão de cada foto.
                </p>
                <label htmlFor="limit">Limite de fotos (0 = todas)</label>
                <input
                  id="limit"
                  disabled={busy}
                  type="number"
                  min="0"
                  value={limit}
                  onChange={(e) => setLimit(Number(e.target.value))}
                />
              </details>
              {limit>0 && <p className="notice">Limite ativo: serão usadas somente {Math.min(limit,selected?.events || limit)} fotos deste dataset. <button type="button" disabled={busy} onClick={()=>setLimit(0)}>Usar todas as fotos</button></p>}
              <label htmlFor="comparison">O que comparar?</label>
              <select id="comparison" disabled={busy} value={comparison} onChange={e=>setComparison(e.target.value)}>
                <option value="baseline">Modelos com o formato padrão</option>
                <option value="prompts">Variações de pergunta e formato de saída</option>
                <option value="requests">Perguntas juntas × uma chamada por atributo</option>
                <option value="all">Todas as variantes de prompt e request</option>
              </select>
              {preview && <p className="dataset-summary">{preview.photos} fotos × {preview.variants} variantes × {repetitions} repetições = {preview.planned} avaliações de foto. Cada avaliação pode incluir várias chamadas internas.</p>}
              {previewError && <p className="error">{previewError}</p>}
              {preview && <details className="method"><summary>Perguntas, requests e métricas de cada variante</summary>
                <p>Todos recebem as mesmas imagens e classes. O BLIP gera texto; o Laya escolhe alternativas nativas. Solicitar JSON ao BLIP é uma condição experimental e pode falhar.</p>
                {preview.protocols.map(p=><section key={p.model}><h3>{p.model}</h3>{p.examples.map((x,i)=><details key={i}><summary>{x.domain} · {x.profile} · {x.calls_per_photo} chamada(s) por foto</summary><p>{x.question_type} · {x.request_mode} · {x.output_format}</p><p>Métricas: {x.metrics.join('; ')}.</p><pre>{JSON.stringify(x.request,null,2)}</pre></details>)}</section>)}
              </details>}
              <div className="actions">
                <button
                  className="primary"
                  disabled={busy || !selected || !preview || selected.available===false || Boolean(previewError)}
                  onClick={() =>
                    onRun({ kind: "benchmark", contract:effectiveContract, dataset, repetitions, limit, comparison })
                  }
                >
                  Iniciar avaliação do dataset
                </button>
              </div>
              <p className="hint">
                Inclui aquecimento e relatório por modelo. Sem gabarito, medimos
                apenas desempenho e falhas.
              </p>
            </>
          )}
        </section>
        <section>
          <h2>{job && !busy ? 'Última avaliação' : 'Andamento'}</h2>
          {job && !busy && <p className="hint">Resultado da execução anterior, não da seleção atual. Dataset: {catalog.datasets.find(d=>d.id===(job.experiment?.dataset || job.dataset))?.name || job.dataset}. Inicie uma nova avaliação para processar a seleção atual.</p>}
          <JobState job={job} cancel={onCancel} />
          <details className="method">
            <summary>Ver registro técnico</summary>
            <pre>{job?.log || "Nenhuma avaliação iniciada."}</pre>
          </details>
        </section>
      </div>
      <Evaluation job={job} />
    </>
  );
}

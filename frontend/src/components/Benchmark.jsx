import React, { useState } from "react";
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
}) {
  const [dataset, setDataset] = useState("portable"),
    [repetitions, setRepetitions] = useState(3),
    [limit, setLimit] = useState(0),
    [importing, setImporting] = useState(false);
  const selected = catalog.datasets.find((d) => d.id === dataset);
  async function imported(result) {
    await onCatalogRefresh();
    setDataset(result.id);
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
            onChange={(e) => setDataset(e.target.value)}
          >
            {catalog.datasets.map((d) => (
              <option key={d.id} value={d.id}>
                {d.name} — {d.events} fotos
              </option>
            ))}
          </select>
          {selected && (
            <p className="dataset-summary">
              {selected.events} fotos · {selected.annotated} com gabarito ·{" "}
              {selected.complete_labels} com todos os atributos anotados
            </p>
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
              <details className="method">
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
              <div className="actions">
                <button
                  className="primary"
                  disabled={busy || !selected}
                  onClick={() =>
                    onRun({ kind: "benchmark", dataset, repetitions, limit })
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
          <h2>Andamento</h2>
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

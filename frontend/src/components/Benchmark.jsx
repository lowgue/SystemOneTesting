import React, { useState } from "react";
import JobState from "./JobState.jsx";

export default function Benchmark({ catalog, job, busy, onRun, onCancel }) {
  const [repetitions, setRepetitions] = useState(3),
    [limit, setLimit] = useState(0);
  return (
    <div className="benchmark-layout">
      <section>
        <h2>Nova avaliação</h2>
        <p>
          Dataset portátil: {catalog.datasets[0]?.events} eventos. Rótulos
          exploratórios.
        </p>
        <label htmlFor="repetitions">Repetições</label>
        <input
          id="repetitions"
          disabled={busy}
          type="number"
          min="1"
          max="100"
          value={repetitions}
          onChange={(e) => setRepetitions(Number(e.target.value))}
        />
        <label htmlFor="limit">Limite de imagens (0 = todas)</label>
        <input
          id="limit"
          disabled={busy}
          type="number"
          min="0"
          value={limit}
          onChange={(e) => setLimit(Number(e.target.value))}
        />
        <div className="actions">
          <button
            className="primary"
            disabled={busy}
            onClick={() =>
              onRun({
                kind: "benchmark",
                dataset: catalog.datasets[0]?.id,
                repetitions,
                limit,
              })
            }
          >
            Iniciar benchmark
          </button>
        </div>
        <p className="hint">
          Aquecimento, ordem alternada, configuração congelada e relatório por
          computador. Falhas permanecem nos denominadores.
        </p>
      </section>
      <section>
        <h2>Andamento</h2>
        <JobState job={job} cancel={onCancel} />
        <pre>{job?.log || "O registro aparecerá aqui."}</pre>
        <a href="#results">Consultar gráficos e métricas</a>
      </section>
    </div>
  );
}

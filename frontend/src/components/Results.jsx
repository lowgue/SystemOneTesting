import React, { useState, useEffect } from "react";
import { api } from "../api.js";
import { stateText } from "../presentation.js";

export default function Results({ inspect }) {
  const [files, setFiles] = useState([]),
    [history, setHistory] = useState([]),
    [filter, setFilter] = useState(""),
    [error, setError] = useState("");
  function refresh() {
    Promise.all([api("/api/artifacts"), api("/api/jobs")])
      .then(([f, h]) => {
        setFiles(f);
        setHistory(h);
      })
      .catch((e) => setError(e.message));
  }
  useEffect(refresh, []);
  return (
    <>
      <div className="section-heading">
        <h2>Histórico de tarefas</h2>
        <button onClick={refresh}>Atualizar lista</button>
      </div>
      {error && <p className="error">{error}</p>}
      <div className="history">
        {history.length ? (
          history.map((job) => (
            <button key={job.id} onClick={() => inspect(job.id)}>
              {job.kind === "live"
                ? "Imagem"
                : job.kind === "benchmark"
                  ? "Dataset"
                  : "Download"}{" "}
              — {stateText(job)} <small>{job.id.slice(0, 8)}</small>
            </button>
          ))
        ) : (
          <p>Nenhuma tarefa registrada.</p>
        )}
      </div>
      <h2>Gráficos e arquivos</h2>
      <label htmlFor="filter">Filtrar arquivos</label>
      <input
        id="filter"
        type="search"
        value={filter}
        onChange={(e) => setFilter(e.target.value)}
        placeholder=".png, .csv ou nome da execução"
      />
      <div className="artifacts">
        {files
          .filter((f) =>
            `${f.source}/${f.name}`
              .toLowerCase()
              .includes(filter.toLowerCase()),
          )
          .map((f) => (
            <div className="artifact" key={f.url}>
              {f.name.endsWith(".png") ? (
                <img src={f.url} alt="Prévia do gráfico" loading="lazy" />
              ) : (
                <span className="file-icon">
                  {f.name.split(".").pop().toUpperCase()}
                </span>
              )}
              <a href={f.url} target="_blank" rel="noreferrer">
                {f.source} / {f.name}
              </a>
            </div>
          ))}
      </div>
    </>
  );
}

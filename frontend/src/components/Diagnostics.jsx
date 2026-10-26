import React, { useState, useEffect } from "react";
import { api } from "../api.js";
import JobState from "./JobState.jsx";

export default function Diagnostics({ job, busy, pull, cancel }) {
  const [data, setData] = useState(null),
    [error, setError] = useState("");
  function refresh() {
    api("/api/diagnostics")
      .then(setData)
      .catch((e) => setError(e.message));
  }
  useEffect(refresh, []);
  useEffect(() => {
    if (job?.kind === "ollama-pull" && job.state === "complete") refresh();
  }, [job?.id, job?.state]);
  return (
    <>
      <div className="section-heading">
        <h2>Ollama e Clef-Flash 9B</h2>
        <button onClick={refresh}>Verificar novamente</button>
      </div>
      {error && <p className="error">{error}</p>}
      {data ? (
        <>
          <dl className="diagnostic-list">
            <dt>Serviço Ollama</dt>
            <dd>{data.ollama === "online" ? "Respondendo" : "Indisponível"}</dd>
            <dt>Versão</dt>
            <dd>{data.version || "Não identificada"}</dd>
            <dt>Clef-Flash 9B</dt>
            <dd>{data.clef_installed ? "Instalado" : "Ainda não instalado"}</dd>
            <dt>RAM total / disponível</dt>
            <dd>
              {data.ram_total_gb} / {data.ram_available_gb} GiB
            </dd>
            <dt>Modelos carregados</dt>
            <dd>
              {data.loaded?.map((m) => m.name).join(", ") ||
                "Nenhum. Isso não informa a causa de uma falha."}
            </dd>
          </dl>
          {data.error && <p className="error">{data.error}</p>}
          <button
            className="primary"
            disabled={busy || data.ollama !== "online"}
            onClick={pull}
          >
            Baixar ou atualizar Clef-Flash 9B
          </button>
          <p className="hint">
            O download pode ocupar mais de 10 GB. Os pesos permanecem no volume
            do Ollama. Nenhuma inferência será iniciada.
          </p>
          <details>
            <summary>Diagnóstico completo</summary>
            <pre>{JSON.stringify(data, null, 2)}</pre>
          </details>
        </>
      ) : (
        <p>Verificando…</p>
      )}
      {job?.kind === "ollama-pull" && <JobState job={job} cancel={cancel} />}
      <div className="help">
        <h3>Investigar erros de inferência</h3>
        <p>
          Um serviço online e um modelo instalado não garantem que o
          carregamento funcionará. Execute um teste de imagem. O resultado
          mostra a mensagem do servidor; o JSON e o histórico preservam os
          detalhes.
        </p>
        <p>
          No Docker, o Ollama da aplicação é independente do Ollama instalado no
          sistema e não ocupa a porta 11434 do computador.
        </p>
      </div>
      <pre>{job?.log || "Nenhum registro nesta sessão."}</pre>
    </>
  );
}

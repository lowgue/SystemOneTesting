import React from "react";
import { labels, text } from "../presentation.js";

export default function Answers({ job }) {
  if (!job?.results?.length)
    return (
      <p className="empty">
        As respostas aparecerão aqui conforme cada modelo terminar.
      </p>
    );
  return job.results.map((result) => (
    <article className="answer" key={result.model}>
      <h3>{result.model}</h3>
      {result.simulated && (
        <p className="simulated">
          Simulação de infraestrutura. Não é um modelo real.
        </p>
      )}
      {result.error ? (
        <p role="alert" className="error">
          {result.error}
        </p>
      ) : (
        <>
          <dl>
            {Object.entries(result.predicted || {}).map(([key, value]) => (
              <React.Fragment key={key}>
                <dt>{labels[key] || key}</dt>
                <dd>{text(value)}</dd>
              </React.Fragment>
            ))}
          </dl>
          <p className="hint">
            {result.valid
              ? "Formato válido"
              : "Formato inválido; consulte a resposta completa."}
          </p>
          <p className="timing">
            Inferência: {result.elapsed_ms?.toFixed(0)} ms · Preparação:{" "}
            {result.load_ms?.toFixed(0)} ms
          </p>
        </>
      )}
      <details>
        <summary>Resposta completa</summary>
        <pre>{JSON.stringify(result, null, 2)}</pre>
      </details>
    </article>
  ));
}

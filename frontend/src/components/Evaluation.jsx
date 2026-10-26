import React, { useEffect, useState } from "react";
import { api } from "../api.js";
import { labels, text } from "../presentation.js";
const percent = (value) =>
  value == null
    ? "Sem gabarito"
    : new Intl.NumberFormat("pt-BR", {
        style: "percent",
        minimumFractionDigits: 1,
        maximumFractionDigits: 1,
      }).format(value);
const millis = (value) =>
  value == null
    ? "Sem sucesso"
    : `${value.toLocaleString("pt-BR", { minimumFractionDigits: 1, maximumFractionDigits: 1 })} ms`;

function Confusion({ field }) {
  const columns = [
    ...new Set(
      Object.values(field.confusion).flatMap((row) => Object.keys(row)),
    ),
  ];
  return (
    <details>
      <summary>Matriz de confusão e métricas por classe</summary>
      <div className="table-scroll">
        <table>
          <caption>Linhas: gabarito. Colunas: resposta prevista.</caption>
          <thead>
            <tr>
              <th>Gabarito / previsão</th>
              {columns.map((c) => (
                <th key={c}>{text(c)}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {Object.entries(field.confusion).map(([expected, row]) => (
              <tr key={expected}>
                <th>{text(expected)}</th>
                {columns.map((c) => (
                  <td key={c} className={c === expected ? "correct-cell" : ""}>
                    {row[c] || 0}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
        <table>
          <thead>
            <tr>
              <th>Classe</th>
              <th>Fotos</th>
              <th>Precisão</th>
              <th>Recall</th>
              <th>F1</th>
            </tr>
          </thead>
          <tbody>
            {field.classes.map((c) => (
              <tr key={c.label}>
                <td>{text(c.label)}</td>
                <td>{c.support}</td>
                <td>{percent(c.precision)}</td>
                <td>{percent(c.recall)}</td>
                <td>{percent(c.f1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </details>
  );
}

function PhotoResults({ data, job }) {
  const [search, setSearch] = useState(""),
    [filter, setFilter] = useState("all"),
    [page, setPage] = useState(0);
  const rows = data.models
    .flatMap((m) =>
      m.photo_results.map((photo) => ({ ...photo, model: m.model })),
    )
    .filter(
      (row) =>
        row.filename.toLowerCase().includes(search.toLowerCase()) &&
        (filter === "all" ||
          (filter === "errors" && row.error) ||
          (filter === "mistakes" &&
            Object.values(row.correct).some((v) => !v)) ||
          (filter === "unlabeled" && !Object.keys(row.expected).length)),
    );
  const pages = Math.max(1, Math.ceil(rows.length / 25)),
    current = Math.min(page, pages - 1);
  return (
    <section className="photo-results">
      <h3>Resultado de cada foto</h3>
      <p className="hint">
        Respostas e tempos da primeira repetição. O resumo acima usa todas as
        repetições para medir desempenho.
      </p>
      <div className="photo-filters">
        <div>
          <label htmlFor="photo-search">Buscar foto</label>
          <input
            id="photo-search"
            type="search"
            value={search}
            onChange={(e) => {
              setSearch(e.target.value);
              setPage(0);
            }}
            placeholder="Nome do arquivo"
          />
        </div>
        <div>
          <label htmlFor="photo-filter">Mostrar</label>
          <select
            id="photo-filter"
            value={filter}
            onChange={(e) => {
              setFilter(e.target.value);
              setPage(0);
            }}
          >
            <option value="all">Todas</option>
            <option value="mistakes">Com atributos incorretos</option>
            <option value="errors">Falhas de execução</option>
            <option value="unlabeled">Sem gabarito</option>
          </select>
        </div>
      </div>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th>Foto</th>
              <th>Modelo</th>
              <th>Tempo</th>
              <th>Avaliação</th>
              <th>Gabarito e resposta</th>
            </tr>
          </thead>
          <tbody>
            {rows.slice(current * 25, current * 25 + 25).map((row) => (
              <tr key={row.model + row.id}>
                <td>
                  <a
                    href={`/api/datasets/${job.dataset || "portable"}/photos/${row.id}`}
                    target="_blank"
                    rel="noreferrer"
                  >
                    <img
                      className="photo-thumb"
                      loading="lazy"
                      src={`/api/datasets/${job.dataset || "portable"}/photos/${row.id}?thumbnail=true`}
                      alt=""
                    />
                    {row.filename}
                  </a>
                </td>
                <td>{row.model}</td>
                <td>{millis(row.elapsed_ms)}</td>
                <td>
                  {row.error
                    ? "Falha"
                    : !Object.keys(row.expected).length
                      ? "Sem gabarito"
                      : Object.values(row.correct).every(Boolean)
                        ? "Acertou os campos anotados"
                        : "Há atributos incorretos"}
                </td>
                <td>
                  {row.error && <p className="error">{row.error}</p>}
                  <details>
                    <summary>Ver atributos</summary>
                    <table>
                      <thead>
                        <tr>
                          <th>Atributo</th>
                          <th>Esperado</th>
                          <th>Previsto</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.keys(row.predicted).map((key) => (
                          <tr key={key}>
                            <td>{labels[key] || key}</td>
                            <td>
                              {key in row.expected
                                ? text(row.expected[key])
                                : "Sem gabarito"}
                            </td>
                            <td
                              className={
                                row.correct[key] === false
                                  ? "incorrect-cell"
                                  : row.correct[key] === true
                                    ? "correct-cell"
                                    : ""
                              }
                            >
                              {text(row.predicted[key])}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </details>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="pagination">
        <button disabled={current === 0} onClick={() => setPage(current - 1)}>
          Anterior
        </button>
        <span>
          {rows.length} resultados · Página {current + 1} de {pages}
        </span>
        <button
          disabled={current >= pages - 1}
          onClick={() => setPage(current + 1)}
        >
          Próxima
        </button>
      </div>
    </section>
  );
}

export default function Evaluation({ job }) {
  const [data, setData] = useState(null),
    [error, setError] = useState("");
  useEffect(() => {
    setData(null);
    setError("");
    if (job?.state === "complete")
      api(`/api/jobs/${job.id}/evaluation`)
        .then(setData)
        .catch((e) => setError(e.message));
  }, [job?.id, job?.state]);
  if (!job || job.state !== "complete") return null;
  if (error) return <p className="error">{error}</p>;
  if (!data) return <p role="status">Abrindo o relatório…</p>;
  const base = `/files/studio/${job.id}/report/`;
  return (
    <section className="evaluation" aria-label="Relatório do dataset">
      <div className="section-heading">
        <h2>Resultado da avaliação</h2>
        <strong>{data.dataset_name}</strong>
        <span>
          {data.photos} fotos · {data.unique_images} imagens únicas ·{" "}
          {data.repetitions} repetições
        </span>
      </div>
      {data.simulated && (
        <p className="notice simulated">
          Simulação: estes números verificam a aplicação, não representam
          desempenho de um modelo real.
        </p>
      )}
      {!data.models.some((m) => m.field_n) && (
        <p className="notice">
          Este dataset não tem gabarito. Acurácia, precisão, recall e F1 não
          podem ser calculados. Os tempos e falhas estão disponíveis abaixo.
        </p>
      )}
      {data.label_status.startsWith("exploratory") && (
        <p className="hint">
          O dataset de exemplo tem gabaritos exploratórios, sem adjudicação
          humana. Use seu dataset para avaliar o seu cenário.
        </p>
      )}
      {data.models.map((model) => (
        <section className="model-evaluation" key={model.model}>
          <h3>{model.model}</h3>
          <div className="metric-grid">
            <div>
              <span>Acurácia completa por foto</span>
              <strong>{percent(model.exact_accuracy)}</strong>
              <small>
                {model.exact_correct} de {model.complete_labeled_photos} fotos
                com gabarito completo
              </small>
            </div>
            <div>
              <span>Acurácia por atributo</span>
              <strong>{percent(model.field_accuracy)}</strong>
              <small>
                {model.field_correct} de {model.field_n} atributos anotados
              </small>
            </div>
            <div>
              <span>Tempo por foto (mediana)</span>
              <strong>{millis(model.latency.p50_ms)}</strong>
              <small>P95: {millis(model.latency.p95_ms)}</small>
            </div>
            <div>
              <span>Falhas de execução</span>
              <strong>
                {model.errors} / {model.requests}
              </strong>
              <small>Formato válido: {percent(model.schema_valid_rate)}</small>
            </div>
          </div>
          {model.fields.length > 0 && (
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>Atributo</th>
                    <th>Acertos / fotos</th>
                    <th>Acurácia</th>
                    <th>F1 macro</th>
                  </tr>
                </thead>
                <tbody>
                  {model.fields.map((field) => (
                    <tr key={field.domain + field.field}>
                      <th>{labels[field.field] || field.field}</th>
                      <td>
                        {field.correct} / {field.n}
                      </td>
                      <td>{percent(field.accuracy)}</td>
                      <td>{percent(field.macro_f1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          <details className="method">
            <summary>Ver métricas detalhadas e matrizes de confusão</summary>
            <dl className="diagnostic-list">
              <dt>Tempo médio / desvio-padrão</dt>
              <dd>
                {millis(model.latency.mean_ms)} / {millis(model.latency.std_ms)}
              </dd>
              <dt>Menor / maior tempo</dt>
              <dd>
                {millis(model.latency.min_ms)} / {millis(model.latency.max_ms)}
              </dd>
              <dt>Chamadas com sucesso por segundo</dt>
              <dd>
                {model.successful_calls_s?.toFixed(2) ?? "Não disponível"}
              </dd>
              <dt>Carregamento / aquecimento</dt>
              <dd>
                {model.load_s.toFixed(2)} s / {model.warmup_s.toFixed(2)} s
              </dd>
              <dt>Memória RSS de pico (amostrada)</dt>
              <dd>{model.peak_rss_mib.toFixed(1)} MiB</dd>
              <dt>CPU média por chamada</dt>
              <dd>{model.mean_cpu_s?.toFixed(3)} s</dd>
            </dl>
            {model.resource_scope.includes("client_only") && (
              <p className="hint">
                CPU e memória acima são do cliente HTTP. Não medem os recursos
                consumidos pelo servidor Ollama.
              </p>
            )}
            {model.fields.map((field) => (
              <section key={field.domain + field.field}>
                <h4>{labels[field.field] || field.field}</h4>
                <Confusion field={field} />
              </section>
            ))}
          </details>
        </section>
      ))}
      <div className="report-downloads">
        <h3>Baixar resultados</h3>
        <a href={base + "models.csv"}>Métricas por modelo (CSV)</a>
        <a
          href={`/files/studio/${job.id}/run/predictions.jsonl`}
          target="_blank"
          rel="noreferrer"
        >
          Todas as chamadas (JSONL)
        </a>
        <a href={base + "evaluation.json"} target="_blank" rel="noreferrer">
          Relatório JSON
        </a>
        <a href={base + "photos.csv"}>Resultados por foto (CSV)</a>
        <a href={base + "attributes.csv"}>Métricas por atributo (CSV)</a>
        <a href={base + "comparison_1.pdf"} target="_blank" rel="noreferrer">
          Gráfico de desempenho (PDF)
        </a>
        {data.models.some((m) => m.field_n > 0) && (
          <a href={base + "accuracy.pdf"} target="_blank" rel="noreferrer">
            Gráfico de acurácia (PDF)
          </a>
        )}
      </div>
      <details className="method">
        <summary>Como estas métricas são calculadas?</summary>
        {Object.values(data.methodology).map((note) => (
          <p key={note}>{note}</p>
        ))}
        <p>
          A acurácia usa uma previsão por foto; repetir a inferência não aumenta
          a amostra. O P95 é o tempo abaixo do qual ficaram 95% das chamadas com
          sucesso. Tempos incluem leitura, preparo e pós-processamento da
          imagem. “Sem gabarito” não significa 0% de acurácia.
        </p>
      </details>
      <details className="report-charts">
        <summary>Ver gráficos de desempenho e acurácia</summary>
        <img
          loading="lazy"
          src={base + "comparison_1.png"}
          alt="Distribuições de latência e CPU por modelo"
        />
        {data.models.some((model) => model.field_n > 0) && (
          <img
            loading="lazy"
            src={base + "accuracy.png"}
            alt="Acurácia por atributo e modelo"
          />
        )}
      </details>
      <PhotoResults data={data} job={job} />
    </section>
  );
}

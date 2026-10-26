import React, { useState } from "react";
import { download } from "../api.js";
import JobState from "./JobState.jsx";
import Answers from "./Answers.jsx";

export default function Live({
  job,
  busy,
  onRun,
  onCancel,
  domain,
  setDomain,
}) {
  const [image, setImage] = useState(null);
  const [error, setError] = useState("");
  return (
    <>
      <div className="workbench">
        <section className="image-panel">
          <div className="section-heading">
            <h2>Imagem de entrada</h2>
            <label className="sr-only" htmlFor="domain">
              Tipo de cena
            </label>
            <select
              id="domain"
              disabled={busy}
              value={domain}
              onChange={(e) => setDomain(e.target.value)}
            >
              <option value="vehicle">Veículo</option>
              <option value="person">Pessoa</option>
            </select>
          </div>
          <label className="image-stage" htmlFor="image">
            {image ? (
              <img alt="Imagem selecionada" src={image.data} />
            ) : (
              <span>
                <strong>Escolha uma imagem</strong>
                <span>JPEG, PNG ou WebP</span>
              </span>
            )}
          </label>
          <input
            id="image"
            type="file"
            disabled={busy}
            accept="image/jpeg,image/png,image/webp"
            onChange={(e) => {
              const file = e.target.files[0];
              if (!file) return;
              if (file.size > 16 * 1024 * 1024) {
                setError("Use uma imagem de até 16 MiB.");
                return;
              }
              const reader = new FileReader();
              reader.onload = () => {
                setImage({ data: reader.result, name: file.name });
                setError("");
              };
              reader.readAsDataURL(file);
            }}
          />
          <p className="hint">{image?.name || "Nenhuma imagem selecionada."}</p>
          {error && (
            <p className="error" role="alert">
              {error}
            </p>
          )}
          <div className="actions">
            <button
              className="primary"
              disabled={busy || !image}
              onClick={() =>
                onRun({ kind: "live", domain, image_data: image.data })
              }
            >
              Avaliar imagem
            </button>
          </div>
          <p className="hint">
            As imagens e respostas ficam armazenadas localmente. A inferência
            começa somente ao clicar no botão.
          </p>
        </section>
        <section className="answers-panel">
          <div className="section-heading">
            <h2>Respostas</h2>
            <button disabled={!job} onClick={() => download(job)}>
              Baixar JSON
            </button>
          </div>
          <JobState job={job?.kind === "live" ? job : null} cancel={onCancel} />
          <Answers job={job?.kind === "live" ? job : null} />
        </section>
      </div>
      <details className="method">
        <summary>Como interpretar este teste?</summary>
        <p>
          Mesma imagem e perguntas para todos os modelos selecionados. Sem
          gabarito e sem aquecimento. A primeira inferência no Ollama pode
          incluir o carregamento do modelo. Use Benchmark para medições com
          repetições; validade do formato não é acurácia.
        </p>
      </details>
    </>
  );
}

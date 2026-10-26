import React, { useEffect, useState } from "react";
import { api } from "../api.js";
import { labels } from "../presentation.js";

const fileData = (file) =>
  new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () =>
      reject(new Error(`Não foi possível ler ${file.name}`));
    reader.readAsDataURL(file);
  });
const csvCell = (text) => `"${text.replaceAll('"', '""')}"`;
export default function DatasetImport({ contract, busy, onImported }) {
  const [files, setFiles] = useState([]),
    [csv, setCsv] = useState(null),
    [name, setName] = useState("Minhas fotos"),
    [domain, setDomain] = useState("vehicle");
  const [working, setWorking] = useState(false),
    [error, setError] = useState(""),
    [sent, setSent] = useState(0),
    [draft, setDraft] = useState(null),
    [questions, setQuestions] = useState(null);
  useEffect(() => {
    api(`/api/contracts/${contract}`)
      .then(setQuestions)
      .catch((e) => setError(e.message));
  }, [contract]);
  const fields = questions?.[domain]?.questions || {};
  function template() {
    const text = [
      ["arquivo", ...Object.keys(fields)].join(","),
      ...files.map((file) =>
        [csvCell(file.name), ...Object.keys(fields).map(() => "")].join(","),
      ),
    ].join("\n");
    const url = URL.createObjectURL(
      new Blob(["\ufeff" + text], { type: "text/csv;charset=utf-8" }),
    );
    const link = document.createElement("a");
    link.href = url;
    link.download = "gabarito.csv";
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  async function prepare() {
    setWorking(true);
    setError("");
    let id = draft;
    try {
      if (!files.length) throw new Error("Selecione as fotos do dataset.");
      if (files.length > 2000)
        throw new Error("Selecione até 2.000 fotos por dataset.");
      if (new Set(files.map((file) => file.name)).size !== files.length)
        throw new Error(
          "Renomeie as fotos com nomes de arquivo repetidos antes de importar.",
        );
      const oversized = files.find((file) => file.size > 16 * 1024 * 1024);
      if (oversized)
        throw new Error(`${oversized.name}: use fotos de até 16 MiB.`);
      if (!id) {
        const result = await api("/api/datasets", { name, domain });
        id = result.id;
        setDraft(id);
      }
      for (let index = sent; index < files.length; index++) {
        if (files[index].size > 16 * 1024 * 1024)
          throw new Error(`${files[index].name}: use fotos de até 16 MiB.`);
        await api(`/api/datasets/${id}/images`, {
          filename: files[index].name,
          image_data: await fileData(files[index]),
        });
        setSent(index + 1);
      }
      const result = await api(`/api/datasets/${id}/finalize`, {
        contract,
        csv: csv ? await csv.text() : "",
      });
      await onImported(result);
      setDraft(null);
      setFiles([]);
      setCsv(null);
      setSent(0);
    } catch (e) {
      setError(e.message);
    } finally {
      setWorking(false);
    }
  }
  return (
    <div className="dataset-import">
      <h3>Importar minhas fotos</h3>
      <p>
        Selecione várias fotos. O gabarito é opcional e pode conter apenas os
        atributos que você quer avaliar.
      </p>
      <div className="import-options">
        <div>
          <label htmlFor="dataset-name">Nome do dataset</label>
          <input
            id="dataset-name"
            value={name}
            disabled={working || Boolean(draft)}
            onChange={(e) => setName(e.target.value)}
          />
        </div>
        <div>
          <label htmlFor="dataset-domain">O que aparece nas fotos?</label>
          <select
            id="dataset-domain"
            value={domain}
            disabled={working || Boolean(draft)}
            onChange={(e) => setDomain(e.target.value)}
          >
            <option value="vehicle">Veículos</option>
            <option value="person">Pessoas</option>
          </select>
        </div>
      </div>
      <label htmlFor="dataset-images">Fotos do dataset</label>
      <input
        id="dataset-images"
        type="file"
        accept="image/jpeg,image/png,image/webp"
        multiple
        disabled={working || Boolean(draft)}
        onChange={(e) => {
          setFiles([...e.target.files]);
          setSent(0);
        }}
      />
      <p className="hint">
        {files.length
          ? `${files.length} fotos selecionadas`
          : "Até 2.000 fotos por dataset. JPEG, PNG ou WebP."}
      </p>
      <label htmlFor="dataset-csv">Gabarito CSV (opcional)</label>
      <input
        id="dataset-csv"
        type="file"
        accept=".csv,text/csv"
        disabled={working}
        onChange={(e) => setCsv(e.target.files[0] || null)}
      />
      <button
        type="button"
        className="quiet"
        disabled={!files.length || working}
        onClick={template}
      >
        Baixar modelo de gabarito
      </button>
      <details className="method">
        <summary>Como preencher o gabarito?</summary>
        <p>
          Baixe o modelo com os nomes das fotos. Preencha as respostas corretas
          e envie o CSV acima. Deixe em branco os campos que não conhece. Para
          medir acurácia completa por foto, preencha todos os atributos.
        </p>
        <p>
          Exemplo para veículos:{" "}
          <code>
            presença = sim; marca = Toyota; cor = preto; carroceria = sedã
          </code>
          . Também são aceitos os códigos abaixo.
        </p>
        {Object.entries(fields).map(([key, q]) => (
          <p key={key}>
            <strong>{labels[key] || key}:</strong>{" "}
            <span className="enum-options">
              {Object.keys(q.criteria).join(", ")}
            </span>
          </p>
        ))}
      </details>
      {!csv && (
        <p className="notice">
          Sem gabarito, mediremos tempo, falhas e formato das respostas. A
          acurácia ficará como “sem gabarito”.
        </p>
      )}
      {working && (
        <p role="status">
          Preparando dataset: {sent} de {files.length} fotos enviadas…
        </p>
      )}
      {error && (
        <p role="alert" className="error">
          {error}
        </p>
      )}
      {draft && !working && (
        <button
          onClick={async () => {
            try {
              await api(`/api/datasets/${draft}/discard`, {});
              setDraft(null);
              setSent(0);
              setError("");
            } catch (e) {
              setError(e.message);
            }
          }}
        >
          Descartar importação e escolher novamente
        </button>
      )}
      <button
        className="primary"
        disabled={busy || working || !files.length}
        onClick={prepare}
      >
        {draft ? "Concluir importação" : "Preparar dataset"}
      </button>
    </div>
  );
}

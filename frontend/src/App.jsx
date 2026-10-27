import React, { useState, useEffect } from "react";
import { api } from "./api.js";
import { pages, isActive } from "./presentation.js";
import Configuration from "./components/Configuration.jsx";
import Live from "./components/Live.jsx";
import Benchmark from "./components/Benchmark.jsx";
import Results from "./components/Results.jsx";
import Diagnostics from "./components/Diagnostics.jsx";

export default function App() {
  const [view, setView] = useState(
      pages[location.hash.slice(1)] ? location.hash.slice(1) : "live",
    ),
    [catalog, setCatalog] = useState(null),
    [error, setError] = useState(""),
    [job, setJob] = useState(null),
    [jobId, setJobId] = useState(null),
    [domain, setDomain] = useState("vehicle");
  const [settings, setSettings] = useState({
    config: "ollama",
    contract: "ollama",
    models: ["clef-flash-ollama"],
    device: "cpu",
    threads: 8,
  });
  const busy = isActive(job);
  useEffect(() => {
    const navigate = () =>
      setView(pages[location.hash.slice(1)] ? location.hash.slice(1) : "live");
    window.addEventListener("hashchange", navigate);
    return () => window.removeEventListener("hashchange", navigate);
  }, []);
  useEffect(() => {
    api("/api/catalog")
      .then((c) => {
        setCatalog(c);
        const previous =
          c.current_job || localStorage.getItem("vision-studio-job");
        if (previous)
          api(`/api/jobs/${previous}`)
            .then((j) => {
              setJob(j);
              setJobId(j.id);
              if (j.settings) setSettings(j.settings);
            })
            .catch(() => localStorage.removeItem("vision-studio-job"));
      })
      .catch((e) => setError(e.message));
  }, []);
  useEffect(() => {
    if (!jobId) return;
    const source = new EventSource(`/api/jobs/${jobId}/events`);
    source.onmessage = (e) => {
      const j = JSON.parse(e.data);
      setJob(j);
      if (!isActive(j)) source.close();
    };
    source.onerror = () => {
      api(`/api/jobs/${jobId}`)
        .then((j) => {
          setJob(j);
          if (!isActive(j)) source.close();
        })
        .catch((e) => setError(`Conexão interrompida: ${e.message}`));
    };
    return () => source.close();
  }, [jobId]);
  async function begin(body, path = "/api/jobs") {
    setError("");
    try {
      if (path === "/api/jobs" && !settings.models.length)
        throw new Error("Selecione pelo menos um modelo.");
      const j = await api(path, { ...settings, ...body });
      setJob(j);
      setJobId(j.id);
      localStorage.setItem("vision-studio-job", j.id);
    } catch (e) {
      setError(e.message);
    }
  }
  async function cancel() {
    try {
      setJob(await api(`/api/jobs/${job.id}/cancel`, {}));
    } catch (e) {
      setError(e.message);
    }
  }
  async function inspect(id) {
    try {
      const j = await api(`/api/jobs/${id}`);
      setJob(j);
      setJobId(id);
      if (j.settings) setSettings(j.settings);
      location.hash =
        j.kind === "live"
          ? "live"
          : j.kind === "benchmark"
            ? "benchmark"
            : "diagnostics";
    } catch (e) {
      setError(e.message);
    }
  }
  const page = pages[view] || pages.live;
  return (
    <>
      <a className="skip" href="#workspace">
        Ir para o conteúdo
      </a>
      <aside>
        <a className="brand" href="#live">
          <span className="brand-mark" aria-hidden="true">
            ◈
          </span>
          <span>
            System One<small>Laboratório visual</small>
          </span>
        </a>
        <nav aria-label="Áreas do laboratório">
          {Object.entries(pages).map(([key, value]) => (
            <a
              key={key}
              href={`#${key}`}
              aria-current={key === view ? "page" : undefined}
            >
              {value[0]}
            </a>
          ))}
        </nav>
        <div className="aside-note">
          <strong>Aplicação web local</strong>
          <p>React + FastAPI</p>
          <span>{catalog ? "API conectada" : "Conectando…"}</span>
        </div>
      </aside>
      <main id="workspace">
        <header>
          <div>
            <h1>{page[0]}</h1>
            <p>{page[1]}</p>
          </div>
          <span className="badge">
            {busy ? "Tarefa em execução" : "Pronto para testar"}
          </span>
        </header>
        {error && (
          <p className="error" role="alert">
            {error}
          </p>
        )}
        {catalog && (
          <>
            {["live", "benchmark"].includes(view) && (
              <Configuration
                catalog={catalog}
                settings={settings}
                setSettings={setSettings}
                busy={busy}
              />
            )}
            <div hidden={view !== "live"}>
              <Live
                job={job}
                busy={busy}
                onRun={begin}
                onCancel={cancel}
                domain={domain}
                setDomain={setDomain}
              />
            </div>
            {view === "benchmark" && (
              <Benchmark
                catalog={catalog}
                contract={settings.contract}
                settings={settings}
                setSettings={setSettings}
                onCatalogRefresh={async () =>
                  setCatalog(await api("/api/catalog"))
                }
                job={job?.kind === "benchmark" ? job : null}
                busy={busy}
                onRun={begin}
                onCancel={cancel}
              />
            )}
            {view === "results" && <Results inspect={inspect} />}
            {view === "diagnostics" && (
              <Diagnostics
                job={job}
                busy={busy}
                pull={() => begin({}, "/api/ollama/pull")}
                cancel={cancel}
              />
            )}
          </>
        )}
        <footer>
          System One Testing{" "}
          <span>Interface React · API FastAPI · Atualizações ao vivo</span>
        </footer>
      </main>
    </>
  );
}

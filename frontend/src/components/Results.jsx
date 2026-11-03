import React, { useState, useEffect, useMemo } from "react";
import { api } from "../api.js";
import { stateText } from "../presentation.js";

export const CHART_GROUPS = [
  {
    id: "latency",
    title: "Latência",
    description: "Tempos de resposta, velocidade de inferência e consistência temporal entre modelos.",
  },
  {
    id: "accuracy",
    title: "Acurácia",
    description: "Taxas globais de acerto, precisão por atributo e robustez sob condições adversas.",
  },
  {
    id: "hardware",
    title: "Recursos CPU / GPU",
    description: "Esforço computacional de processamento (CPU/GPU) e perfil de consumo de memória RAM.",
  },
  {
    id: "inference",
    title: "Inferência & Throughput",
    description: "Vazão de requisições por segundo, comportamento em fila e orçamento de tokens.",
  },
  {
    id: "all",
    title: "Todos os 10 Tipos",
    description: "Visualização consolidada de todos os 10 tipos de gráficos essenciais em PDF.",
  },
];

export const CHART_TYPES = [
  // ================= Grupo 1: Latência (Tipos 1 a 3) =================
  {
    id: "eval_time",
    typeNumber: 1,
    group: "latency",
    groupLabel: "Latência",
    title: "Tempo de Avaliação por Imagem (Latência Mediana e P95)",
    subtitle: "Mede o tempo de inferência por foto em milissegundos (mediana e percentil 95), excluindo carga inicial e aquecimento.",
    matches: (name) => {
      const l = name.toLowerCase();
      return (
        l.includes("simple_latency") ||
        l.includes("01_desempenho_distribuicoes") ||
        l.includes("desempenho_distribuicoes") ||
        (l.includes("comparison_") && !l.includes("comparisons.csv"))
      );
    },
  },
  {
    id: "paired_latency",
    typeNumber: 2,
    group: "latency",
    groupLabel: "Latência",
    title: "Comparação Pareada de Latência",
    subtitle: "Compara diretamente a diferença de tempo de resposta entre Laya e BLIP sobre a exata mesma imagem em teste pareado.",
    matches: (name) => {
      const l = name.toLowerCase();
      return l.includes("02_latencia_pareada") || l.includes("latencia_pareada");
    },
  },
  {
    id: "temporal_repeatability",
    typeNumber: 3,
    group: "latency",
    groupLabel: "Latência",
    title: "Repetibilidade e Estabilidade Temporal",
    subtitle: "Mede a consistência e o coeficiente de variação dos tempos de inferência ao longo de repetições e blocos consecutivos.",
    matches: (name) => {
      const l = name.toLowerCase();
      return l.includes("03_repetibilidade") || l.includes("repetibilidade");
    },
  },

  // ================= Grupo 2: Acurácia (Tipos 4 a 6) =================
  {
    id: "overall_accuracy",
    typeNumber: 4,
    group: "accuracy",
    groupLabel: "Acurácia",
    title: "Acurácia Geral dos Modelos",
    subtitle: "Percentual global de acertos em fotos e atributos com gabarito conhecido, comparando diretamente os modelos avaliados.",
    matches: (name) => {
      const l = name.toLowerCase();
      return (
        l.includes("simple_overview") ||
        (l.endsWith("accuracy.pdf") && !l.includes("attribute_")) ||
        l.includes("01_desempenho")
      );
    },
  },
  {
    id: "attribute_accuracy",
    typeNumber: 5,
    group: "accuracy",
    groupLabel: "Acurácia",
    title: "Acurácia por Atributo e Matrizes de Confusão",
    subtitle: "Precisão, recall, suporte e matrizes de confusão detalhadas por atributo específico (cor, tipo de veículo, marca e presença).",
    matches: (name) => {
      const l = name.toLowerCase();
      return (
        l.includes("attribute_") ||
        l.includes("classes_") ||
        l.includes("confusion_") ||
        l.includes("simple_vehicle_") ||
        (l.includes("simple_") && !l.includes("simple_latency") && !l.includes("simple_overview"))
      );
    },
  },
  {
    id: "robustness_quality",
    typeNumber: 6,
    group: "accuracy",
    groupLabel: "Acurácia",
    title: "Robustez e Sensibilidade a Degradações",
    subtitle: "Analisa a estabilidade das respostas dos modelos quando submetidos a baixa luz, desfoque, compressão, ruído e oclusão.",
    matches: (name) => {
      const l = name.toLowerCase();
      return l.includes("06_robustez") || l.includes("robustez");
    },
  },

  // ================= Grupo 3: Recursos CPU / GPU (Tipos 7 e 8) =================
  {
    id: "cpu_gpu_usage",
    typeNumber: 7,
    group: "hardware",
    groupLabel: "Recursos CPU / GPU",
    title: "Consumo de Processador (CPU e GPU)",
    subtitle: "Tempo de esforço computacional e perfil de processamento demandado pelos modelos por inferência realizada.",
    matches: (name) => {
      const l = name.toLowerCase();
      return (
        l.includes("12_perfil_laya") ||
        l.includes("13_perfil_blip") ||
        l.includes("perfil_") ||
        (l.includes("comparison_") && l.endsWith(".pdf"))
      );
    },
  },
  {
    id: "memory_usage",
    typeNumber: 8,
    group: "hardware",
    groupLabel: "Recursos CPU / GPU",
    title: "Uso de Memória (RAM e RSS do Processo)",
    subtitle: "Série temporal e picos amostrados de memória residente (RSS) do processo e memória RAM disponível no sistema.",
    matches: (name) => {
      const l = name.toLowerCase();
      return l.includes("11_memoria_processo") || l.includes("memoria");
    },
  },

  // ================= Grupo 4: Inferência & Throughput (Tipos 9 e 10) =================
  {
    id: "throughput_queue",
    typeNumber: 9,
    group: "inference",
    groupLabel: "Inferência & Throughput",
    title: "Throughput e Comportamento em Fila",
    subtitle: "Taxa de vazão de requisições por segundo e tempo de resposta sob concorrência e carga de múltiplos clientes.",
    matches: (name) => {
      const l = name.toLowerCase();
      return l.includes("09_fila") || l.includes("fila") || l.includes("throughput");
    },
  },
  {
    id: "inference_profile_tokens",
    typeNumber: 10,
    group: "inference",
    groupLabel: "Inferência & Throughput",
    title: "Perfil de Inferência e Orçamento de Tokens",
    subtitle: "Efeito do volume de tokens de entrada, tamanho das instruções e formatações de prompt no tempo de inferência.",
    matches: (name) => {
      const l = name.toLowerCase();
      return (
        l.includes("08_orcamento_laya") ||
        l.includes("orcamento") ||
        l.includes("04_prompts_laya") ||
        l.includes("05_prompts_blip") ||
        l.includes("prompts_")
      );
    },
  },
];

function formatSourceBadge(file) {
  if (file.source === "scientific") {
    return "Atlas Científico";
  }
  if (file.source === "studio") {
    const parts = file.name.split("/");
    const jobId = parts[0] ? parts[0].slice(0, 8) : "";
    return jobId ? `Execução Studio · ${jobId}` : "Execução Studio";
  }
  return file.source || "Geral";
}

function cleanFileName(fullName) {
  const parts = fullName.split("/");
  return parts[parts.length - 1];
}

export default function Results({ inspect }) {
  const [files, setFiles] = useState([]);
  const [history, setHistory] = useState([]);
  const [activeGroup, setActiveGroup] = useState("latency");
  const [filter, setFilter] = useState("");
  const [error, setError] = useState("");

  function refresh() {
    Promise.all([api("/api/artifacts"), api("/api/jobs")])
      .then(([f, h]) => {
        setFiles(f);
        setHistory(h);
      })
      .catch((e) => setError(e.message));
  }

  useEffect(refresh, []);

  // Classificar os arquivos nos 10 tipos de gráficos
  const { categorizedCharts, groupCounts, rawDataFiles } = useMemo(() => {
    const pdfFiles = files.filter((f) => f.name.toLowerCase().endsWith(".pdf"));
    const otherFiles = files.filter(
      (f) => !f.name.toLowerCase().endsWith(".pdf")
    );

    const counts = {
      all: 0,
      latency: 0,
      accuracy: 0,
      hardware: 0,
      inference: 0,
    };

    const charts = CHART_TYPES.map((chartType) => {
      const matchedPdfs = pdfFiles.filter((pdf) =>
        chartType.matches(pdf.name)
      );

      counts[chartType.group] += matchedPdfs.length;
      counts.all += matchedPdfs.length;

      return {
        ...chartType,
        files: matchedPdfs,
      };
    });

    return {
      categorizedCharts: charts,
      groupCounts: counts,
      rawDataFiles: otherFiles,
    };
  }, [files]);

  // Filtrar os gráficos por pasta ativa e termo de busca
  const visibleCharts = useMemo(() => {
    return categorizedCharts.filter((chart) => {
      if (activeGroup !== "all" && chart.group !== activeGroup) {
        return false;
      }

      if (!filter.trim()) return true;

      const term = filter.toLowerCase();
      const matchesMeta =
        chart.title.toLowerCase().includes(term) ||
        chart.subtitle.toLowerCase().includes(term) ||
        chart.groupLabel.toLowerCase().includes(term);

      const matchesAnyFile = chart.files.some((f) =>
        `${f.source}/${f.name}`.toLowerCase().includes(term)
      );

      return matchesMeta || matchesAnyFile;
    });
  }, [categorizedCharts, activeGroup, filter]);

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

      <div className="section-heading" style={{ marginTop: "32px" }}>
        <div>
          <h2>Gráficos de Resultados (10 Tipos em PDF)</h2>
          <p className="results-header-desc">
            Métricas essenciais organizadas por abas temáticas, sem excesso de arquivos.
          </p>
        </div>
      </div>

      {/* Navegação por Abas / Tabs horizontais */}
      <nav className="chart-folders-nav" aria-label="Abas de Gráficos">
        {CHART_GROUPS.map((group) => {
          const count = groupCounts[group.id] || 0;
          const isActive = activeGroup === group.id;
          return (
            <button
              key={group.id}
              className={`chart-folder-pill ${isActive ? "active" : ""}`}
              onClick={() => setActiveGroup(group.id)}
              type="button"
            >
              <span>{group.title}</span>
              <span className="pill-count">{count}</span>
            </button>
          );
        })}
      </nav>

      {/* Barra de Pesquisa / Filtro */}
      <div className="chart-filter-bar">
        <label htmlFor="chart-filter" style={{ fontWeight: 600 }}>
          Filtrar:
        </label>
        <input
          id="chart-filter"
          type="search"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          placeholder="Filtrar por nome, execução ou modelo..."
        />
        {filter && (
          <button onClick={() => setFilter("")} style={{ padding: "8px 12px" }}>
            Limpar
          </button>
        )}
      </div>

      {/* Grid com os Cards dos Tipos de Gráficos */}
      <div className="chart-cards-grid">
        {visibleCharts.length > 0 ? (
          visibleCharts.map((chart) => {
            const chartFiles = filter.trim()
              ? chart.files.filter((f) =>
                  `${f.source}/${f.name}`
                    .toLowerCase()
                    .includes(filter.toLowerCase())
                )
              : chart.files;

            return (
              <div key={chart.id} className="chart-card">
                <div className="chart-card-top">
                  <div className="chart-title-area">
                    <span className="chart-type-num">
                      Tipo {chart.typeNumber} de 10
                    </span>
                    <h3>{chart.title}</h3>
                  </div>
                  <span className="chart-group-tag">
                    {chart.groupLabel}
                  </span>
                </div>

                <p className="chart-subtitle">{chart.subtitle}</p>

                <div className="chart-files-section">
                  <div className="chart-files-label">
                    Arquivos em PDF disponíveis ({chartFiles.length}):
                  </div>

                  {chartFiles.length > 0 ? (
                    <div className="chart-pdf-list">
                      {chartFiles.map((file) => (
                        <div key={file.url} className="chart-pdf-item">
                          <div className="chart-pdf-info">
                            <span className="pdf-badge">PDF</span>
                            <span className="chart-source-tag">
                              {formatSourceBadge(file)}:
                            </span>
                            <span
                              className="chart-pdf-name"
                              title={cleanFileName(file.name)}
                            >
                              {cleanFileName(file.name)}
                            </span>
                          </div>
                          <a
                            href={file.url}
                            target="_blank"
                            rel="noreferrer"
                            className="pdf-action-link"
                          >
                            Visualizar PDF
                          </a>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="no-files-notice">
                      {chart.files.length === 0
                        ? "Nenhum PDF gerado para este tipo ainda. Execute uma rodada de teste no painel para criar."
                        : "Nenhum arquivo corresponde ao filtro de busca atual."}
                    </p>
                  )}
                </div>
              </div>
            );
          })
        ) : (
          <p className="no-files-notice" style={{ padding: "20px 0" }}>
            Nenhum gráfico encontrado para os critérios selecionados.
          </p>
        )}
      </div>

      {/* Seção Secundária Recolhível para Dados Brutos (CSV / JSON) */}
      <details className="raw-data-details">
        <summary>
          Dados Brutos e Tabelas (CSV, JSON, Markdown) — {rawDataFiles.length} arquivos
        </summary>
        <div className="raw-data-content">
          <p style={{ color: "var(--muted)", fontSize: "0.88rem", marginTop: 0 }}>
            Tabelas tabulares, predições brutas por foto e relatórios gerados
            pelas execuções para análise detalhada.
          </p>
          <div className="raw-data-list">
            {rawDataFiles.map((f) => (
              <div key={f.url} className="raw-data-item">
                <span
                  style={{
                    fontWeight: 700,
                    fontSize: "0.75rem",
                    padding: "2px 6px",
                    borderRadius: "4px",
                    background: "#e2e8f0",
                    color: "#475569",
                  }}
                >
                  {f.name.split(".").pop().toUpperCase()}
                </span>
                <span style={{ fontWeight: 600, color: "var(--ink)" }}>
                  {formatSourceBadge(f)}:
                </span>
                <a href={f.url} target="_blank" rel="noreferrer">
                  {cleanFileName(f.name)}
                </a>
              </div>
            ))}
          </div>
        </div>
      </details>
    </>
  );
}

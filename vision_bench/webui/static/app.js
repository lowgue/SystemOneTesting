"use strict";
const $ = (id) => document.getElementById(id);
const titles = {
  live: [
    "Teste de imagem",
    "Inspecione uma imagem e compare as respostas dos modelos.",
  ],
  benchmark: [
    "Benchmark",
    "Meça desempenho com dados, perguntas e configurações registrados.",
  ],
  results: [
    "Resultados",
    "Consulte os relatórios e gráficos das suas avaliações.",
  ],
  diagnostics: [
    "Diagnóstico",
    "Entenda o estado do serviço antes de investigar uma falha.",
  ],
};
let catalog,
  imageData,
  currentJob,
  jobResult,
  pollTimer,
  files = [];
function node(tag, text, cls) {
  const n = document.createElement(tag);
  if (text !== undefined) n.textContent = text;
  if (cls) n.className = cls;
  return n;
}
function showError(message) {
  $("global-error").textContent = message;
  $("global-error").hidden = !message;
}
async function api(path, body) {
  const response = await fetch(
    path,
    body === undefined
      ? {}
      : {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        },
  );
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}
function options(select, entries, chosen) {
  select.replaceChildren(
    ...entries.map(([value, label]) => {
      const o = node("option", label);
      o.value = value;
      return o;
    }),
  );
  if (entries.some((e) => e[0] === chosen)) select.value = chosen;
}
function chooseModels() {
  const config = catalog.configs.find((c) => c.id === $("config").value);
  const field = $("models");
  field.replaceChildren(node("legend", "Modelos que serão avaliados"));
  for (const model of config.models) {
    const label = node("label");
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.value = model.id;
    checkbox.name = "model";
    checkbox.checked = model.id === "clef-flash-ollama" || model.simulated;
    label.append(
      checkbox,
      node("span", model.id + (model.simulated ? " (simulação)" : "")),
    );
    field.append(label);
  }
}
function chooseDomain() {
  const contract = catalog.contracts.find((c) => c.id === $("contract").value);
  options(
    $("domain"),
    contract.domains.map((d) => [d, d === "vehicle" ? "Veículo" : "Pessoa"]),
    $("domain").value,
  );
}
function configuration() {
  return {
    config: $("config").value,
    contract: $("contract").value,
    models: [...document.querySelectorAll("input[name=model]:checked")].map(
      (n) => n.value,
    ),
    device: $("device").value,
    threads: Number($("threads").value),
  };
}
function navigate() {
  let view = location.hash.slice(1) || "live";
  if (!titles[view]) view = "live";
  document
    .querySelectorAll(".view")
    .forEach((el) => (el.hidden = el.id !== `view-${view}`));
  document.querySelectorAll("nav a").forEach((el) => {
    if (el.dataset.view === view) el.setAttribute("aria-current", "page");
    else el.removeAttribute("aria-current");
  });
  $("configuration").hidden = !["live", "benchmark"].includes(view);
  [$("page-title").textContent, $("page-subtitle").textContent] = titles[view];
  if (view === "results") refreshFiles();
  if (view === "diagnostics") diagnostics();
}
function busy(value) {
  ["run-live", "run-benchmark"].forEach((id) => ($(id).disabled = value));
  document.querySelectorAll(".cancel").forEach((el) => (el.hidden = !value));
}
async function start(kind) {
  showError("");
  if (kind === "live" && !imageData) {
    showError("Escolha uma imagem antes de iniciar.");
    return;
  }
  const body = { ...configuration(), kind };
  if (!body.models.length) {
    showError("Selecione pelo menos um modelo.");
    return;
  }
  if (kind === "live")
    Object.assign(body, { domain: $("domain").value, image_data: imageData });
  else
    Object.assign(body, {
      dataset: $("dataset").value,
      repetitions: Number($("repetitions").value),
      limit: Number($("limit").value),
    });
  busy(true);
  try {
    const job = await api("/api/jobs", body);
    currentJob = job.id;
    jobResult = null;
    localStorage.setItem("vision-studio-job", job.id);
    poll();
  } catch (error) {
    busy(false);
    showError(error.message);
  }
}
const names = {
  presence: "Presença",
  make: "Marca",
  color: "Cor",
  body: "Carroceria",
  helmet: "Capacete",
  carrying: "Objeto transportado",
  clothing: "Vestimenta",
};
const values = {
  yes: "Sim",
  no: "Não",
  unknown: "Não identificado",
  not_visible: "Não visível",
  parse_error: "Resposta não reconhecida",
  black: "Preto",
  white: "Branco",
  gray_silver: "Cinza / prata",
  red: "Vermelho",
  blue: "Azul",
  green: "Verde",
  yellow: "Amarelo",
  orange: "Laranja",
  brown: "Marrom",
  purple: "Roxo",
  pink: "Rosa",
  other: "Outro",
  sedan: "Sedã",
  suv: "SUV",
  hatch: "Hatchback",
  truck: "Picape",
  coupe: "Cupê",
  convertible: "Conversível",
  wagon: "Perua",
  van: "Van",
  motorcycle: "Moto",
  bus: "Ônibus",
  backpack: "Mochila",
  bag_or_box: "Bolsa ou caixa",
  none: "Nenhum",
  safety_vest: "Colete de segurança",
  jacket_coat: "Jaqueta ou casaco",
  shirt_tshirt: "Camisa ou camiseta",
};
function displayValue(key, value) {
  if (key === "make" && !["unknown", "parse_error", "other"].includes(value))
    return value.replaceAll("_", " ").replace(/\b\w/g, (c) => c.toUpperCase());
  return values[value] || value;
}
function renderAnswers(job) {
  const container = $("answers");
  container.replaceChildren();
  for (const result of job.results || []) {
    const article = node("article", undefined, "answer");
    article.append(node("h3", result.model));
    if (result.simulated)
      article.append(
        node(
          "p",
          "Simulação de infraestrutura. Não é um modelo real.",
          "simulated",
        ),
      );
    if (result.error) article.append(node("p", result.error, "error"));
    else {
      const list = node("dl");
      for (const [key, value] of Object.entries(result.predicted || {}))
        list.append(
          node("dt", names[key] || key),
          node("dd", displayValue(key, value)),
        );
      article.append(list);
      article.append(
        node(
          "p",
          result.valid
            ? "Formato válido"
            : "Há campos inválidos ou não reconhecidos",
          result.valid ? "hint" : "error",
        ),
      );
      if (result.description) {
        const facts = Object.entries(result.facts || {}).map(
          ([k, v]) => `${names[k] || k}: ${displayValue(k, v)}`,
        );
        article.append(
          node(
            "p",
            `${job.domain === "vehicle" ? "Veículo" : "Pessoa"}${facts.length ? " — " + facts.join("; ") : ""}.`,
          ),
        );
      }
      article.append(
        node(
          "p",
          `Inferência: ${result.elapsed_ms.toFixed(0)} ms | Preparação: ${result.load_ms.toFixed(0)} ms`,
          "timing",
        ),
      );
    }
    const details = node("details");
    details.append(
      node("summary", "Resposta completa"),
      node("pre", JSON.stringify(result, null, 2)),
    );
    article.append(details);
    container.append(article);
  }
  if (!container.children.length)
    container.append(node("p", "Aguardando o primeiro resultado.", "empty"));
}
async function poll() {
  clearTimeout(pollTimer);
  if (!currentJob) return;
  try {
    const job = await api(`/api/jobs/${currentJob}`);
    jobResult = job;
    const running = ["queued", "running"].includes(job.state);
    busy(running);
    const labels = {
      queued: "Preparando tarefa",
      running: job.phase || "Executando benchmark",
      complete: job.errors ? "Concluído com falhas" : "Concluído",
      failed: "Falha na execução",
      cancelled: "Tarefa cancelada",
    };
    const label =
      (labels[job.state] || job.state) +
      (job.current_model ? ` — ${job.current_model}` : "");
    if (job.kind === "live") {
      $("live-state").textContent = label;
      renderAnswers(job);
      $("download").disabled = false;
    } else {
      $("benchmark-state").textContent = label;
      const p = job.progress || {};
      $("progress").max = p.planned || 1;
      $("progress").value = p.completed || 0;
      $("job-log").textContent = job.log || "Preparando modelos e aquecimento…";
    }
    $("diagnostic-log").textContent = job.log || "Sem mensagens no registro.";
    if (job.error) showError(job.error);
    if (running) pollTimer = setTimeout(poll, 900);
  } catch (error) {
    busy(false);
    showError(`Não foi possível acompanhar a tarefa: ${error.message}`);
  }
}
async function refreshFiles() {
  try {
    files = await api("/api/artifacts");
    renderFiles();
  } catch (error) {
    showError(error.message);
  }
}
function renderFiles() {
  const filter = $("artifact-filter").value.toLowerCase();
  const container = $("artifacts");
  container.replaceChildren();
  const selected = files.filter((f) =>
    `${f.source}/${f.name}`.toLowerCase().includes(filter),
  );
  if (!selected.length) {
    container.append(
      node(
        "p",
        "Nenhum arquivo encontrado. Execute um benchmark ou ajuste o filtro.",
        "empty",
      ),
    );
    return;
  }
  for (const file of selected) {
    const row = node("div", undefined, "artifact");
    if (file.name.endsWith(".png")) {
      const img = document.createElement("img");
      img.src = file.url;
      img.alt = "Prévia do gráfico";
      img.loading = "lazy";
      row.append(img);
    } else
      row.append(
        node("span", file.name.split(".").pop().toUpperCase(), "file-icon"),
      );
    const link = node("a", `${file.source} / ${file.name}`);
    link.href = file.url;
    link.target = "_blank";
    link.rel = "noopener";
    row.append(link, node("small", "Abrir arquivo"));
    container.append(row);
  }
}
async function diagnostics() {
  $("diagnostics").textContent = "Verificando o serviço…";
  try {
    const data = await api("/api/diagnostics");
    const dl = node("dl", undefined, "diagnostic-list");
    const rows = [
      [
        "Serviço Ollama",
        data.ollama === "online"
          ? "Respondendo"
          : "Indisponível ou não verificado",
      ],
      ["Versão do Ollama", data.version || "Não identificada"],
      [
        "Clef-Flash 9B instalado",
        data.clef_installed
          ? "Sim"
          : "Não confirmado — execute ollama pull clef-flash:9b",
      ],
      ["RAM total", `${data.ram_total_gb} GiB`],
      ["RAM disponível", `${data.ram_available_gb} GiB`],
      [
        "Modelos carregados",
        (data.loaded || []).map((m) => m.name).join(", ") ||
          "Nenhum; isso não indica a causa de uma falha.",
      ],
      [
        "Inferência verificada",
        "Não. Execute um teste de imagem para verificar.",
      ],
    ];
    for (const [label, value] of rows)
      dl.append(node("dt", label), node("dd", value));
    $("diagnostics").replaceChildren(dl);
    if (data.error) $("diagnostics").append(node("p", data.error, "error"));
    const details = node("details");
    details.append(
      node("summary", "Informações completas"),
      node("pre", JSON.stringify(data, null, 2)),
    );
    $("diagnostics").append(details);
  } catch (error) {
    $("diagnostics").replaceChildren(node("p", error.message, "error"));
  }
}
$("image").addEventListener("change", () => {
  const file = $("image").files[0];
  if (!file) return;
  if (file.size > 16 * 1024 * 1024) {
    showError("Use uma imagem de até 16 MiB.");
    return;
  }
  const reader = new FileReader();
  reader.onload = () => {
    imageData = reader.result;
    $("preview").src = imageData;
    $("preview").hidden = false;
    $("image-placeholder").hidden = true;
    $("image-name").textContent = file.name;
    showError("");
  };
  reader.readAsDataURL(file);
});
$("config").onchange = chooseModels;
$("contract").onchange = chooseDomain;
$("run-live").onclick = () => start("live");
$("run-benchmark").onclick = () => start("benchmark");
document.querySelectorAll(".cancel").forEach(
  (button) =>
    (button.onclick = async () => {
      if (!currentJob) return;
      try {
        await api(`/api/jobs/${currentJob}/cancel`, {});
        poll();
      } catch (error) {
        showError(error.message);
      }
    }),
);
$("download").onclick = () => {
  if (!jobResult) return;
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(jobResult, null, 2)], {
      type: "application/json",
    }),
  );
  const a = node("a");
  a.href = url;
  a.download = `teste-${jobResult.id}.json`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
$("refresh-results").onclick = refreshFiles;
$("artifact-filter").oninput = renderFiles;
$("refresh-diagnostics").onclick = diagnostics;
window.addEventListener("hashchange", navigate);
(async () => {
  try {
    catalog = await api("/api/catalog");
    options(
      $("config"),
      catalog.configs.map((c) => [c.id, c.id]),
      "ollama",
    );
    options(
      $("contract"),
      catalog.contracts.map((c) => [
        c.id,
        c.id === "ollama"
          ? "Compartilhado Ollama (até 26 alternativas)"
          : "Original (49 alternativas de marca)",
      ]),
      "ollama",
    );
    options(
      $("dataset"),
      catalog.datasets.map((d) => [d.id, `${d.id} — ${d.events} eventos`]),
    );
    chooseModels();
    chooseDomain();
    $("connection").textContent = "Painel conectado";
    currentJob =
      catalog.current_job || localStorage.getItem("vision-studio-job");
    if (currentJob) {
      try {
        await api(`/api/jobs/${currentJob}`);
        poll();
      } catch {
        currentJob = null;
        localStorage.removeItem("vision-studio-job");
      }
    }
    navigate();
  } catch (error) {
    showError(error.message);
    $("connection").textContent = "Falha de conexão";
  }
})();

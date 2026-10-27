export const pages = {
  live: ["Avaliar foto", "Envie uma foto e veja as respostas dos modelos."],
  benchmark: [
    "Avaliar dataset",
    "Avalie várias fotos automaticamente e compare acurácia, tempo e falhas.",
  ],
  results: ["Resultados", "Histórico de tarefas, gráficos e métricas."],
  diagnostics: [
    "Ambiente e modelos",
    "Prepare o Clef e verifique o serviço Ollama.",
  ],
};
export const labels = {
  presence: "Presença",
  make: "Marca",
  color: "Cor",
  body: "Carroceria",
  helmet: "Capacete",
  carrying: "Objeto transportado",
  clothing: "Vestimenta",
};
const values = {
  __error__: "Falha de execução",
  __missing__: "Ausente",
  yes: "Sim",
  no: "Não",
  unknown: "Não identificado",
  not_visible: "Não visível",
  parse_error: "Resposta inválida",
  black: "Preto",
  white: "Branco",
  red: "Vermelho",
  blue: "Azul",
  gray_silver: "Cinza / prata",
  other: "Outro",
  sedan: "Sedã",
  truck: "Truck",
  pickup: "Picape",
  bus: "Ônibus",
  motorcycle: "Motocicleta",
  suv: "SUV",
};
export const text = (value) => values[value] || value?.replaceAll("_", " ");
export const isActive = (job) =>
  job && ["queued", "running"].includes(job.state);
export function stateText(job) {
  if (!job) return "Nenhuma tarefa iniciada.";
  return (
    {
      queued: "Preparando tarefa",
      running:
        job.progress?.status === "complete" ? job.phase || "Gerando relatório"
        : job.progress?.phase === "warmup"
          ? `Preparando modelo ${job.progress.model} e aquecimento…`
          : job.progress?.model
            ? `Avaliando fotos com ${job.progress.model}`
            : job.phase || "Executando",
      complete: job.errors ? "Concluído com falhas" : "Concluído",
      failed: job.error || "Falha na tarefa",
      cancelled: "Cancelado",
    }[job.state] || job.state
  );
}

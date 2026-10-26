export const pages = {
  live: ["Teste de imagem", "Compare modelos sobre a mesma imagem."],
  benchmark: [
    "Benchmark",
    "Execute avaliações reproduzíveis e acompanhe os resultados.",
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
  truck: "Picape",
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
      running: job.phase || "Executando",
      complete: job.errors ? "Concluído com falhas" : "Concluído",
      failed: job.error || "Falha na tarefa",
      cancelled: "Cancelado",
    }[job.state] || job.state
  );
}

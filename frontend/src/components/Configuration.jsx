import React from "react";
const modelNames = {
  "clef-flash-ollama": "Clef-Flash 9B (Ollama)",
  "blip-vqa": "BLIP VQA",
  "laya-vision-head512": "Laya Vision",
  "clef-flash-local-q4": "Clef-Flash (llama-server)",
};
const sourceNames = {
  ollama: "Ollama local",
  local: "Modelos locais + llama-server",
  cloudflare: "Cloudflare",
  smoke_mock: "Simulação para testar a aplicação",
};
export default function Configuration({
  catalog,
  settings,
  setSettings,
  busy,
}) {
  const models =
    catalog.configs.find((c) => c.id === settings.config)?.models || [];
  const update = (key, value) => setSettings((s) => ({ ...s, [key]: value }));
  return (
    <section
      className="configuration simple-config"
      aria-label="Configuração da avaliação"
    >
      <div className="model-source">
        <label htmlFor="config">Onde executar os modelos?</label>
        <select
          id="config"
          disabled={busy}
          value={settings.config}
          onChange={(e) => {
            const config = catalog.configs.find((c) => c.id === e.target.value);
            setSettings((s) => ({
              ...s,
              config: config.id,
              models: config.models
                .filter((m) => m.id === "clef-flash-ollama" || m.simulated)
                .map((m) => m.id),
            }));
          }}
        >
          {catalog.configs.map((c) => (
            <option key={c.id} value={c.id}>
              {sourceNames[c.id] || c.id}
            </option>
          ))}
        </select>
      </div>
      <fieldset disabled={busy}>
        <legend>Quais modelos avaliar?</legend>
        {models.map((m) => (
          <label key={m.id}>
            <input
              name="model"
              type="checkbox"
              checked={settings.models.includes(m.id)}
              onChange={(e) =>
                update(
                  "models",
                  e.target.checked
                    ? [...settings.models, m.id]
                    : settings.models.filter((id) => id !== m.id),
                )
              }
            />
            {modelNames[m.id] || m.id}
            {m.simulated ? " (simulação)" : ""}
          </label>
        ))}
      </fieldset>
      <details className="advanced-config">
        <summary>Configuração avançada</summary>
        <div className="advanced-fields">
          <div>
            <label htmlFor="contract">Perguntas compartilhadas</label>
            <select
              id="contract"
              disabled={busy}
              value={settings.contract}
              onChange={(e) => update("contract", e.target.value)}
            >
              {catalog.contracts.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.id === "ollama"
                    ? "Compatível com Ollama (26 alternativas)"
                    : c.id === "historical" ? "Testes históricos (inclui ônibus, moto e picape)"
                    : "Original (49 alternativas de marca)"}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label htmlFor="device">Dispositivo Python</label>
            <select
              id="device"
              disabled={busy}
              value={settings.device}
              onChange={(e) => update("device", e.target.value)}
            >
              <option value="cpu">CPU</option>
              <option value="cuda">CUDA</option>
            </select>
          </div>
          <div>
            <label htmlFor="threads">Threads Python</label>
            <input
              id="threads"
              type="number"
              disabled={busy}
              min="1"
              max="256"
              value={settings.threads}
              onChange={(e) => update("threads", Number(e.target.value))}
            />
          </div>
        </div>
        <p className="hint">
          Ollama gerencia seu próprio hardware. BLIP e Laya precisam do backend
          Python instalado. Todos os modelos recebem o mesmo contrato de
          perguntas.
        </p>
      </details>
    </section>
  );
}

import React from "react";

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
    <section className="configuration" aria-label="Configuração da avaliação">
      <div>
        <label htmlFor="config">Fonte dos modelos</label>
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
              {c.id}
            </option>
          ))}
        </select>
      </div>
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
          disabled={busy}
          type="number"
          min="1"
          max="256"
          value={settings.threads}
          onChange={(e) => update("threads", Number(e.target.value))}
        />
      </div>
      <fieldset disabled={busy}>
        <legend>Modelos avaliados em sequência</legend>
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
            {m.id}
            {m.simulated ? " (simulação)" : ""}
          </label>
        ))}
      </fieldset>
      <p className="hint">
        Ollama gerencia o próprio hardware. BLIP/Laya exigem o backend Python
        com pesos instalado; a imagem Docker padrão usa backends HTTP.
      </p>
    </section>
  );
}

import React from "react";
import { isActive, stateText } from "../presentation.js";

export default function JobState({ job, cancel }) {
  return (
    <div className="job-state">
      <p role="status" aria-live="polite">
        {stateText(job)}
      </p>
      {job?.progress && (
        <p className="hint">
          {job.progress.completed} de {job.progress.planned} previsões
          concluídas (
          {Math.round(
            (100 * job.progress.completed) / (job.progress.planned || 1),
          )}
          %).
          {job.progress.current_filename
            ? ` Última foto: ${job.progress.current_filename}.`
            : ""}
        </p>
      )}
      {job?.progress && (
        <progress
          aria-label="Progresso da avaliação"
          max={job.progress.planned || 1}
          value={job.progress.completed || 0}
        />
      )}
      {Boolean(job?.progress?.errors) && (
        <p className="error">
          {job.progress.errors} falhas até agora. Consulte o registro para os
          detalhes.
        </p>
      )}
      {job?.download?.total > 0 && (
        <progress
          aria-label="Download do modelo"
          max={job.download.total}
          value={job.download.completed || 0}
        />
      )}
      {isActive(job) && <button onClick={cancel}>Cancelar tarefa</button>}
    </div>
  );
}

import React from "react";
import { isActive, stateText } from "../presentation.js";

export default function JobState({ job, cancel }) {
  return (
    <div className="job-state">
      <p role="status" aria-live="polite">
        {stateText(job)}
      </p>
      {job?.progress && (
        <progress
          aria-label="Progresso da avaliação"
          max={job.progress.planned || 1}
          value={job.progress.completed || 0}
        />
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

export async function api(path, body) {
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
  if (!response.ok)
    throw new Error(data.error || data.detail || `HTTP ${response.status}`);
  return data;
}
export function download(job) {
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(job, null, 2)], { type: "application/json" }),
  );
  const link = document.createElement("a");
  link.href = url;
  link.download = `teste-${job.id}.json`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

#!/usr/bin/env bash
# Start one warmed vLLM server for the Clef/System One benchmark.
# Environment variables below keep caches and the virtualenv on /models.

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/models/cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/models/cache/uv-python}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/models/cache/pip}"
export HF_HOME="${HF_HOME:-/models/cache/huggingface}"
export VLLM_ENV="${VLLM_ENV:-/models/venvs/systemone-vllm-cu128}"
export PYTHON312="${PYTHON312:-$(uv python find 3.12)}"

: "${PYTHON312:?Defina PYTHON312 ou instale Python 3.12 com uv}"
[[ -x "$VLLM_ENV/bin/vllm" ]] || {
  echo "ERRO: vLLM não encontrado em $VLLM_ENV/bin/vllm" >&2
  exit 2
}

HOST="${VLLM_HOST:-127.0.0.1}"
PORT="${VLLM_PORT:-8000}"
MODEL="${VLLM_MODEL:-Cloudflare/clef-flash}"

echo "Servidor vLLM externo: $MODEL em http://$HOST:$PORT/v1/systemone" >&2
exec "$VLLM_ENV/bin/vllm" serve "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --max-model-len 4096 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.9 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --enable-per-request-metrics \
  --logprobs-mode raw_logprobs \
  --trust-remote-code

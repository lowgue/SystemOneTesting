#!/usr/bin/env bash
set -euo pipefail

clef_model_path="$(hf download ggml-org/Clef-Flash-GGUF Clef-Flash-Q4_K_M.gguf --quiet)"
clef_mmproj_path="$(hf download ggml-org/Clef-Flash-GGUF mmproj-Clef-Flash-Q8_0.gguf --quiet)"

exec /usr/local/lib/ollama/llama-server \
  --model "$clef_model_path" \
  --mmproj "$clef_mmproj_path" \
  --host 127.0.0.1 --port 11437 \
  -c 2048 -np 1 -b 1024 -ub 1024 \
  -t 8 -tb 8 --load-mode mmap --fit off

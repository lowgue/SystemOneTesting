#!/usr/bin/env bash
set -euo pipefail
# Resolve from this checkout instead of a machine-specific executable path.
clef_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
clef_python="${BENCH_PYTHON:-$clef_repo_root/.venv/bin/python}"
cd -- "$clef_repo_root"
exec "$clef_python" -m vision_bench clef-server "$@"

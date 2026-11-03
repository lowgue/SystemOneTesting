#!/usr/bin/env bash
# Quick automated setup script for Linux / macOS
# Clones submodules, installs dependencies, caches model weights and verifies readiness.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODE="cpu"
MODELS="laya d1-omni clef-gguf"
RUN_SMOKE=0

usage() {
    echo "Uso: $0 [opções]"
    echo ""
    echo "Opções:"
    echo "  --cpu               Configura ambiente para inferência em CPU (padrão)"
    echo "  --gpu               Instala dependências com suporte a GPU/CUDA"
    echo "  --models <lista>    Modelos para baixar (padrão: 'laya d1-omni clef-gguf')"
    echo "                      Presets: vision-local, vision-ollama, vision-all, all"
    echo "  --smoke             Executa um smoke test (1 imagem) após a instalação"
    echo "  -h, --help          Exibe esta ajuda"
    echo ""
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpu)
            MODE="cpu"
            shift
            ;;
        --gpu)
            MODE="gpu"
            shift
            ;;
        --models)
            MODELS="$2"
            shift 2
            ;;
        --smoke)
            RUN_SMOKE=1
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Opção desconhecida: $1" >&2
            usage
            ;;
    esac
done

echo "======================================================================"
echo " SystemOne Vision Benchmark - Instalação Rápida Automatizada"
echo "======================================================================"
echo "Diretório do projeto : $ROOT_DIR"
echo "Modo de aceleração   : $MODE"
echo "Modelos selecionados : $MODELS"
echo "======================================================================"

# 1. Checar Python
PYTHON_BIN=""
for cmd in python3.12 python3 python; do
    if command -v "$cmd" >/dev/null 2>&1; then
        ver=$("$cmd" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        major=$(echo "$ver" | cut -d. -f1)
        minor=$(echo "$ver" | cut -d. -f2)
        if [[ "$major" -eq 3 && "$minor" -ge 12 ]]; then
            PYTHON_BIN=$(command -v "$cmd")
            echo "✓ Python compatível encontrado: $PYTHON_BIN ($ver)"
            break
        fi
    fi
done

if [[ -z "$PYTHON_BIN" ]]; then
    echo "ERRO: Python 3.12 ou superior é obrigatório." >&2
    exit 1
fi

# 2. Inicializar submódulos Git
echo "--> Atualizando submódulos Git..."
git submodule update --init --recursive

# 3. Criar / reutilizar ambiente virtual
VENV_DIR="$ROOT_DIR/.venv"
VENV_PY="$VENV_DIR/bin/python"

if [[ ! -x "$VENV_PY" ]]; then
    echo "--> Criando ambiente virtual em $VENV_DIR..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# 4. Instalar dependências
echo "--> Atualizando pip e instalando dependências base..."
"$VENV_PY" -m pip install --upgrade pip

if [[ "$MODE" == "cpu" ]]; then
    echo "--> Instalando dependências de CPU..."
    "$VENV_PY" -m pip install -r requirements/cpu.txt -e .
else
    echo "--> Instalando dependências de GPU (PyTorch CUDA + bibliotecas)..."
    "$VENV_PY" -m pip install -r requirements/models.txt -e . -e ./vendor/laya-vision
fi

# 5. Baixar e verificar modelos
echo "--> Baixando e verificando pesos dos modelos selecionados..."
# shellcheck disable=SC2086
"$VENV_PY" "$ROOT_DIR/scripts/download_models.py" --models $MODELS

# 6. Rodar o doctor
echo "--> Executando diagnóstico do ambiente (doctor)..."
"$VENV_PY" -m vision_bench doctor --config benchmarks/configs/local.json

# 7. Smoke test opcional
if [[ "$RUN_SMOKE" -eq 1 ]]; then
    echo "--> Executando smoke test rápido (1 imagem)..."
    "$VENV_PY" -m vision_bench run \
        --config benchmarks/configs/local.json \
        --models laya-vision-head512 d1-omni-600M \
        --limit 1 --repetitions 1 \
        --output "$ROOT_DIR/.artifacts/runs/quick-smoke"
    echo "✓ Smoke test concluído com sucesso!"
fi

echo ""
echo "======================================================================"
echo " Instalação concluída com sucesso!"
echo " Para rodar o benchmark comparativo completo:"
echo "   .venv/bin/python -m vision_bench run \\"
echo "     --config benchmarks/configs/local.json \\"
echo "     --models laya-vision-head512 d1-omni-600M \\"
echo "     --repetitions 3 --output .artifacts/runs/meu-teste"
echo "======================================================================"

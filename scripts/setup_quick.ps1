# Quick automated setup script for Windows (PowerShell)
# Clones submodules, installs dependencies, caches model weights and verifies readiness.
param (
    [switch]$Gpu = $false,
    [string]$Models = "laya d1-omni clef-gguf",
    [switch]$Smoke = $false,
    [switch]$Help = $false
)

if ($Help) {
    Write-Host "Uso: .\scripts\setup_quick.ps1 [-Gpu] [-Models '<lista>'] [-Smoke]"
    Write-Host ""
    Write-Host "Opções:"
    Write-Host "  -Gpu               Instala dependências com suporte a GPU"
    Write-Host "  -Models '<lista>'  Modelos a baixar (default: 'laya d1-omni clef-gguf')"
    Write-Host "  -Smoke             Executa teste rápido de 1 imagem ao final"
    exit 0
}

$ErrorActionPreference = "Stop"
$RootDir = Split-Path -Parent $PSScriptRoot
Set-Location $RootDir

Write-Host "======================================================================"
Write-Host " SystemOne Vision Benchmark - Instalador Rapido Windows"
Write-Host "======================================================================"

# 1. Procurar Python 3.12
$PythonCmd = $null
if (Get-Command py -ErrorAction SilentlyContinue) {
    $PythonCmd = "py -3.12"
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $PythonCmd = "python"
} else {
    Write-Error "Python 3.12 nao encontrado. Instale Python 3.12 ou use o inicializador 'py'."
}

# 2. Submodulos Git
Write-Host "--> Atualizando submodulos Git..."
git submodule update --init --recursive

# 3. Ambiente Virtual
$VenvDir = Join-Path $RootDir ".venv"
$VenvPy = Join-Path $VenvDir "Scripts\python.exe"

if (-not (Test-Path $VenvPy)) {
    Write-Host "--> Criando ambiente virtual em $VenvDir..."
    Invoke-Expression "$PythonCmd -m venv `"$VenvDir`""
}

# 4. Instalar Dependencias
Write-Host "--> Atualizando pip..."
& $VenvPy -m pip install --upgrade pip

if ($Gpu) {
    $TorchIndexUrl = if ($env:PYTORCH_INDEX_URL) { $env:PYTORCH_INDEX_URL } else { "https://download.pytorch.org/whl/cu128" }
    Write-Host "--> Instalando PyTorch CUDA pelo índice oficial: $TorchIndexUrl"
    & $VenvPy -m pip install torch torchvision --index-url $TorchIndexUrl
    Write-Host "--> Instalando bibliotecas dos modelos..."
    & $VenvPy -m pip install -r requirements/models.txt -e . -e ./modelos/laya/laya_vision
    Write-Host "--> Validando GPU CUDA encontrada..."
    & $VenvPy -c "import torch; assert torch.cuda.is_available(), 'PyTorch não encontrou CUDA; confira o driver NVIDIA'; print(torch.cuda.get_device_name(0), torch.version.cuda)"
} else {
    Write-Host "--> Instalando pacotes para CPU..."
    & $VenvPy -m pip install -r requirements/cpu.txt -e .
}

# 5. Baixar e verificar modelos
Write-Host "--> Baixando e verificando modelos..."
$ModelArgs = $Models.Split(' ')
& $VenvPy "$RootDir\scripts\download_models.py" --models @ModelArgs

# 6. Doctor
Write-Host "--> Executando diagnostic doctor..."
& $VenvPy -m vision_bench doctor --config benchmarks/configs/local.json

# 7. Smoke test
if ($Smoke) {
    Write-Host "--> Executando smoke test (1 imagem)..."
    & $VenvPy -m vision_bench run --config benchmarks/configs/local.json --models laya-vision-head512 d1-omni-600M --device auto --threads auto --limit 1 --repetitions 1 --output "$RootDir\.artifacts\runs\quick-smoke"
    Write-Host "✓ Smoke test concluido!"
}

Write-Host "======================================================================"
Write-Host " Instalacao concluida com sucesso!"
Write-Host " Execute o runner com:"
Write-Host "   .venv\Scripts\python.exe -m vision_bench run --config benchmarks/configs/local.json --models blip-vqa laya-vision-head512 d1-omni-600M --device auto --threads auto --output .artifacts\runs\teste"
Write-Host "======================================================================"

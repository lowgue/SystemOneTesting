# System One — Comparação portátil de modelos visuais

CLI para comparar modelos visuais (BLIP VQA, Laya Vision, d1-omni-600M e Clef via System One/Ollama) e gerar relatórios com gráficos salvos. Sem interface web.

## Instalar

Python 3.12.

```bash
git clone --recurse-submodules https://github.com/lowgue/SystemOneTesting.git
cd SystemOneTesting
python3.12 scripts/setup_benchmark.py --backend cpu   # ou --backend http (somente Clef/HTTP)
```

Rápido em Linux/macOS: `./scripts/setup_quick.sh` (instala e baixa os modelos por padrão). Windows: `./scripts/setup_quick.ps1`. Veja [QUICK_START_MODELS.md](docs/QUICK_START_MODELS.md).

## Uso

```bash
.venv/bin/python -m vision_bench doctor            # valida ambiente, sem carregar pesos nem inferir
.venv/bin/python -m vision_bench run --dry-run --output .artifacts/runs/validacao

.venv/bin/python -m vision_bench run \
  --config benchmarks/configs/local.json \
  --models blip-vqa laya-vision-head512 \
  --repetitions 5 --output .artifacts/runs/meu-run

.venv/bin/python -m vision_bench report .artifacts/runs/meu-run --output .artifacts/reports/meu-relatorio
```

O `report` gera `README.md`, `metrics.csv`, `metrics.json` e gráficos PNG (300 dpi)/SVG/PDF.

## Comandos

| Comando | Função |
|---|---|
| `doctor` | Valida arquivos, dependências e ambiente, sem carregar pesos nem enviar inferência |
| `run` | Executa os modelos em sequência com configuração congelada; `--resume` retoma |
| `report` | Gera relatório e gráficos de uma ou mais execuções, sem misturar máquinas |
| `pack-data` / `unpack-data` | Exporta/importa datasets verificados (ZIP com hashes, sem pesos) |
| `clef-server` | Baixa os pesos do Clef e inicia um `llama-server` compatível |

## Estrutura

```text
vision_bench/   Motor de benchmarks e relatórios (CLI, runner, gráficos)
benchmarks/     Configurações de modelos, contratos e dataset portátil
eval/           Experimentos e evidências científicas anteriores preservados
scripts/        Instalação e download de modelos
requirements/   Dependências Python por backend
docs/           Guias de comparação e metodologia
tests/          Testes do motor
```

## Documentação

- [Comparações reproduzíveis](docs/COMPARISON_SETUP.md)
- [Datasets e métricas](docs/DATASETS.md)
- [Metodologia das comparações](docs/PROMPT_COMPARISONS.md)
- [Modelos e início rápido](docs/QUICK_START_MODELS.md)
- [Atlas estatístico anterior](eval/figures/scientific_20261026_final/README.md)

Os gabaritos existentes são exploratórios; o benchmark separa execuções e computadores. Recursos do cliente HTTP não medem o consumo do servidor Ollama.

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

## Testar e gerar gráficos

Depois de clonar o repositório, rode um comando para criar `.venv`, instalar as dependências leves dos testes, executar a suíte inteira e gerar relatórios para todas as execuções completas encontradas em `.artifacts/runs/`:

```bash
python3 scripts/test_all.py
```

No Windows, use `py -3.12 scripts/test_all.py`. Os relatórios e gráficos ficam em `charts/reports/AAAAmmdd-HHMMSS/`; cada execução tem gráficos de desempenho e avaliação em PNG/PDF, com os dados CSV/JSON correspondentes. O script não baixa pesos nem executa inferência. Para gerar gráficos a partir de execuções específicas:

```bash
.venv/bin/python scripts/report_runs.py .artifacts/runs/minha-execucao
```

Se as dependências já estiverem instaladas em `.venv`, use `python3 scripts/test_all.py --no-install`.

### Computador com GPU NVIDIA (ex.: RTX 3090)

Com driver NVIDIA instalado e `nvidia-smi` funcionando, rode:

```bash
python3 scripts/test_all.py --gpu
```

No Windows, use `py -3.12 scripts/test_all.py --gpu`. Esse fluxo instala a build CUDA oficial do PyTorch, verifica que a GPU está acessível, executa toda a suíte unitária e roda o benchmark completo das 22 imagens portáteis com BLIP, Laya e D1. O device é selecionado automaticamente; `--threads auto` usa os núcleos físicos reportados pelo sistema, pois os modelos são executados um por vez. TF32 fica habilitado na GPU NVIDIA para acelerar operações de ponto flutuante. A rodada real baixa os pesos dos três modelos quando necessário e salva execução, dados e gráficos em `.artifacts/runs/` e `charts/reports/`.

O Clef não entra nessa rodada: ele roda em um servidor externo e precisa de configuração própria de `llama-server`/CUDA. Para preparar esse caminho separado, use `scripts/setup_quick.sh --gpu` no Linux/macOS ou `scripts/setup_quick.ps1 -Gpu` no Windows. O índice CUDA pode ser trocado pela variável `PYTORCH_INDEX_URL`.

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
vision_bench/   Motor de benchmarks (CLI, execução, adapters e protocolos)
  reporting/    Relatórios e métricas
    charts/     Gráficos por categoria: qualidade, desempenho, atributos, classes e confusão
benchmarks/     Configurações, tarefas, datasets e resultados
  configs/      Configuração de modelos e endpoints
  contracts/    Tipos de request, prompts e critérios de resposta
  datasets/     Dados e imagens agrupados por domínio/tipo
  tasks/        Execução e relatórios dos benchmarks históricos
  results/      Resultados e evidências por execução
models/        Experimentos e bibliotecas organizados por modelo
  laya/laya_vision/  Submódulo do código Laya Vision
charts/       Figuras e PDFs históricos, organizados por característica
scripts/        Instalação, download de modelos e preparação de datasets
requirements/   Dependências Python por backend
docs/           Guias de comparação e metodologia
tests/          Testes do motor e dos benchmarks
```

## Documentação

- [Comparações reproduzíveis](docs/COMPARISON_SETUP.md)
- [Datasets e métricas](docs/DATASETS.md)
- [Metodologia das comparações](docs/PROMPT_COMPARISONS.md)
- [Modelos e início rápido](docs/QUICK_START_MODELS.md)
- [Atlas estatístico anterior](charts/runs/scientific_20261026_final/README.md)

Os gabaritos existentes são exploratórios; o benchmark separa execuções e computadores. Recursos do cliente HTTP não medem o consumo do servidor Ollama.

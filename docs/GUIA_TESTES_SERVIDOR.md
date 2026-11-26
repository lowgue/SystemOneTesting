# Guia de testes e setup do servidor GPU

Este guia reúne o setup recomendado para Linux com GPU NVIDIA, os comandos para cada tipo de teste e a localização dos resultados. O caminho principal usa o vLLM para os quatro modelos do repositório — Clef-Flash, BLIP, Laya e D1 — em batches dimensionados automaticamente pela GPU. O modo serial em processo continua disponível como baseline opcional.

## 1. O que cada teste cobre

Há três grupos diferentes de testes neste repositório:

1. **Testes automatizados do código**: não baixam modelos, não usam a GPU e não chamam APIs pagas. Validam parsing, métricas, runtime, adapters e contratos.
2. **Benchmarks de modelos**: fazem inferência em imagens e medem qualidade e desempenho. Precisam dos pesos e/ou de um servidor de modelo.
3. **Relatórios**: leem execuções concluídas; não fazem inferência.

### Testes automatizados

O runner `scripts/test_all.py` executa todos os arquivos `tests/test_*.py` com `unittest`:

| Arquivo | O que verifica |
|---|---|
| `test_automated_description_core.py` | Normalização, validação e renderização das descrições automatizadas. |
| `test_dataset_metrics.py` | Métricas de qualidade, latência, erros e comparações de dataset. |
| `test_gpu_concurrency.py` | Dimensionamento automático do batch pela memória livre da GPU. |
| `test_legacy_model_runtime.py` | Integrações do runner legado e adapters de modelos. |
| `test_paired_power.py` | Limiares exatos do teste pareado por enumeração binomial. |
| `test_portable_benchmark.py` | CLI do benchmark portátil e fluxo HTTP simulado, sem baixar modelos. |
| `test_runtime_selection.py` | Seleção de CPU/CUDA e dimensionamento de threads. |
| `test_vllm_adapter.py` | Formatação e validação de requests vLLM, inclusive concorrência e métricas de batch. |

Execute só a suíte automatizada, sem iniciar inferências:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

### Benchmarks padronizados por modelo

`models/` contém os mesmos 11 experimentos para `blip`, `clef`, `laya` e `d1_omni`: oito de dataset e três de prompt. A tabela descreve o que os arquivos numerados avaliam.

| Arquivo | Pergunta do experimento |
|---|---|
| `dataset_tests/test_01_cor_veiculo.py` | Classificação da cor predominante do veículo. |
| `dataset_tests/test_02_tipo_veiculo.py` | Classificação da carroceria, como sedan, SUV, hatch e pickup. |
| `dataset_tests/test_03_marca_veiculo.py` | Reconhecimento da marca/fabricante. |
| `dataset_tests/test_04_presenca_veiculo.py` | Presença ou ausência de veículo, incluindo controles negativos. |
| `dataset_tests/test_05_bundle_completo.py` | Resposta simultânea para presença, marca, cor e carroceria. |
| `dataset_tests/test_06_atributos_pessoa.py` | Atributos visíveis de pessoas, como capacete, vestimenta e itens carregados. |
| `dataset_tests/test_07_robustez_imagem.py` | Efeito de baixa luz, ruído e compressão. |
| `dataset_tests/test_08_multiplos_objetos.py` | Associação das respostas aos alvos em cenas com vários objetos. |
| `prompt_tests/test_01_formatos_prompt.py` | Pergunta aberta, opções, descrição longa e JSON estruturado. |
| `prompt_tests/test_02_variacao_instrucoes.py` | Instruções concisas, detalhadas ou restritivas. |
| `prompt_tests/test_03_orcamento_opcoes.py` | Efeito da quantidade de alternativas/taxonomia. |

Para listar e executar um desses experimentos pelo CLI legado:

```bash
.venv/bin/python models/executar.py --listar
.venv/bin/python models/executar.py --modelo clef --teste 01 --limit 2 --mock
```

`--mock` valida o caminho do experimento com respostas simuladas; o resultado não mede o modelo. Para inferência real, use o runner portátil abaixo, que tem setup e resultados mais reprodutíveis. Os gabaritos dos experimentos históricos são exploratórios.

## 2. Preparar o servidor

### 2.1. Conferir disco, GPU e ferramentas

Coloque o clone, os ambientes Python e os caches na partição grande. Neste servidor, use `/models`, não o filesystem raiz:

```bash
df -h / /models
nvidia-smi
git --version
```

`nvidia-smi` precisa listar a RTX 3090 sem erro. O setup deste guia usa Python 3.12, `uv` e um driver NVIDIA funcional. O vLLM inclui seu runtime CUDA via pacotes Python; o teste normal não exige instalar o CUDA Toolkit do sistema.

### 2.2. Obter o repositório

Se o servidor alcança o Git remoto:

```bash
mkdir -p /models/SystemOneTests
cd /models/SystemOneTests
git clone https://github.com/lowgue/SystemOneTesting.git
cd SystemOneTesting
```

Se o servidor só pode ser acessado por SSH, rode no computador que tem o clone atualizado. O destino fica explicitamente em `/models`; a cópia não apaga arquivos remotos. O ambiente `.venv` e os resultados locais não são transferidos:

```bash
cd /caminho/local/SystemOneTesting
ssh usuario@servidor 'mkdir -p /models/SystemOneTests/SystemOneTesting'
rsync -av \
  --exclude='/.local/' \
  --exclude='/.venv/' \
  --exclude='/.artifacts/' \
  --exclude='__pycache__/' \
  ./ usuario@servidor:/models/SystemOneTests/SystemOneTesting/
```

O `/` no início de `/.local/` evita copiar uma pasta pessoal acidental para dentro do clone. Substitua `usuario@servidor` e `/caminho/local/SystemOneTesting`. Não copie `.venv` de outro computador: ambientes virtuais guardam caminhos e dependências específicos da máquina. O fluxo Clef/vLLM padrão não precisa do submódulo Laya. Para os benchmarks locais BLIP/Laya/D1, inicialize-o no servidor quando houver acesso ao remoto Git:

```bash
git submodule update --init --recursive
```

Confira que o clone está na partição com espaço livre:

```bash
cd /models/SystemOneTests/SystemOneTesting
df -h / /models
```

### 2.3. Instalar `uv` e Python 3.12

Se `uv` já estiver disponível, pule a instalação. Com acesso à internet:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
export UV_PYTHON_INSTALL_DIR=/models/cache/uv-python
uv python install 3.12
PYTHON312="$(uv python find 3.12)"
"$PYTHON312" --version
```

`UV_PYTHON_INSTALL_DIR` mantém o Python gerenciado pelo `uv` fora da partição raiz. O último comando deve mostrar Python 3.12. Guarde essa sessão do terminal ou defina `PYTHON312` e `UV_PYTHON_INSTALL_DIR` novamente após reconectar por SSH.

### 2.4. Direcionar caches para `/models`

Isso evita encher `/` com downloads do `uv`, do Python gerenciado, do pip ou do Hugging Face:

```bash
mkdir -p /models/cache/uv /models/cache/uv-python /models/cache/pip /models/cache/huggingface /models/venvs
export UV_CACHE_DIR=/models/cache/uv
export UV_PYTHON_INSTALL_DIR=/models/cache/uv-python
export PIP_CACHE_DIR=/models/cache/pip
export HF_HOME=/models/cache/huggingface
```

Repita os `export` após cada nova sessão SSH. Para torná-los permanentes, adicione essas quatro linhas ao `~/.bashrc` do usuário que executará os testes.

### 2.5. Criar ambiente dedicado do vLLM

O vLLM fica separado do ambiente do projeto. Isso evita conflito entre as dependências do servidor e as dos relatórios:

```bash
export VLLM_ENV=/models/venvs/systemone-vllm
uv venv --python "$PYTHON312" --seed "$VLLM_ENV"
uv pip install --python "$VLLM_ENV/bin/python" --upgrade vllm --torch-backend=auto
```

Valide que o PyTorch desse ambiente enxerga a GPU:

```bash
"$VLLM_ENV/bin/python" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0), torch.version.cuda)'
"$VLLM_ENV/bin/vllm" --version
```

Se a instalação falhar com `No space left on device`, confira `df -h / /models`, confirme os três caches em `/models` e libere espaço antes de repetir a instalação. Não apague diretórios pessoais para liberar espaço.

## 3. Rodar os testes no servidor

### 3.1. Primeira execução: suíte + smoke dos quatro modelos

Na raiz do projeto e com as variáveis de cache da seção anterior exportadas, rode primeiro uma repetição. O script instala as dependências leves de teste em `.venv`, valida a CUDA do vLLM, consulta a memória livre com `nvidia-smi` e inicia um servidor vLLM por modelo em uma porta livre. O manifesto `benchmarks/configs/vllm-models.json` define Clef-Flash, BLIP, Laya e D1; cada modelo é servido, medido e encerrado antes do próximo.

```bash
cd /models/SystemOneTests/SystemOneTesting
"$PYTHON312" scripts/test_all.py \
  --gpu \
  --repetitions 1 \
  --vllm-executable "$VLLM_ENV/bin/vllm"
```

Na primeira execução, os pacotes Python e os pesos precisam estar disponíveis por rede. A porta padrão 8000 não precisa estar livre: o script escolhe outra porta local e encerra somente os processos vLLM que ele próprio iniciou. Cada `vllm.log` e cada `gpu.csv` ficam ao lado da execução correspondente.

### 3.2. Benchmark completo com o maior batch que a GPU permite

Depois do smoke, rode as três repetições padrão. O padrão é `--concurrency auto`: a concorrência é calculada como `(VRAM livre - reserva) / 512 MiB`, com teto de 64 por padrão. Em uma RTX 3090 com 22 imagens no dataset, o runner forma um lote com todas as imagens; o número exato aparece no início da saída e fica registrado em `frozen_config.json`.

```bash
"$PYTHON312" scripts/test_all.py \
  --gpu \
  --repetitions 3 \
  --vllm-executable "$VLLM_ENV/bin/vllm"
```

Use `--concurrency 8` para fixar um valor ou `--max-concurrency 128` para erguer o teto automático. O mesmo número controla as threads HTTP do cliente e o `--max-num-seqs` do servidor vLLM. Use `--models clef-flash-vllm d1-omni-vllm` para limitar a execução a alguns IDs do manifesto. Repetições não aumentam o número de imagens independentes.

Para executar apenas a suíte automatizada sem benchmark de modelo:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Para pular instalação depois que `.venv` já estiver funcional:

```bash
"$PYTHON312" scripts/test_all.py --gpu --no-install --repetitions 3 \
  --vllm-executable "$VLLM_ENV/bin/vllm"
```

Não use `--no-install` no primeiro uso, nem com `.venv` ausente, copiado ou quebrado.

### 3.3. Baseline serial em processo (opcional)

O modo `--include-local-benchmarks` acrescenta uma execução com BLIP, Laya e D1 carregados diretamente no `.venv`, serialmente, usando o mesmo contrato dos modelos servidos. Ele instala PyTorch CUDA, inicializa o submódulo Laya e serve como controle metodológico; não substitui a rodada padrão com vLLM.

```bash
git submodule update --init --recursive
"$PYTHON312" scripts/test_all.py \
  --gpu \
  --include-local-benchmarks \
  --repetitions 1 \
  --vllm-executable "$VLLM_ENV/bin/vllm"
```

BLIP, Laya e D1 são carregados pelo vLLM com `--trust-remote-code`. Nem toda versão do vLLM expõe chat multimodal para arquiteturas customizadas; se algum modelo falhar, o erro aparece na saída e o log completo fica em `.artifacts/runs/gpu-*/<id>-vllm.log`. O script continua com os demais modelos e gera os gráficos das execuções concluídas. Nesse caso, use o baseline serial desta seção para não perder a comparação do modelo.

### 3.4. Rodar outro modelo multimodal servido pelo vLLM

Esse é um fluxo separado do teste Clef padrão. Inicie um modelo que aceite imagens no API OpenAI do vLLM em um terminal:

```bash
"$VLLM_ENV/bin/vllm" serve nome-do-modelo \
  --host 127.0.0.1 --port 8000 --max-model-len 4096
```

No segundo terminal, exporte a URL e o nome exposto pelo servidor e rode um smoke:

```bash
cd /models/SystemOneTests/SystemOneTesting
export VLLM_CHAT_COMPLETIONS_URL=http://127.0.0.1:8000/v1/chat/completions
export VLLM_MODEL=nome-do-modelo
.venv/bin/python -m vision_bench doctor \
  --config benchmarks/configs/vllm.json \
  --contract benchmarks/contracts/description_ollama_v1.json
.venv/bin/python -m vision_bench run \
  --config benchmarks/configs/vllm.json \
  --contract benchmarks/contracts/description_ollama_v1.json \
  --models vllm-vision --limit 2 --repetitions 1 --concurrency auto \
  --output .artifacts/runs/vllm-smoke
```

O processo vLLM precisa continuar ativo no primeiro terminal. `--concurrency auto` usa a mesma regra de dimensionamento por VRAM do `test_all.py`; um inteiro positivo mantém o controle explícito. A concorrência envia requests ao mesmo tempo; o endpoint pode formar batches internamente. Para o perfil específico Clef System One e os demais modelos do manifesto, use `scripts/test_all.py --gpu`.

## 4. Outros comandos do runner portátil

### Verificar ambiente sem inferência

```bash
.venv/bin/python -m vision_bench doctor
```

### Smoke dos modelos em processo

Esses comandos exigem BLIP/PyTorch e o submódulo Laya. O modo `--include-local-benchmarks` da seção 3.3 instala essas dependências; se ainda não rodou esse modo, prepare-as uma vez:

```bash
git submodule update --init --recursive
.venv/bin/python -m pip install torch torchvision \
  --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install \
  -r requirements/models.txt \
  -e . -e ./models/laya/laya_vision
```

Com esses requisitos instalados, use o runner:

```bash
.venv/bin/python -m vision_bench run \
  --config benchmarks/configs/local.json \
  --models blip-vqa laya-vision-head512 d1-omni-600M \
  --limit 2 --repetitions 1 --device auto --threads auto \
  --output .artifacts/runs/local-smoke
```

### Rodada maior no dataset portátil

O dataset padrão versionado tem 22 eventos exploratórios. Sem `--limit`, execute todos:

```bash
.venv/bin/python -m vision_bench run \
  --config benchmarks/configs/local.json \
  --models blip-vqa laya-vision-head512 d1-omni-600M \
  --repetitions 3 --device auto --threads auto \
  --machine-id servidor-3090 \
  --output .artifacts/runs/local-servidor-3090
```

Para incluir Clef/vLLM em uma rodada feita manualmente, use o script `test_all.py` da seção 3.2. `--device auto` seleciona CUDA se o PyTorch instalado no `.venv` oferecer suporte; o servidor de vLLM tem ambiente/device próprios.

### Gerar ou refazer gráficos de uma execução já concluída

```bash
.venv/bin/python scripts/report_runs.py .artifacts/runs/gpu-AAAAMMDD-HHMMSS/d1-omni-vllm
```

Ou comparar várias execuções sem misturar as máquinas. O relatório gera gráficos por execução e, quando há mais de uma execução com o mesmo contrato, dataset e máquina, um conjunto adicional intitulado **Todos os modelos · mesmo hardware**:

```bash
.venv/bin/python -m vision_bench report \
  .artifacts/runs/local-servidor-3090 \
  .artifacts/runs/gpu-AAAAMMDD-HHMMSS/clef-flash-vllm \
  .artifacts/runs/gpu-AAAAMMDD-HHMMSS/d1-omni-vllm \
  --output .artifacts/reports/comparacao
```

## 5. Resultados e coleta

- Execuções brutas e snapshots de configuração: `.artifacts/runs/gpu-*/<id>/`.
- Logs de inicialização e telemetria por modelo: `.artifacts/runs/gpu-*/<id>-vllm.log` e `<id>-gpu.csv`.
- Relatórios, gráficos e tabelas de cada rodada de `test_all.py`: `charts/reports/execucao-*/`.
- `README.md`: resumo de qualidade e desempenho, com tabelas de latência, throughput, CPU, RAM e VRAM (média / mínimo / máximo).
- `metrics.csv` / `metrics.json`: métricas tabulares por modelo e por lote.
- PNG, PDF e SVG: box plots com pontos por imagem, histogramas, séries cartesianas, mínimo/média/máximo, latência × throughput, acurácia por atributo com IC 95% de Wilson e telemetria `nvidia-smi` com utilização, memória e potência.

Copie resultados para seu computador depois de identificar o diretório exato da execução. Rode isto no computador local:

```bash
rsync -av usuario@servidor:/models/SystemOneTests/SystemOneTesting/charts/reports/execucao-AAAAMMDD-HHMMSS/ \
  ./charts/reports/execucao-AAAAMMDD-HHMMSS/
```

Para preservar a comparação e permitir refazer relatórios, copie também o diretório correspondente sob `.artifacts/runs/`.

## 6. Se o servidor não tiver internet

O primeiro setup precisa obter Python/`uv`, pacotes Python e o snapshot `Cloudflare/clef-flash`. A cópia do Git por SSH transfere o código, mas não esses artefatos. Prepare os pacotes e o cache do modelo em uma máquina Linux compatível com a mesma arquitetura, versão Python e tipo de GPU, transfira-os por SSH para `/models` e mantenha `HF_HOME` apontando para o cache transferido. Não copie o `.venv` entre máquinas. Sem wheels compatíveis e o snapshot completo no cache, a inferência offline não inicia. A flag `HF_HUB_OFFLINE=1` só deve ser usada depois de conferir que o modelo já está no cache.

## 7. Problemas comuns

| Sintoma | Ação |
|---|---|
| `No space left on device` | Rode `df -h / /models`; confirme que `UV_CACHE_DIR`, `UV_PYTHON_INSTALL_DIR`, `PIP_CACHE_DIR`, `HF_HOME`, o clone e `VLLM_ENV` estão em `/models`. |
| `No virtual environment ... .venv` ou `No module named encodings` | Não reaproveite/copiei `.venv` de outro Python ou computador. Rode `scripts/test_all.py` sem `--no-install` para o script criar um ambiente válido. |
| PyTorch diz que CUDA não está disponível | Rode `nvidia-smi` e o teste Python do ambiente vLLM da seção 2.5. O driver precisa funcionar antes da instalação do vLLM. |
| Erro de memória CUDA | Pare outros processos GPU e reduza o batch com `--concurrency 8` ou `--max-concurrency 16`. |
| O primeiro start do vLLM não termina | Aguarde o download/carregamento inicial; confira `.artifacts/runs/gpu-*/<id>-vllm.log` e o espaço em `/models`. |
| Um modelo customizado não inicia no vLLM | Confira o log dele; arquiteturas sem chat multimodal compatível exigem o baseline serial (`--include-local-benchmarks`). |
| `/v1/models` responde `Not Found` | A porta consultada pode pertencer a outro serviço. No fluxo recomendado, deixe `test_all.py` escolher a porta e iniciar o vLLM. |
| Benchmark não encontra vLLM | Passe `--vllm-executable /models/venvs/systemone-vllm/bin/vllm` e confira se o arquivo existe. |

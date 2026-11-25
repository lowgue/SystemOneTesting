# Comparação portátil de modelos visuais

Este é o ponto de entrada para novos testes. Os scripts históricos foram organizados em `benchmarks/tasks/`, os dados em `benchmarks/datasets/` e os resultados em `benchmarks/results/`; eles servem como evidência histórica e não são necessários para adicionar modelos ao runner portátil.

> [!TIP]
> **Instalação Rápida em Novo Computador**: Para automatizar a clonagem, instalação de dependências e download de todos os modelos visuais de decisão (**Laya Vision**, **d1-omni-600M** e **Clef**), execute `./scripts/setup_quick.sh` (Linux/macOS) ou `.\scripts\setup_quick.ps1` (Windows). Veja o [QUICK_START_MODELS.md](QUICK_START_MODELS.md).

A interface `python -m vision_bench` aceita qualquer quantidade de modelos registrados em JSON. Há adaptadores para BLIP VQA, Laya Vision, d1-omni-600M (Liquid AI), endpoints System One, vLLM via chat multimodal compatível com OpenAI e extensão Python para outros protocolos. Os testes sempre solicitam o conjunto completo de atributos definido no contrato selecionado.

## 1. Clonar e instalar no outro computador

Baseline testada: Python 3.12. Os pacotes diretos estão fixados em `requirements/`; a versão de todas as distribuições instaladas é registrada em cada execução. Dependências transitivas podem variar; comparar o snapshot de ambiente antes de atribuir diferenças ao modelo.

```bash
git clone --recurse-submodules https://github.com/lowgue/SystemOneTesting.git
cd SystemOneTesting
```

Se já clonou sem submódulos:

```bash
git submodule update --init --recursive
```

**Somente Clef/API HTTP**, sem instalar torch ou transformers:

```bash
python3.12 scripts/setup_benchmark.py --backend http
.venv/bin/python -m vision_bench doctor --models clef-flash-local-q4
```

**BLIP + Laya em CPU**, além do cliente HTTP:

```bash
python3.12 scripts/setup_benchmark.py --backend cpu
.venv/bin/python -m vision_bench doctor
```

O setup cria/reutiliza `.venv` e instala o projeto. Não baixa pesos nem inicia inferência. No Windows, use `py -3.12` para o setup e `.venv\Scripts\python.exe` nos comandos seguintes. O launcher Python do Clef também funciona no Windows; o wrapper `run_clef_local.sh` exige Bash/WSL.

Para GPU, o fluxo padrão `python3.12 scripts/test_all.py --gpu` roda a suíte e mede o conjunto de imagens com Clef-Flash pelo vLLM, em batches concorrentes de 4. Ele não roda os benchmarks BLIP/Laya/D1 serialmente antes do Clef. Para incluí-los, acrescente `--include-local-benchmarks`; esses adaptadores locais não oferecem uma interface comum segura para batch de imagens. Instale uma versão atual do vLLM em um ambiente separado em `~/venvs/systemone-vllm`; o vLLM recente oferece o endpoint System One usado pelo Clef:

```bash
uv venv --python 3.12 --seed --managed-python ~/venvs/systemone-vllm
uv pip install --python ~/venvs/systemone-vllm/bin/python --upgrade vllm --torch-backend=auto
```

O driver NVIDIA precisa estar instalado. A build pip do PyTorch inclui o runtime CUDA necessário para os modelos locais opcionais; para selecionar outra build compatível, use `--torch-index-url` ou consulte o [seletor oficial do PyTorch](https://pytorch.org/get-started/locally/).

```bash
.venv/bin/python -m pip install -r requirements/models.txt -e . -e ./models/laya/laya_vision
```

O runner usa CUDA automaticamente quando PyTorch detecta uma GPU; `--device cpu` força CPU, e `--device cuda` exige CUDA disponível. `--threads auto` seleciona um Torch thread por núcleo físico, sem contar SMT como núcleos adicionais. O runner sincroniza o dispositivo para medir tempo completo e registra o pico de memória CUDA alocada pelo processo. Para métricas comparáveis, o número de threads, device e flags TF32 ficam congelados nos metadados. Os parâmetros de GPU do servidor Clef são configurados separadamente; o device do runner não altera um servidor externo.

## 2. Dados disponíveis no clone e conjunto completo

O padrão `benchmarks/datasets/vehicles/portable.json` contém **22 eventos de veículos**, com as imagens já rastreadas no Git. Serve para teste inicial sem baixar dados. São rótulos exploratórios, não uma amostra confirmatória. O contrato também tem perguntas de pessoas, mas essa amostra padrão não testa pessoas.

### Conjunto maior e mais diverso

Para uma rodada com mais imagens distintas, gere o conjunto Stanford Cars balanceado. O downloader usa o split de teste, seleciona até 10 fotos por marca e até 3 por modelo, baixa somente as imagens selecionadas e grava um manifesto pronto para o runner. O comando abaixo gera 300 imagens e mede marca e carroceria; não há gabarito de cor.

```bash
.venv/bin/python scripts/datasets/download_stanford_cars_sample.py \
  --runner-ready --count 300 --per-make 10 --per-class 3
.venv/bin/python -m vision_bench doctor \
  --manifest .artifacts/datasets/stanford_cars_diverse/manifest.json \
  --data-root .artifacts/datasets/stanford_cars_diverse
.venv/bin/python -m vision_bench run \
  --manifest .artifacts/datasets/stanford_cars_diverse/manifest.json \
  --data-root .artifacts/datasets/stanford_cars_diverse \
  --models blip-vqa laya-vision-head512 d1-omni-600M \
  --device auto --threads auto --repetitions 1 \
  --machine-id computador-b \
  --output .artifacts/runs/stanford-cars-diverse
```

Use `--count`, `--per-make`, `--per-class`, `--scan-limit`, `--seed` ou `--output-dir` para ajustar a seleção. Sem `--runner-ready`, o downloader mantém o formato histórico utilizado pelos scripts antigos. As anotações vêm do dataset de origem e são exploratórias, sem revisão independente neste projeto. Use uma repetição para avaliar qualidade em imagens distintas; aumente `--repetitions` quando a pergunta for latência ou estabilidade. Repetições adicionais processam novamente as mesmas imagens e não aumentam a amostra independente.

As imagens de pessoas e outros datasets antigos são ignoradas pelo Git. Para usar exatamente o Teste 7 nas duas máquinas, exporte na máquina que possui as imagens:

```bash
.venv/bin/python -m vision_bench pack-data \
  --manifest benchmarks/datasets/historical/mixed/manifests/test_7_complete_tags_manifest.json \
  --output .artifacts/datasets/test7.zip
```

Copie o ZIP para o outro computador e importe:

```bash
.venv/bin/python -m vision_bench unpack-data test7.zip \
  --destination .artifacts/datasets/test7
```

O bundle inclui manifesto, rótulos e imagens; não contém pesos ou credenciais. Imagens repetidas são reutilizadas. O import verifica hashes e caminhos antes de escrever arquivos e recusa sobrescrever um diretório existente. A exportação local deste conjunto foi conferida com **41 eventos / 39 imagens únicas**. Preserve atribuição/licenças de origem ao transportar os dados; o bundle não altera permissões sobre as imagens.

Ao usar esse conjunto, acrescente em todos os comandos `doctor` e `run`:

```text
--manifest .artifacts/datasets/test7/manifest.json --data-root .artifacts/datasets/test7
```

Para dados próprios, copie um manifesto e forneça para cada evento: `id` único, `domain` (`vehicle` ou `person`), `rel_path` relativo ao data-root e `sha256` do arquivo. `expected` é opcional; quando presente deve conter todos os campos do domínio. `group_id` pode identificar cenas relacionadas. Registre `label_status` na raiz do manifesto, por exemplo `unlabeled` ou `exploratory_agent_reviewed`; não rotule dados automaticamente como adjudicados por humanos.

## 3. Clef-Flash local

Use um `llama-server` que suporte o modelo Clef e a rota `/v1/systemone`. Não basta qualquer build de llama.cpp: o [model card GGUF](https://huggingface.co/ggml-org/Clef-Flash-GGUF) aponta a implementação de suporte ao Clef. Instale esse runtime para o sistema/GPU do computador e informe o executável.

O arquivo `benchmarks/configs/clef_weights.json` fixa o commit do repositório e os SHA-256 dos dois arquivos. Os pesos selecionados ocupam cerca de **7,35 GB em disco** no total (modelo + projetor); RAM/VRAM necessária inclui também runtime e cache. Não iniciar simultaneamente Clef, BLIP e Laya numa máquina sem memória suficiente para todos.

Verifique o comando, sem baixar pesos:

```bash
.venv/bin/python -m vision_bench clef-server \
  --llama-server /caminho/para/llama-server --dry-run
```

No primeiro terminal, inicie o servidor. Esta etapa baixa os pesos com `hf download --revision`, verifica hashes por streaming e inicia o processo:

```bash
.venv/bin/python -m vision_bench clef-server \
  --llama-server /caminho/para/llama-server \
  --threads 8 --context 4096 --gpu-layers 0
```

Para offload em GPU use, por exemplo, `--gpu-layers -1` se o runtime e a memória comportarem. O comando e a versão observada do runtime ficam no arquivo de metadados. O servidor permanece nesse terminal e para com Ctrl+C. O wrapper antigo continua disponível: `LLAMA_SERVER=/caminho/para/llama-server bash run_clef_local.sh`.

No segundo terminal, configure o cliente e faça uma execução curta:

```bash
export CLEF_LOCAL_SYSTEMONE_URL=http://127.0.0.1:11437/v1/systemone
export CLEF_SERVER_METADATA="$PWD/.artifacts/clef-server.json"
.venv/bin/python -m vision_bench run \
  --models clef-flash-local-q4 --limit 2 --repetitions 1 \
  --output .artifacts/runs/clef-smoke
```

PowerShell: use `$env:CLEF_LOCAL_SYSTEMONE_URL="http://127.0.0.1:11437/v1/systemone"` e `$env:CLEF_SERVER_METADATA="$PWD/.artifacts/clef-server.json"`.

Com os metadados, o runner mede CPU/RSS do cliente mais o processo local do servidor e seus filhos já existentes no início da requisição. Ele verifica PID, instante de criação, endpoint e revisão para rejeitar metadados obsoletos. RSS somado não representa memória física exclusiva de processos com páginas compartilhadas. GPU de um servidor externo não é medida pelo contador CUDA do cliente.

Sem `CLEF_SERVER_METADATA`, o teste HTTP funciona, mas registra `client_only`: CPU e memória do servidor não foram medidas. A revisão/quantização declarada no JSON não é confirmação dos pesos carregados num servidor arbitrário.

Não altere servidor, pesos, quantização ou flags durante uma execução. O setup não faz download automático de llama.cpp nem escolhe driver/GPU pelo usuário.

## 3b. Modelo multimodal servido pelo vLLM

O perfil `benchmarks/configs/vllm.json` usa a API de chat compatível com OpenAI do vLLM. O servidor precisa carregar um modelo multimodal e estar ativo antes da rodada. O vLLM documenta a API de chat e imagens no [servidor compatível com OpenAI](https://docs.vllm.ai/en/stable/serving/online_serving/openai_compatible_server/).

Configure a URL completa de chat e o nome do modelo servido:

```bash
export VLLM_CHAT_COMPLETIONS_URL=http://127.0.0.1:8000/v1/chat/completions
export VLLM_MODEL=nome-do-modelo-servido
.venv/bin/python -m vision_bench doctor \
  --config benchmarks/configs/vllm.json \
  --contract benchmarks/contracts/description_ollama_v1.json
.venv/bin/python -m vision_bench run \
  --config benchmarks/configs/vllm.json \
  --contract benchmarks/contracts/description_ollama_v1.json \
  --models vllm-vision --repetitions 1 --concurrency 4 \
  --output .artifacts/runs/vllm-smoke
```

O valor de `--concurrency` limita quantas imagens são enviadas em paralelo; cada imagem continua sendo uma requisição independente e o servidor pode agrupá-las durante a inferência. O padrão é `1`. A concorrência maior que `1` só é aceita para adaptadores HTTP (`systemone_http` e `vllm_openai`). As linhas registram `batch_size`, `batch_elapsed_ms` e `batch_id`; o relatório calcula throughput pelo tempo total de cada lote. CPU e RSS são compartilhados pelo lote e identificados como `client_batch_shared`. O uso de GPU do servidor continua fora dos contadores CUDA do cliente.

Para o endpoint System One nativo do vLLM, o perfil `benchmarks/configs/vllm-systemone.json` usa `Cloudflare/clef-flash`:

```bash
vllm serve Cloudflare/clef-flash --max-model-len 8192
```

Em outro terminal:

```bash
export VLLM_SYSTEMONE_URL=http://127.0.0.1:8000/v1/systemone
.venv/bin/python -m vision_bench run \
  --config benchmarks/configs/vllm-systemone.json \
  --contract benchmarks/contracts/description_ollama_v1.json \
  --models clef-flash-vllm --limit 2 --repetitions 1 --concurrency 2 \
  --output .artifacts/runs/vllm-systemone-smoke
```

O Clef-Flash usa um schema fixo e o endpoint vLLM rejeita `instructions` por pergunta; o perfil remove esse campo e mantém as opções do contrato. O vLLM documenta esse modelo no endpoint `/v1/systemone` e entrada de imagens. Reduza `--concurrency` ou `--max-model-len` se a GPU ficar sem memória.

Para autenticação, defina `token_env` no arquivo de configuração e exporte a variável correspondente. Não inclua credenciais na URL ou no JSON. O runner registra a URL e o nome de modelo resolvidos no snapshot congelado da execução.

## 3c. Perfil legado do Ollama

O perfil `benchmarks/configs/ollama.json` continua disponível para reproduzir execuções antigas. O novo fluxo recomendado para serviço externo usa vLLM acima. Para reproduzir o perfil histórico, instale o [Ollama](https://ollama.com/download). O [modelo oficial Clef-Flash](https://ollama.com/library/clef-flash) exige Ollama 0.35.1 ou superior. No outro computador:

```bash
ollama --version
ollama pull clef-flash:9b
ollama list
```

Se o serviço não estiver ativo, execute `ollama serve` em um terminal e mantenha-o aberto. A API local padrão é `http://127.0.0.1:11434/v1/systemone`.

Na pasta do projeto atualizado, instale apenas o cliente HTTP e faça o smoke:

```bash
git pull
python3.12 scripts/setup_benchmark.py --backend http
.venv/bin/python -m vision_bench run \
  --config benchmarks/configs/ollama.json \
  --contract benchmarks/contracts/description_ollama_v1.json \
  --models clef-flash-ollama --limit 2 --repetitions 1 \
  --output .artifacts/runs/clef-ollama-smoke
```

No Windows, use `py -3.12` no setup e `.venv\Scripts\python.exe` nos comandos do runner. Para a rodada completa, retire `--limit 2`, use `--repetitions 5` e outro diretório de saída. Gere o relatório com `python -m vision_bench report <diretorio-do-run> --output <diretorio-novo>`.

Diferenças de protocolo documentadas pelo Ollama: `model` é obrigatório; as imagens são Base64 puro, sem prefixo de data URL; perguntas `choice` aceitam até 26 opções. A configuração `ollama.json` já aplica essas regras. O contrato original de 49 opções de marca é recusado no preflight, antes da inferência.

O contrato alternativo tem 24 marcas nomeadas + `other` + `unknown`, mantendo os quatro campos por domínio. É uma taxonomia exploratória fixa; não exclui opções com base na imagem. Para comparar com BLIP e Laya, use esse mesmo arquivo de contrato para todos os modelos de `ollama.json`. Os resultados não são diretamente equivalentes aos históricos com 49 opções. Se um manifesto próprio contiver outra marca no gabarito, produza uma cópia explicitamente remapeada para `other`; não altere o gabarito histórico original nem trate resposta não interpretável como `other`.

Ollama gerencia os pesos, a quantização e o processo de inferência; o launcher `clef-server` não é usado nesse caminho. Registre `ollama --version` e o digest do modelo (`ollama list` ou `/api/tags`) com a rodada, pois a tag `:9b` pode mudar. O escopo de recursos neste perfil é `client_only`: CPU/RAM do daemon e GPU não são medidos pelo processo Python. Para liberar o modelo entre sessões, use `ollama stop clef-flash:9b`.

Referências: [System One no Ollama](https://docs.ollama.com/api/systemone) e [catálogo Clef-Flash](https://ollama.com/library/clef-flash). O protocolo foi verificado por teste HTTP local simulado; este preparo não baixou nem executou o Clef real no Ollama.

## 4. Rodada comparativa

Validar arquivos e dependências, sem carregar pesos nem enviar inferência:

```bash
.venv/bin/python -m vision_bench doctor
.venv/bin/python -m vision_bench run --dry-run \
  --output .artifacts/runs/validation
```

`doctor` não comprova que uma API está online ou que implementa o protocolo. O smoke com `--limit 2 --repetitions 1` verifica isso de fato. O modo `--dry-run` não cria o diretório de resultados.

Na mesma máquina, com os backends disponíveis:

```bash
.venv/bin/python -m vision_bench run \
  --config benchmarks/configs/local.json \
  --models blip-vqa laya-vision-head512 clef-flash-local-q4 \
  --repetitions 5 --threads 8 --machine-id computador-b \
  --output .artifacts/runs/computador-b-all
```

Cada bloco altera a ordem dos modelos; as imagens recebem ordem pseudoaleatória com seed registrada. Um adaptador é carregado por vez. O servidor Clef externo não é encerrado automaticamente: se a RAM for limitada, execute cada modelo em uma sessão separada e encerre o servidor antes de BLIP/Laya.

```bash
# Após encerrar o servidor externo, se necessário para liberar memória:
.venv/bin/python -m vision_bench run --models blip-vqa \
  --repetitions 5 --threads 8 --machine-id computador-b \
  --output .artifacts/runs/computador-b-blip

.venv/bin/python -m vision_bench run --models laya-vision-head512 \
  --repetitions 5 --threads 8 --machine-id computador-b \
  --output .artifacts/runs/computador-b-laya
```

A primeira execução dos adaptadores Hub pode baixar pesos e componentes do backbone. Depois de preencher o cache, `--offline` proíbe downloads do Hub. Essa flag não desliga o HTTP necessário a backends remotos. Se o cache estiver incompleto, o carregamento falha; faça uma execução online antes. Carga e warm-up ficam fora da latência por evento e são registrados à parte. Inicialização ou warm-up inválidos abortam a rodada antes de tratá-los como dados de avaliação.

A entrada comum é RGB, com redução proporcional ao máximo de 1.024 pixels por lado (`--max-edge`), passada a todos os adaptadores. No HTTP o transporte usa PNG sem nova compressão com perda. O processamento interno de cada modelo pode ser diferente e faz parte do pipeline nativo. O tempo por evento inclui leitura, preparação, inferência, parser, filtro de ausência e composição determinística da descrição; no HTTP inclui rede e servidor.

**Laya:** o ID `laya-vision-head512` identifica explicitamente o orçamento experimental de 512 tokens para preservar a lista completa de marcas. O adaptador usa `strict=True`: truncamento provoca falha em vez de uma comparação silenciosamente degradada. Isso resolve a integridade da entrada, mas não demonstra qualidade fora do orçamento treinado. O contrato exploratório v1 contém 48 marcas mais desconhecida para todos os modelos; nenhuma opção é escolhida com base no gabarito.

**Timeouts:** o HTTP tem timeout por requisição. A chamada Python local não oferece cancelamento preemptivo por inferência; Ctrl+C interrompe o processo e preserva linhas concluídas. Erros de inferência mensurada ficam registrados e não desaparecem do denominador.

Retomar uma execução interrompida:

```bash
# Repita EXATAMENTE os parâmetros anteriores e acrescente --resume.
.venv/bin/python -m vision_bench run --models blip-vqa \
  --repetitions 5 --threads 8 --machine-id computador-b \
  --output .artifacts/runs/computador-b-blip --resume
```

A retomada recusa mudanças de código, contrato, imagens, seleção de eventos, dependências, máquina, endpoint ou parâmetros. Não resume no computador diferente como se fosse a mesma medição. Uma última linha JSON interrompida por falha de escrita deve ser investigada; o runner não a descarta silenciosamente. Diretórios de execução existentes exigem `--resume`; resultados históricos não são sobrescritos.

## 5. Relatório com qualquer número de modelos

```bash
.venv/bin/python -m vision_bench report \
  .artifacts/runs/computador-b-all \
  --output .artifacts/reports/computador-b
```

Ou informe várias execuções, inclusive de computadores diferentes:

```bash
.venv/bin/python -m vision_bench report \
  .artifacts/runs/computador-b-blip \
  .artifacts/runs/computador-b-laya \
  .artifacts/runs/clef-smoke \
  --output .artifacts/reports/multiplas-execucoes
```

Cada execução/máquina recebe seção e gráfico próprios; não se misturam tempos de hardwares diferentes. Use mesmos hashes de contrato/dataset, versões e quantidade de repetições para uma comparação direta. Um smoke de duas imagens não substitui a rodada completa. Preserve o diretório inteiro de cada run ao copiar resultados para consolidar.

Saídas: tabelas Markdown, `metrics.json`, `metrics.csv` e gráficos científicos de acurácia, latência, RAM e GPU em PNG 300 dpi, PDF e SVG. Pontos de distribuição são medianas por evento entre repetições, não imagens novas. O relatório portátil é descritivo, sem intervalos populacionais fabricados para uma amostra arbitrária. O README, os dados congelados e a proveniência das execuções permanecem disponíveis para conferência.

Não compare CPU de HTTP `client_only` à CPU do modelo em processo como custo total. Não compare custo local ao tempo remoto sem informar rede/hardware/faturamento. As respostas brutas preservam `usage` quando fornecido, mas não há preço monetário presumido. A métrica de positivos completos com os rótulos históricos é exploratória e deve permanecer identificada assim.

## Executar na máquina remota e trazer resultados

Os exemplos abaixo usam a máquina remota `lospampa@HOST_REMOTO` e o clone remoto `~/ProjetosJev/Luan/SystemOneTesting`. Substitua `HOST_REMOTO` pelo alias SSH que funciona ou pelo IP da máquina. Rode os comandos de cópia no terminal local, depois de entrar na raiz do clone local. Assim, `charts/reports/` vai para o projeto, não para `~/charts`.

### Rodar testes e o benchmark padrão na GPU

Na máquina remota, com o repositório atualizado e o driver NVIDIA funcionando:

```bash
cd ~/ProjetosJev/Luan/SystemOneTesting
git pull
python3.12 scripts/test_all.py --gpu --repetitions 1 --concurrency 4
```

Esse fluxo verifica CUDA, executa os testes automatizados e mede o Clef-Flash no vLLM com quatro imagens simultâneas. Requer o executável `~/venvs/systemone-vllm/bin/vllm` e internet no primeiro download do modelo. O script escolhe uma porta local livre, sem exigir que a porta 8000 esteja livre. Os resultados ficam em `.artifacts/runs/` e `charts/reports/`; `clef-gpu.csv` registra amostras de utilização, memória e potência obtidas pelo `nvidia-smi`. Se o ambiente `.venv` e as dependências já estiverem instalados, acrescente `--no-install` para pular a instalação do projeto. Aumente `--concurrency` para medir mais solicitações simultâneas; reduza para 2 se houver erro de memória CUDA. Acrescente `--include-local-benchmarks` para também rodar BLIP/Laya/D1 serialmente.

Para rodar o dataset maior e diverso documentado acima, use o downloader e o runner em vez do benchmark padrão de 22 imagens:

```bash
cd ~/ProjetosJev/Luan/SystemOneTesting
.venv/bin/python scripts/datasets/download_stanford_cars_sample.py \
  --runner-ready --count 300 --per-make 10 --per-class 3
.venv/bin/python -m vision_bench doctor \
  --manifest .artifacts/datasets/stanford_cars_diverse/manifest.json \
  --data-root .artifacts/datasets/stanford_cars_diverse
.venv/bin/python -m vision_bench run \
  --manifest .artifacts/datasets/stanford_cars_diverse/manifest.json \
  --data-root .artifacts/datasets/stanford_cars_diverse \
  --models blip-vqa laya-vision-head512 d1-omni-600M \
  --device auto --threads auto --repetitions 1 \
  --machine-id maquina-remota \
  --output .artifacts/runs/stanford-cars-diverse
.venv/bin/python scripts/report_runs.py \
  .artifacts/runs/stanford-cars-diverse
```

O último comando imprime a pasta gerada, por exemplo `charts/reports/execucao-20261109-122411`. Anote esse nome para conferir a cópia. Para outro run, troque `stanford-cars-diverse` pelo nome usado em `--output`.

### Copiar os dados e gráficos para o clone local

No computador local, primeiro entre na raiz do repositório. Consulte o IP na máquina remota com `hostname -I` se o nome do host não resolver no seu computador. Se a conexão SSH usa uma porta diferente da padrão, acrescente `-e "ssh -p PORTA"` aos comandos `rsync`.

```bash
cd /caminho/para/o/clone/local/SystemOneTesting

mkdir -p .artifacts/runs/stanford-cars-diverse
rsync -av --progress \
  lospampa@HOST_REMOTO:~/ProjetosJev/Luan/SystemOneTesting/.artifacts/runs/stanford-cars-diverse/ \
  .artifacts/runs/stanford-cars-diverse/

mkdir -p .artifacts/datasets/stanford_cars_diverse
rsync -av --progress \
  lospampa@HOST_REMOTO:~/ProjetosJev/Luan/SystemOneTesting/.artifacts/datasets/stanford_cars_diverse/ \
  .artifacts/datasets/stanford_cars_diverse/

mkdir -p charts/reports
rsync -av --progress \
  lospampa@HOST_REMOTO:~/ProjetosJev/Luan/SystemOneTesting/charts/reports/ \
  charts/reports/
```

Para copiar só um relatório, troque o último caminho remoto por `charts/reports/execucao-AAAAMMDD-HHMMSS/` e o destino por `charts/reports/execucao-AAAAMMDD-HHMMSS/`, usando o nome impresso no servidor. O `rsync` mescla o conteúdo no `charts/reports/` do projeto e não apaga outros relatórios locais. As pastas `.artifacts/` e `charts/reports/` são saídas locais ignoradas pelo Git; os arquivos ficam disponíveis no clone, mas não entram em commits.

Também é possível transferir os dados brutos e gerar os gráficos localmente, depois de copiar um run completo:

```bash
.venv/bin/python scripts/report_runs.py \
  .artifacts/runs/stanford-cars-diverse
```

Para copiar runs com outros nomes, use seus diretórios correspondentes dentro de `.artifacts/runs/`. Preserve o diretório inteiro de cada run para manter logs, previsões e metadados junto dos gráficos.

## 6. Clef hospedado na Cloudflare (opcional)

Configure as variáveis no shell; `.env.example` é referência e não é carregado automaticamente:

```bash
export CLOUDFLARE_ACCOUNT_ID=seu_account_id
export CLOUDFLARE_API_TOKEN=seu_token
.venv/bin/python -m vision_bench doctor --config benchmarks/configs/cloudflare.json
.venv/bin/python -m vision_bench run \
  --config benchmarks/configs/cloudflare.json --limit 2 --repetitions 1 \
  --output .artifacts/runs/clef-cloud-smoke
```

A chamada real pode ser faturada pelo provedor; o setup/doctor/dry-run não enviam inferência. Este backend usa imagens embutidas e perguntas tipadas conforme a [documentação oficial](https://developers.cloudflare.com/workers-ai/models/clef-flash/). O serviço hospedado não oferece aqui uma revisão de pesos fixada por commit; o relatório registra essa diferença em relação aos backends locais.

## 7. Adicionar outro modelo

Se expõe System One, basta um novo objeto em uma cópia do JSON:

```json
{
  "schema_version": 1,
  "models": [{
    "id": "meu-modelo",
    "adapter": "systemone_http",
    "url_env": "MY_MODEL_SYSTEMONE_URL",
    "url": "http://127.0.0.1:11438/v1/systemone",
    "model": "identificador-do-modelo",
    "revision": "revisao-declarada",
    "timeout_s": 300
  }]
}
```

Se exigir autenticação, acrescente `"token_env": "MY_MODEL_API_TOKEN"`. Não coloque o token literal nem credenciais/query strings na URL. O runner não grava o valor da variável de autenticação.

Para outro protocolo, implemente uma classe Python e configure `"adapter": "python:meu_pacote.adapters:MeuAdapter"`. O pacote precisa estar importável no venv. Interface:

```python
class MeuAdapter:
    def __init__(self, spec, contract, device="cpu", offline=False):
        # Carregar o modelo somente aqui; registrar sua identidade no spec.
        self.contract = contract

    def predict(self, image, domain):
        # image é PIL RGB; consultar TODOS os campos do domínio.
        # Retornar labels do contrato e resposta bruta (JSON serializável).
        answers = {"presence": "yes", "make": "unknown", "color": "black", "body": "suv"}
        return answers, {"original_response": "..."}

    def synchronize(self):
        # Sincronizar o dispositivo se a inferência for assíncrona.
        pass

    def close(self):
        # Liberar recursos do adaptador.
        pass
```

O exemplo ilustra o domínio veículo, não implementa inferência. Pessoas exigem seus quatro campos próprios. Respostas fora da taxonomia tornam o evento inválido; atributos inventados não são convertidos em desconhecidos corretos. Não acessar os gabaritos dentro do adaptador.

## 8. Testes de infraestrutura

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m vision_bench run \
  --config benchmarks/configs/smoke_mock.json --limit 2 --repetitions 2 \
  --output .artifacts/runs/infra-fixture
.venv/bin/python -m vision_bench report .artifacts/runs/infra-fixture \
  --output .artifacts/reports/infra-fixture
```

O adapter mock é marcado como **SIMULAÇÃO** nos metadados e relatório. Ele testa a infraestrutura e não produz evidência sobre capacidade de modelos.

Validação nesta preparação: 20 testes passaram em um venv HTTP novo, sem torch/transformers: parser/estatística e integração HTTP local, pack/unpack com hashes, rejeição de arquivos corrompidos/caminhos externos, retomada sem duplicação, rejeição de mudança de endpoint e relatório com três modelos simulados. O diagnóstico também funcionou fora da pasta do projeto. O teste real de BLIP/Laya offline encontrou cache de pesos incompleto; não se afirma nova inferência validada por esses adaptadores nesta etapa. Clef local/cloud, CUDA e instalação em um segundo sistema operacional precisam do smoke na máquina de destino. Os pesos grandes do Clef não foram baixados nesta preparação.

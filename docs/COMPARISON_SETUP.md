# Painel web atual

Para testes interativos, benchmarks com progresso, gráficos e diagnóstico, execute `docker compose up -d --build` e abra `http://localhost:7860`. Veja [STUDIO.md](STUDIO.md) para os fluxos da interface. Os comandos abaixo continuam sendo a referência de comparação por CLI.

# Comparação portátil de modelos visuais

Este é o ponto de entrada para novos testes. Os scripts em `eval/run_test_*` e os resultados existentes são evidência histórica; não precisam ser movidos ou editados para adicionar modelos.

A interface `python -m vision_bench` aceita qualquer quantidade de modelos registrados em JSON. Há adaptadores para BLIP VQA, Laya Vision, Clef local/Cloudflare via System One HTTP e extensão Python para outros protocolos. Os testes sempre solicitam o conjunto completo de atributos definido em `benchmarks/contracts/description_v1.json`.

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

Para GPU, crie o venv e instale uma combinação compatível de torch/torchvision usando o [seletor oficial do PyTorch](https://pytorch.org/get-started/locally/). Depois instale:

```bash
.venv/bin/python -m pip install -r requirements/models.txt -e . -e ./vendor/laya-vision
```

Use `--device cuda` no runner para os adaptadores locais. O runner sincroniza o dispositivo para medir tempo completo e registra o pico de memória CUDA alocada pelo processo. Os parâmetros de GPU do servidor Clef são configurados separadamente; `--device cuda` não altera um servidor externo. Esta preparação não executou validação de GPU nesta máquina.

## 2. Dados disponíveis no clone e conjunto completo

O padrão `benchmarks/datasets/portable_vehicles.json` contém **22 eventos de veículos**, com as imagens já rastreadas no Git. Serve para teste inicial sem baixar dados. São rótulos exploratórios, não uma amostra confirmatória. O contrato também tem perguntas de pessoas, mas essa amostra padrão não testa pessoas.

As imagens de pessoas e outros datasets antigos são ignoradas pelo Git. Para usar exatamente o Teste 7 nas duas máquinas, exporte na máquina que possui as imagens:

```bash
.venv/bin/python -m vision_bench pack-data \
  --manifest eval/datasets/test_7_complete_tags_manifest.json \
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

## 3b. Clef-Flash pelo Ollama

Instale/atualize o [Ollama](https://ollama.com/download). O [modelo oficial Clef-Flash](https://ollama.com/library/clef-flash) exige Ollama 0.35.1 ou superior. No outro computador:

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

Saídas: tabelas Markdown, `metrics.json`, `metrics.csv` e gráficos PNG 300 dpi/SVG/PDF. Pontos de distribuição são medianas por evento entre repetições, não imagens novas. O relatório portátil é descritivo, sem intervalos populacionais fabricados para uma amostra arbitrária. O atlas científico histórico continua em `eval/figures/scientific_20261026_final` e usa seus dados congelados; o gerador histórico não deve ser usado como se aceitasse automaticamente novos modelos.

Não compare CPU de HTTP `client_only` à CPU do modelo em processo como custo total. Não compare custo local ao tempo remoto sem informar rede/hardware/faturamento. As respostas brutas preservam `usage` quando fornecido, mas não há preço monetário presumido. A métrica de positivos completos com os rótulos históricos é exploratória e deve permanecer identificada assim.

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
.venv/bin/python -m unittest discover -s eval/tests -v
.venv/bin/python -m vision_bench run \
  --config benchmarks/configs/smoke_mock.json --limit 2 --repetitions 2 \
  --output .artifacts/runs/infra-fixture
.venv/bin/python -m vision_bench report .artifacts/runs/infra-fixture \
  --output .artifacts/reports/infra-fixture
```

O adapter mock é marcado como **SIMULAÇÃO** nos metadados e relatório. Ele testa a infraestrutura e não produz evidência sobre capacidade de modelos.

Validação nesta preparação: 20 testes passaram em um venv HTTP novo, sem torch/transformers: parser/estatística e integração HTTP local, pack/unpack com hashes, rejeição de arquivos corrompidos/caminhos externos, retomada sem duplicação, rejeição de mudança de endpoint e relatório com três modelos simulados. O diagnóstico também funcionou fora da pasta do projeto. O teste real de BLIP/Laya offline encontrou cache de pesos incompleto; não se afirma nova inferência validada por esses adaptadores nesta etapa. Clef local/cloud, CUDA e instalação em um segundo sistema operacional precisam do smoke na máquina de destino. Os pesos grandes do Clef não foram baixados nesta preparação.

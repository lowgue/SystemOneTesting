# Aplicação web

A aplicação atual usa React/Vite no frontend e FastAPI/Uvicorn na API. A execução recomendada é `docker compose up -d --build`, seguida de acesso a **http://localhost:7860**. Nenhum script `.sh` faz parte desse fluxo.

## Preparar o modelo pela interface

Abra **Ambiente e modelos**, verifique se o Ollama responde e clique em **Baixar ou atualizar Clef-Flash 9B**. O progresso chega à página por Server-Sent Events. O download usa `/api/pull` do Ollama, não comandos shell. É possível cancelar o worker; o servidor Ollama pode continuar processando uma requisição já recebida.

O Compose executa um Ollama separado, sem publicar a porta 11434. Pesos já instalados no sistema não aparecem automaticamente nessa instância. O volume `ollama-models` preserva o download entre reinicializações. O modelo precisa de vários GB de disco e memória; consulte erros de alocação reais em vez de deduzir a causa somente da RAM total.

## Testar imagens e comparar

Selecione a fonte de modelos, o contrato e os modelos desejados. Use o contrato compartilhado Ollama ao comparar com Clef. Escolha veículo ou pessoa, envie a imagem e clique em **Avaliar imagem**. Não há inferência automática ao selecionar um arquivo.

A aplicação mostra os estados de preparação/inferência, cada resposta e cada erro. O botão **Baixar JSON** exporta a tarefa. As imagens são guardadas localmente no volume de avaliações; não envie esse volume a terceiros se contiver imagens privadas.

As chamadas usam a mesma imagem RGB com até 1024 px e as mesmas perguntas para cada modelo. Os modelos são avaliados sequencialmente. A primeira inferência HTTP pode incluir carregamento do modelo; este modo não tem aquecimento nem gabarito e não estima acurácia.

O container padrão inclui BLIP/Laya em CPU, com as dependências e o código do submódulo. Execute `git submodule update --init --recursive` antes de construir a imagem. Os pesos são baixados na primeira avaliação e persistidos no volume `huggingface-models`. Para GPU Python, use o ambiente específico do guia de comparação; o override de GPU do Ollama não fornece CUDA ao backend Python. Threads da interface afetam os modelos Python, não a configuração do Ollama.

## Avaliação automática de datasets e resultados

Em **Avaliar dataset**, importe suas fotos e um CSV de gabarito opcional, ou use o conjunto de exemplo. Veja [DATASETS.md](DATASETS.md) para o formato do CSV e a definição de cada métrica. O relatório inclui acurácia completa e por atributo, precisão, recall, F1, distribuição de latência, falhas, formato e consumo de recursos, além de revisão e exportação por foto.

A avaliação usa o runner existente, incluindo aquecimento, alternância e configuração congelada. O dataset portátil tem 22 eventos e rótulos exploratórios. Ao terminar, gera gráficos e métricas. Em **Resultados**, abra os arquivos ou selecione uma tarefa anterior para consultar o estado e as respostas.

O histórico e as avaliações ficam em `/app/.artifacts` no volume `evaluations`. A aplicação usa um processo Uvicorn e um worker de inferência por vez. Não aumente `--workers`: o gerenciador de tarefas é local ao processo.

`docker compose restart` reinicia os serviços. Um encerramento normal cancela o worker da API; os arquivos parciais ficam preservados. Não há retomada automática após encerramento forçado. O CLI de benchmark continua disponível para `--resume` com os mesmos dados e configurações.

## Investigar problemas

1. Abra **Ambiente e modelos** e confira a versão do Ollama, instalação do Clef e memória disponível.
2. Execute uma avaliação e leia o erro individual do modelo. O JSON preserva essa mensagem.
3. Consulte `docker compose logs -f web ollama` para os registros do servidor.

`ollama ps` no terminal do sistema consulta o Ollama do sistema, não necessariamente o serviço da aplicação. Para consultar a instância do Compose, use `docker compose exec ollama ollama ps`.

O endpoint `/api/health` verifica a aplicação; não carrega modelos. `/api/diagnostics` consulta o Ollama, mas não prova que a inferência funcionará. Se o erro for de memória, confira os recursos disponibilizados ao Docker Desktop e o uso de GPU. Ter 40 GB no computador não garante que todos estejam disponíveis ao runtime.

## Desenvolvimento e API

Os comandos completos estão no README. Use Vite em desenvolvimento e Uvicorn no backend. Em produção local, o FastAPI serve o build React. Documentação REST em `/docs`; schema em `/openapi.json`.

Endpoints principais: `/api/catalog`, `/api/jobs`, `/api/jobs/{id}`, `/api/jobs/{id}/events`, `/api/jobs/{id}/cancel`, `/api/ollama/pull`, `/api/diagnostics` e `/api/artifacts`.

Referências oficiais: [FastAPI em Docker](https://fastapi.tiangolo.com/deployment/docker/), [Ollama em Docker](https://docs.ollama.com/docker) e [API de download do Ollama](https://docs.ollama.com/api/pull).

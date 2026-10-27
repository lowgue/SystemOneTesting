# System One — Laboratório visual

Aplicação web para testar Clef-Flash 9B, comparar modelos visuais e executar benchmarks. Frontend **React/Vite**, API **FastAPI/Uvicorn**, eventos ao vivo e histórico persistente.

## Executar a aplicação

Instale Docker Desktop (Windows/macOS) ou Docker Engine com Compose (Linux). Na pasta do projeto:

```bash
git pull
git submodule update --init --recursive
docker compose up -d --build
```

Abra **http://localhost:7860**.

1. Entre em **Ambiente e modelos**.
2. Clique em **Baixar ou atualizar Clef-Flash 9B** e acompanhe o progresso.
3. Em **Avaliar foto**, envie uma imagem e clique em **Avaliar foto**.

Não é necessário executar `.sh`, instalar Python/Node no computador de destino ou iniciar `ollama serve`. O Compose sobe a aplicação e um Ollama próprio. Os pesos do Clef ficam em `ollama-models`, os pesos do BLIP/Laya em `huggingface-models` e as avaliações em `evaluations`; esses volumes persistem ao reiniciar.

O primeiro build baixa as imagens Docker. O download do modelo é separado e iniciado pela interface. O Ollama da aplicação não compartilha automaticamente os pesos do Ollama instalado no sistema, e não usa a porta 11434 do computador.

## Telas

| Tela | Função |
|---|---|
| Avaliar foto | Inspecionar uma foto com os modelos selecionados; respostas e tempo por modelo |
| Avaliar dataset | Importar várias fotos e gabarito CSV opcional; avaliar automaticamente e ver acurácia, F1, latência, falhas e gráficos |
| Resultados | Histórico das tarefas, JSON, CSV, gráficos PNG/PDF e atlas científico anterior |
| Ambiente e modelos | Download do Clef pelo navegador, versão do Ollama, memória e diagnóstico |

A imagem Docker padrão inclui BLIP e Laya em CPU (torch, torchvision, transformers e o código do submódulo), além de Clef pelo Ollama. Os pesos do BLIP/Laya são baixados na primeira avaliação e reutilizados do volume `huggingface-models`. Para validar a aplicação sem pesos, escolha `smoke_mock`: os resultados aparecem explicitamente como simulação. Para desenvolvimento e GPU, consulte o [guia de comparação](docs/COMPARISON_SETUP.md).

## Avaliar meu dataset

O catálogo também oferece os sete testes históricos, com conjuntos de até **340 fotos** e recortes separados de avaliação/desenvolvimento. Em **O que comparar?**, escolha modelos, prompts, requests conjuntos ou chamadas por atributo. A tela mostra perguntas, formatos e orçamento antes de executar; o relatório inclui gráficos por atributo/classe, matrizes e diferenças pareadas. Consulte [datasets](docs/DATASETS.md) e [metodologia das comparações](docs/PROMPT_COMPARISONS.md). Imagens históricas não estão no Git; o painel permite exportar ZIP verificado para outro computador.

Abra **Avaliar dataset → Importar minhas fotos**. Selecione as fotos, anexe um CSV de gabarito se disponível e clique em **Preparar dataset**. Depois, clique em **Iniciar avaliação do dataset**.

O relatório mostra acurácia completa por foto, acurácia por atributo, precisão/recall/F1, mediana e P95 do tempo, falhas, validade do formato e recursos medidos. Inclui revisão por foto, matrizes de confusão e exportação CSV/JSON/PDF. Sem gabarito, ficam disponíveis somente as métricas de desempenho e falhas.

O botão **Baixar modelo de gabarito** facilita a criação do CSV. Veja [como importar e interpretar as métricas](docs/DATASETS.md).

## Operação

```bash
docker compose logs -f web ollama
docker compose restart
docker compose down
```

`down` preserva os volumes. `down -v` apaga pesos e avaliações; não use para uma reinicialização comum.

Para NVIDIA com suporte GPU configurado no Docker:

```bash
docker compose -f compose.yaml -f compose.gpu.yaml up -d --build
```

Sem o override, a configuração é CPU. No Docker Desktop, confira a RAM disponibilizada à VM: a RAM total do computador não é necessariamente a memória disponível ao container. Um serviço online ou modelo instalado não comprova que a inferência carregará; a tela de teste mostra os erros do runtime.

## Desenvolvimento sem Docker

Python 3.12 e Node 24. Em um terminal na raiz:

```bash
python -m pip install -r requirements/web.txt
python -m uvicorn vision_bench.webui.api:app --host 127.0.0.1 --port 8000 --reload
```

Em outro terminal:

```bash
npm --prefix frontend ci
npm --prefix frontend run dev
```

Abra o endereço exibido pelo Vite (normalmente `http://localhost:5173`). O proxy encaminha API, arquivos e eventos ao backend. Neste modo, o endpoint Ollama padrão é `http://127.0.0.1:11434/v1/systemone`; pode ser configurado pela variável `OLLAMA_SYSTEMONE_URL`.

Para servir o frontend compilado diretamente pelo FastAPI:

```bash
npm --prefix frontend run build
python -m uvicorn vision_bench.webui.api:app --host 127.0.0.1 --port 7860
```

API documentada em **http://localhost:7860/docs** (ou porta 8000 no desenvolvimento).

## Estrutura e documentação

```text
frontend/                Aplicação React e build Vite
vision_bench/webui/      API FastAPI, serviços e workers de inferência
vision_bench/            Motor compartilhado de benchmarks e relatórios
benchmarks/              Configurações, contratos e dataset portátil
requirements/           Dependências Python por backend
compose.yaml            Aplicação + Ollama + volumes persistentes
Dockerfile              Build do frontend e servidor web
tests/                 Testes de API, integração e navegador
docs/                  Guias de uso, arquitetura e validação
eval/                  Experimentos e evidências anteriores preservados
```

- [Uso da aplicação e solução de erros](docs/STUDIO.md)
- [Arquitetura e responsabilidades](docs/PROJECT_STRUCTURE.md)
- [Comparações reproduzíveis](docs/COMPARISON_SETUP.md)
- [Atlas estatístico anterior](eval/figures/scientific_20261026_final/README.md)
- [Plano de avaliação robusta](eval/plans/robust_visual_description_evaluation.md)

O contrato compartilhado mantém até 26 alternativas por pergunta, compatível com Ollama. O modo interativo não estima acurácia. Os gabaritos existentes são exploratórios; o benchmark separa execuções e computadores. Recursos do cliente HTTP não medem o consumo do servidor Ollama.

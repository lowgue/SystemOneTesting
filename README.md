# System One — Laboratório visual

Aplicação web para testar Clef-Flash 9B, comparar modelos visuais e executar benchmarks. Frontend **React/Vite**, API **FastAPI/Uvicorn**, eventos ao vivo e histórico persistente.

## Executar a aplicação

Instale Docker Desktop (Windows/macOS) ou Docker Engine com Compose (Linux). Na pasta do projeto:

```bash
git pull
docker compose up -d --build
```

Abra **http://localhost:7860**.

1. Entre em **Ambiente e modelos**.
2. Clique em **Baixar ou atualizar Clef-Flash 9B** e acompanhe o progresso.
3. Em **Teste de imagem**, envie uma imagem e clique em **Avaliar imagem**.

Não é necessário executar `.sh`, instalar Python/Node no computador de destino ou iniciar `ollama serve`. O Compose sobe a aplicação e um Ollama próprio. Os pesos ficam no volume `ollama-models` e as avaliações no volume `evaluations`; ambos persistem ao reiniciar.

O primeiro build baixa as imagens Docker. O download do modelo é separado e iniciado pela interface. O Ollama da aplicação não compartilha automaticamente os pesos do Ollama instalado no sistema, e não usa a porta 11434 do computador.

## Telas

| Tela | Função |
|---|---|
| Teste de imagem | Avaliar a mesma imagem com os modelos selecionados; respostas e erros por modelo |
| Benchmark | Rodadas com repetições, aquecimento, progresso e cancelamento |
| Resultados | Histórico das tarefas, JSON, CSV, gráficos PNG/PDF e atlas científico anterior |
| Ambiente e modelos | Download do Clef pelo navegador, versão do Ollama, memória e diagnóstico |

A imagem Docker padrão contém o backend HTTP, incluindo Clef pelo Ollama. Para validar a aplicação sem pesos, escolha `smoke_mock`: os resultados aparecem explicitamente como simulação. BLIP/Laya precisam do ambiente Python específico, documentado no [guia de comparação](docs/COMPARISON_SETUP.md).

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

# System One Testing

Laboratório local para testar modelos visuais, comparar respostas e medir desempenho. O painel reúne testes com imagens, benchmarks reproduzíveis, gráficos e diagnóstico do Ollama.

## Começar pelo navegador

Instale **Python 3.12** e [Ollama](https://ollama.com/download) **0.35.1 ou superior**. Na pasta do projeto, atualize com `git pull` e execute:

**Windows:** dê dois cliques em `iniciar_clef_ollama.bat`.

**Linux/macOS:**

```bash
bash iniciar_clef_ollama.sh
```

O iniciador prepara o ambiente, verifica/inicia o Ollama, baixa `clef-flash:9b` se necessário e abre **http://127.0.0.1:7860**. O primeiro download pode demorar. Mantenha o terminal aberto e use `Ctrl+C` para encerrar.

O modelo pode estar instalado e ainda falhar ao carregar. Use **Diagnóstico** e **Teste de imagem** para verificar a inferência no seu hardware. Neste computador de 7,5 GiB de RAM, o Clef 9B falhou ao tentar alocar aproximadamente 12,6 GB. Isso não determina a causa de um erro em outro computador.

## O que há no painel

| Área | Uso |
|---|---|
| Teste de imagem | Enviar uma imagem, selecionar um ou mais modelos e acompanhar respostas e erros individualmente |
| Benchmark | Executar o dataset portátil com repetições, aquecimento, progresso e cancelamento |
| Resultados | Abrir gráficos PNG/PDF, métricas CSV/JSON e o atlas científico histórico |
| Diagnóstico | Consultar versão do Ollama, modelos instalados/carregados, RAM disponível e registro da tarefa |

A configuração inicial usa **Clef-Flash 9B pelo Ollama**. BLIP e Laya estão na mesma lista e precisam do ambiente Python para modelos. Para experimentar a aplicação sem pesos, selecione `smoke_mock`: os resultados são identificados como **simulação**.

## Abrir apenas o painel

Se o ambiente já está preparado, não é necessário reiniciar nem baixar modelos:

```bash
.venv/bin/python -m vision_bench studio --open
```

No Windows, substitua `.venv/bin/python` por `.venv\Scripts\python.exe`. Outra porta: acrescente `--port 7861`. A interface funciona sem CDN, Node ou compilação de frontend.

## Instalação e comparações

Para preparar um ambiente somente HTTP:

```bash
python3.12 scripts/setup_benchmark.py --backend http
```

Para BLIP/Laya em CPU, use `--backend cpu`. Para GPU, modelos adicionais, datasets externos, uso por CLI e execução em outros computadores, veja o [guia de comparação](docs/COMPARISON_SETUP.md).

O contrato padrão compartilhado tem até 26 alternativas por pergunta, compatível com Ollama. Todos os modelos da mesma comparação usam a mesma imagem e o mesmo contrato. Testes interativos não estimam acurácia; os gabaritos históricos são exploratórios. O benchmark mantém as máquinas e execuções separadas e não estima custo monetário ou energia.

## Organização

```text
vision_bench/             Motor atual: contratos, adapters, execução e relatórios
  webui/                 Painel: servidor HTTP, gerência de tarefas e worker
    static/              HTML, CSS e JavaScript locais
benchmarks/              Configurações, contratos e dataset portátil
scripts/                 Preparação e inicialização
tests/                  Testes da aplicação atual
docs/                   Guias, arquitetura e referência histórica
eval/                   Experimentos, datasets, testes e evidências históricas
web/                    Interfaces anteriores mantidas por compatibilidade
examples/               Exemplos de uso
.artifacts/              Imagens enviadas, tarefas, logs e relatórios locais (fora do Git)
```

- [Como usar o painel e investigar erros](docs/STUDIO.md)
- [Mapa do projeto e arquitetura](docs/PROJECT_STRUCTURE.md)
- [Comparações reproduzíveis](docs/COMPARISON_SETUP.md)
- [Atlas estatístico histórico](eval/figures/scientific_20261026_final/README.md)
- [Plano de avaliação robusta](eval/plans/robust_visual_description_evaluation.md)
- [Guia anterior e exemplos](docs/LEGACY_GUIDE.md)
- [Skills encontradas para interface e testes](docs/SKILLS.md)

## Verificar alterações

```bash
.venv/bin/python -m unittest discover -s tests
.venv/bin/python -m unittest discover -s eval/tests
```

Os testes automatizados usam modelos simulados e servidores HTTP locais. Não baixam pesos nem executam APIs pagas. O teste de navegador é opcional; veja [tests/README.md](tests/README.md).

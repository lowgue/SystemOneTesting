# System One Testing

Experimentos para avaliar modelos de decisão System 1 (Laya) em tarefas reais de classificação, roteamento e inspeção visual.

## Estrutura do projeto

```
SystemOneTesting/
├── web/
│   ├── index.html          # Interface visual em tempo real (Tailwind / responsiva)
│   └── server.py           # Servidor local para inspeção interativa de imagens
├── examples/
│   ├── laya_text.py        # Exemplos de uso da API textual (/v1/systemone)
│   └── laya_vision.py      # Exemplo com imagens (thaitea/laya-vision)
├── eval/
│   ├── harness.py          # Harness de avaliação em batch textual
│   ├── vision_cars.py      # Avaliação de imagens de carros (acurácia, latência)
│   ├── datasets/
│   │   └── default.jsonl   # Dataset anotado: 15 casos, 5 domínios
│   ├── images/cars/        # Imagens de teste (carros e paisagens)
│   └── results/            # Resultados JSON por execução (auto-gerado)
└── vendor/laya-vision/     # Código de inferência do modelo visual
```

## Laya textual no Ollama

O Laya textual está instalado no Ollama local. A API de decisões é `/v1/systemone`; ela aceita um `state` (texto ou JSON estruturado) e perguntas tipadas como `noul` (sim/não), `choice` (opções) e `score` (rubrica ordenada). A versão `laya` do Ollama é somente texto.

Confirme que o Ollama está ativo e que o modelo aparece na lista:

```bash
ollama --version
ollama list
```

O Laya exige Ollama 0.40.0 ou mais recente. Se ainda não estiver baixado:

```bash
ollama pull laya
```

Este modelo textual é independente do exemplo visual abaixo.

## Laya Vision com imagens

`thaitea/laya-vision` é uma variante independente e experimental que recebe a imagem diretamente e responde perguntas tipadas (`noul`, `choice`, `score`) com probabilidades. Não roda como o `laya` já instalado no Ollama: o model card orienta usar o pacote Python próprio, que baixa os pesos do Hugging Face. O exemplo abaixo pergunta diretamente à imagem se há um carro e sobre a marca e a cor, em uma chamada.

Crie um ambiente Python e instale o código de inferência publicado pelo autor:

```bash
python3 -m venv .venv
source .venv/bin/activate
git clone https://github.com/r33drichards/laya-vision vendor/laya-vision
python -m pip install -e ./vendor/laya-vision torchvision pillow
```

Execute com uma imagem local:

```bash
python examples/laya_vision.py ./imagem.jpg
```

O checkpoint tem cerca de 201 milhões de parâmetros e processa uma imagem por estado, redimensionada para um tile de 512 px. É pesquisa experimental; confira a licença CC BY-NC-SA 4.0 antes de usar fora de testes não comerciais. As probabilidades devem ser avaliadas com imagens anotadas do seu próprio domínio.

## Interface Web em Tempo Real (Browser)

Para visualizar e testar imagens interativamente com interface didática:

```bash
source .venv/bin/activate
python web/server.py --port 7860
```

Acesse no navegador: **[http://localhost:7860](http://localhost:7860)**

O playground executa **um modelo por sessão**. Para usar Laya Vision (rápido), inicie:

```bash
.venv/bin/python web/server.py --model laya --port 7860
```

Para usar o Clef-Flash hospedado no Cloudflare Workers AI, configure as credenciais no terminal e inicie outra sessão web:

```bash
export CLOUDFLARE_ACCOUNT_ID="seu_account_id"
export CLOUDFLARE_API_TOKEN="seu_api_token_com_permissao_workers_ai"
.venv/bin/python web/server.py --model clef --port 7860
```

Encerre uma sessão com `Ctrl+C` antes de iniciar a outra na mesma porta.

Para testar Clef-Flash localmente em Q4_K_M, rode o launcher em um terminal. Ele baixa os pesos quantizados e inicia `llama-server` na porta `11437`:

```bash
bash run_clef_local.sh
```

Em outro terminal, abra o playground usando essa instância local:

```bash
.venv/bin/python web/server.py --model clef-local --port 7860
```

## Teste isolado de cada modelo

Os scripts abaixo avaliam a mesma imagem de Audi TT e fazem as mesmas duas perguntas. Rode apenas um por vez:

```bash
.venv/bin/python test_laya_real_image.py
.venv/bin/python test_clef_local.py
.venv/bin/python test_clef_real_image.py
```

`test_clef_local.py` exige o servidor iniciado por `run_clef_local.sh`; `test_clef_real_image.py` usa a API hospedada da Cloudflare e exige `CLOUDFLARE_ACCOUNT_ID` e `CLOUDFLARE_API_TOKEN`. Os testes aceitam outra imagem como argumento posicional.

## Experimentos

Registre, para cada caso, a entrada, as regras/opções, a resposta correta anotada, a resposta do modelo, as probabilidades, a latência e a versão do modelo. Separe os dados usados para ajustar limiares do conjunto final de avaliação. As probabilidades declaradas pelo modelo não garantem calibração no seu domínio.

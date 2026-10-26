# Testes

Instale as dependências de desenvolvimento em um ambiente Python 3.12:

```bash
python -m pip install -r requirements/dev.txt
python -m unittest discover -s tests
python -m unittest discover -s eval/tests
```

A suíte atual verifica o motor, workers, API FastAPI, streaming de eventos, histórico, download do modelo por API simulada, validação de pedidos, concorrência, cancelamento e geração de relatórios. Não baixa pesos nem usa APIs pagas.

## Navegador da aplicação React

```bash
npm --prefix frontend ci
npm --prefix frontend run build
python -m playwright install chromium
python -m tests.browser_web
```

O teste inicia uma API Uvicorn própria em uma porta livre e abre Chromium headless. Exercita upload, dois modelos simulados, eventos SSE, JSON, benchmark, gráficos, histórico, download do modelo simulado, erro detalhado, recarregamento e layout móvel sem overflow. Não usa o Ollama real para baixar pesos.

Capturas e resumo ficam em `.artifacts/validation/react-web/`.

`tests/browser_studio.py` verifica apenas a interface anterior, mantida como referência de compatibilidade. Não é o teste da aplicação React atual.

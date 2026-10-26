# Verificação da aplicação

Execute os comandos a partir da raiz, com o ambiente virtual ativo:

```bash
python -m unittest discover -s tests
python -m unittest discover -s eval/tests
```

A suíte atual usa servidor HTTP real, workers em subprocessos e adapters simulados. Cobre comparação de dois modelos com contrato/imagem compartilhados, benchmark com gráficos, validação de entradas, controle de concorrência, cancelamento e restrição de arquivos acessíveis. A suíte em `eval/tests` também verifica parser, proveniência, portabilidade e erros de inferência.

## Navegador (opcional)

```bash
python -m pip install -r requirements/dev.txt
python -m playwright install chromium
python -m tests.browser_studio
```

O teste usa Chromium headless e um servidor próprio em porta livre. Verifica upload, respostas de dois modelos simulados, download JSON, benchmark, relatório, diagnóstico, erro detalhado de um backend local e layout móvel sem rolagem horizontal. Não requer pesos ou API paga. O diagnóstico pode consultar o Ollama local, sem carregar modelos ou executar inferência.

Capturas e resumo ficam em `.artifacts/validation/studio/`. A ausência de pesos reais limita o que esse teste comprova: valida a integração da aplicação, não a qualidade ou compatibilidade de inferência dos modelos reais.

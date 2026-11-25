# Testes

Ambiente Python 3.12 ou superior. Do clone, o comando recomendado cria um ambiente virtual local, instala só as dependências necessárias à suíte, executa todos os testes e gera relatórios para execuções completas já salvas:

```bash
python3 scripts/test_all.py
```

No Windows: `py -3.12 scripts/test_all.py`. Para rodar a suíte num ambiente já instalado, use `python3 scripts/test_all.py --no-install`. A suíte cobre o motor, as métricas de dataset e as tarefas históricas migradas. O fluxo não baixa pesos nem usa APIs pagas. Os gráficos dos relatórios são salvos em `charts/reports/`.

Em uma máquina Linux com NVIDIA, `python3.12 scripts/test_all.py --gpu` executa a suíte e o benchmark do Clef-Flash pelo vLLM, enviando quatro imagens simultaneamente por padrão. Instale vLLM em `~/venvs/systemone-vllm` antes. `--include-local-benchmarks` também roda BLIP/Laya/D1, que permanecem seriais; `--skip-benchmark` pula todas as inferências.

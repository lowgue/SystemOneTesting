# Testes

Ambiente Python 3.12 ou superior. Do clone, o comando recomendado cria um ambiente virtual local, instala só as dependências necessárias à suíte, executa todos os testes e gera relatórios para execuções completas já salvas:

```bash
python3 scripts/test_all.py
```

No Windows: `py -3.12 scripts/test_all.py`. Para rodar a suíte num ambiente já instalado, use `python3 scripts/test_all.py --no-install`. A suíte cobre o motor, as métricas de dataset e as tarefas históricas migradas. O fluxo não baixa pesos nem usa APIs pagas. Os gráficos dos relatórios são salvos em `charts/reports/`.

Em uma máquina NVIDIA com o driver instalado, `python3 scripts/test_all.py --gpu` instala a build CUDA do PyTorch, verifica a placa, executa a mesma suíte e roda o benchmark dos três adapters locais para gerar gráficos. Adicione `--skip-benchmark` para validar somente a instalação CUDA e a suíte unitária.

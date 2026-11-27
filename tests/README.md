# Testes

Para o inventário de cada experimento e o setup completo do servidor GPU, veja [Guia de testes e setup do servidor](../docs/GUIA_TESTES_SERVIDOR.md).

Ambiente Python 3.12 ou superior. Do clone, o comando recomendado cria um ambiente virtual local, instala só as dependências necessárias à suíte, executa todos os testes e gera relatórios para execuções completas já salvas:

```bash
python3 scripts/test_all.py
```

No Windows: `py -3.12 scripts/test_all.py`. Para rodar a suíte num ambiente já instalado, use `python3 scripts/test_all.py --no-install`. A suíte cobre o motor, as métricas de dataset e as tarefas históricas migradas. O fluxo não baixa pesos nem usa APIs pagas. Os gráficos dos relatórios são salvos em `charts/reports/`.

Em uma máquina Linux com NVIDIA, `python3.12 scripts/test_all.py --gpu` executa a suíte e mede os quatro modelos de `benchmarks/configs/vllm-models.json` pelo vLLM. O lote padrão é `--concurrency auto`: no fluxo gerenciado ele usa a VRAM livre; com `--external-model clef-flash-vllm`, usa `--max-num-seqs 32` e as métricas reais de KV cache do servidor aquecido. Instale o vLLM 0.34 com CUDA 12.8 em `/models/venvs/systemone-vllm-cu128` e use `scripts/start_vllm_clef.sh` para manter o Clef aquecido. `--include-local-benchmarks` também roda BLIP/Laya/D1, que permanecem seriais; `--skip-benchmark` pula todas as inferências. Para validar mudanças de configuração, compare answers e logprobs com `scripts/smoke_compare.py`.

# Laya Vision (thaitea/laya-vision)

Modelo de decisão visual com saída estruturada (choice/noul), treinado sobre SmolVLM-256M-Instruct.

## Teste Standalone

```bash
python test_laya_real_image.py [imagem]
```

## Subpastas

- **laya_vision/** — Submódulo Git com a implementação externa do Laya Vision.
- **dataset_tests/** — Benchmarks empíricos (Testes 1-5, 7): cor, tipo de veículo, atributos de pessoa, robustez, multi-objeto, tags completas. Inclui comparação Laya vs BLIP.
- **prompt_tests/** — Teste 6: comparação de formatos de critério (native_short, native_descriptive) + auditoria de orçamento de opções.

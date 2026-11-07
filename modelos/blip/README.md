# BLIP (Salesforce/blip-vqa-base)

Modelo generativo de Visual Question Answering da Salesforce.

## Teste Standalone

```bash
python test_blip_vqa.py [imagem] --question "What color is the car?"
```

## Subpastas

- **testes_datasets/** — Benchmarks empíricos (Testes 1-5, 7): cor, tipo de veículo, atributos de pessoa, robustez, multi-objeto, tags completas
- **testes_prompts/** — Teste 6: comparação de formatos de prompt (short_open, short_options, long_descriptions, structured_json)

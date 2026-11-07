# D1-Omni (LiquidAI)

Modelo de decisão multi-modal da LiquidAI com `trust_remote_code`, usando o método nativo `.system_one()` para inferência com zero tokens gerados.

## Integração

Executado via framework `vision_bench/` com o adaptador `d1_omni`:

```python
from vision_bench.adapters import D1Omni
```

## Subpastas

- **testes_datasets/** — Testes de dataset via framework `vision_bench/`
- **testes_prompts/** — Testes de variação de prompt via `vision_bench/`

> Os benchmarks do D1-Omni são integrados pelo adapter em `vision_bench/adapters/d1_omni.py` (classe `D1Omni`). A configuração fica em `benchmarks/configs/`.

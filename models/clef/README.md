# Clef-Flash (Q4_K_M)

Modelo Clef-Flash via System One API — local (llama-server) ou Cloudflare Workers AI.

## Testes Standalone

```bash
# Local (llama-server)
python test_clef_local.py [imagem] --url http://127.0.0.1:11437/v1/systemone

# Cloudflare Workers AI
export CLOUDFLARE_ACCOUNT_ID=...
export CLOUDFLARE_API_TOKEN=...
python test_clef_real_image.py [imagem]
```

## Subpastas

- **dataset_tests/** — Testes de dataset via framework `vision_bench/` (adaptador `systemone_http`)
- **prompt_tests/** — Testes de variação de prompt via `vision_bench/`

> Os benchmarks do Clef são executados pelo adapter HTTP em `vision_bench/adapters/http.py` (classe `SystemOneHTTP`), não como scripts standalone.

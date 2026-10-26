# Validação da aplicação web

Verificação em 26/10/2026, Linux, Python 3.12 e Node 24. Windows/macOS não foram executados nesta máquina.

| Verificação | Resultado |
|---|---|
| `python -m unittest discover -s tests` | 21 testes passaram, incluindo API, importação, gabarito e métricas |
| `python -m unittest discover -s eval/tests` | 25 testes passaram |
| `npm --prefix frontend run build` | Build React/Vite gerado |
| `python -m tests.browser_web` | Chromium headless passou, sem erros JavaScript |
| `docker compose config --quiet` | Configuração válida |
| `docker compose build web` | Imagem gerada com frontend compilado e API Uvicorn |
| `docker compose up -d --wait` | Aplicação e Ollama saudáveis; API confirmou Ollama 0.40.6 |
| Navegador na instalação Docker | Teste simulado e diagnóstico real entre containers passaram |
| Dataset próprio no Docker | Upload de 2 fotos, CSV parcial em português e 3 repetições: acurácia por atributo de 80% conferida, gráficos e download CSV passaram |
| Links locais | Sem destinos ausentes |
| Evidências históricas em `eval/` | Não modificadas |

O navegador verifica uma foto, importação de várias fotos, gabarito parcial CSV em português, avaliação automática, acurácia conferida com resultado esperado, revisão de fotos incorretas, dois modelos simulados, eventos SSE, exportações, gráficos, histórico, download do modelo via servidor simulado, diagnóstico, detalhes de erro 500, recarregamento e layout de 390 px sem overflow.

Os testes de métricas confirmam que repetições não aumentam o denominador de acurácia, falhas não são descartadas e ausência de gabarito não vira 0%. Também verificam precisão/recall/F1, matriz de confusão, imutabilidade do dataset, reenvio idempotente e continuidade após falha de aquecimento.

As capturas locais ficam em `.artifacts/validation/react-web/`. Os testes não comprovam a qualidade dos modelos reais. O download do modelo foi validado com um servidor HTTP simulado, sem transferir novos pesos Clef nesta etapa.

A tentativa anterior de inferência real no host de 7,5 GiB falhou por falta de memória. Isso não diagnostica outro computador. Para verificar a inferência no destino, use a aplicação e consulte os detalhes de erro do serviço Ollama.

# Validação da aplicação web

Verificação em 26/10/2026, Linux, Python 3.12 e Node 24. Windows/macOS não foram executados nesta máquina.

| Verificação | Resultado |
|---|---|
| `python -m unittest discover -s tests` | 10 testes passaram, incluindo API FastAPI |
| `python -m unittest discover -s eval/tests` | 25 testes passaram |
| `npm --prefix frontend run build` | Build React/Vite gerado |
| `python -m tests.browser_web` | Chromium headless passou, sem erros JavaScript |
| `docker compose config --quiet` | Configuração válida |
| `docker compose build web` | Imagem gerada com frontend compilado e API Uvicorn |
| `docker compose up -d --wait` | Aplicação e Ollama saudáveis; API confirmou Ollama 0.40.6 |
| Navegador na instalação Docker | Teste simulado e diagnóstico real entre containers passaram |
| Benchmark no container web | Rodada simulada concluída e gráfico salvo no volume persistente |
| Links locais | Sem destinos ausentes |
| Evidências históricas em `eval/` | Não modificadas |

O navegador verifica upload, dois modelos simulados, eventos SSE, exportação JSON, benchmark, gráficos, histórico, download do modelo via servidor simulado, diagnóstico, detalhes de erro 500, recuperação da tarefa ao recarregar a página e layout de 390 px sem overflow.

As capturas locais ficam em `.artifacts/validation/react-web/`. Os testes não comprovam a qualidade dos modelos reais. O download do modelo foi validado com um servidor HTTP simulado, sem transferir novos pesos Clef nesta etapa.

A tentativa anterior de inferência real no host de 7,5 GiB falhou por falta de memória. Isso não diagnostica outro computador. Para verificar a inferência no destino, use a aplicação e consulte os detalhes de erro do serviço Ollama.

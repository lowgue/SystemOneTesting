# Validação da reorganização

Verificação em 26/10/2026, Linux e Python 3.12. O ambiente Windows não foi executado nesta máquina.

| Verificação | Resultado |
|---|---|
| `python -m unittest discover -s tests` | 6 testes passaram |
| `python -m unittest discover -s eval/tests` | 25 testes passaram |
| `python -m tests.browser_studio` | Chromium headless: passou, sem erros JavaScript |
| Iniciador Ollama/painel | Abriu o painel, catálogo e diagnóstico responderam |
| Encerramento com benchmark ativo | Worker cancelado e estado persistido como `cancelled` |
| Empacotamento Python | Wheel gerado com HTML, CSS e JavaScript |
| Links locais da documentação | Verificados, sem destinos ausentes |
| Evidências em `eval/` | Nenhum arquivo histórico modificado nesta reorganização |

A suíte atual cobre inferência com dois adapters simulados, a continuidade após falha de dependência de um modelo, benchmark com relatório, entrada inválida, concorrência, cancelamento e restrição de arquivos. O navegador verifica upload, comparação, download JSON, benchmark, arquivos, diagnóstico, apresentação do erro 500 e layout de 390 px sem rolagem horizontal.

As capturas locais estão em `.artifacts/validation/studio/`: `desktop.png`, `mobile.png`, `results.png` e `error.png`. São evidências de teste da interface com modelos simulados, não resultados científicos de modelos reais.

Não foi comprovada a inferência real do Clef nesta máquina: a tentativa anterior falhou por memória insuficiente. Essa falha não diagnostica outro computador. O diagnóstico do painel não carrega modelos; o teste de imagem é necessário para verificar a inferência no hardware de destino.

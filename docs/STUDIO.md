# Painel local

O painel está em `vision_bench/webui/`. Abra com o iniciador Clef da raiz ou, no ambiente já instalado:

```bash
python -m vision_bench studio --open
```

O servidor escuta apenas em `127.0.0.1`. Não há etapa de build, dependência de Node, CDN ou serviço remoto para abrir a interface. Consultas a backends hospedados dependem da configuração escolhida e das credenciais do terminal.

## Testar uma imagem

1. Escolha a fonte dos modelos. `ollama` inclui Clef 9B, BLIP e Laya; `local` usa Clef via llama-server. `smoke_mock` é apenas simulação.
2. Use o contrato compartilhado Ollama quando comparar com Clef no Ollama. O contrato original tem 49 alternativas de marca e não cabe no limite de 26.
3. Marque os modelos. CPU/CUDA e threads afetam apenas os modelos Python; Ollama gerencia o próprio hardware.
4. Selecione veículo ou pessoa, envie uma imagem e clique em **Avaliar imagem**.
5. Acompanhe cada resultado. Baixe o JSON para guardar respostas, erros e tempos.

A imagem enviada fica salva em `.artifacts/studio/<id>/input.png`. A aplicação padroniza RGB e tamanho máximo de 1024 px para inferência. Uma única tarefa roda por vez; os modelos são avaliados sequencialmente e liberados ao terminar. É possível cancelar a tarefa sem fechar o painel. Isso encerra o worker local; uma requisição já recebida pelo Ollama ou outro servidor pode continuar lá.

O tempo de preparação é separado da chamada de inferência. Não há aquecimento no modo interativo; o primeiro pedido HTTP pode incluir o carregamento interno do servidor. Não compare esses números diretamente com os tempos aquecidos do benchmark. Sem gabarito, o painel avalia validade do formato, não acurácia.

## Benchmark

O dataset portátil inclui 22 eventos com imagens versionadas. Escolha repetições e limite; zero usa todas as imagens. O worker chama o runner existente e o gerador de relatórios. Há aquecimento e alternância da ordem dos modelos conforme as regras do runner.

O painel mostra o registro enquanto os pesos carregam e o progresso conforme as previsões terminam. Ao concluir, **Resultados** lista PNG/PDF, CSV/JSON e o relatório Markdown. Arquivos de uma rodada cancelada permanecem em disco; não são apresentados como avaliação completa. Uma falha antes da primeira previsão aparece no registro.

O dataset portátil serve para validar o fluxo. Não é um conjunto confirmatório de qualidade. Para datasets externos e novos modelos, use o CLI documentado no [guia de comparação](COMPARISON_SETUP.md); o painel usa somente os datasets explicitamente cadastrados em `Studio.manifests`.

## Diagnóstico de erros

- **Serviço indisponível:** verifique `ollama list`. Se necessário, abra o aplicativo Ollama ou o serviço do sistema.
- **Endereço em uso:** pode já haver um Ollama em execução. Não inicie uma segunda instância sem verificar.
- **Modelo ausente:** execute `ollama pull clef-flash:9b`.
- **`ollama ps` vazio:** nenhum modelo está carregado. Não prova falta de RAM ou erro de instalação.
- **HTTP 500:** veja o erro individual do modelo e o registro. No backend HTTP local sem token, o painel mostra os detalhes devolvidos pelo servidor. Erros remotos continuam sanitizados.
- **Memória insuficiente:** identifique a mensagem de alocação do servidor. RAM total, RAM disponível, VRAM e configuração do runtime influenciam o carregamento; um erro em outra máquina não diagnostica a sua.
- **BLIP/Laya não carregam:** instale o backend CPU/GPU e confira os pesos e o submódulo. O painel não instala pesos de modelos Python automaticamente.

O diagnóstico consulta `/api/version`, `/api/tags` e `/api/ps` do Ollama local. Não faz inferência e informa isso explicitamente. Para compartilhar um problema, baixe o JSON do teste e copie a mensagem do modelo. Não é necessário reiniciar o Ollama repetidamente.

## Arquivos e retomada

Cada tarefa recebe uma pasta nova em `.artifacts/studio/`, fora do Git. Contém pedido, estado, registro e imagem enviada (modo interativo), ou execução congelada e relatório (benchmark). Não publique imagens privadas junto com os registros.

Recarregar a página mantém o acompanhamento da última tarefa pelo identificador salvo no navegador. Um fechamento normal do servidor cancela o worker que ele iniciou. Não há retomada automática de workers após encerramento forçado do servidor/sistema; para retomar uma avaliação parcial, use o CLI com `--resume` e as mesmas configurações, conforme o guia de comparação.

## Interfaces anteriores

`web/server.py` e `web/benchmark_server.py` continuam disponíveis como referência e compatibilidade. O ponto de entrada atual é `python -m vision_bench studio`. O iniciador `iniciar_clef_ollama` já abre o painel atual.

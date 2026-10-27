# Comparações de perguntas e requests

Em **Avaliar dataset**, selecione os modelos e escolha **O que comparar?**. Antes de executar, abra **Perguntas, requests e métricas de cada variante** para ver as entradas efetivas, alternativas, formato esperado e quantidade de avaliações. O relatório repete essas informações e o JSONL registra o request sem pixels em cada evento.

| Modelo | Variante | Entrada e execução | Saída |
|---|---|---|---|
| BLIP | short_open | Pergunta VQA curta; batch das perguntas da foto | Texto, até 12 tokens, normalização estrita |
| BLIP | short_options | Pergunta com nomes das alternativas | Texto, até 12 tokens |
| BLIP | long_descriptions | Alternativas e descrições mais longas | Texto, até 12 tokens |
| BLIP | structured_json | Pergunta pedindo objeto com uma chave e classe válida | JSON textual, até 64 tokens; sem conserto de JSON inválido |
| BLIP | short_serial | Mesma pergunta curta, uma geração por atributo | Texto, até 12 tokens |
| Laya | native_short | Perguntas `choice`, alternativas curtas, request conjunto | `answers.<atributo>.choice` |
| Laya | native_descriptive | Mesmas chaves; descrições mais longas das alternativas | Escolha nativa |
| Laya | native_serial | Mesmas alternativas curtas, um request por atributo | Escolha nativa |

BLIP e Laya têm arquiteturas e interfaces distintas. O batch do BLIP repete a imagem por pergunta; o request conjunto do Laya entrega as perguntas juntas. O número de chamadas é o número de invocações de `generate`/`predict`, não pacotes HTTP ou forwards internos. Para tarefas históricas isoladas, somente os atributos daquela tarefa são perguntados; a pergunta de alvo/caixa vermelha é preservada.

A variante JSON usa orçamento de geração maior; portanto a comparação mede o protocolo completo, incluindo esse orçamento, e não isola exclusivamente o texto do prompt. Todas as gerações usam `do_sample=False`. Laya registra `head_max_len`; excesso de orçamento produz falha explícita, sem truncamento silencioso. As variantes carregam o mesmo checkpoint fixado por SHA de cada modelo.

## Métricas

- Qualidade: acurácia por atributo, precisão macro, recall macro e F1 macro; classes com suporte no gabarito, divisão por zero igual a zero. Acurácia completa só para fotos com todos os atributos anotados. Campos sem gabarito são excluídos.
- Formato: aderência bruta às classes, formato válido e ambiguidade antes da normalização/filtro de ausência. Não confundir escolha nativa do Laya com habilidade de gerar JSON. Aderência pode ser alta por construção nessa interface. Ambiguidade detecta nomes explícitos de classes conflitantes; não é análise semântica humana.
- Desempenho: média, desvio, p50/p95, throughput sequencial, chamadas de inferência, tempo de carga/aquecimento separado, CPU e RSS amostrada. Tempo por atributo só é apresentado para chamadas isoladas; tempo conjunto não é dividido artificialmente.
- Comparações: diferenças B − A nas mesmas fotos, acertos exclusivos e IC 95% por bootstrap de grupos de origem (2.000 reamostragens, seed 20261027). Não há teste de significância, correção por múltiplas comparações ou seleção automática do melhor modelo.

Qualidade usa somente a primeira repetição de cada evento; falhas contam como erro. Desempenho usa todas as repetições medidas, com latência resumida sobre chamadas bem-sucedidas. Gráficos exportam PNG a 300 dpi e PDF: qualidade/formato/tempo por atributo, F1 por classe e matriz de confusão por variante. CSV e JSON preservam os dados subjacentes. Custo monetário e energia não são medidos.

Escolha prompts no recorte de desenvolvimento e confirme a escolha no recorte de avaliação. O teste 6 contém imagens também presentes no teste de cores; seus resultados não constituem uma validação independente. Os gabaritos históricos continuam exploratórios.

## Apresentação dos gráficos

Os gráficos simples usam Laya azul (`#3378FF`) e BLIP laranja (`#F78C25`), barras horizontais e acertos/total por classe. A comparação simples identifica os protocolos padrão na legenda e não agrega variantes distintas. Tempo mediano fica em um gráfico separado. O seletor **Técnicos** mantém F1, matrizes e comparações de variantes. PNG 300 dpi e PDF estão disponíveis nos dois modos. Para atualizar somente a apresentação dos relatórios concluídos, preservando inferências e métricas, execute `python -m scripts.refresh_charts`.

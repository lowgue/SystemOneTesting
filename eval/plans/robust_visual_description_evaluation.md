# Plano de avaliação empírica de descrições visuais

Estado: etapa automática preliminar executada; avaliação confirmatória ainda não executada. Data: 2026-10-26.

## Resultados preliminares sem anotação humana

O [piloto automatizado](../experiments/automated_pilot_20261026/report.md) executou 452 requisições exploratórias: cinco blocos de desempenho com ordem alternada, seis configurações completas por modelo, degradações, controles sintéticos e fila. Os dados históricos continuam fora do futuro teste confirmatório.

Dois achados afetam a preparação da próxima rodada:

- A lista de 48 marcas mais desconhecida provoca truncamento da instrução e de opções no orçamento padrão do Laya. A [auditoria de orçamento](../experiments/laya_option_budget_20261026/report.md) testa a integridade da entrada; qualquer nova configuração ainda precisa de validação de qualidade.
- No lote adicional de seis imagens, aumentar a cabeça de 256 para 512 tokens eliminou os avisos de truncamento (6/6 para 0/6), alterou três conjuntos de respostas e aumentou a mediana de 2,47 s para 3,08 s. Esse teste confirma integridade de entrada, não melhora semântica. Não declarar uma vantagem definitiva de custo a partir da configuração que recebia entradas cortadas.
- A [simulação de poder](../experiments/paired_power_20261026/report.md), sem usar gabaritos, mostra que 600 positivos por domínio não garantem 80% de poder. Para detectar uma diferença real de 5 pontos contra zero, com α=0,025 por domínio e 20% de discordância entre modelos, o poder simulado foi 67,2%. Na grade testada, 900 pares foi o primeiro tamanho com pelo menos 80% nesse cenário. Com 30% de discordância, foi 1.200; com 50%, 2.400. São cenários de planejamento, não estimativas da discordância real.

Assim, os 600 positivos continuam sendo uma meta inicial condicionada ao piloto anotado. Não abrir o teste final antes de fechar a análise de poder e a família de critérios de aceitação. A simulação distingue detectar diferença contra zero de demonstrar vantagem superior à margem prática de 5 pontos.

## 1. Perguntas que o estudo deve responder

1. Qual configuração de cada modelo entrega uma descrição completa e factual com maior frequência?
2. Qual oferece menor custo por descrição que atende ao contrato, mantendo a mesma exigência de qualidade?
3. Quando o modelo deve omitir um atributo, rejeitar a descrição ou encaminhar o caso para revisão?
4. A conclusão muda entre pessoas, veículos, condições difíceis e execução operacional?

BLIP VQA e Laya Vision serão avaliados como pipelines completos. A interface pode ser diferente: perguntas abertas para BLIP e perguntas tipadas para Laya. Os atributos exigidos, as imagens, as regras de aceitação e os gabaritos serão iguais. Não presumir que nenhum dos modelos serve para geração de legenda livre; o caminho principal extrai atributos e compõe frases determinísticas.

## 2. Problemas dos testes anteriores a corrigir

- O Teste 7 tem apenas 32 imagens positivas; diferença de duas imagens não sustenta uma escolha definitiva.
- Imagens usadas para explorar prompts foram reutilizadas na avaliação.
- Há rótulos antigos derivados da categoria de origem e inconsistências entre metadados e conteúdo visível.
- A revisão visual do agente não substitui anotação humana independente.
- O normalizador histórico pode confundir resposta incompreendida com abstenção. Respostas como `car` não devem virar uma carroceria desconhecida corretamente reconhecida.
- Tempo de parede em CPU não mede diretamente energia, custo monetário ou tempo efetivo de CPU.
- A execução anterior não mediu detecção, recorte, filas ou integração real com CaptionerManager.

Os Testes 1–7 permanecem como exploração histórica. Suas imagens não entram no teste final novo, incluindo cópias, recortes e frames próximos da mesma origem.

## 3. Contrato da descrição e gabarito

Congelar `description_contract_v1.json` antes de coletar resultados. Os campos abaixo são a proposta inicial baseada no repositório; não representam uma integração já validada com CaptionerManager.

| Domínio | Conteúdo exigido no caso positivo |
|---|---|
| Veículo | Presença, cor predominante, carroceria e marca quando identificável visualmente |
| Pessoa | Presença, capacete, objeto transportado e vestimenta superior quando visíveis |

Cada evento positivo exige a tentativa de todos os atributos. O relatório principal pontua o conjunto inteiro; métricas individuais explicam as falhas. Não escolher o modelo somente por cor ou outro campo isolado.

Regras de anotação:

- Identificar o objeto-alvo com bounding box; em cenas com vários objetos, não usar o conceito ambíguo de “mais proeminente” como único identificador.
- Para cada atributo registrar valor, visibilidade (`visible`, `not_visible`, `ambiguous`), valores aceitáveis quando houver ambiguidade e justificativa curta.
- A marca registrada nos metadados de origem não é automaticamente uma informação legível na imagem. Separar identidade conhecida da fonte de evidência visual disponível ao modelo.
- Distinguir `none` (não transporta objeto), `not_visible` (não se pode observar), `other_visible` (objeto visível fora das classes) e `parse_error` (resposta não interpretável).
- Permitir múltiplos objetos transportados e combinações de vestimenta, quando exigidas pelo contrato. Não obrigar “mochila” e “bolsa” a serem alternativas exclusivas se ambas estiverem visíveis.
- Se o contrato exigir múltiplos valores, ambos os pipelines devem suportá-los. No Laya, usar decisões binárias por categoria ou outra representação nativa equivalente; no BLIP, perguntas adicionais se necessárias. Contabilizar todas essas decisões no tempo e custo. Quatro campos semânticos não significam necessariamente quatro inferências.
- Definir se ferramenta, telefone e capacete na mão contam como transporte. Recomenda-se incluí-los em categorias próprias ou em `other_visible`, com a regra congelada antes da avaliação.
- Definir exemplos de fronteira para cor, SUV/hatch, camisa/jaqueta, colete/refletivo, capacete/boné e roupa não observável.
- Para marca/cor/carroceria incertas, a saída deve declarar a abstenção estruturada e a frase omitir o atributo. Uma resposta não interpretável não conta como abstenção correta.
- A frase não pode acrescentar ação, identidade, localização ou outro fato não coberto por evidência anotada.

Exemplos: `SUV preto da Jeep.`; `Pessoa de jaqueta, com mochila e sem capacete.`; `Veículo preto; marca não identificável.` A opção entre omitir o campo e declarar sua indisponibilidade será uma regra única do renderizador.

Dois anotadores humanos rotulam cada imagem de forma independente, sem ver respostas dos modelos. Um terceiro resolve divergências. Registrar concordância bruta, divergências por campo e alterações após adjudicação. Se esses revisores não estiverem disponíveis, o trabalho continua como piloto exploratório, sem promover rótulos do agente a gabarito confirmatório.

## 4. Amostra e separação dos dados

Meta inicial: **2.400 eventos independentes**, metade de cada domínio.

| Partição | Pessoas | Veículos | Total | Finalidade |
|---|---:|---:|---:|---|
| Desenvolvimento | 150 | 150 | 300 | Prompts, normalização e política de recorte |
| Calibração | 150 | 150 | 300 | Limiares e regra de aceitação/rejeição |
| Teste final reservado | 900 | 900 | 1.800 | Comparação confirmatória |

Em cada domínio do teste final: **600 casos positivos e 300 controles negativos**. Nas outras duas partições: 120 positivos e 30 negativos por domínio. O controle negativo significa ausência do objeto daquele domínio; pode conter objetos do outro domínio.

Unidade independente: cena/evento de origem. Para a estimativa principal, usar um evento-alvo por cena. Se vários eventos vierem de um vídeo, câmera, objeto ou sequência relacionada, agrupá-los e preservar o grupo inteiro em uma única partição. Deduplicar por origem, hash e similaridade perceptual; auditar os pares suspeitos. Dividir por grupos antes de produzir recortes ou degradações.

Fontes e composição:

- Priorizar cenas reais do uso pretendido, com origem e autorização documentadas; complementar lacunas com imagens externas de origem registrada.
- Incluir dia/noite, iluminação ruim, baixa resolução, oclusão, objeto pequeno, vários objetos, fundos semelhantes e classes raras.
- Negativos difíceis: capacetes sem pessoa, manequins, cartazes/fotos de pessoas, brinquedos e imagens de veículos sem veículo físico presente. O contrato deve definir como tratar representações.
- Fixar quotas de classe e dificuldade antes de avaliar. Registrar contagens e interseções; não afirmar que todas as combinações possíveis estão balanceadas.
- Reportar a amostra controlada separadamente de uma amostra operacional representativa. Só calcular a média ponderada de produção quando a frequência real dos estratos for conhecida e fixada previamente.
- Não chamar o teste de “inédito para o pré-treinamento”: é novo para esta avaliação; contaminação de pré-treinamento pode ser desconhecida.

Justificativa do tamanho: 600 positivos independentes por domínio permitem aproximadamente ±4 pontos percentuais em uma proporção próxima de 50%, com 95% de confiança. Isso é uma referência de precisão, não garantia de poder para detectar uma diferença de 5 pontos. Correlação entre eventos pode alargar o intervalo.

Nos 300 negativos por domínio, zero erros corresponde a limite superior binomial unilateral de 95% de aproximadamente 0,994% (`1 - 0.05**(1/300)`). Isso não prova taxa zero e só se aplica à população amostrada e à hipótese de independência.

O piloto estima a frequência de discordância entre modelos e a correlação por fonte. Simular o poder do contraste pareado para uma diferença prática de 5 pontos, meta de 80%, antes de abrir o teste final. Se a amostra prevista for insuficiente, ampliar o plano e congelar o novo tamanho nessa etapa. Não coletar sucessivamente até obter significância.

## 5. Escolha de prompts e modo de inferência

Usar somente desenvolvimento. Teto inicial: seis configurações completas por modelo, três variações de formulação combinadas com dois modos de execução. Todas devem tentar gerar todos os campos do domínio.

| Modelo | Formulações candidatas | Modos candidatos |
|---|---|---|
| BLIP | Perguntas curtas genéricas; perguntas curtas específicas do objeto; perguntas curtas com instrução de incerteza | Quatro perguntas em batch; presença primeiro e demais perguntas apenas se positivo |
| Laya | Critérios curtos; critérios descritivos curtos; critérios com regra explícita de visibilidade | Todos os campos em uma chamada; presença primeiro e demais campos apenas se positivo |

Os exemplos de melhor desempenho anterior servem como candidatos, não como vencedores presumidos. Para Laya, manter a mesma taxonomia em todos os candidatos; reduzir opções com base no gabarito é proibido. Lista operacional menor só vale em uma tarefa separada, aplicável igualmente à normalização do BLIP e incluindo `other_visible` e abstenção.

No desenvolvimento, comparar quadro completo e recorte do alvo anotado com a configuração-base. Selecionar a política de entrada antes da rodada final de prompts, evitando cruzar toda a matriz e escolher por ruído. Um recorte anotado mede o extrator em condições controladas; não comprova a qualidade do detector.

Selecionar uma configuração por modelo e domínio usando acerto do evento completo, erros factuais e custo, nessa ordem. Documentar todas as tentativas. Congelar prompts, critérios, parser, renderizador e versões em arquivo com hash. Não usar o conjunto de teste final para corrigir o parser.

Se uma geração livre por BLIP for incluída, tratá-la como braço exploratório separado: a variante VQA atual não demonstrou capacidade de gerar a descrição completa. Não comparar uma frase curta incompleta com um conjunto completo de atributos como se fossem a mesma entrega.

## 6. Calibração e política de saída

Usar somente calibração para escolher a regra que aceita, omite ou rejeita resultados. Avaliar o pipeline sem limiares e o pipeline com rejeição em separado.

- Não pressupor que probabilidades do Laya estejam calibradas.
- Caso se use score de confiança do BLIP, definir o cálculo antes e validar sua relação com erro. Não compará-lo numericamente à probabilidade do Laya sem validação.
- Se um modelo não tiver score útil, registrar essa limitação; não inventar confiança nem usar o gabarito para decidir revisão.
- Congelar os limiares por atributo/domínio e informar quanto dado de calibração sustenta cada um. Classes escassas permanecem sem conclusão específica.
- O filtro de ausência deve seguir o mesmo contrato para ambos; guardar respostas anteriores e posteriores ao filtro. Uma falsa ausência em caso positivo conta como falha do evento inteiro.
- Informar precisão e cobertura juntas. Rejeitar quase todas as imagens não pode produzir um “vencedor” por precisão aparente.

## 7. Teste final: qualidade da descrição

Executar uma vez cada configuração congelada em todos os eventos reservados. Comparar modelos de forma pareada: a mesma imagem e o mesmo alvo para ambos. As repetições de desempenho não acrescentam observações independentes à qualidade.

Métricas principais, separadas por domínio:

1. **Acerto do evento positivo completo:** todos os atributos visíveis obrigatórios corretos, abstenções corretas nos não observáveis e frase sem afirmação falsa. Denominador: todos os positivos, inclusive rejeitados, falhas e timeouts.
2. **Afirmações indevidas em negativos:** frequência de descrição ou atributo positivo quando o alvo não existe.
3. **Descrição positiva aceita corretamente / todos os eventos positivos:** taxa útil efetiva da política de aceitação. Rejeitados permanecem no denominador.
4. **Custo por descrição positiva aceita e correta:** custo total do lote, incluindo negativos, dividido pelo número de descrições positivas que passam o contrato e a política de aceitação. Um controle corretamente descartado não conta como descrição útil entregue.

Métricas secundárias:

- Acurácia, precisão, recall e matriz de confusão por atributo/classe; micro e macro identificados explicitamente.
- Cobertura dos atributos visíveis; abstenções desnecessárias; respostas inválidas; erros e timeouts.
- Erros por recorte errado, alvo confundido, percepção, parser, regra de ausência e renderização.
- Precisão das descrições positivas emitidas e aceitas, cobertura nos positivos, curva erro versus cobertura e calibração quando houver scores válidos. Falsas descrições emitidas nos negativos entram como erros na precisão; descartes corretos de negativos não a aumentam.
- Resultados por dificuldade, origem e classe; não ocultar desempenho ruim em pessoas numa média dominada por veículos.
- Controles negativos fora do denominador principal de extração positiva.
- Auditoria humana cega da frase e dos atributos de todos os eventos finais, com ordem aleatória e sem identidade do modelo. Para o renderizador determinístico, a tabela de fatos é a evidência primária; a auditoria detecta acréscimos ou omissões da frase.

## 8. Incerteza e decisão estatística

- Publicar numerador, denominador e intervalo de 95% para cada proporção principal.
- Usar intervalo Wilson como resumo de proporções independentes e intervalo binomial exato para eventos raros. Se houver agrupamento, reportar também intervalos por reamostragem de grupos.
- Para diferença entre modelos, calcular intervalo pareado: reamostrar os mesmos grupos/cenas para os dois modelos e manter os estratos. Usar 10.000 reamostragens e seed registrada. Não decidir pela sobreposição de intervalos individuais.
- Definir duas comparações primárias, pessoas e veículos. Para uma afirmação conjunta sobre os dois domínios, ajustar as duas comparações, por exemplo usando intervalos de 97,5% em cada uma. As demais análises são exploratórias.
- Margem de relevância prática proposta: 5 pontos percentuais no acerto positivo completo. Declarar superioridade prática quando o limite inferior do intervalo da diferença superar +5 pontos; superioridade estatística apenas acima de zero deve ser descrita como tal.
- Para equivalência prática, exigir que o intervalo inteiro fique entre −5 e +5 pontos. Nesse caso, custo pode desempatar, desde que os requisitos de qualidade sejam atendidos.
- Se o intervalo permitir vantagem relevante para qualquer lado, a conclusão é inconclusiva. Não forçar um vencedor.

Referências metodológicas: [NIST: intervalos de confiança binomiais](https://itl.nist.gov/div898/handbook/prc/section2/prc241.htm) e [SciPy: bootstrap e amostragem pareada](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html). O bootstrap por grupos é uma decisão deste protocolo para respeitar a estrutura dos dados.

## 9. Desempenho, custo e operação

Dois cenários de qualidade devem ser identificados no relatório:

- **Extrator:** alvo e recorte corretos fornecidos por anotação; isola a capacidade do modelo.
- **Fluxo operacional:** quadro → detector fixo → recorte previsto → atributos → normalização → descrição. Usar o mesmo detector, configuração e associação alvo/evento nos dois modelos. Falhas de detecção entram no custo e na qualidade desse cenário.

A configuração é escolhida em desenvolvimento e congelada. Se não houver integração disponível, chamar o segundo cenário de harness operacional simulado, nunca de integração real com CaptionerManager.

Medição de desempenho após congelamento, em subconjunto fixo de 200 eventos finais (100 por domínio), com distribuição de dificuldade registrada:

- Cinco blocos completos por modelo, alternando a ordem AB/BA e randomizando a ordem das imagens com seeds conhecidas. Um modelo por vez no ensaio de CPU; não disputar recursos entre modelos.
- CPU, modelo de processador, RAM, sistema, threads, precisão, temperatura, frequência, versões e revisões dos pesos registradas. Anotar processos concorrentes relevantes.
- Separar carga do modelo, warm-up, execução aquecida e primeira requisição. Comparação de carga fria exige controlar cache do sistema; sem isso, chamar de tempo de carga observado.
- Medir latência p50/p90/p95/p99, tempo de CPU, pico de RAM, throughput e taxa de erro. Indicar a incerteza do p99; não prometer cauda confiável com poucas cenas independentes.
- Medir concorrência 1, 2 e 4 em teste separado para fila/throughput, sem misturar com a latência isolada.
- Registrar tempo por etapa e tempo total operacional. Incluir tentativas, rejeições e fallback no custo, quando existirem.
- Não executar cinco inferências em produção e pontuar a melhor delas. Se repetição/votação virar parte do produto, testar como pipeline adicional com todo o custo contabilizado.

Tabela de custo:

| Medida | Cálculo |
|---|---|
| Tempo por 1.000 eventos | Soma dos tempos operacionais / eventos × 1.000 |
| Energia por evento | Energia total medida / eventos, com escopo do sensor informado |
| Energia incremental | Energia medida menos baseline ocioso medido em janela equivalente |
| Custo local | kWh × tarifa informada; custo de hardware opcional separado, com amortização explícita |
| Custo em instância | Horas efetivas faturadas × preço da instância, com configuração e data |
| Custo por descrição aceita correta | Custo total do lote / descrições positivas aceitas e corretas |

Usar medidor de tomada para máquina inteira ou contador disponível com escopo identificado; energia de pacote CPU não é energia da máquina inteira. Se não houver medição, reportar tempo e RAM, deixando energia e reais como não medidos. Não converter redução de latência automaticamente em redução de conta elétrica.

Sem prevalência operacional conhecida, calcular custos por estrato e cenários explícitos de mistura; não usar a proporção artificial de controles como estimativa de custo de produção. Se nenhum evento passar o contrato, custo por entrega correta é indefinido/inviável, nunca zero.

## 10. Robustez adicional

Depois da escolha final, executar um conjunto separado de desafios: degradação de resolução, JPEG, brilho, blur, oclusão, troca de fundo e múltiplos objetos. As transformações devem preservar o rótulo ou receber nova anotação de visibilidade. Cada variante continua ligada à imagem original e não aumenta o número de cenas independentes.

Para fontes/câmeras não vistas, reservar grupos inteiros antes de qualquer ajuste. Relatar transferência de domínio separadamente. Se o pipeline for alterado depois desses resultados, a nova versão precisa de um novo teste reservado.

## 11. Critérios de aceitação propostos

Estes são alvos iniciais de produto, não propriedades demonstradas dos modelos:

- Limite inferior unilateral de 95% do acerto positivo completo ≥80%, por domínio.
- Limite inferior unilateral de 95% da precisão das descrições aceitas ≥95%, com cobertura positiva ≥80%.
- Limite superior unilateral de 95% de descrições indevidas em negativos ≤1%, por domínio.
- Latência e RAM dentro do orçamento operacional, que será fixado antes de abrir o teste final. Enquanto esse orçamento não existir, comparar qualidade/custo sem declarar atendimento ao SLA.

Esses limites individuais não fornecem confiança conjunta de 95% para todos os critérios. Se a conclusão formal for “todos os requisitos atendidos”, registrar antecipadamente o ajuste de multiplicidade e recalcular o tamanho dos controles; 300 negativos pode deixar de ser suficiente.

Um modelo pode vencer a comparação e reprovar nesses critérios. Se ambos reprovarem, o resultado é “nenhum atende”; o próximo experimento pode incluir outro modelo ou revisão humana, com custo explícito. Não relaxar as metas após observar o teste para aprovar o vencedor.

## 12. Execução e entregáveis

| Etapa | Saída verificável | Condição para avançar |
|---|---|---|
| Contrato | Taxonomias, exemplos de fronteira, regras de frase e métricas | Sem campos ambíguos ou requisitos implícitos |
| Coleta/anotação | Manifesto, hashes, fontes, grupos, rótulos independentes e adjudicação | Qualidade revisada; splits sem vazamento |
| Desenvolvimento | Matriz completa de candidatos e resultados | Uma configuração congelada por modelo/domínio |
| Calibração | Regras de aceitação, análise de poder e tamanho final | Nenhum limiar pendente; teste final ainda fechado |
| Teste final | Predições brutas por evento, atributos, frase, erros e métricas | Integridade conferida; nenhuma edição por resultados |
| Custo/robustez | Logs de recursos, tempos por etapa e desafios | Escopo de cada medição documentado |
| Decisão | Relatório com intervalos, custo/qualidade e reprovações | Conclusão compatível com incerteza e metas |

Arquivos propostos em diretório de execução novo, sem sobrescrever resultados anteriores:

```text
eval/experiments/<run_id>/
  description_contract_v1.json
  manifest.json
  annotation_audit.jsonl
  split_groups.json
  frozen_config.json
  predictions.jsonl
  performance.jsonl
  energy.jsonl                 # somente se houver sensor
  metrics.json
  report.md
```

Persistir progresso atomicamente e permitir retomada por evento, registrando tentativas anteriores. Fixar commits dos modelos e dependências, hashes do código/configuração/imagem, sementes, prompts, opções e parâmetros de geração. Timeouts e erros entram no denominador; nenhuma linha pode desaparecer silenciosamente.

Estimativa de inferência: os tempos atuais somados são aproximadamente 5,5 s por evento para os dois modelos. Uma passagem pelos 1.800 eventos finais corresponde a cerca de **2,75 horas** na máquina atual. Esse número é só uma ordem de grandeza: não inclui coleta/anotação, detector, novos atributos, busca de prompts, repetição de desempenho, carga ou contenção. Seis candidatos por modelo nos 300 eventos de desenvolvimento correspondem a aproximadamente mais 2,75 horas usando a mesma aproximação; devem ser confirmadas no piloto. A revisão humana é uma etapa de trabalho própria, sem prazo estimado sem conhecer os revisores.

Primeira ação de execução: fechar contrato e manual de anotação, montar as 300 cenas de desenvolvimento e validar o harness com um pequeno lote já adjudicado. Não iniciar a rodada final grande com os gabaritos antigos.

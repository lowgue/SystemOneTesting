# Relatório empírico: Laya Vision x BLIP VQA

## Escopo e leitura rápida

Esta avaliação compara `thaitea/laya-vision` e `Salesforce/blip-vqa-base` em 397 imagens locais de veículos. Ambos receberam as mesmas quatro perguntas base em cada imagem: presença de veículo, marca, cor predominante e carroceria. Só respostas com rótulo de referência foram pontuadas. Há 538 campos rotulados no total: 67 de presença, 67 de marca, 337 de cor e 67 de carroceria. Portanto, a acurácia agregada é micro, ponderada por cada resposta rotulada; a cor representa 62,6% dos rótulos.

| Medida | Laya Vision | BLIP VQA |
|---|---:|---:|
| Acertos agregados | 370/538 (68,77%) | 254/538 (47,21%) |
| Média simples das quatro acurácias por pergunta | 65,7% | 45,5% |
| Latência média por imagem | 1.976 ms | 3.980 ms |
| Latência mediana por imagem | 1.892 ms | 3.845 ms |
| p90 por imagem | 2.296 ms | 4.494 ms |
| p95 por imagem | 2.472 ms | 4.698 ms |
| Erros de inferência na rodada | 0/397 | 0/397 |

Na mediana, o BLIP levou aproximadamente 2,03 vezes o tempo do Laya (cerca de 103% mais tempo); equivalentemente, o Laya usou aproximadamente 50,8% menos tempo. Estes valores são de uma máquina específica e da implementação descrita aqui.

## Histórico de todas as rodadas salvas

| Etapa | Imagens | Campos pontuados | Laya | BLIP | Mediana Laya / BLIP |
|---|---:|---:|---:|---:|---:|
| Conjunto curado inicial | 7 | 28 | 17/28 (60,71%) | 16/28 (57,14%) | 1.416 / 3.017 ms |
| Veículos Stanford + locais | 67 | 208 | 116/208 (55,77%) | 140/208 (67,31%) | 2.228 / 2.759 ms |
| Primeira expansão de cor | 247 | 388 | 227/388 (58,51%) | 285/388 (73,45%) | 1.517 / 945 ms |
| Conjunto colorido atual | 397 | 538 | 370/538 (68,77%) | 254/538 (47,21%) | 1.892 / 3.845 ms |

Os quatro pontos estão no [gráfico histórico](laya-blip-history.svg) e nos dados [laya-blip-history.json](laya-blip-history.json). Não são quatro medições de uma mesma configuração: os dados, as respostas pontuadas por imagem, as cores e as instruções mudaram. O gráfico registra o que foi observado em cada etapa, não uma curva de progresso ou regressão do modelo.

### O que mudou de uma etapa para a seguinte

- **7 → 67 imagens:** a amostra foi ampliada para incluir fotos Stanford e locais. O conjunto de sete era curado; o de 67 tem apenas sete imagens com gabarito de cor. A acurácia agregada do Laya caiu 4,94 pontos percentuais e a do BLIP subiu 10,17 pontos. Isso é compatível com a troca de amostra e o aumento de rótulos, não comprova uma mudança de qualidade do modelo.
- **67 → 247 imagens:** foram adicionadas 180 imagens de cor. A rodada passa a ter 187 campos de cor (incluindo sete imagens curadas) entre 388 campos. BLIP ficou com 73,45% no agregado e 80,21% em cor; Laya, 58,51% e 61,50%. A proporção maior de perguntas de cor mudou o peso da acurácia agregada. Nessa rodada, imagens do bloco `car_color` guardam somente a previsão de cor, ao passo que os demais registros têm respostas de tarefas múltiplas. Logo, as latências combinam cargas diferentes.
- **247 → 397 imagens:** o bloco de cores passou de 180 para 330 imagens, distribuídas em 11 classes com 30 cada. A rodada atual executa as quatro perguntas base em todas as imagens, embora só pontue os campos anotados; nos registros da etapa anterior, as imagens `car_color` contêm apenas a previsão de cor. O BLIP mudou de 80,21% (150/187) para 48,96% (165/337) na métrica agregada de cor, e de 50,75% para 8,96% em marca. O prompt atual acrescenta a lista longa de alternativas à pergunta textual. As respostas brutas e o diagnóstico mostram que BLIP frequentemente ignora essa instrução. Além disso, revisões exatas dos modelos não foram registradas, então não é possível atribuir qualquer diferença de saída exclusivamente ao prompt ou ao dataset.

Na passagem de 247 para 397, o Laya passou de 58,51% para 68,77% no agregado; o BLIP passou de 73,45% para 47,21%. Essa diferença não significa que o Laya “aprendeu” ou que o BLIP “piorou”: houve alteração no mix de tarefas, no protocolo de inferência das imagens coloridas e no formato do prompt. Para comparação de latência, a mudança é particularmente relevante: a mediana BLIP de 945 ms na rodada antiga é dominada por imagens `car_color` cuja execução salvou só a pergunta de cor; na atual, a mediana de 3.845 ms inclui as quatro perguntas processadas para cada imagem. O hardware está registrado para a rodada atual, mas não com o mesmo nível de detalhe nos arquivos históricos.

Também comparei arquivos com o mesmo nome e o mesmo rótulo entre os JSON das etapas 247 e 397. Nas **154 imagens compartilhadas** de cinco classes de cor, BLIP passou de 125/154 (81,2%) para 78/154 (50,6%); Laya, de 112/154 (72,7%) para 101/154 (65,6%). Nas 60 imagens compartilhadas com marca, presença e carroceria anotadas (180 campos), BLIP passou de 124/180 (68,9%) para 82/180 (45,6%), e Laya de 102/180 (56,7%) para 116/180 (64,4%). Portanto, acrescentar novas imagens/classes, por si só, não explica a queda de BLIP: as previsões ou o protocolo também diferiram nas imagens que se repetem. O diagnóstico da pergunta curta contra a lista de opções longa confirma sensibilidade do BLIP ao prompt; ainda assim, sem revisão dos pesos e prompt histórico exato, não dá para repartir causalmente a diferença entre prompt, checkpoint e execução.

### Resultados por pergunta em cada comparação visual

| Etapa | Modelo | Presença | Marca | Cor | Carroceria |
|---|---|---:|---:|---:|---:|
| 7 imagens | Laya | 7/7 | 4/7 | 4/7 | 2/7 |
| 7 imagens | BLIP | 7/7 | 1/7 | 5/7 | 3/7 |
| 67 imagens | Laya | 64/67 | 16/67 | 4/7 | 32/67 |
| 67 imagens | BLIP | 67/67 | 34/67 | 5/7 | 34/67 |
| 247 imagens | Laya | 64/67 | 16/67 | 115/187 | 32/67 |
| 247 imagens | BLIP | 67/67 | 34/67 | 150/187 | 34/67 |
| 397 imagens | Laya | 64/67 | 32/67 | 242/337 | 32/67 |
| 397 imagens | BLIP | 67/67 | 6/67 | 165/337 | 16/67 |

O aumento de exemplos anotados de cor também eleva o peso de cor no agregado: passou de 7/208 campos na etapa de 67 imagens para 187/388 na primeira expansão, e depois 337/538 na rodada atual. A tabela por tarefa é mais apropriada quando se quer comparar o perfil do que um único percentual agregado.

### Outros experimentos salvos no projeto

Estes não são rodadas Laya x BLIP e, por isso, não entram no gráfico de comparação entre os dois:

- **Laya Vision, 20 fotos Stanford (arquivo histórico):** marca 11/20, carroceria 11/20, presença 9/20 com limiar 0,5 ou 19/20 com limiar 0,2; média 1.439 ms/imagem, 4 threads. O limiar muda substancialmente a métrica de presença.
- **Laya Vision, 7 imagens de controle:** 19/28 (67,86%), com presença 6/7, marca 6/7, cor 4/7 e tipo 3/7; média 1.336 ms/imagem. É uma execução distinta do comparativo curado 7-imagens, que marcou 17/28 para Laya.
- **Laya textual, 33 casos em cinco domínios:** 16/33 (48,48%), média 1.001 ms por caso. Não é tarefa visual nem comparação com BLIP.
- **Primeira tentativa do teste visual de 7 imagens:** 0 respostas pontuadas por incompatibilidade de processador (`'list' object has no attribute 'to'`); não há métrica válida.
- **Clef-Flash:** o histórico mostra inicialização e chamadas exploratórias, mas não existe no repositório um resultado completo por imagem com acurácia/latência que possa ser comparado às tabelas acima. A latência de 38,8 ms mencionada anteriormente foi uma informação trazida pelo usuário sobre a API, não medição local deste benchmark.

## Resultados por pergunta

| Pergunta base | Laya | BLIP | n por modelo |
|---|---:|---:|---:|
| Há um carro? | 64/67 (95,52%) | 67/67 (100%) | 67 |
| Marca | 32/67 (47,76%) | 6/67 (8,96%) | 67 |
| Cor predominante | 242/337 (71,81%) | 165/337 (48,96%) | 337 |
| Carroceria | 32/67 (47,76%) | 16/67 (23,88%) | 67 |

Os intervalos Wilson de 95% são descritivos e não corrigem dependência entre imagens ou viés de seleção. Para presença: Laya 87,6–98,5%; BLIP 94,6–100%. Marca: Laya 36,3–59,5%; BLIP 4,2–18,2%. Cor: Laya 66,8–76,4%; BLIP 43,7–54,3%. Carroceria: Laya 36,3–59,5%; BLIP 15,3–35,3%.

## Subconjunto balanceado de cor

O conjunto dedicado a cor contém exatamente 330 imagens (`source=car_color`), 30 por classe: preto, azul, marrom, cinza/prata, verde, laranja, rosa, roxo, vermelho, branco e amarelo. Como as classes têm o mesmo tamanho, a acurácia micro e a média macro por classe coincidem nesse subconjunto.

| Classe | Laya | BLIP |
|---|---:|---:|
| Preto | 13/30 (43,3%) | 3/30 (10,0%) |
| Azul | 22/30 (73,3%) | 25/30 (83,3%) |
| Marrom | 20/30 (66,7%) | 0/30 (0%) |
| Cinza/prata | 23/30 (76,7%) | 19/30 (63,3%) |
| Verde | 24/30 (80,0%) | 25/30 (83,3%) |
| Laranja | 24/30 (80,0%) | 4/30 (13,3%) |
| Rosa | 23/30 (76,7%) | 7/30 (23,3%) |
| Roxo | 22/30 (73,3%) | 25/30 (83,3%) |
| Vermelho | 21/30 (70,0%) | 15/30 (50,0%) |
| Branco | 18/30 (60,0%) | 15/30 (50,0%) |
| Amarelo | 28/30 (93,3%) | 26/30 (86,7%) |
| **Total balanceado** | **238/330 (72,1%)** | **164/330 (49,7%)** |

Há ainda sete imagens locais/curadas com rótulo de cor, incluídas na pergunta de cor agregada (337); por isso o total de cor da tabela anterior difere do subconjunto balanceado. O gráfico por classe usa estritamente as 330 imagens, mantendo n=30.

## Por que a velocidade difere

- **Tipo de saída e decodificação:** Laya Vision é um modelo de decisão visual com respostas tipadas/seleção de alternativas; o caminho utilizado não precisa gerar uma frase. BLIP VQA é generativo: após codificar imagem e pergunta, `generate` produz tokens de resposta sequencialmente. A geração autorregressiva acrescenta custo por token.
- **Trabalho por imagem neste código:** Laya responde às quatro perguntas numa chamada do modelo por imagem. O código BLIP prepara quatro pares imagem/pergunta em lote, repetindo a imagem quatro vezes, e executa geração para cada resposta. O lote compartilha uma chamada, mas não elimina a codificação/atenção dos quatro pares nem a geração.
- **Tamanho e implementação:** a ficha do Laya informa cerca de 201 milhões de parâmetros e arquitetura visual compacta; a ficha do BLIP VQA base descreve backbone ViT-base. O custo real depende também de resolução, kernels, precisão numérica, tokenização e código de inferência.
- **Hardware e threads:** a rodada usou CPU Intel Core i5-12450H, 8 threads PyTorch, sem CUDA, PyTorch 2.14.1+cpu e Transformers 5.20.2. GPU, quantização, lote, número de threads e concorrência podem mudar latências e até inverter comparações.
- **O que foi cronometrado:** latência por imagem inclui processamento de imagem/perguntas e inferência; não deve ser comparada com um número de latência do model card ou API medido em hardware/protocolo diferente. A rodada completa levou 2.394,4 s (39 min 54 s), incluindo as duas passagens e custos de inicialização.

## Por que a acurácia difere e fatores que a afetam

1. **O prompt afeta especialmente o BLIP.** Para tarefas com alternativas, o benchmark acrescenta ao texto da pergunta uma lista extensa com rótulos e descrições, seguida da instrução para responder com o rótulo. A lista contém, conforme a tarefa, 12 marcas, 13 cores ou 10 carrocerias. Embora a quantidade de alternativas seja igual para os dois modelos, a interface não é equivalente: Laya recebe `choice` tipado e retorna uma das opções; BLIP recebe texto livre e precisa interpretar a lista e obedecer ao formato.
2. **Há evidência observável de não aderência.** No conjunto completo, BLIP produziu a resposta bruta `yes` para marca em 285/397 imagens (71,8%), que foi normalizada para `other` em grande parte dos casos; entre as 67 marcas pontuadas, só 6 ficaram corretas. `convertible` aparece com frequência nas respostas de carroceria. Em diagnóstico com a imagem `toyota_red.jpg`, a pergunta base de marca gerou `nissan`, mas ao acrescentar as alternativas a resposta virou `yes`; cor mudou de `red` para `blue`, e carroceria de `sedan` para `convertible`. Isso aponta para sensibilidade ao formato/prompt, não prova de limite intrínseco do BLIP.
3. **Contagem igual de alternativas não garante dificuldade igual.** O mesmo número de escolhas não equaliza o comprimento dos nomes, descrições, frequência de classes, clareza semântica, habilidade de seguir instruções nem o mecanismo de saída. As listas longas podem competir com a pergunta e com o conteúdo visual no contexto textual do BLIP.
4. **Acurácia depende dos rótulos e da imagem.** Reflexos, sombra, iluminação, balanço de branco, carro parcialmente visível, cor de fundo, múltiplos veículos e cor de pintura ambígua afetam a identificação. “Cinza” versus “prata” e “marrom” versus “laranja” também são fronteiras subjetivas. Marca/modelo e carroceria são atributos mais difíceis quando a imagem não mostra detalhes distintivos.
5. **Amostra e métrica importam.** As 330 imagens de cores são balanceadas, mas marca/carroceria/presença têm apenas 67 exemplos anotados, e o conjunto mistura Stanford Cars, fotos locais e um dataset de cores complementado. A acurácia agregada dá peso maior à cor. Os intervalos nas tarefas com n=67 são largos. Uma mudança de poucos exemplos nesses grupos altera bastante o percentual.
6. **Normalização também pode afetar o escore.** BLIP gera texto livre que é convertido para o vocabulário de rótulos por regras. Resposta válida expressa de outra forma pode virar `other`/erro; regras excessivamente permissivas também poderiam criar acertos falsos. A comparação exige auditoria das respostas brutas, não só do rótulo normalizado.

## Interpretação e como tornar a comparação mais justa

O resultado sustenta que, **neste CPU, dataset, prompt e pipeline**, Laya foi mais rápido e teve maior acurácia agregada; BLIP foi perfeito na pergunta binária de presença e superou Laya em azul, verde e roxo no subconjunto balanceado. Não sustenta que Laya seja universalmente mais preciso, nem que BLIP seja fraco em VQA geral. O resultado de marca do BLIP é especialmente contaminado pela baixa aderência às alternativas no prompt testado.

Uma próxima comparação de capacidade deve preservar as quatro perguntas, mas separar duas medidas: (a) VQA aberto usando apenas a pergunta base e avaliando a resposta semântica; (b) escolha fechada usando um protocolo nativo compatível com cada modelo (pontuação das alternativas para Laya e uma estratégia documentada para BLIP). Para BLIP, pode-se testar uma pergunta curta que mencione as categorias sem descrições prolixas, restringir/validar a saída e auditar respostas brutas. Deve-se executar a mesma ablação de prompt para ambos, manter as mesmas imagens, medir latência após aquecimento, registrar precisão e resolução e repetir execuções. Qualquer novo prompt constitui uma nova condição experimental e os resultados devem ficar separados desta rodada.

## Reprodutibilidade e limites

- Artefato completo por imagem/modelo: `laya-blip-car-color.json` ao lado deste relatório.
- Dataset: 397 imagens; quatro perguntas enviadas para cada imagem; campos pontuados apenas quando há referência anotada.
- Opções configuradas: presença 2, marca 12, cor 13, carroceria 10.
- Dispositivo `cpu`; PyTorch `2.14.1+cpu`; Transformers `5.20.2`; 8 threads. Os IDs de revisão dos checkpoints não foram salvos durante a execução; como os nomes Hub não foram fixados por hash, a identificação exata do snapshot usado não pode ser confirmada retrospectivamente.
- A latência não inclui o tempo de download dos pesos; os pesos estavam disponíveis em cache. A medição específica de `latency_ms` do código deve ser considerada ao reproduzir.
- Resultados são um benchmark exploratório de domínio restrito, não uma validação externa independente. A procedência/licenças das imagens e o manifesto devem ser consultados antes de redistribuição.

## Referências técnicas

Os números de parâmetros e a descrição de arquitetura abaixo referem-se às fichas públicas consultadas; como a revisão exata dos pesos desta execução não foi registrada, eles não substituem um manifesto de checkpoint fixado.

- [Model card Laya Vision](https://huggingface.co/thaitea/laya-vision)
- [Documentação oficial Transformers: BLIP](https://huggingface.co/docs/transformers/en/model_doc/blip)
- [Model card Salesforce/blip-vqa-base](https://huggingface.co/Salesforce/blip-vqa-base)

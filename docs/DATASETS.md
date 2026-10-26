# Avaliar uma foto ou um dataset

## Uma foto

Abra **Avaliar foto**, escolha os modelos, selecione a imagem e clique em **Avaliar foto**. O painel mostra os atributos previstos, tempo e resposta de cada modelo. Não precisa de gabarito. Esse fluxo é para inspecionar respostas; não estima acurácia.

## Várias fotos automaticamente

1. Abra **Avaliar dataset**.
2. Use o exemplo ou clique em **Importar minhas fotos**.
3. Selecione as fotos (até 2.000, até 16 MiB e 25 megapixels cada) e o tipo de cena: veículos ou pessoas.
4. Se tiver respostas esperadas, anexe um CSV. O botão **Baixar modelo de gabarito** cria um arquivo com os nomes das fotos selecionadas.
5. Clique em **Preparar dataset**. A interface mostra quantas fotos têm gabarito e quantas estão completamente anotadas.
6. Clique em **Iniciar avaliação do dataset**. Todas as fotos serão processadas; acompanhe progresso e falhas ou cancele se necessário.

Por padrão, cada foto é avaliada três vezes por modelo para medir desempenho. Em **Opções de medição**, altere as repetições ou limite a quantidade de fotos para uma rodada pequena. A acurácia sempre usa a primeira repetição por foto. Todos os modelos recebem o mesmo contrato e as mesmas imagens.

## Gabarito CSV

Exemplo de veículos, com vírgula ou ponto e vírgula:

```csv
arquivo,presenca,marca,cor,carroceria
carro1.jpg,sim,Toyota,preto,sedã
carro2.jpg,sim,Ford,branco,suv
paisagem.jpg,não,unknown,unknown,unknown
foto_sem_marca.jpg,,,azul,
```

São aceitos os cabeçalhos `filename,presence,make,color,body` ou os equivalentes em português acima. Para pessoas, use `arquivo,presenca,capacete,objeto,roupa` (ou `filename,presence,helmet,carrying,clothing`). Os valores aceitos aparecem na ajuda da importação; códigos como `yes`, `no`, `black`, `sedan` e respostas em português como `sim`, `não`, `preto` e `sedã` são reconhecidos.

O nome em `arquivo` deve coincidir com o nome da foto enviada. Deixe em branco os atributos sem gabarito. Fotos que não aparecem no CSV são avaliadas normalmente, mas não entram nos denominadores de acurácia. Um CSV pode ser parcial: por exemplo, apenas `arquivo,cor`. Valores inválidos e nomes desconhecidos são rejeitados antes de publicar o dataset.

“Sem gabarito” é diferente de uma previsão `unknown`: uma célula em branco não entra na avaliação daquele atributo. `unknown`/`not_visible` são respostas esperadas explícitas e contam no cálculo. Não use `unknown` apenas para preencher um campo que você não anotou.

Sem CSV, a aplicação calcula desempenho, falhas e validade do formato. **Não calcula acurácia, precisão, recall ou F1 sem respostas esperadas.**

## Interpretar o relatório

| Métrica | O que significa |
|---|---|
| Acurácia completa por foto | Proporção de fotos em que todos os atributos estão corretos. Só usa fotos com gabarito completo. |
| Acurácia por atributo | Acertos divididos pelo total de atributos anotados. Pode usar gabarito parcial. |
| Acurácia por campo | Acertos de presença, marca, cor etc., cada qual com seu denominador. |
| Precisão, recall e F1 | Métricas por classe; veja as matrizes de confusão nos detalhes de cada modelo. |
| F1 macro | Média do F1 nas classes com suporte no gabarito; divisões indefinidas contam como zero. |
| Tempo mediano e P95 | Mediana e percentil 95 das chamadas com sucesso, incluindo leitura, pré e pós-processamento; carga e aquecimento excluídos. |
| Tempo médio, mínimo, máximo e desvio | Detalhamento da distribuição de latências em todas as repetições medidas com sucesso. |
| Falhas | Chamadas que terminaram com erro, em todas as repetições. Falhas na primeira repetição também contam como erro de qualidade. |
| Formato válido | Proporção de chamadas que entregam todos os campos válidos, sem erro de execução; não é acurácia. |
| CPU e memória RSS | Consumo do processo medido. Em HTTP, normalmente é o cliente, não o servidor Ollama. |

As matrizes de confusão comparam gabarito (linhas) e resposta prevista (colunas). Falhas aparecem como uma categoria de erro, sem desaparecer do denominador. A tabela por foto mostra a primeira repetição; o resumo de desempenho usa todas as repetições. O filtro **Com atributos incorretos** ajuda a revisar os casos problemáticos.

Resultados com `smoke_mock` são explicitamente identificados como **simulação**. Os rótulos do dataset de exemplo são exploratórios. Gabaritos enviados pelo usuário são identificados como fornecidos pelo usuário, sem alegar validação humana independente.

## Arquivos para baixar

- `evaluation.json`: relatório completo, denominadores, matrizes e metadados.
- `models.csv`: métricas por modelo, incluindo desempenho e qualidade. Acurácias e taxas são frações de 0 a 1.
- `photos.csv`: primeira previsão por foto/modelo, gabarito, tempo e falha.
- `attributes.csv`: métricas de cada atributo por modelo.
- `predictions.jsonl`: todas as chamadas e repetições, disponível no relatório e em Resultados.
- Gráficos de desempenho e acurácia em PNG/PDF.

Imagens, gabaritos e relatórios ficam no volume persistente de avaliações. O dataset finalizado é imutável; para corrigir fotos ou rótulos, importe outro dataset. Importações incompletas podem ser descartadas sem apagar datasets finalizados ou avaliações existentes.

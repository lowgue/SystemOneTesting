# Auditoria automática do orçamento de opções — Laya

Orçamento padrão do checkpoint: **256 tokens de cabeça**, sequência máxima 1024.

Esta auditoria constrói entradas usando o tokenizer e as regras do checkpoint. Não usa gabarito nem executa avaliação de acurácia. Há um piloto adicional em seis imagens de veículos comparando a entrada padrão e a cabeça de 512 tokens, mantendo todas as marcas. As listas menores da tabela são prefixos alfabéticos fixos; não são propostas de excluir a marca correta de uma tarefa real.

| Critérios | Marcas + desconhecida | Orçamento | Instrução cortada | Opções cortadas | Truncamento total |
|---|---:|---:|---|---:|---|
| descriptive | 9 | 256 | False | 0 | False |
| descriptive | 9 | 384 | False | 0 | False |
| descriptive | 9 | 512 | False | 0 | False |
| descriptive | 13 | 256 | False | 0 | False |
| descriptive | 13 | 384 | False | 0 | False |
| descriptive | 13 | 512 | False | 0 | False |
| descriptive | 17 | 256 | False | 0 | False |
| descriptive | 17 | 384 | False | 0 | False |
| descriptive | 17 | 512 | False | 0 | False |
| descriptive | 25 | 256 | False | 0 | False |
| descriptive | 25 | 384 | False | 0 | False |
| descriptive | 25 | 512 | False | 0 | False |
| descriptive | 33 | 256 | False | 17 | True |
| descriptive | 33 | 384 | False | 0 | False |
| descriptive | 33 | 512 | False | 0 | False |
| descriptive | 49 | 256 | True | 45 | True |
| descriptive | 49 | 384 | False | 25 | True |
| descriptive | 49 | 512 | False | 0 | False |
| short | 9 | 256 | False | 0 | False |
| short | 9 | 384 | False | 0 | False |
| short | 9 | 512 | False | 0 | False |
| short | 13 | 256 | False | 0 | False |
| short | 13 | 384 | False | 0 | False |
| short | 13 | 512 | False | 0 | False |
| short | 17 | 256 | False | 0 | False |
| short | 17 | 384 | False | 0 | False |
| short | 17 | 512 | False | 0 | False |
| short | 25 | 256 | False | 0 | False |
| short | 25 | 384 | False | 0 | False |
| short | 25 | 512 | False | 0 | False |
| short | 33 | 256 | False | 14 | True |
| short | 33 | 384 | False | 0 | False |
| short | 33 | 512 | False | 0 | False |
| short | 49 | 256 | True | 42 | True |
| short | 49 | 384 | False | 23 | True |
| short | 49 | 512 | False | 0 | False |
| visible | 9 | 256 | False | 0 | False |
| visible | 9 | 384 | False | 0 | False |
| visible | 9 | 512 | False | 0 | False |
| visible | 13 | 256 | False | 0 | False |
| visible | 13 | 384 | False | 0 | False |
| visible | 13 | 512 | False | 0 | False |
| visible | 17 | 256 | False | 0 | False |
| visible | 17 | 384 | False | 0 | False |
| visible | 17 | 512 | False | 0 | False |
| visible | 25 | 256 | False | 0 | False |
| visible | 25 | 384 | False | 0 | False |
| visible | 25 | 512 | False | 0 | False |
| visible | 33 | 256 | False | 17 | True |
| visible | 33 | 384 | False | 0 | False |
| visible | 33 | 512 | False | 0 | False |
| visible | 49 | 256 | True | 45 | True |
| visible | 49 | 384 | False | 25 | True |
| visible | 49 | 512 | False | 0 | False |

## Inferência com a taxonomia completa

| Orçamento de cabeça | Eventos | Eventos com truncamento | Mediana (ms) |
|---:|---:|---:|---:|
| 256 | 6 | 6/6 | 2472.0 |
| 512 | 6 | 0/6 | 3079.7 |

Conjuntos de respostas alterados entre os dois orçamentos: 3/6. Mudança não significa melhora. As medições deste pequeno lote não substituem os blocos alternados do ensaio de desempenho.

**Implicação:** validar o orçamento antes de medir qualidade. Uma decisão tipada pode ser perfeitamente válida em JSON e ainda ter recebido uma instrução ou opções cortadas.

Aumentar o orçamento no construtor não demonstra que o modelo mantenha qualidade fora da configuração treinada. Reduzir a taxonomia também muda a tarefa. Qualquer decomposição em grupos deve preservar todas as opções necessárias e ser avaliada posteriormente com gabarito confiável.

[Entradas, textos efetivos e detalhes dos cortes](budget_audit.json)

# Simulação de poder — comparação pareada

Simulação sem imagens ou gabaritos: diferença real de **5 pontos percentuais**, 20.000 experimentos por cenário, teste binomial pareado exato bilateral. A discordância é a fração de eventos em que só um dos modelos acerta; aqui ela é um parâmetro hipotético.

O teste detecta diferença contra zero. Não demonstra que o limite inferior da vantagem supere a margem prática de 5 pontos: isso requer uma diferença real maior que essa margem.

| Pares independentes por domínio | Discordância | α por domínio | Poder estimado | Erro Monte Carlo (±1 SE) |
|---|---:|---:|---:|---:|
| 300 | 10% | 0.05 | 74.4% | 0.31 pp |
| 600 | 10% | 0.05 | 97.4% | 0.11 pp |
| 900 | 10% | 0.05 | 99.9% | 0.03 pp |
| 1200 | 10% | 0.05 | 100.0% | 0.00 pp |
| 1800 | 10% | 0.05 | 100.0% | 0.00 pp |
| 2400 | 10% | 0.05 | 100.0% | 0.00 pp |
| 300 | 20% | 0.05 | 44.6% | 0.35 pp |
| 600 | 20% | 0.05 | 75.8% | 0.30 pp |
| 900 | 20% | 0.05 | 91.3% | 0.20 pp |
| 1200 | 20% | 0.05 | 97.1% | 0.12 pp |
| 1800 | 20% | 0.05 | 99.7% | 0.04 pp |
| 2400 | 20% | 0.05 | 100.0% | 0.01 pp |
| 300 | 30% | 0.05 | 31.3% | 0.33 pp |
| 600 | 30% | 0.05 | 58.6% | 0.35 pp |
| 900 | 30% | 0.05 | 77.3% | 0.30 pp |
| 1200 | 30% | 0.05 | 87.4% | 0.23 pp |
| 1800 | 30% | 0.05 | 97.0% | 0.12 pp |
| 2400 | 30% | 0.05 | 99.4% | 0.06 pp |
| 300 | 50% | 0.05 | 21.1% | 0.29 pp |
| 600 | 50% | 0.05 | 38.6% | 0.34 pp |
| 900 | 50% | 0.05 | 55.0% | 0.35 pp |
| 1200 | 50% | 0.05 | 67.6% | 0.33 pp |
| 1800 | 50% | 0.05 | 84.7% | 0.25 pp |
| 2400 | 50% | 0.05 | 93.1% | 0.18 pp |
| 300 | 10% | 0.025 | 65.7% | 0.34 pp |
| 600 | 10% | 0.025 | 95.0% | 0.15 pp |
| 900 | 10% | 0.025 | 99.4% | 0.05 pp |
| 1200 | 10% | 0.025 | 100.0% | 0.01 pp |
| 1800 | 10% | 0.025 | 100.0% | 0.00 pp |
| 2400 | 10% | 0.025 | 100.0% | 0.00 pp |
| 300 | 20% | 0.025 | 34.4% | 0.34 pp |
| 600 | 20% | 0.025 | 67.2% | 0.33 pp |
| 900 | 20% | 0.025 | 85.5% | 0.25 pp |
| 1200 | 20% | 0.025 | 94.4% | 0.16 pp |
| 1800 | 20% | 0.025 | 99.4% | 0.06 pp |
| 2400 | 20% | 0.025 | 99.9% | 0.02 pp |
| 300 | 30% | 0.025 | 22.2% | 0.29 pp |
| 600 | 30% | 0.025 | 47.5% | 0.35 pp |
| 900 | 30% | 0.025 | 66.9% | 0.33 pp |
| 1200 | 30% | 0.025 | 80.8% | 0.28 pp |
| 1800 | 30% | 0.025 | 94.5% | 0.16 pp |
| 2400 | 30% | 0.025 | 98.6% | 0.08 pp |
| 300 | 50% | 0.025 | 13.2% | 0.24 pp |
| 600 | 50% | 0.025 | 28.2% | 0.32 pp |
| 900 | 50% | 0.025 | 43.8% | 0.35 pp |
| 1200 | 50% | 0.025 | 56.8% | 0.35 pp |
| 1800 | 50% | 0.025 | 76.6% | 0.30 pp |
| 2400 | 50% | 0.025 | 88.2% | 0.23 pp |

α=0,025 por domínio corresponde a Bonferroni para duas comparações primárias dentro de uma família de 5%. Dependência entre cenas exige tratar grupos e pode reduzir o poder; a simulação não resolve a representatividade da coleta.

## Controles negativos: limite superior unilateral de 1% com zero falhas

| Comparações na família (exemplo Bonferroni) | Negativos necessários | Limite superior com 0/300 |
|---:|---:|---:|
| 1 | 299 | 0.994% |
| 2 | 368 | 1.222% |
| 6 | 477 | 1.583% |

A fórmula usa o limite binomial exato unilateral `1 − α**(1/n)`. Zero falhas não significa taxa zero. O caso de seis comparações é um cenário de ajuste, não uma seleção definitiva da família de testes.

Esses resultados orientam a coleta antes de abrir o teste final. Não há conclusão sobre qual modelo é melhor e não se deve ampliar a amostra até obter um resultado favorável.

[Resultados e parâmetros](power.json)

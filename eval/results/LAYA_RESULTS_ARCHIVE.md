# Arquivo Histórico de Resultados: Laya Vision

Este arquivo preserva todos os resultados de detecção e classificação visual obtidos com o modelo **Laya Vision** (`thaitea/laya-vision`, 201M parâmetros) para comparação com o **Clef-Flash** (9.1B).

---

## 1. Benchmark no Dataset Stanford Cars (Fotos Reais)

- **Dataset:** 20 fotografias reais do Stanford Cars Dataset
- **Backbone:** SmolVLM-256M Instruct + Head de Decisão Laya
- **Dispositivo:** CPU (4 threads)
- **Tokens de Saída:** 0 tokens (Decisão direta por logits)
- **Latência Média:** 1439 ms/imagem

### Métricas Gerais:
| Métrica | Resultado |
|---|:---:|
| **Classificação de Marca (12 opções)** | **55.0%** (11/20) |
| **Classificação de Carroceria (8 opções)** | **55.0%** (11/20) |
| **Presença de Carro (Limiar padrão 0.5)** | **45.0%** (9/20) |
| **Presença de Carro (Limiar calibrado 0.2)** | **95.0%** (19/20) |
| **Latência Média** | **1439 ms** |

### Tabela Caso a Caso (Laya Vision):
| # | Veículo (Stanford Cars) | Presença (p) | Marca Predita | Marca Real | Correto? | Tipo Predito | Tipo Real | Latência |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | Audi Tt Rs Coupe 2012 | ✓ (83%) | **audi** | audi | ✓ | **coupe** | coupe | 1608 ms |
| 2 | Dodge Dakota Club Cab 2007 | ✗ (23%) | toyota | dodge | ✗ | suv | truck | 1756 ms |
| 3 | Ford F-450 Super Duty Crew Cab 2012 | ✗ (18%) | **ford** | ford | ✓ | **truck** | truck | 1534 ms |
| 4 | Dodge Journey Suv 2012 | ✗ (42%) | **dodge** | dodge | ✓ | **suv** | suv | 1330 ms |
| 5 | Chevrolet Traverse Suv 2012 | ✗ (31%) | **chevrolet** | chevrolet | ✓ | **suv** | suv | 1299 ms |
| 6 | Toyota Sequoia Suv 2012 | ✗ (21%) | **toyota** | toyota | ✓ | **suv** | suv | 1511 ms |
| 7 | Toyota Corolla Sedan 2012 | ✓ (78%) | honda | toyota | ✗ | coupe | sedan | 1361 ms |
| 8 | Audi A5 Coupe 2012 | ✓ (77%) | **audi** | audi | ✓ | **coupe** | coupe | 1339 ms |
| 9 | Chevrolet Malibu Sedan 2007 | ✗ (20%) | toyota | chevrolet | ✗ | coupe | sedan | 1348 ms |
| 10 | Volkswagen Beetle Hatchback 2012 | ✓ (72%) | toyota | volkswagen | ✗ | coupe | hatch | 1405 ms |
| 11 | Bmw M6 Convertible 2010 | ✓ (82%) | **bmw** | bmw | ✓ | suv | convertible | 1379 ms |
| 12 | Hyundai Sonata Hybrid Sedan 2012 | ✓ (72%) | audi | hyundai | ✗ | hatch | sedan | 1404 ms |
| 13 | Hyundai Elantra Touring Hatchback 2012 | ✓ (51%) | honda | hyundai | ✗ | coupe | hatch | 1426 ms |
| 14 | Ford Expedition El Suv 2009 | ✗ (40%) | other | ford | ✗ | **suv** | suv | 1508 ms |
| 15 | Bmw Z4 Convertible 2012 | ✓ (84%) | **bmw** | bmw | ✓ | **convertible** | convertible | 1424 ms |
| 16 | Volkswagen Golf Hatchback 1991 | ✗ (46%) | chevrolet | volkswagen | ✗ | coupe | hatch | 1404 ms |
| 17 | Jeep Grand Cherokee Suv 2012 | ✗ (27%) | **jeep** | jeep | ✓ | **suv** | suv | 1456 ms |
| 18 | Honda Odyssey Minivan 2012 | ✗ (26%) | **honda** | honda | ✓ | suv | van | 1424 ms |
| 19 | Jeep Liberty Suv 2012 | ✓ (82%) | **jeep** | jeep | ✓ | **suv** | suv | 1440 ms |
| 20 | Honda Accord Coupe 2012 | ✗ (15%) | unknown | honda | ✗ | **coupe** | coupe | 1433 ms |

---

## 2. Benchmark em Imagens Sintéticas e Cenários de Controle

- **Acurácia Global (4 perguntas por imagem):** 67.9% (19/28)
- **Detecção de Presença (Sim/Não):** 6/7 (85.7%)
- **Marca:** 5/5 nos casos com carro
- **Cor:** Acertos sólidos em cores saturadas (vermelho, azul, preto, branco)
- **Viés identificado no Laya:** Forte preferência por classificar silhuetas genéricas como SUV (~50% dos casos).

---

## 3. Arquivo Bruto
Os dados brutos em JSON estão persistidos em:
- `eval/results/laya_vision_all_saved_results.json`
- `eval/results/vision-cars-20261020T213744Z.json`

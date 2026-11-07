# Modelos System One — Testes Padronizados

Estrutura unificada e padronizada para os 4 modelos visuais do System One:
- **`blip`** — Salesforce/blip-vqa-base
- **`clef`** — Clef-Flash Q4_K_M (via System One HTTP / Cloudflare / local llama-server)
- **`laya`** — thaitea/laya-vision
- **`d1_omni`** — LiquidAI/d1-omni-600M

---

## 📁 Estrutura de Diretórios Padronizada

Cada um dos 4 modelos possui **exatamente os mesmos testes**, com o mesmo padrão de nomenclatura e interface:

```text
modelos/
├── executar.py                     # CLI unificado para rodar qualquer teste em 1 ou em todos os modelos
├── comum/                          # Motor compartilhado (adapters, estatísticas e métricas)
│   ├── motor.py                    # Executor central reutilizável
│   ├── model_specs.py              # Especificações dos 4 modelos
│   └── metricas.py                 # Wilson 95%, latências percentis e formatação
│
├── blip/                           # Modelo BLIP VQA
│   ├── testes_datasets/            # 8 testes padronizados de dataset
│   └── testes_prompts/             # 3 testes padronizados de formulação de prompt
│
├── clef/                           # Modelo Clef-Flash (System One HTTP)
│   ├── testes_datasets/            # 8 testes padronizados de dataset
│   └── testes_prompts/             # 3 testes padronizados de formulação de prompt
│
├── laya/                           # Modelo Laya Vision
│   ├── laya_vision/                # Submódulo Git com a implementação Laya
│   ├── testes_datasets/            # 8 testes padronizados de dataset
│   └── testes_prompts/             # 3 testes padronizados de formulação de prompt
│
└── d1_omni/                        # Modelo LiquidAI D1-Omni
    ├── testes_datasets/            # 8 testes padronizados de dataset
    └── testes_prompts/             # 3 testes padronizados de formulação de prompt
```

---

## 🏷️ Padrão de Nomenclatura dos Testes

### 1. Testes de Datasets (`testes_datasets/`)
Todos os modelos possuem os seguintes arquivos:
- `test_01_cor_veiculo.py` — Classificação da cor predominante do veículo (11 classes + desconhecido).
- `test_02_tipo_veiculo.py` — Classificação da carroceria (sedan, SUV, hatch, pickup, van, coupe, etc.).
- `test_03_marca_veiculo.py` — Reconhecimento da marca/fabricante (Audi, BMW, Ford, Toyota, Honda, etc.).
- `test_04_presenca_veiculo.py` — Detecção binária de presença do veículo (yes/no) e controles negativos.
- `test_05_bundle_completo.py` — Inferência simultânea dos 4 atributos (presença, marca, cor, carroceria).
- `test_06_atributos_pessoa.py` — Atributos visíveis de pessoa (capacete, itens carregados, vestimenta).
- `test_07_robustez_imagem.py` — Avaliação sob ruído e degradações visuais (baixa luz, compressão).
- `test_08_multiplos_objetos.py` — Associação de resposta com múltiplos alvos na mesma cena.

### 2. Testes de Prompts (`testes_prompts/`)
Todos os modelos possuem os seguintes arquivos:
- `test_01_formatos_prompt.py` — Comparação de formatos: pergunta aberta (`short_open`), com alternativas (`short_options`), descrições longas (`long_descriptions`) e JSON estruturado (`structured_json`).
- `test_02_variacao_instrucoes.py` — Impacto de instruções concisas vs detalhadas vs restritivas.
- `test_03_orcamento_opcoes.py` — Impacto do tamanho da taxonomia / lista de opções no contexto.

---

## 🚀 Como Executar

### Opção 1: Executar um arquivo de teste diretamente (Recomendado)

Você pode entrar no diretório ou chamar diretamente o script Python desejado:

```bash
# Executar teste de cor no BLIP
python modelos/blip/testes_datasets/test_01_cor_veiculo.py

# Executar teste de tipo de veículo no Laya (apenas 5 imagens para teste rápido)
python modelos/laya/testes_datasets/test_02_tipo_veiculo.py --limit 5

# Executar teste de cor no Clef via mock (sem precisar do llama-server ativo)
python modelos/clef/testes_datasets/test_01_cor_veiculo.py --mock

# Executar teste de formatos de prompt no D1-Omni
python modelos/d1_omni/testes_prompts/test_01_formatos_prompt.py --limit 3
```

**Argumentos CLI aceitos por todos os testes:**
- `--limit` / `-n`: Limita a quantidade de imagens avaliadas.
- `--repeticoes` / `-r`: Número de repetições por imagem (padrão: 1).
- `--device`: `cpu` ou `cuda`.
- `--threads`: Número de threads PyTorch (padrão: 4).
- `--mock`: Executa com adaptador simulado (ideal para validar pipelines sem baixar pesos).
- `--url`: Sobrescreve o endpoint HTTP (para Clef).
- `--output` / `-o`: Caminho customizado para salvar o relatório JSON.

---

### Opção 2: Via CLI unificado (`modelos/executar.py`)

```bash
# Listar todos os testes e modelos
python modelos/executar.py --listar

# Rodar um teste específico em um modelo
python modelos/executar.py --modelo blip --teste test_01_cor_veiculo.py

# Rodar o mesmo teste em TODOS os 4 modelos para comparação lado a lado
python modelos/executar.py --modelo todos --teste 01 --limit 5 --mock
```

---

## ✏️ Como Editar os Testes Facilmente

Cada arquivo de teste foi desenhado para ser **autoexplicativo e declarativo**. No topo de qualquer arquivo de teste (ex: `test_01_cor_veiculo.py`), você encontrará o bloco de configuração:

```python
# ==============================================================================
# CONFIGURAÇÃO DO TESTE (EDITE AQUI FACILMENTE)
# ==============================================================================
CONFIG = {
    "nome": "01 - Cor do Veículo",
    "campo_alvo": "color",
    "dominio": "vehicle",
    "pergunta": "What is the predominant color of this vehicle?",
    "criterios": {
        "black": "Black",
        "white": "White",
        "gray_silver": "Gray or silver",
        "red": "Red",
        "blue": "Blue",
        # Adicione ou remova opções aqui
    },
    "dataset": "benchmarks/datasets/vehicles/portable.json",
    "repeticoes": 1,
}
```

Para editar um teste:
1. **Alterar a pergunta/prompt:** Modifique o valor da chave `"pergunta"`.
2. **Adicionar/remover classes aceitas:** Edite o dicionário `"criterios"`.
3. **Mudar o dataset:** Altere o caminho na chave `"dataset"`.
4. **Mudar número padrão de repetições:** Altere `"repeticoes"`.

Todo o código pesado de carregamento de pesos, cálculo de acurácia, latência, matrizes e formatação é gerenciado automaticamente pelo motor compartilhado em `modelos/comum/`.

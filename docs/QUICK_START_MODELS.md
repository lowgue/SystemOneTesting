# Guia Rápido de Instalação e Benchmark de Modelos Visuais

Este guia explica como configurar e executar rapidamente os modelos de decisão visual (**Laya Vision**, **Clef-Flash** e **d1-omni-600M**) em qualquer computador (Linux, Windows ou macOS).

---

## 1. Escopo dos Modelos com Encoder de Visão

O benchmark avalia modelos capazes de receber uma **imagem** e um conjunto de perguntas tipadas no protocolo **System One**, retornando decisões calibradas sem geração de texto livre.

| Modelo | Arquitetura | Parâmetros | Entrada | Tamanho Pesos | Runtime |
|---|---|---|---|---|---|
| **Laya Vision** (`laya-vision-head512`) | SmolVLM-256M + Readout Head | ~300M | Imagem + Perguntas | ~1,3 GB (com backbone) | PyTorch / transformers (local) |
| **d1-omni-600M** (`d1-omni-600M`) | LFM2.5 + SigLIP2 + Decision Head | ~587M | Imagem + Perguntas | ~2,35 GB | PyTorch / transformers (local) |
| **Clef-Flash (GGUF)** (`clef-flash-local-q4`) | Clef-Flash Q4_K_M + Projector | ~9B | Imagem + Perguntas | ~7,35 GB | `llama-server` (local) |
| **Clef-Flash (Ollama)** (`clef-flash-ollama`) | Clef-Flash 9B | ~9B | Imagem + Perguntas | ~5,5 GB | Serviço Ollama (local) |
| *(Opcional)* **BLIP VQA** (`blip-vqa`) | ViT-B/16 + BERT VQA | ~385M | Imagem + Prompt texto | ~1,9 GB | PyTorch / transformers (baseline) |

> [!NOTE]
> Modelos como `kev-0.5b` foram intencionalmente excluídos deste escopo porque são estritamente text-only (baseados em Qwen text) e não possuem encoder visual nativo.

---

## 2. Instalação em 1 Comando (Novo Computador)

Clone o repositório no novo computador:

```bash
git clone --recurse-submodules https://github.com/lowgue/SystemOneTesting.git
cd SystemOneTesting
```

### No Linux ou macOS

Execute o script rápido de bootstrap:

```bash
# Para inferência em CPU:
./scripts/setup_quick.sh --cpu

# Ou para máquinas com GPU NVIDIA:
./scripts/setup_quick.sh --gpu
```

Para já executar um teste rápido de 1 imagem ao final da instalação:
```bash
./scripts/setup_quick.sh --cpu --smoke
```

### No Windows (PowerShell)

```powershell
# Para CPU:
.\scripts\setup_quick.ps1

# Ou para GPU:
.\scripts\setup_quick.ps1 -Gpu -Smoke
```

O script executa automaticamente:
1. Criação do ambiente virtual Python 3.12 (`.venv`).
2. Atualização dos submódulos Git (`modelos/laya/laya_vision`).
3. Instalação das dependências pip do projeto.
4. Download e validação dos pesos dos 3 modelos (`laya`, `d1-omni`, `clef-gguf`).
5. Diagnóstico de integridade (`vision_bench doctor`).

---

## 3. Instalação Modular via `download_models.py`

Se preferir controlar manualmente o download ou selecionar apenas modelos específicos, utilize o utilitário Python dedicado:

### Ver espaço em disco e requisitos (sem baixar nada)
```bash
.venv/bin/python scripts/download_models.py --dry-run
```

### Baixar apenas Laya e d1-omni (inferência 100% local em Python)
```bash
.venv/bin/python scripts/download_models.py --models laya d1-omni
```

### Baixar para quem usa Ollama
```bash
.venv/bin/python scripts/download_models.py --models vision-ollama
```
*(Faz o download de Laya, d1-omni e executa `ollama pull clef-flash:9b` automaticamente se o Ollama estiver ativo)*.

### Baixar todos os modelos visuais locais
```bash
.venv/bin/python scripts/download_models.py --models vision-local
```

### Presets disponíveis
- `vision-local`: `laya`, `d1-omni`, `clef-gguf`
- `vision-ollama`: `laya`, `d1-omni`, `clef-ollama`
- `vision-all`: `laya`, `d1-omni`, `clef-gguf`, `clef-ollama`
- `all`: todos os anteriores mais o baseline `blip`

---

## 4. Como Executar os Benchmarks

### Passo 1: Validar ambiente com o `doctor`

```bash
.venv/bin/python -m vision_bench doctor --config benchmarks/configs/local.json
```
Deve exibir `"status": "ready"` e `"issues": []`.

---

### Passo 2: Teste curto (Smoke Test com 2 imagens)

#### Testar apenas os modelos em processo (Laya e d1-omni):
```bash
.venv/bin/python -m vision_bench run \
  --config benchmarks/configs/local.json \
  --models laya-vision-head512 d1-omni-600M \
  --limit 2 --repetitions 1 \
  --output .artifacts/runs/smoke-local
```

#### Testar comparando com Clef no Ollama:
Certifique-se de que o Ollama está rodando (`ollama serve`). Então execute:
```bash
.venv/bin/python -m vision_bench run \
  --config benchmarks/configs/ollama.json \
  --contract benchmarks/contracts/description_ollama_v1.json \
  --models laya-vision-head512 d1-omni-600M clef-flash-ollama \
  --limit 2 --repetitions 1 \
  --output .artifacts/runs/smoke-ollama
```

---

### Passo 3: Rodada de Avaliação Completa

Para avaliar todos os 22 eventos do dataset padrão com 3 repetições:

```bash
.venv/bin/python -m vision_bench run \
  --config benchmarks/configs/local.json \
  --models laya-vision-head512 d1-omni-600M \
  --repetitions 3 \
  --threads 8 \
  --device cpu \
  --output .artifacts/runs/computador-novo-local
```

*(Se estiver em máquina com GPU, use `--device cuda`).*

---

### Passo 4: Gerar Relatório e Gráficos

```bash
.venv/bin/python -m vision_bench report \
  .artifacts/runs/computador-novo-local \
  --output .artifacts/reports/computador-novo
```

Saídas geradas no diretório de relatório:
- `README.md`: resumo das métricas, acurácia por atributo e aderência.
- `metrics.json` e `metrics.csv`: dados tabulares para exportação.
- Gráficos em PNG (300 dpi) e SVG comparando latências, consumo de CPU/RAM e acurácia.

---

## 5. Execução em Modo Offline

Após rodar o script `scripts/download_models.py`, todos os modelos têm seus pesos, tokenizadores e backbones gravados no cache local (`~/.cache/huggingface/hub`).

Para garantir que nenhuma requisição saia para a internet durante os testes:

```bash
.venv/bin/python -m vision_bench run \
  --config benchmarks/configs/local.json \
  --models laya-vision-head512 d1-omni-600M \
  --offline \
  --output .artifacts/runs/teste-offline
```

A flag `--offline` instrui o Hugging Face Hub e os tokenizadores a recusarem qualquer chamada externa.

"""Motor central e padronizado para execução de testes em modelos System One."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Dict, List

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vision_bench.adapters import create_adapter
from vision_bench.core import load_json, write_json
from vision_bench.protocols import PROFILES
from .metricas import imprimir_tabela_resumo, intervalo_wilson_95, resumir_latencias
from .model_specs import obter_spec_modelo


def criar_argument_parser(descricao: str) -> argparse.ArgumentParser:
    """Cria o parser CLI padrão compartilhado por todos os testes."""
    parser = argparse.ArgumentParser(description=descricao)
    parser.add_argument("--limit", "-n", type=int, default=None, help="Limita o número de imagens testadas (útil para testes rápidos/smoke)")
    parser.add_argument("--repeticoes", "-r", type=int, default=1, help="Número de repetições por imagem (padrão: 1)")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Dispositivo para execução (cpu ou cuda)")
    parser.add_argument("--threads", type=int, default=4, help="Número de threads para PyTorch")
    parser.add_argument("--mock", action="store_true", help="Executa com modelo simulado/mock (sem carregar pesos pesados)")
    parser.add_argument("--url", type=str, default=None, help="Sobrescreve a URL do endpoint HTTP (para Clef)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Caminho do arquivo JSON para salvar os resultados")
    return parser


def carregar_itens_dataset(caminho_relativo: str) -> List[Dict[str, Any]]:
    """Carrega os itens do dataset a partir do caminho relativo ao ROOT."""
    caminho = ROOT / caminho_relativo
    if not caminho.is_file():
        # Fallback para o dataset portátil padrão se o arquivo não for encontrado
        caminho_fallback = ROOT / "benchmarks/datasets/vehicles/portable.json"
        print(f"Aviso: Dataset '{caminho_relativo}' não encontrado. Usando '{caminho_fallback.name}'.")
        caminho = caminho_fallback

    dados = load_json(caminho)
    itens = dados.get("items", dados.get("images", []))
    return itens


def construir_contrato(dominio: str, campo: str, pergunta: str, criterios: Dict[str, str]) -> Dict[str, Any]:
    """Constrói o contrato no formato padrão System One consumido pelos adapters."""
    contrato_base = load_json(ROOT / "benchmarks/contracts/description_v1.json")

    if campo == "bundle_completo":
        return contrato_base

    return {
        "schema_version": 1,
        "domains": {
            dominio: {
                "prompts": {campo: pergunta},
                "questions": {
                    campo: {
                        "type": "choice",
                        "instructions": pergunta,
                        "criteria": criterios,
                    }
                },
            }
        },
        "aliases": contrato_base.get("aliases", {}),
        "absence": contrato_base.get("absence", {}),
    }


def executar_teste_dataset(modelo: str, config: Dict[str, Any], args_custom: List[str] | None = None) -> Dict[str, Any]:
    """Executa um teste padronizado de dataset para um modelo específico."""
    parser = criar_argument_parser(f"Teste: {config.get('nome', 'Dataset Test')} no modelo {modelo.upper()}")
    args = parser.parse_args(args_custom)

    try:
        import torch
        torch.set_num_threads(args.threads)
    except ImportError:
        pass

    # 1. Obter especificação do modelo
    spec = obter_spec_modelo(modelo, usar_mock=args.mock, url_custom=args.url)

    # 2. Configurar contrato e domínio
    dominio = config.get("dominio", "vehicle")
    campo_alvo = config.get("campo_alvo", "color")
    pergunta = config.get("pergunta", "Describe the visible attribute.")
    criterios = config.get("criterios", {})
    contrato = construir_contrato(dominio, campo_alvo, pergunta, criterios)

    # 3. Carregar e filtrar dataset
    dataset_path = config.get("dataset", "benchmarks/datasets/vehicles/portable.json")
    itens_brutos = carregar_itens_dataset(dataset_path)

    # Validar quais imagens realmente existem no disco
    itens_validos = []
    for item in itens_brutos:
        rel_p = item.get("rel_path", "")
        img_p = ROOT / rel_p
        if img_p.is_file():
            itens_validos.append(item)

    if not itens_validos:
        if args.mock:
            print("Nenhuma imagem física encontrada, mas executando com imagens sintéticas via --mock.")
            itens_validos = itens_brutos[: (args.limit or 5)]
        else:
            raise FileNotFoundError(
                f"Nenhuma imagem encontrada no disco para '{dataset_path}'. "
                f"Verifique se o dataset foi baixado ou use --mock para testar a lógica."
            )

    if args.limit:
        itens_validos = itens_validos[: args.limit]

    total_avaliar = len(itens_validos)
    repeticoes = args.repeticoes or config.get("repeticoes", 1)

    print(f"\nIniciando: {config.get('nome')} | Modelo: {spec['nome']}")
    print(f"Imagens válidas: {total_avaliar} | Repetições: {repeticoes} | Dispositivo: {args.device}")

    # 4. Inicializar adaptador
    try:
        adapter = create_adapter(spec, contrato, device=args.device)
    except Exception as exc:
        if spec.get("adapter") == "systemone_http":
            print(f"\n[ERRO DE CONEXÃO] Não foi possível conectar ao servidor Clef em '{spec.get('url')}'.")
            print("Dica: Inicie o servidor local primeiro ou execute este teste com a flag '--mock'.")
        raise exc

    # 5. Warm-up
    print("Executando warm-up...", end="", flush=True)
    try:
        with Image.open(ROOT / itens_validos[0]["rel_path"]) as src:
            _ = adapter.predict(src.convert("RGB"), dominio)
    except Exception:
        dummy = Image.new("RGB", (100, 100), color="gray")
        _ = adapter.predict(dummy, dominio)
    print(" Concluído.")

    # 6. Execução das inferências
    resultados_imagens = []
    latencias_todas = []
    acertos_totais = 0
    erros_totais = 0
    detalhes_classe: Dict[str, Dict[str, int]] = {}

    for idx, item in enumerate(itens_validos, 1):
        rel_p = item.get("rel_path", "")
        img_p = ROOT / rel_p

        try:
            if img_p.is_file():
                with Image.open(img_p) as src:
                    img = src.convert("RGB")
            else:
                img = Image.new("RGB", (100, 100), color="blue")
        except Exception:
            erros_totais += 1
            continue

        esperado = item.get("expected", {})
        if not isinstance(esperado, dict):
            esperado = {campo_alvo: str(item.get("expected_color", item.get("color", "unknown")))}

        perguntas = contrato["domains"][dominio]["questions"]
        campos_bundle = tuple(perguntas)
        esperado_bundle = None
        if campo_alvo == "bundle_completo":
            campos_ausentes = [campo for campo in campos_bundle if campo not in esperado]
            if campos_ausentes:
                raise ValueError(
                    "O dataset não tem os atributos necessários para avaliar o bundle: "
                    + ", ".join(campos_ausentes)
                )
            esperado_bundle = {
                campo: str(esperado[campo]).lower() for campo in campos_bundle
            }
            valor_esperado = json.dumps(
                esperado_bundle, ensure_ascii=False, sort_keys=True
            )
        else:
            valor_esperado = esperado.get(campo_alvo, "unknown")

        preds_item = []
        lats_item = []

        for rep in range(repeticoes):
            t0 = time.perf_counter()
            try:
                answers, raw = adapter.predict(img, dominio)
                lat_ms = (time.perf_counter() - t0) * 1000
                if esperado_bundle is None:
                    pred = str(answers.get(campo_alvo, "parse_error")).lower()
                else:
                    pred_bundle = {
                        campo: str(answers.get(campo, "parse_error")).lower()
                        for campo in campos_bundle
                    }
                    pred = json.dumps(
                        pred_bundle, ensure_ascii=False, sort_keys=True
                    )
            except Exception:
                lat_ms = (time.perf_counter() - t0) * 1000
                pred = "error"
                erros_totais += 1

            preds_item.append(pred)
            lats_item.append(lat_ms)
            latencias_todas.append(lat_ms)

        pred_final = max(set(preds_item), key=preds_item.count) if preds_item else "error"
        eh_correto = (pred_final == valor_esperado)
        if eh_correto:
            acertos_totais += 1

        detalhes_classe.setdefault(valor_esperado, {"acertos": 0, "total": 0})
        detalhes_classe[valor_esperado]["total"] += 1
        if eh_correto:
            detalhes_classe[valor_esperado]["acertos"] += 1

        resultados_imagens.append({
            "id": item.get("id", str(idx)),
            "rel_path": rel_p,
            "esperado": esperado_bundle if esperado_bundle is not None else valor_esperado,
            "predito": (
                json.loads(pred_final)
                if esperado_bundle is not None and pred_final != "error"
                else pred_final
            ),
            "correto": eh_correto,
            "latencia_mediana_ms": round(float(statistics.median(lats_item)), 1) if lats_item else 0.0,
        })

        if idx % 10 == 0 or idx == total_avaliar:
            print(f"Progresso: {idx}/{total_avaliar} imagens processadas.")

    if hasattr(adapter, "close"):
        adapter.close()

    # 7. Resumo e Métricas
    latencias_resumo = resumir_latencias(latencias_todas)
    imprimir_tabela_resumo(
        titulo=config.get("nome", "Teste de Dataset"),
        modelo_nome=spec["nome"],
        total=total_avaliar,
        acertos=acertos_totais,
        erros=erros_totais,
        latencias=latencias_resumo,
        detalhes_classe=detalhes_classe,
        repeticoes=repeticoes,
        dispositivo=args.device,
    )

    resultado_final = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "modelo": spec,
        "teste": config.get("nome"),
        "campo_alvo": campo_alvo,
        "total_imagens": total_avaliar,
        "acertos": acertos_totais,
        "acuracia": round(acertos_totais / total_avaliar, 4) if total_avaliar else 0.0,
        "ic_wilson_95": list(intervalo_wilson_95(acertos_totais, total_avaliar)),
        "latencias_ms": latencias_resumo,
        "por_classe": detalhes_classe,
        "itens": resultados_imagens,
    }

    # 8. Salvar resultados
    if args.output:
        caminho_saida = Path(args.output)
    else:
        dir_res = ROOT / "modelos" / modelo / "resultados"
        dir_res.mkdir(parents=True, exist_ok=True)
        timestamp_slug = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug_teste = config.get("nome", "teste").lower().replace(" ", "_")
        caminho_saida = dir_res / f"{slug_teste}_{timestamp_slug}.json"

    write_json(caminho_saida, resultado_final)
    print(f"Resultados detalhados salvos em: {caminho_saida}")
    return resultado_final


def executar_teste_prompt(modelo: str, config: Dict[str, Any], args_custom: List[str] | None = None) -> Dict[str, Any]:
    """Executa teste padronizado comparando diferentes formulações de prompt."""
    parser = criar_argument_parser(f"Teste de Prompts: {config.get('nome')} no modelo {modelo.upper()}")
    args = parser.parse_args(args_custom)

    try:
        import torch
        torch.set_num_threads(args.threads)
    except ImportError:
        pass

    spec_base = obter_spec_modelo(modelo, usar_mock=args.mock, url_custom=args.url)
    formatos = config.get("formatos", ["short_open", "short_options", "long_descriptions", "structured_json"])
    adapter_profile = obter_spec_modelo(modelo)["adapter"]
    perfis_suportados = {
        profile["id"] for profile in PROFILES.get(adapter_profile, [])
    }
    formatos_invalidos = sorted(set(formatos) - perfis_suportados)
    if formatos_invalidos:
        suportados = ", ".join(sorted(perfis_suportados)) or "nenhum"
        raise ValueError(
            f"Perfis de prompt não suportados pelo adaptador '{adapter_profile}': "
            f"{', '.join(formatos_invalidos)}. Perfis suportados: {suportados}."
        )

    dominio = config.get("dominio", "vehicle")
    campo_alvo = config.get("campo_alvo", "color")
    pergunta_base = config.get("pergunta_base", "What is the predominant color of this vehicle?")
    criterios = config.get("criterios", {})

    dataset_path = config.get("dataset", "benchmarks/datasets/vehicles/portable.json")
    itens = [it for it in carregar_itens_dataset(dataset_path) if (ROOT / it.get("rel_path", "")).is_file()]
    if args.mock and not itens:
        itens = carregar_itens_dataset(dataset_path)[: (args.limit or 5)]

    if args.limit:
        itens = itens[: args.limit]

    total_imagens = len(itens)
    print(f"\nIniciando Teste de Prompts: {config.get('nome')} | Modelo: {spec_base['nome']}")
    print(f"Formatos a testar: {formatos} | Imagens: {total_imagens}")

    resumo_formatos = {}

    for fmt in formatos:
        print(f"\n--- Avaliando Formato de Prompt: '{fmt}' ---")
        spec_fmt = deepcopy(spec_base)
        spec_fmt["profile"] = fmt

        contrato = construir_contrato(dominio, campo_alvo, pergunta_base, criterios)
        try:
            adapter = create_adapter(spec_fmt, contrato, device=args.device)
        except Exception as e:
            print(f"Pulo de formato '{fmt}' devido a erro: {e}")
            continue

        latencias = []
        acertos = 0
        aderentes = 0
        erros = 0

        try:
            for item in itens:
                t0 = time.perf_counter()
                try:
                    img_p = ROOT / item.get("rel_path", "")
                    img = (
                        Image.open(img_p).convert("RGB")
                        if img_p.is_file()
                        else Image.new("RGB", (100, 100), "red")
                    )
                    exp = item.get("expected", {}).get(campo_alvo, "unknown")
                    answers, _ = adapter.predict(img, dominio)
                    pred = str(answers.get(campo_alvo, "unknown")).lower()
                except Exception:
                    lat = (time.perf_counter() - t0) * 1000
                    latencias.append(lat)
                    erros += 1
                    continue

                lat = (time.perf_counter() - t0) * 1000
                latencias.append(lat)
                if pred == exp:
                    acertos += 1
                if pred in criterios:
                    aderentes += 1
        finally:
            if hasattr(adapter, "close"):
                adapter.close()

        acc = (acertos / total_imagens) if total_imagens else 0.0
        taxa_aderencia = (aderentes / total_imagens) if total_imagens else 0.0
        resumo_formatos[fmt] = {
            "acuracia": round(acc, 4),
            "aderencia": round(taxa_aderencia, 4),
            "erros": erros,
            "latencia_mediana_ms": round(float(statistics.median(latencias)), 1) if latencias else 0.0,
        }

    # Exibe tabela comparativa
    print("\n" + "=" * 80)
    print(f" COMPARAÇÃO DE FORMATOS DE PROMPT | MODELO: {spec_base['nome']}")
    print("=" * 80)
    print(f" {'Formato':<25} | {'Acurácia':<12} | {'Aderência Taxonomia':<22} | {'Erros':<7} | {'Latência (ms)':<12}")
    print("-" * 80)
    for fmt, dados in resumo_formatos.items():
        print(f" {fmt:<25} | {dados['acuracia']*100:>10.1f}% | {dados['aderencia']*100:>20.1f}% | {dados['erros']:>5} | {dados['latencia_mediana_ms']:>10.1f} ms")
    print("=" * 80 + "\n")

    return resumo_formatos

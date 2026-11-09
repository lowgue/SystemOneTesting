"""Cálculo padronizado de métricas e formatação de relatórios em terminal."""

from __future__ import annotations
import math
import statistics
from typing import Any, Dict, List, Tuple


def intervalo_wilson_95(acertos: int, total: int) -> Tuple[float, float]:
    """Calcula o Intervalo de Confiança Wilson de 95% para proporções."""
    if total <= 0:
        return 0.0, 0.0
    z = 1.959963984540054
    p = acertos / total
    denom = 1 + (z * z) / total
    centro = (p + (z * z) / (2 * total)) / denom
    margem = (z * math.sqrt((p * (1 - p)) / total + (z * z) / (4 * total * total))) / denom
    lim_inf = max(0.0, centro - margem)
    lim_sup = min(1.0, centro + margem)
    return round(float(lim_inf), 4), round(float(lim_sup), 4)


def _calcular_percentil(valores: List[float], percentil: float) -> float:
    """Calcula percentil sem exigir numpy."""
    if not valores:
        return 0.0
    s = sorted(valores)
    k = (len(s) - 1) * (percentil / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    d0 = s[int(f)] * (c - k)
    d1 = s[int(c)] * (k - f)
    return d0 + d1


def resumir_latencias(latencias_ms: List[float]) -> Dict[str, float]:
    """Calcula estatísticas descritivas de latência em milissegundos."""
    if not latencias_ms:
        return {"media": 0.0, "mediana": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
    return {
        "media": round(float(statistics.mean(latencias_ms)), 1),
        "mediana": round(float(statistics.median(latencias_ms)), 1),
        "p90": round(float(_calcular_percentil(latencias_ms, 90.0)), 1),
        "min": round(float(min(latencias_ms)), 1),
        "max": round(float(max(latencias_ms)), 1),
    }


def imprimir_tabela_resumo(
    titulo: str,
    modelo_nome: str,
    total: int,
    acertos: int,
    erros: int,
    latencias: Dict[str, float],
    detalhes_classe: Dict[str, Dict[str, int]] | None = None,
    repeticoes: int = 1,
    dispositivo: str = "cpu",
) -> None:
    """Imprime um resumo visual claro e estruturado no terminal."""
    acuracia = (acertos / total) if total > 0 else 0.0
    ic_inf, ic_sup = intervalo_wilson_95(acertos, total)

    sep_larga = "=" * 80
    sep_fina = "-" * 80

    print("\n" + sep_larga)
    print(f" TESTE: {titulo.upper()} | MODELO: {modelo_nome}")
    print(sep_larga)
    print(f" Imagens avaliadas: {total:<6} | Repetições: {repeticoes:<3} | Dispositivo: {dispositivo}")
    print(sep_fina)
    print(f" Acurácia Geral:    {acuracia * 100:>6.1f}% ({acertos}/{total})")
    print(f" IC Wilson 95%:     [{ic_inf * 100:.1f}%, {ic_sup * 100:.1f}%]")
    print(f" Latência Mediana:  {latencias['mediana']:>6.1f} ms/imagem")
    print(f" Latência Média:    {latencias['media']:>6.1f} ms (p90: {latencias['p90']:.1f} ms)")
    print(f" Falhas / Erros:    {erros:>6}")

    if detalhes_classe:
        print(sep_fina)
        print(" Detalhamento por Classe (Acurácia por rótulo):")
        for rotulo, contagem in sorted(detalhes_classe.items()):
            n = contagem.get("total", 0)
            c = contagem.get("acertos", 0)
            taxa = (c / n * 100) if n > 0 else 0.0
            print(f"   • {rotulo:<18}: {c:>2}/{n:<2} ({taxa:>5.1f}%)")

    print(sep_larga + "\n")

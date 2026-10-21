"""Evaluation harness for Laya textual model on real-world decision tasks.

Usage:
    python eval/harness.py [--dataset eval/datasets/default.jsonl] [--model laya] [--out eval/results/]
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

OLLAMA_URL = "http://localhost:11434"


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------

def query_laya(
    state: str,
    questions: dict[str, Any],
    model: str = "laya",
    base_url: str = OLLAMA_URL,
) -> tuple[dict[str, Any], float]:
    """Returns (response_json, latency_ms)."""
    payload = json.dumps({"model": model, "state": state, "questions": questions}).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/systemone",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read())
    latency_ms = (time.perf_counter() - t0) * 1000
    return result, round(latency_ms, 2)


# ---------------------------------------------------------------------------
# Prediction extraction & scoring
# ---------------------------------------------------------------------------

def extract_prediction(answer: dict[str, Any], threshold: float = 0.5) -> str:
    """Converts a raw Laya answer dict into a comparable string label.

    Response shapes (as returned by /v1/systemone):
      noul   → {"type":"noul", "noul": 0.93}
      choice → {"type":"choice", "choice": "alto", "probabilities": {...}, "confidence": 0.7}
      score  → {"type":"score", "score": 2.3, "legend": {"0":"Mínima",...}, "probabilities": {...}}
    """
    atype = answer.get("type")
    if atype == "noul":
        return "true" if answer["noul"] >= threshold else "false"
    if atype == "choice":
        # The winner is a plain string under "choice"
        return str(answer["choice"])
    if atype == "score":
        # Winner is the index (string) with highest probability
        probs = answer.get("probabilities", {})
        if probs:
            return max(probs, key=probs.get)
        # Fallback: round the continuous score to nearest index
        legend = answer.get("legend", {})
        s = answer.get("score", 0)
        idx = str(round(s))
        return idx if idx in legend else str(min(legend.keys(), key=lambda k: abs(int(k) - round(s))))
    return ""


def is_correct(prediction: str, expected: str, atype: str) -> bool:
    if atype == "noul":
        pos = {"true", "yes", "1", "sim", "s"}
        pred_pos = prediction == "true"
        exp_pos = str(expected).lower() in pos
        return pred_pos == exp_pos
    if atype == "score":
        # Accept the predicted index (0-based string) OR the label text
        return str(prediction) == str(expected)
    return str(prediction).lower() == str(expected).lower()


def score_noul(answer: dict[str, Any], expected: str) -> float:
    """Returns probability assigned to the correct class (Brier-compatible)."""
    pos = {"true", "yes", "1", "sim", "s"}
    p = answer["noul"]
    correct_is_pos = str(expected).lower() in pos
    return p if correct_is_pos else 1.0 - p


# ---------------------------------------------------------------------------
# Per-sample evaluation
# ---------------------------------------------------------------------------

def evaluate_sample(sample: dict[str, Any], model: str, base_url: str) -> dict[str, Any]:
    case_id = sample.get("id", "unknown")
    state = sample["state"]
    questions = sample["questions"]
    expected = sample.get("expected", {})

    try:
        response, latency_ms = query_laya(state, questions, model=model, base_url=base_url)
    except Exception as exc:
        return {
            "id": case_id,
            "domain": sample.get("domain", ""),
            "error": str(exc),
            "latency_ms": None,
        }

    answers = response.get("answers", {})
    usage = response.get("usage", {})

    per_question: list[dict] = []
    for qkey, ans in answers.items():
        atype = ans.get("type", "")
        prediction = extract_prediction(ans)
        exp_val = expected.get(qkey, None)
        correct = is_correct(prediction, exp_val, atype) if exp_val is not None else None

        q_result: dict[str, Any] = {
            "question": qkey,
            "type": atype,
            "prediction": prediction,
            "expected": exp_val,
            "correct": correct,
            "confidence": ans.get("confidence"),
            "probabilities": ans.get("probabilities"),
            "raw": ans,
        }
        if atype == "noul" and exp_val is not None:
            q_result["p_correct"] = round(score_noul(ans, exp_val), 4)
        per_question.append(q_result)

    return {
        "id": case_id,
        "domain": sample.get("domain", ""),
        "latency_ms": latency_ms,
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "questions": per_question,
        "error": None,
    }


# ---------------------------------------------------------------------------
# Aggregation & reporting
# ---------------------------------------------------------------------------

def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute overall and per-domain accuracy, mean latency, and mean p_correct."""
    total, correct_total, errors = 0, 0, 0
    latencies: list[float] = []
    p_corrects: list[float] = []
    domain_stats: dict[str, dict] = {}

    for r in results:
        domain = r.get("domain", "unknown")
        if domain not in domain_stats:
            domain_stats[domain] = {"total": 0, "correct": 0}

        if r.get("error"):
            errors += 1
            continue

        if r.get("latency_ms") is not None:
            latencies.append(r["latency_ms"])

        for q in r.get("questions", []):
            total += 1
            domain_stats[domain]["total"] += 1
            if q.get("correct") is True:
                correct_total += 1
                domain_stats[domain]["correct"] += 1
            if "p_correct" in q:
                p_corrects.append(q["p_correct"])

    accuracy = correct_total / total if total else None
    mean_latency = sum(latencies) / len(latencies) if latencies else None
    mean_p_correct = sum(p_corrects) / len(p_corrects) if p_corrects else None

    per_domain = {}
    for dom, stats in domain_stats.items():
        t = stats["total"]
        c = stats["correct"]
        per_domain[dom] = {"accuracy": round(c / t, 4) if t else None, "total": t, "correct": c}

    return {
        "total_questions": total,
        "correct": correct_total,
        "accuracy": round(accuracy, 4) if accuracy is not None else None,
        "errors": errors,
        "mean_latency_ms": round(mean_latency, 1) if mean_latency else None,
        "mean_p_correct": round(mean_p_correct, 4) if mean_p_correct else None,
        "per_domain": per_domain,
    }


def print_report(summary: dict[str, Any], results: list[dict[str, Any]]) -> None:
    sep = "─" * 60
    print(f"\n{'═'*60}")
    print("  RELATÓRIO DE AVALIAÇÃO – Laya Textual")
    print(f"{'═'*60}")
    print(f"  Questões avaliadas : {summary['total_questions']}")
    print(f"  Corretas           : {summary['correct']}")
    acc = summary["accuracy"]
    print(f"  Acurácia global    : {acc*100:.1f}%" if acc is not None else "  Acurácia global    : –")
    mp = summary["mean_p_correct"]
    print(f"  Média p(correto)   : {mp:.4f}" if mp is not None else "  Média p(correto)   : –")
    lat = summary["mean_latency_ms"]
    print(f"  Latência média     : {lat:.0f} ms" if lat else "  Latência média     : –")
    if summary["errors"]:
        print(f"  Erros              : {summary['errors']}")

    print(f"\n  Por domínio:")
    for dom, stats in summary["per_domain"].items():
        a = stats["accuracy"]
        print(f"    {dom:20s} {stats['correct']}/{stats['total']}  {a*100:.0f}%" if a is not None else f"    {dom}")

    print(f"\n{sep}")
    print("  Detalhamento por caso:")
    print(sep)
    for r in results:
        if r.get("error"):
            print(f"  [{r['id']}] ERRO: {r['error']}")
            continue
        print(f"  [{r['id']}] latência={r['latency_ms']} ms")
        for q in r.get("questions", []):
            status = "✓" if q.get("correct") is True else ("✗" if q.get("correct") is False else "?")
            pc = f"  p={q['p_correct']:.3f}" if "p_correct" in q else ""
            print(f"    {status} {q['question']}: {q['prediction']} (esperado: {q['expected']}){pc}")
    print()


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

DEFAULT_DATASET = Path(__file__).parent / "datasets" / "default.jsonl"


def load_dataset(path: Path) -> list[dict[str, Any]]:
    samples = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"AVISO: linha {lineno} inválida em {path}: {exc}", file=sys.stderr)
    return samples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Laya textual on decision tasks")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", default="laya")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--base-url", default=OLLAMA_URL)
    parser.add_argument("--no-save", action="store_true", help="Do not save results to disk")
    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"Dataset não encontrado: {args.dataset}", file=sys.stderr)
        sys.exit(1)

    samples = load_dataset(args.dataset)
    print(f"Carregando {len(samples)} casos de '{args.dataset}'...")
    print(f"Modelo: {args.model}  |  Ollama: {args.base_url}\n")

    all_results = []
    for i, sample in enumerate(samples, 1):
        sid = sample.get("id", f"sample_{i}")
        print(f"[{i}/{len(samples)}] {sid} ...", end=" ", flush=True)
        result = evaluate_sample(sample, model=args.model, base_url=args.base_url)
        all_results.append(result)
        if result.get("error"):
            print(f"ERRO: {result['error']}")
        else:
            n_q = len(result.get("questions", []))
            n_ok = sum(1 for q in result["questions"] if q.get("correct") is True)
            print(f"{n_ok}/{n_q} corretas  {result['latency_ms']} ms")

    summary = aggregate(all_results)
    print_report(summary, all_results)

    if not args.no_save:
        args.out.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_name = f"{args.model}-{ts}"
        out_file = args.out / f"{run_name}.json"
        payload = {
            "run": run_name,
            "model": args.model,
            "dataset": str(args.dataset),
            "timestamp": ts,
            "summary": summary,
            "results": all_results,
        }
        out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        print(f"Resultados salvos em: {out_file}")


if __name__ == "__main__":
    main()

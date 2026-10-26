"""Recalculate metrics for Test 4 results with updated normalization."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS_FILE = ROOT / "eval/results/test_4_robustness_results.json"

COLOR_ANSWERS = {
    "black": ("black",), "white": ("white",),
    "gray_silver": ("gray", "grey", "silver"),
    "red": ("red",), "blue": ("blue",), "green": ("green",),
    "yellow": ("yellow",), "orange": ("orange",), "brown": ("brown",),
    "purple": ("purple", "violet"), "pink": ("pink",),
    "unknown": ("unknown", "none", "no", "not visible", "unidentifiable", "cannot"),
}

TYPE_ANSWERS = {
    "sedan": ("sedan", "saloon"), "hatchback": ("hatchback", "compact"),
    "suv": ("suv", "crossover"), "pickup": ("pickup", "pick up", "pick-up"),
    "van": ("van", "minivan"), "coupe": ("coupe", "coupé"),
    "convertible": ("convertible", "cabriolet"),
    "truck": ("truck", "lorry"), "bus": ("bus", "coach"),
    "motorcycle": ("motorcycle", "scooter", "motorbike"),
    "unknown": ("unknown", "none", "cannot", "not"),
}

HELMET_ANSWERS = {
    "yes": ("yes", "helmet", "hard hat", "hardhat"),
    "no": ("no", "bare", "hat", "cap"),
    "not_visible": ("not visible", "cannot", "unclear", "unknown", "not_visible"),
}

def _normalise_blip(answer: str, answer_map: dict[str, tuple[str, ...]]) -> str:
    cleaned = re.sub(r"[^a-z0-9 ]+", " ", answer.lower()).strip()
    words = set(cleaned.split())
    for label, terms in answer_map.items():
        if any(term in cleaned for term in terms):
            return label
    return "other" if cleaned else "unknown"

def _get_blip_normaliser(task: str):
    if task == "color":
        return lambda ans: _normalise_blip(ans, COLOR_ANSWERS)
    elif task == "vehicle_type":
        return lambda ans: _normalise_blip(ans, TYPE_ANSWERS)
    elif task == "helmet":
        return lambda ans: _normalise_blip(ans, HELMET_ANSWERS)
    else:
        raise ValueError(f"Unknown task: {task}")

def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if not n:
        return 0.0, 0.0
    p = k / n
    denom = 1 + (z * z) / n
    center = (p + (z * z) / (2 * n)) / denom
    half = (z * np.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / denom
    return round(float(max(0.0, center - half)), 4), round(float(min(1.0, center + half)), 4)

def main():
    with open(RESULTS_FILE) as f:
        data = json.load(f)

    strata = data["strata"]

    # Re-normalize blip answers
    for rec in data["per_image_results"]:
        norm_fn = _get_blip_normaliser(rec["task"])
        blip_data = rec["models"]["blip"]
        blip_data["normalized"] = [norm_fn(raw) for raw in blip_data["raw_answers"]]

        # Update consensus and is_correct
        for m_key in ["laya", "blip"]:
            m_data = rec["models"][m_key]
            preds = m_data["normalized"]
            m_data["pred_consensus"] = max(set(preds), key=preds.count) if preds else "unknown"
            m_data["is_correct"] = (m_data["pred_consensus"] == rec["expected"])

    # Recompute metrics per model
    for m_key in ["laya", "blip"]:
        stratum_metrics = {}
        task_metrics = {}

        for stratum in strata:
            stratum_items = [r for r in data["per_image_results"] if r["stratum"] == stratum]
            correct = sum(1 for r in stratum_items if r["models"][m_key]["is_correct"])
            total = len(stratum_items)
            acc = round(correct / total, 4) if total else 0.0
            wl, wh = wilson_interval(correct, total)

            stratum_lats = []
            for r in stratum_items:
                stratum_lats.extend(r["models"][m_key]["latencies_ms"])

            unknowns = sum(1 for r in stratum_items
                           if r["models"][m_key]["pred_consensus"] in ("unknown", "not_visible"))
            unknown_rate = round(unknowns / total, 4) if total else 0.0

            stratum_metrics[stratum] = {
                "correct": correct,
                "total": total,
                "accuracy": acc,
                "wilson_ci_95": [wl, wh],
                "unknown_rate": unknown_rate,
                "unknowns": unknowns,
                "latency_ms": {
                    "median": round(float(np.median(stratum_lats)), 1) if stratum_lats else 0.0,
                    "p90": round(float(np.percentile(stratum_lats, 90)), 1) if stratum_lats else 0.0,
                    "p95": round(float(np.percentile(stratum_lats, 95)), 1) if stratum_lats else 0.0,
                },
            }

        # Per-task breakdown
        for task in ["color", "vehicle_type", "helmet"]:
            task_items = [r for r in data["per_image_results"] if r["task"] == task]
            if not task_items:
                continue
            task_stratum_data = {}
            for stratum in strata:
                s_items = [r for r in task_items if r["stratum"] == stratum]
                if not s_items:
                    continue
                correct = sum(1 for r in s_items if r["models"][m_key]["is_correct"])
                total = len(s_items)
                acc = round(correct / total, 4) if total else 0.0
                unknowns = sum(1 for r in s_items
                               if r["models"][m_key]["pred_consensus"] in ("unknown", "not_visible"))
                task_stratum_data[stratum] = {
                    "correct": correct, "total": total, "accuracy": acc,
                    "unknown_rate": round(unknowns / total, 4) if total else 0.0,
                }
            task_metrics[task] = task_stratum_data

        # Accuracy deltas vs original baseline
        orig_acc = stratum_metrics.get("original", {}).get("accuracy", 0.0)
        accuracy_deltas = {}
        for stratum in strata:
            if stratum == "original":
                continue
            s_acc = stratum_metrics[stratum]["accuracy"]
            accuracy_deltas[stratum] = round(s_acc - orig_acc, 4)

        data["models"][m_key]["overall_accuracy"] = {
            "correct": sum(s["correct"] for s in stratum_metrics.values()),
            "total": sum(s["total"] for s in stratum_metrics.values()),
            "accuracy": round(
                sum(s["correct"] for s in stratum_metrics.values()) /
                max(1, sum(s["total"] for s in stratum_metrics.values())), 4),
        }
        data["models"][m_key]["per_stratum"] = stratum_metrics
        data["models"][m_key]["accuracy_deltas_vs_original"] = accuracy_deltas
        data["models"][m_key]["per_task_per_stratum"] = task_metrics

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("Recalculation complete!")

if __name__ == "__main__":
    main()

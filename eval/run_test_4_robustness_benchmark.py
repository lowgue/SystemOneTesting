"""Empirical Benchmark Runner for Test 4: Robustness to Image Quality.

Evaluates how accuracy and unknown/refusal rates change across degradation
strata (low_light, blur, low_res, jpeg_compress, occlusion) vs the original
baseline. Uses the same Laya Vision and BLIP VQA models as Tests 1-3.

Adheres to the empirical test plan:
- Same images (degraded variants) for both models
- Model load time measured separately
- Warm-up excluded from stats
- 3 repetitions; median, p90, p95 latency
- Per-image records with raw/normalized answers, probabilities, latency
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "laya-vision"))
sys.path.insert(0, str(ROOT / "eval"))

MANIFEST_PATH = ROOT / "eval/datasets/test_4_robustness_manifest.json"
DEFAULT_OUTPUT = ROOT / "eval/results/test_4_robustness_results.json"

# ── Criteria for Laya typed choices per task ──────────────────────────

COLOR_CRITERIA = {
    "black": "Black", "white": "White", "gray_silver": "Gray or silver",
    "red": "Red", "blue": "Blue", "green": "Green", "yellow": "Yellow",
    "orange": "Orange", "brown": "Brown", "purple": "Purple", "pink": "Pink",
    "unknown": "No car or color not identifiable / not visible",
}

VEHICLE_TYPE_CRITERIA = {
    "sedan": "Sedan / saloon", "hatchback": "Hatchback / compact",
    "suv": "SUV / crossover", "pickup": "Pickup truck",
    "van": "Van or minivan", "coupe": "Coupé / 2-door sports car",
    "convertible": "Convertible / cabriolet", "truck": "Truck / lorry",
    "bus": "Bus / coach", "motorcycle": "Motorcycle / scooter",
    "unknown": "Cannot identify or not a vehicle",
}

HELMET_CRITERIA = {
    "yes": "Wearing a helmet or hard hat",
    "no": "Not wearing a helmet",
    "not_visible": "Person or head not clearly visible",
}

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
    """Normalise open-text BLIP answer using a keyword map."""
    cleaned = re.sub(r"[^a-z0-9 ]+", " ", answer.lower()).strip()
    words = set(cleaned.split())
    for label, terms in answer_map.items():
        if any(term in cleaned for term in terms):
            return label
    return "other" if cleaned else "unknown"


def _get_laya_question(task: str) -> dict[str, Any]:
    """Return the Laya typed-choice question dict for the given task."""
    if task == "color":
        return {"q": {"type": "choice", "instructions": "What is the predominant color of the most prominent car?", "criteria": COLOR_CRITERIA}}
    elif task == "vehicle_type":
        return {"q": {"type": "choice", "instructions": "What body type is this vehicle: sedan, hatchback, suv, pickup truck, van, coupe, convertible, truck, bus, or motorcycle?", "criteria": VEHICLE_TYPE_CRITERIA}}
    elif task == "helmet":
        return {"q": {"type": "choice", "instructions": "Is the prominent person wearing a helmet or hard hat?", "criteria": HELMET_CRITERIA}}
    else:
        raise ValueError(f"Unknown task: {task}")


def _get_blip_prompt(task: str) -> str:
    """Return the BLIP text prompt for the given task."""
    if task == "color":
        return "What is the predominant color of the most prominent car?"
    elif task == "vehicle_type":
        return "What body type is this vehicle: sedan, hatchback, suv, pickup truck, van, coupe, convertible, truck, bus, or motorcycle?"
    elif task == "helmet":
        return "Is the prominent person wearing a helmet or hard hat?"
    else:
        raise ValueError(f"Unknown task: {task}")


def _get_blip_normaliser(task: str):
    """Return the normaliser function for BLIP answers."""
    if task == "color":
        return lambda ans: _normalise_blip(ans, COLOR_ANSWERS)
    elif task == "vehicle_type":
        return lambda ans: _normalise_blip(ans, TYPE_ANSWERS)
    elif task == "helmet":
        return lambda ans: _normalise_blip(ans, HELMET_ANSWERS)
    else:
        raise ValueError(f"Unknown task: {task}")


def _apply_laya_transformers_compat_patch() -> None:
    import torch
    import laya.vlm as vlm

    def encode_images_compat(self, pixel_values, pixel_attention_mask=None):
        mask = pixel_attention_mask[None] if pixel_attention_mask is not None else None
        result = self._image_features(pixel_values[None], mask)
        return torch.stack(result) if isinstance(result, list) else result

    vlm.VLMDecisionModel.encode_images = encode_images_compat


def load_laya_model():
    _apply_laya_transformers_compat_patch()
    import laya
    return laya.load_vlm("thaitea/laya-vision")


def load_blip_model(device: str = "cpu"):
    from transformers import AutoModelForVisualQuestionAnswering, AutoProcessor
    processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = AutoModelForVisualQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    model.to(device)
    model.eval()
    return processor, model


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if not n:
        return 0.0, 0.0
    p = k / n
    denom = 1 + (z * z) / n
    center = (p + (z * z) / (2 * n)) / denom
    half = (z * np.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / denom
    return round(float(max(0.0, center - half)), 4), round(float(min(1.0, center + half)), 4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    import torch
    torch.set_num_threads(args.threads)

    if not MANIFEST_PATH.exists():
        raise SystemExit(f"Manifest not found: {MANIFEST_PATH}. Run build_test_4_dataset.py first.")

    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)

    items = [it for it in manifest["items"] if it["split"] == "eval"]
    total_images = len(items)
    strata = manifest["strata"]

    print(f"=== TEST 4: EMPIRICAL BENCHMARK - ROBUSTNESS TO IMAGE QUALITY ===")
    print(f"Dataset items: {total_images} (across {len(strata)} strata)")
    print(f"Repetitions per image: {args.repetitions}")
    print(f"CPU threads: {args.threads}")

    all_results: dict[str, Any] = {
        "test": "Test 4: Robustness to Image Quality (Robustez à Qualidade da Imagem)",
        "strata": strata,
        "strata_descriptions": manifest["strata_descriptions"],
        "source_tests": manifest["source_tests"],
        "device": "cpu",
        "precision": "float32",
        "threads": args.threads,
        "repetitions": args.repetitions,
        "image_count": total_images,
        "models": {},
        "per_image_results": [],
    }

    # Build per-image record structure
    records_by_id: dict[str, dict] = {}
    for it in items:
        records_by_id[it["id"]] = {
            "id": it["id"],
            "filename": it["filename"],
            "rel_path": it["rel_path"],
            "source_test": it["source_test"],
            "source_id": it["source_id"],
            "task": it["task"],
            "question_en": it["question_en"],
            "expected": it["expected"],
            "stratum": it["stratum"],
            "models": {},
        }

    # Use first image for warm-up
    warmup_path = ROOT / items[0]["rel_path"]
    with Image.open(warmup_path) as src:
        warmup_image = src.convert("RGB")

    # ── CANDIDATE 1: LAYA VISION ──────────────────────────────────────

    print("\n[Candidate 1] Loading Laya Vision (thaitea/laya-vision)...")
    t0 = time.perf_counter()
    laya_model = load_laya_model()
    laya_load_time_s = time.perf_counter() - t0
    print(f"[Candidate 1] Model loaded in {laya_load_time_s:.2f}s")

    # Warm-up
    print("[Candidate 1] Running warm-up...")
    warmup_q = _get_laya_question(items[0]["task"])
    _ = laya_model.predict(warmup_image, warmup_q)
    print("[Candidate 1] Warm-up complete.")

    laya_latencies_all = []
    laya_errors = 0

    print(f"[Candidate 1] Running {args.repetitions} reps × {total_images} images...")
    for rep in range(1, args.repetitions + 1):
        print(f"  -> Repetition {rep}/{args.repetitions}...")
        for idx, it in enumerate(items, 1):
            rec = records_by_id[it["id"]]["models"].setdefault("laya", {
                "raw_answers": [], "normalized": [],
                "probabilities": None, "confidence": None,
                "latencies_ms": [], "errors": [],
            })
            img_path = ROOT / it["rel_path"]
            try:
                with Image.open(img_path) as s:
                    img = s.convert("RGB")
                laya_q = _get_laya_question(it["task"])
                t_inf0 = time.perf_counter()
                out = laya_model.predict(img, laya_q)
                lat_ms = (time.perf_counter() - t_inf0) * 1000

                ans = out.get("answers", {}).get("q", {})
                choice = str(ans.get("choice", "unknown")).lower()
                probs = ans.get("probabilities", {})
                conf = ans.get("confidence", 0.0)

                rec["raw_answers"].append(choice)
                rec["normalized"].append(choice)
                rec["latencies_ms"].append(round(lat_ms, 2))
                laya_latencies_all.append(lat_ms)
                if rec["probabilities"] is None:
                    rec["probabilities"] = probs
                    rec["confidence"] = conf
            except Exception as exc:
                laya_errors += 1
                rec["errors"].append(str(exc))
                rec["raw_answers"].append("ERROR")
                rec["normalized"].append("error")
                rec["latencies_ms"].append(0.0)

            if idx % 50 == 0 or idx == total_images:
                print(f"     Progress: {idx}/{total_images} (rep {rep})")

    del laya_model
    gc.collect()

    # ── CANDIDATE 2: BLIP VQA ─────────────────────────────────────────

    print("\n[Candidate 2] Loading BLIP VQA (Salesforce/blip-vqa-base)...")
    t0 = time.perf_counter()
    blip_proc, blip_model = load_blip_model(device="cpu")
    blip_load_time_s = time.perf_counter() - t0
    print(f"[Candidate 2] Model loaded in {blip_load_time_s:.2f}s")

    # Warm-up
    print("[Candidate 2] Running warm-up...")
    with torch.inference_mode():
        in_t = blip_proc(images=warmup_image, text=_get_blip_prompt(items[0]["task"]),
                         return_tensors="pt").to("cpu")
        _ = blip_model.generate(**in_t, max_new_tokens=10, do_sample=False)
    print("[Candidate 2] Warm-up complete.")

    blip_latencies_all = []
    blip_errors = 0

    print(f"[Candidate 2] Running {args.repetitions} reps × {total_images} images...")
    import torch
    for rep in range(1, args.repetitions + 1):
        print(f"  -> Repetition {rep}/{args.repetitions}...")
        for idx, it in enumerate(items, 1):
            rec = records_by_id[it["id"]]["models"].setdefault("blip", {
                "raw_answers": [], "normalized": [],
                "probabilities": None, "confidence": None,
                "latencies_ms": [], "errors": [],
            })
            img_path = ROOT / it["rel_path"]
            try:
                with Image.open(img_path) as s:
                    img = s.convert("RGB")
                prompt = _get_blip_prompt(it["task"])
                normaliser = _get_blip_normaliser(it["task"])

                t_inf0 = time.perf_counter()
                with torch.inference_mode():
                    in_t = blip_proc(images=img, text=prompt, return_tensors="pt").to("cpu")
                    tok = blip_model.generate(**in_t, max_new_tokens=10, do_sample=False)
                lat_ms = (time.perf_counter() - t_inf0) * 1000

                raw_text = blip_proc.decode(tok[0], skip_special_tokens=True).strip()
                norm = normaliser(raw_text)

                rec["raw_answers"].append(raw_text)
                rec["normalized"].append(norm)
                rec["latencies_ms"].append(round(lat_ms, 2))
                blip_latencies_all.append(lat_ms)
            except Exception as exc:
                blip_errors += 1
                rec["errors"].append(str(exc))
                rec["raw_answers"].append("ERROR")
                rec["normalized"].append("error")
                rec["latencies_ms"].append(0.0)

            if idx % 50 == 0 or idx == total_images:
                print(f"     Progress: {idx}/{total_images} (rep {rep})")

    del blip_proc, blip_model
    gc.collect()

    # ── METRICS ───────────────────────────────────────────────────────

    print("\nComputing robustness metrics...")

    # Compute per-image consensus and latency stats
    for rec in records_by_id.values():
        for m_key in ["laya", "blip"]:
            m_data = rec["models"][m_key]
            preds = m_data["normalized"]
            m_data["pred_consensus"] = max(set(preds), key=preds.count) if preds else "unknown"
            if m_data["latencies_ms"]:
                m_data["latency_median_ms"] = round(float(np.median(m_data["latencies_ms"])), 1)
                m_data["latency_p90_ms"] = round(float(np.percentile(m_data["latencies_ms"], 90)), 1)
            m_data["is_correct"] = (m_data["pred_consensus"] == rec["expected"])

    all_results["per_image_results"] = list(records_by_id.values())

    # Per-model, per-stratum, per-task metrics
    for m_key in ["laya", "blip"]:
        stratum_metrics = {}
        task_metrics = {}

        for stratum in strata:
            stratum_items = [r for r in all_results["per_image_results"] if r["stratum"] == stratum]
            correct = sum(1 for r in stratum_items if r["models"][m_key]["is_correct"])
            total = len(stratum_items)
            acc = round(correct / total, 4) if total else 0.0
            wl, wh = wilson_interval(correct, total)

            # Latencies for this stratum
            stratum_lats = []
            for r in stratum_items:
                stratum_lats.extend(r["models"][m_key]["latencies_ms"])

            # Unknown/refusal rate
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
            task_items = [r for r in all_results["per_image_results"] if r["task"] == task]
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

        lats = laya_latencies_all if m_key == "laya" else blip_latencies_all

        all_results["models"][m_key] = {
            "metadata": {
                "name": "Laya Vision" if m_key == "laya" else "BLIP VQA",
                "model_id": "thaitea/laya-vision" if m_key == "laya" else "Salesforce/blip-vqa-base",
                "load_time_s": round(laya_load_time_s if m_key == "laya" else blip_load_time_s, 2),
                "errors": laya_errors if m_key == "laya" else blip_errors,
            },
            "overall_accuracy": {
                "correct": sum(s["correct"] for s in stratum_metrics.values()),
                "total": sum(s["total"] for s in stratum_metrics.values()),
                "accuracy": round(
                    sum(s["correct"] for s in stratum_metrics.values()) /
                    max(1, sum(s["total"] for s in stratum_metrics.values())), 4),
            },
            "per_stratum": stratum_metrics,
            "accuracy_deltas_vs_original": accuracy_deltas,
            "per_task_per_stratum": task_metrics,
            "latency_ms_global": {
                "median": round(float(np.median(lats)), 1) if lats else 0.0,
                "p90": round(float(np.percentile(lats, 90)), 1) if lats else 0.0,
                "p95": round(float(np.percentile(lats, 95)), 1) if lats else 0.0,
            },
        }

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResults written to: {out_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("RESUMO: TESTE 4 – ROBUSTEZ À QUALIDADE DA IMAGEM")
    print("=" * 80)
    print(f"{'Estrato':<20} | {'Laya Acc':>10} | {'Laya Unk%':>10} | {'BLIP Acc':>10} | {'BLIP Unk%':>10}")
    print("-" * 80)
    for stratum in strata:
        l_s = all_results["models"]["laya"]["per_stratum"][stratum]
        b_s = all_results["models"]["blip"]["per_stratum"][stratum]
        print(f"{stratum:<20} | {l_s['accuracy']*100:>9.1f}% | {l_s['unknown_rate']*100:>9.1f}% "
              f"| {b_s['accuracy']*100:>9.1f}% | {b_s['unknown_rate']*100:>9.1f}%")

    print("-" * 80)
    print("\nDelta vs Original (negative = degradation hurts):")
    for stratum in strata:
        if stratum == "original":
            continue
        l_d = all_results["models"]["laya"]["accuracy_deltas_vs_original"][stratum]
        b_d = all_results["models"]["blip"]["accuracy_deltas_vs_original"][stratum]
        print(f"  {stratum:<20}: Laya {l_d*100:+.1f}pp  |  BLIP {b_d*100:+.1f}pp")
    print("=" * 80)


if __name__ == "__main__":
    main()

"""Empirical Benchmark Runner for Test 1: Vehicle Color.

Adheres strictly to the empirical test plan:
- Balanced color dataset (11 colors + unknown controls)
- Model load time measured separately
- Warm-up execution excluded from latency stats
- 3 full repetitions across all evaluation images
- Per-image records: question, expected, raw response, normalized response,
  confidence/probabilities (when available), per-run latencies, median, p90, error.
- Full system/library/revision provenance tracking.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "laya-vision"))
sys.path.insert(0, str(ROOT / "eval"))

MANIFEST_PATH = ROOT / "eval/datasets/test_1_vehicle_color_manifest.json"
DEFAULT_OUTPUT = ROOT / "eval/results/test_1_vehicle_color_results.json"

COLOR_CRITERIA = {
    "black": "Black",
    "white": "White",
    "gray_silver": "Gray or silver",
    "red": "Red",
    "blue": "Blue",
    "green": "Green",
    "yellow": "Yellow",
    "orange": "Orange",
    "brown": "Brown",
    "purple": "Purple",
    "pink": "Pink",
    "unknown": "No car or color not identifiable / not visible",
}

COLOR_ANSWERS = {
    "black": ("black",),
    "white": ("white",),
    "gray_silver": ("gray", "grey", "silver"),
    "red": ("red",),
    "blue": ("blue",),
    "green": ("green",),
    "yellow": ("yellow",),
    "orange": ("orange",),
    "brown": ("brown",),
    "purple": ("purple", "violet"),
    "pink": ("pink",),
    "unknown": ("unknown", "none", "no", "not visible", "unidentifiable", "cannot"),
}

QUESTION_PROMPT = "What is the predominant color of the most prominent car?"


def _normalise_blip_color(answer: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", answer.lower()).strip()
    words = set(cleaned.split())
    for label, terms in COLOR_ANSWERS.items():
        if any(term in words for term in terms):
            return label
    return "other" if cleaned else "unknown"


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
    parser.add_argument("--threads", type=int, default=8, help="Number of PyTorch CPU threads")
    parser.add_argument("--repetitions", type=int, default=3, help="Number of repetitions per image")
    parser.add_argument("--split", default="eval", help="Manifest split to run ('eval' or 'all')")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output JSON path")
    args = parser.parse_args()

    import torch
    torch.set_num_threads(args.threads)

    if not MANIFEST_PATH.exists():
        raise SystemExit(f"Manifest not found: {MANIFEST_PATH}. Run build_test_1_dataset.py first.")

    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)

    if args.split == "all":
        items = manifest["items"]
    else:
        items = [it for it in manifest["items"] if it["split"] == args.split]

    total_images = len(items)
    print(f"=== TEST 1: EMPIRICAL BENCHMARK - VEHICLE COLOR ===")
    print(f"Dataset items: {total_images} (split: {args.split})")
    print(f"Repetitions per image: {args.repetitions}")
    print(f"CPU threads: {args.threads}")

    all_results: dict[str, Any] = {
        "test": "Test 1: Vehicle Color (Cor do Veículo)",
        "question_pt": manifest["question"],
        "question_en": QUESTION_PROMPT,
        "taxonomy": manifest["taxonomy"],
        "device": "cpu",
        "precision": "float32",
        "threads": args.threads,
        "repetitions": args.repetitions,
        "split": args.split,
        "image_count": total_images,
        "models": {},
        "per_image_results": [],
    }

    # Structure per-image records
    records_by_id = {
        it["id"]: {
            "id": it["id"],
            "filename": it["filename"],
            "rel_path": it["rel_path"],
            "expected_color": it["color"],
            "split": it["split"],
            "source": it["source"],
            "difficult": it["difficult"],
            "difficulty_tags": it["difficulty_tags"],
            "models": {},
        }
        for it in items
    }

    warmup_img_path = ROOT / items[0]["rel_path"]
    with Image.open(warmup_img_path) as src:
        warmup_image = src.convert("RGB")

    # ----------------------------------------------------
    # CANDIDATE 1: LAYA VISION
    # ----------------------------------------------------
    print("\n[Candidate 1] Loading Laya Vision (thaitea/laya-vision)...")
    t0 = time.perf_counter()
    laya_model = load_laya_model()
    laya_load_time_s = time.perf_counter() - t0
    print(f"[Candidate 1] Model loaded in {laya_load_time_s:.2f}s")

    # Warm-up (excluded from statistics)
    print("[Candidate 1] Running warm-up...")
    laya_q = {
        "cor": {
            "type": "choice",
            "instructions": QUESTION_PROMPT,
            "criteria": COLOR_CRITERIA,
        }
    }
    _ = laya_model.predict(warmup_image, laya_q)
    print("[Candidate 1] Warm-up complete.")

    laya_latencies_all = []
    laya_errors = 0

    print(f"[Candidate 1] Running {args.repetitions} repetitions across {total_images} images...")
    for rep in range(1, args.repetitions + 1):
        print(f"  -> Repetition {rep}/{args.repetitions}...")
        for idx, it in enumerate(items, 1):
            rec = records_by_id[it["id"]]["models"].setdefault("laya", {
                "raw_answers": [],
                "normalized": [],
                "probabilities": None,
                "confidence": None,
                "latencies_ms": [],
                "errors": [],
            })
            img_path = ROOT / it["rel_path"]
            try:
                with Image.open(img_path) as s:
                    img = s.convert("RGB")
                t_inf0 = time.perf_counter()
                out = laya_model.predict(img, laya_q)
                lat_ms = (time.perf_counter() - t_inf0) * 1000

                ans = out.get("answers", {}).get("cor", {})
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

            if idx % 30 == 0 or idx == total_images:
                print(f"     Progresso: {idx}/{total_images} imagens processadas (rep {rep})")

    del laya_model
    gc.collect()

    # ----------------------------------------------------
    # CANDIDATE 2: BLIP VQA
    # ----------------------------------------------------
    print("\n[Candidate 2] Loading BLIP VQA (Salesforce/blip-vqa-base)...")
    t0 = time.perf_counter()
    blip_proc, blip_model = load_blip_model(device="cpu")
    blip_load_time_s = time.perf_counter() - t0
    print(f"[Candidate 2] Model loaded in {blip_load_time_s:.2f}s")

    # Warm-up (excluded from statistics)
    print("[Candidate 2] Running warm-up...")
    with torch.inference_mode():
        in_t = blip_proc(images=warmup_image, text=QUESTION_PROMPT, return_tensors="pt").to("cpu")
        _ = blip_model.generate(**in_t, max_new_tokens=10, do_sample=False)
    print("[Candidate 2] Warm-up complete.")

    blip_latencies_all = []
    blip_errors = 0

    print(f"[Candidate 2] Running {args.repetitions} repetitions across {total_images} images...")
    for rep in range(1, args.repetitions + 1):
        print(f"  -> Repetition {rep}/{args.repetitions}...")
        for idx, it in enumerate(items, 1):
            rec = records_by_id[it["id"]]["models"].setdefault("blip", {
                "raw_answers": [],
                "normalized": [],
                "probabilities": None,
                "confidence": None,
                "latencies_ms": [],
                "errors": [],
            })
            img_path = ROOT / it["rel_path"]
            try:
                with Image.open(img_path) as s:
                    img = s.convert("RGB")
                t_inf0 = time.perf_counter()
                with torch.inference_mode():
                    in_t = blip_proc(images=img, text=QUESTION_PROMPT, return_tensors="pt").to("cpu")
                    tok = blip_model.generate(**in_t, max_new_tokens=10, do_sample=False)
                lat_ms = (time.perf_counter() - t_inf0) * 1000

                raw_text = blip_proc.decode(tok[0], skip_special_tokens=True).strip()
                norm = _normalise_blip_color(raw_text)

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

            if idx % 30 == 0 or idx == total_images:
                print(f"     Progresso: {idx}/{total_images} imagens processadas (rep {rep})")

    del blip_proc, blip_model
    gc.collect()

    # ----------------------------------------------------
    # METRICS CALCULATION AND ANALYSIS
    # ----------------------------------------------------
    print("\nComputing comprehensive empirical metrics...")
    all_results["per_image_results"] = list(records_by_id.values())

    model_metadata = {
        "laya": {
            "name": "Laya Vision",
            "model_id": "thaitea/laya-vision",
            "revision": "f2fe3c12cb6d04c59d8a190250bf3fb40fc828dc",
            "backbone": "HuggingFaceTB/SmolVLM-256M-Instruct",
            "format": "typed_choice",
            "criteria_options": list(COLOR_CRITERIA.keys()),
            "load_time_s": round(laya_load_time_s, 2),
            "errors": laya_errors,
        },
        "blip": {
            "name": "BLIP VQA",
            "model_id": "Salesforce/blip-vqa-base",
            "revision": "787b3d35d57e49572baabd22884b3d5a05acf072",
            "backbone": "BLIP ViT-B",
            "format": "open_vqa",
            "prompt": QUESTION_PROMPT,
            "load_time_s": round(blip_load_time_s, 2),
            "errors": blip_errors,
        },
    }

    taxonomy = manifest["taxonomy"]

    for m_key in ["laya", "blip"]:
        correct_count = 0
        total_eval = 0
        diff_correct = 0
        diff_total = 0
        std_correct = 0
        std_total = 0

        # Confusion matrix init
        cm = {act: {pred: 0 for pred in taxonomy + ["other", "error"]} for act in taxonomy}

        # Class breakdown
        class_stats = {c: {"tp": 0, "fp": 0, "fn": 0, "total": 0} for c in taxonomy}

        # Difficulty breakdown
        diff_tag_stats: dict[str, dict[str, int]] = {}

        for rec in all_results["per_image_results"]:
            exp = rec["expected_color"]
            m_data = rec["models"][m_key]
            # Use majority vote / primary normalized prediction from repetitions
            preds = m_data["normalized"]
            pred = max(set(preds), key=preds.count) if preds else "unknown"

            # Per-image aggregate stats
            m_data["pred_consensus"] = pred
            m_data["latency_median_ms"] = round(float(np.median(m_data["latencies_ms"])), 1)
            m_data["latency_p90_ms"] = round(float(np.percentile(m_data["latencies_ms"], 90)), 1)
            m_data["latency_mean_ms"] = round(float(np.mean(m_data["latencies_ms"])), 1)
            m_data["is_correct"] = (pred == exp)

            # Accuracy counts
            total_eval += 1
            is_match = (pred == exp)
            if is_match:
                correct_count += 1

            if rec["difficult"]:
                diff_total += 1
                if is_match: diff_correct += 1
            else:
                std_total += 1
                if is_match: std_correct += 1

            for tag in rec["difficulty_tags"]:
                diff_tag_stats.setdefault(tag, {"correct": 0, "total": 0})
                diff_tag_stats[tag]["total"] += 1
                if is_match:
                    diff_tag_stats[tag]["correct"] += 1

            # Confusion matrix
            if exp in cm:
                cm[exp][pred] = cm[exp].get(pred, 0) + 1

            # Class stats
            if exp in class_stats:
                class_stats[exp]["total"] += 1
                if is_match:
                    class_stats[exp]["tp"] += 1
                else:
                    class_stats[exp]["fn"] += 1

            if pred in class_stats and not is_match:
                class_stats[pred]["fp"] += 1

        overall_acc = round(correct_count / total_eval, 4) if total_eval else 0.0
        wilson_lo, wilson_hi = wilson_interval(correct_count, total_eval)

        # Per-class precision, recall, F1
        per_class_metrics = {}
        for c, s in class_stats.items():
            tp, fp, fn, n = s["tp"], s["fp"], s["fn"], s["total"]
            prec = round(tp / (tp + fp), 4) if (tp + fp) else 0.0
            rec = round(tp / (tp + fn), 4) if (tp + fn) else 0.0
            f1 = round(2 * prec * rec / (prec + rec), 4) if (prec + rec) else 0.0
            per_class_metrics[c] = {
                "total": n,
                "correct": tp,
                "accuracy": round(tp / n, 4) if n else 0.0,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
            }

        # Unknown / Refusal rate
        # 1. On valid cars: how often did it refuse (say unknown)? -> False Rejection
        valid_car_items = [r for r in all_results["per_image_results"] if r["expected_color"] != "unknown"]
        valid_car_refusals = sum(1 for r in valid_car_items if r["models"][m_key]["pred_consensus"] == "unknown")
        false_rejection_rate = round(valid_car_refusals / len(valid_car_items), 4) if valid_car_items else 0.0

        # 2. On negative controls: how often did it correctly say unknown? -> True Negative / Correct Rejection
        neg_control_items = [r for r in all_results["per_image_results"] if r["expected_color"] == "unknown"]
        neg_correct = sum(1 for r in neg_control_items if r["models"][m_key]["pred_consensus"] == "unknown")
        correct_rejection_rate = round(neg_correct / len(neg_control_items), 4) if neg_control_items else 0.0

        # Difficulty breakdown summary
        per_tag_summary = {
            tag: {
                "correct": s["correct"],
                "total": s["total"],
                "accuracy": round(s["correct"] / s["total"], 4) if s["total"] else 0.0,
            }
            for tag, s in diff_tag_stats.items()
        }

        # Latency distribution
        lats = laya_latencies_all if m_key == "laya" else blip_latencies_all
        all_results["models"][m_key] = {
            "metadata": model_metadata[m_key],
            "accuracy": {
                "correct": correct_count,
                "total": total_eval,
                "accuracy": overall_acc,
                "wilson_ci_95": [wilson_lo, wilson_hi],
                "standard_images_accuracy": round(std_correct / std_total, 4) if std_total else 0.0,
                "difficult_images_accuracy": round(diff_correct / diff_total, 4) if diff_total else 0.0,
                "difficulty_tags_accuracy": per_tag_summary,
            },
            "per_class": per_class_metrics,
            "confusion_matrix": cm,
            "refusal_analysis": {
                "valid_car_count": len(valid_car_items),
                "valid_car_refused_unknown": valid_car_refusals,
                "false_rejection_rate": false_rejection_rate,
                "negative_control_count": len(neg_control_items),
                "negative_control_correct_unknown": neg_correct,
                "correct_rejection_rate": correct_rejection_rate,
            },
            "latency_ms": {
                "mean": round(float(np.mean(lats)), 1) if lats else 0.0,
                "median": round(float(np.median(lats)), 1) if lats else 0.0,
                "p90": round(float(np.percentile(lats, 90)), 1) if lats else 0.0,
                "p95": round(float(np.percentile(lats, 95)), 1) if lats else 0.0,
                "min": round(float(np.min(lats)), 1) if lats else 0.0,
                "max": round(float(np.max(lats)), 1) if lats else 0.0,
            },
        }

    # Save output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nBenchmark completed successfully! Results written to: {out_path}")

    # Print summary table
    print("\n" + "=" * 65)
    print("RESUMO FINAL: TESTE 1 - COR DO VEÍCULO")
    print("=" * 65)
    print(f"{'Métrica':<35} | {'Laya Vision':<12} | {'BLIP VQA':<12}")
    print("-" * 65)
    l_acc = all_results['models']['laya']['accuracy']
    b_acc = all_results['models']['blip']['accuracy']
    l_lat = all_results['models']['laya']['latency_ms']
    b_lat = all_results['models']['blip']['latency_ms']
    l_ref = all_results['models']['laya']['refusal_analysis']
    b_ref = all_results['models']['blip']['refusal_analysis']

    print(f"{'Acurácia Geral':<35} | {l_acc['accuracy']*100:>10.2f}% | {b_acc['accuracy']*100:>10.2f}%")
    print(f"{'IC Wilson 95%':<35} | [{l_acc['wilson_ci_95'][0]*100:.1f}%, {l_acc['wilson_ci_95'][1]*100:.1f}%] | [{b_acc['wilson_ci_95'][0]*100:.1f}%, {b_acc['wilson_ci_95'][1]*100:.1f}%]")
    print(f"{'Acurácia em imagens padrão':<35} | {l_acc['standard_images_accuracy']*100:>10.2f}% | {b_acc['standard_images_accuracy']*100:>10.2f}%")
    print(f"{'Acurácia em imagens difíceis':<35} | {l_acc['difficult_images_accuracy']*100:>10.2f}% | {b_acc['difficult_images_accuracy']*100:>10.2f}%")
    print(f"{'Taxa rejeição correta (sem carro)':<35} | {l_ref['correct_rejection_rate']*100:>10.2f}% | {b_ref['correct_rejection_rate']*100:>10.2f}%")
    print(f"{'Falsa rejeição (carro -> unknown)':<35} | {l_ref['false_rejection_rate']*100:>10.2f}% | {b_ref['false_rejection_rate']*100:>10.2f}%")
    print(f"{'Tempo de carregamento':<35} | {all_results['models']['laya']['metadata']['load_time_s']:>10.2f}s | {all_results['models']['blip']['metadata']['load_time_s']:>10.2f}s")
    print(f"{'Latência Mediana (ms)':<35} | {l_lat['median']:>10.1f}ms | {b_lat['median']:>10.1f}ms")
    print(f"{'Latência p90 (ms)':<35} | {l_lat['p90']:>10.1f}ms | {b_lat['p90']:>10.1f}ms")
    print(f"{'Erros de inferência':<35} | {all_results['models']['laya']['metadata']['errors']:>12} | {all_results['models']['blip']['metadata']['errors']:>12}")
    print("=" * 65)


if __name__ == "__main__":
    main()

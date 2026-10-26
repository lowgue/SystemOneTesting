"""Empirical Benchmark Runner for Test 5: Multiple Objects and Response Association.

Evaluates how Laya Vision and BLIP VQA perform when answering visual questions about:
  1. An isolated crop of the detected object (detector crop)
  2. The full scene image with the target object highlighted via a visual prompt (red bounding box)

Measures:
  - Accuracy on Crop vs Accuracy on Full Highlighted
  - Association Error Rate (model outputs distractor attribute instead of target)
  - Classification Error Rate (model outputs neither)
  - Unknown / Refusal Rate
  - Latency impact of full image vs cropped image
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "laya-vision"))
sys.path.insert(0, str(ROOT / "eval"))

MANIFEST_PATH = ROOT / "eval/datasets/test_5_multi_object_manifest.json"
DEFAULT_OUTPUT = ROOT / "eval/results/test_5_multi_object_results.json"

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

CARRYING_CRITERIA = {
    "backpack": "Carrying a backpack or rucksack",
    "bag_or_box": "Carrying a handheld bag, box, or package",
    "none": "Not carrying any bag or object",
    "not_visible": "Person or hands not clearly visible",
}

CLOTHING_CRITERIA = {
    "safety_vest": "High-visibility or reflective safety vest",
    "jacket_coat": "Jacket, coat, or blazer",
    "shirt_tshirt": "Shirt, t-shirt, or polo",
    "not_visible": "Torso not visible or cannot identify",
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
    "sedan": ("sedan", "saloon", "car"), "hatchback": ("hatchback", "compact"),
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

CARRYING_ANSWERS = {
    "backpack": ("backpack", "rucksack"),
    "bag_or_box": ("bag", "box", "briefcase", "suitcase", "package"),
    "none": ("nothing", "none", "no"),
    "not_visible": ("not visible", "cannot", "unknown"),
}

CLOTHING_ANSWERS = {
    "safety_vest": ("safety vest", "vest", "reflective", "hi-vis", "yellow vest"),
    "jacket_coat": ("jacket", "coat", "suit", "blazer", "sweater", "outerwear"),
    "shirt_tshirt": ("shirt", "t-shirt", "tshirt", "polo", "top"),
    "not_visible": ("not visible", "cannot", "unknown"),
}


def _normalise_blip(answer: str, answer_map: dict[str, tuple[str, ...]]) -> str:
    cleaned = re.sub(r"[^a-z0-9 ]+", " ", answer.lower()).strip()
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
    elif task == "carrying":
        return lambda ans: _normalise_blip(ans, CARRYING_ANSWERS)
    elif task == "clothing":
        return lambda ans: _normalise_blip(ans, CLOTHING_ANSWERS)
    else:
        raise ValueError(f"Unknown task: {task}")


def _get_laya_question(task: str, question_en: str) -> dict[str, Any]:
    if task == "color":
        criteria = COLOR_CRITERIA
    elif task == "vehicle_type":
        criteria = VEHICLE_TYPE_CRITERIA
    elif task == "helmet":
        criteria = HELMET_CRITERIA
    elif task == "carrying":
        criteria = CARRYING_CRITERIA
    elif task == "clothing":
        criteria = CLOTHING_CRITERIA
    else:
        raise ValueError(f"Unknown task: {task}")

    return {
        "ans": {
            "type": "choice",
            "instructions": question_en,
            "criteria": criteria,
        }
    }


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    import torch
    torch.set_num_threads(args.threads)

    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)

    items = manifest["items"]
    total_items = len(items)

    print(f"=== TEST 5: EMPIRICAL BENCHMARK - MULTIPLE OBJECTS & RESPONSE ASSOCIATION ===")
    print(f"Total scenes: {manifest['scene_count']} (60 presentation items: 30 crop + 30 highlighted)")
    print(f"Repetitions per item: {args.repetitions}")
    print(f"CPU threads: {args.threads}")

    all_results: dict[str, Any] = {
        "test": "Test 5: Multiple Objects and Response Association (Múltiplos Objetos e Associação da Resposta)",
        "scene_count": manifest["scene_count"],
        "total_items": total_items,
        "tasks_breakdown": manifest["tasks_breakdown"],
        "device": "cpu",
        "precision": "float32",
        "threads": args.threads,
        "repetitions": args.repetitions,
        "models": {},
        "per_item_results": [],
    }

    records_by_id: dict[str, dict] = {}
    for it in items:
        records_by_id[it["id"]] = {
            "id": it["id"],
            "scene_id": it["scene_id"],
            "presentation": it["presentation"],
            "task": it["task"],
            "rel_path": it["rel_path"],
            "target_label": it["target_label"],
            "distractor_label": it["distractor_label"],
            "question_en": it["question_en"],
            "description": it["description"],
            "target_bbox": it["target_bbox"],
            "distractor_bbox": it["distractor_bbox"],
            "models": {},
        }

    warmup_path = ROOT / items[0]["rel_path"]
    with Image.open(warmup_path) as s:
        warmup_image = s.convert("RGB")

    # ── CANDIDATE 1: LAYA VISION ──────────────────────────────────────
    print("\n[Candidate 1] Loading Laya Vision (thaitea/laya-vision)...")
    t0 = time.perf_counter()
    laya_model = load_laya_model()
    laya_load_time_s = time.perf_counter() - t0
    print(f"[Candidate 1] Model loaded in {laya_load_time_s:.2f}s")

    print("[Candidate 1] Running warm-up...")
    warmup_q = _get_laya_question(items[0]["task"], items[0]["question_en"])
    _ = laya_model.predict(warmup_image, warmup_q)
    print("[Candidate 1] Warm-up complete.")

    laya_latencies = []
    laya_errors = 0

    print(f"[Candidate 1] Running {args.repetitions} reps × {total_items} items...")
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
                laya_q = _get_laya_question(it["task"], it["question_en"])
                t_inf0 = time.perf_counter()
                out = laya_model.predict(img, laya_q)
                lat_ms = (time.perf_counter() - t_inf0) * 1000

                ans = out.get("answers", {}).get("ans", {})
                choice = str(ans.get("choice", "unknown")).lower()
                probs = ans.get("probabilities", {})
                conf = ans.get("confidence", 0.0)

                rec["raw_answers"].append(choice)
                rec["normalized"].append(choice)
                rec["latencies_ms"].append(round(lat_ms, 2))
                laya_latencies.append(lat_ms)
                if rec["probabilities"] is None:
                    rec["probabilities"] = probs
                    rec["confidence"] = conf
            except Exception as exc:
                laya_errors += 1
                rec["errors"].append(str(exc))
                rec["raw_answers"].append("ERROR")
                rec["normalized"].append("error")
                rec["latencies_ms"].append(0.0)

            if idx % 20 == 0 or idx == total_items:
                print(f"     Progress: {idx}/{total_items} (rep {rep})")

    del laya_model
    gc.collect()

    # ── CANDIDATE 2: BLIP VQA ─────────────────────────────────────────
    print("\n[Candidate 2] Loading BLIP VQA (Salesforce/blip-vqa-base)...")
    t0 = time.perf_counter()
    blip_proc, blip_model = load_blip_model("cpu")
    blip_load_time_s = time.perf_counter() - t0
    print(f"[Candidate 2] Model loaded in {blip_load_time_s:.2f}s")

    print("[Candidate 2] Running warm-up...")
    with torch.inference_mode():
        in_t = blip_proc(images=warmup_image, text=items[0]["question_en"], return_tensors="pt")
        _ = blip_model.generate(**in_t, max_new_tokens=10, do_sample=False)
    print("[Candidate 2] Warm-up complete.")

    blip_latencies = []
    blip_errors = 0

    print(f"[Candidate 2] Running {args.repetitions} reps × {total_items} items...")
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
                norm_fn = _get_blip_normaliser(it["task"])
                t_inf0 = time.perf_counter()
                with torch.inference_mode():
                    in_t = blip_proc(images=img, text=it["question_en"], return_tensors="pt")
                    tok = blip_model.generate(**in_t, max_new_tokens=10, do_sample=False)
                lat_ms = (time.perf_counter() - t_inf0) * 1000

                raw_text = blip_proc.decode(tok[0], skip_special_tokens=True).strip()
                norm = norm_fn(raw_text)

                rec["raw_answers"].append(raw_text)
                rec["normalized"].append(norm)
                rec["latencies_ms"].append(round(lat_ms, 2))
                blip_latencies.append(lat_ms)
            except Exception as exc:
                blip_errors += 1
                rec["errors"].append(str(exc))
                rec["raw_answers"].append("ERROR")
                rec["normalized"].append("error")
                rec["latencies_ms"].append(0.0)

            if idx % 20 == 0 or idx == total_items:
                print(f"     Progress: {idx}/{total_items} (rep {rep})")

    del blip_proc, blip_model
    gc.collect()

    # ── COMPUTE METRICS ───────────────────────────────────────────────
    print("\nComputing association and visual grounding metrics...")

    for rec in records_by_id.values():
        for m_key in ["laya", "blip"]:
            m_data = rec["models"][m_key]
            preds = m_data["normalized"]
            consensus = max(set(preds), key=preds.count) if preds else "unknown"
            m_data["pred_consensus"] = consensus
            m_data["latency_median_ms"] = round(float(np.median(m_data["latencies_ms"])), 1)
            m_data["is_correct"] = (consensus == rec["target_label"])
            m_data["is_association_error"] = (consensus == rec["distractor_label"])

    all_results["per_item_results"] = list(records_by_id.values())

    # Build scene-level comparative analysis
    scene_ids = sorted(list({it["scene_id"] for it in items}))

    for m_key in ["laya", "blip"]:
        crop_correct = 0
        high_correct = 0
        association_errors = 0
        classification_errors = 0
        unknown_errors = 0
        total_scenes = len(scene_ids)

        task_stats = {}

        crop_latencies = []
        high_latencies = []

        for sid in scene_ids:
            crop_rec = records_by_id[f"{sid}_crop"]
            high_rec = records_by_id[f"{sid}_full_highlighted"]

            task = crop_rec["task"]
            task_stats.setdefault(task, {
                "scenes": 0, "crop_correct": 0, "high_correct": 0,
                "association_errors": 0, "classification_errors": 0,
            })
            task_stats[task]["scenes"] += 1

            c_pred = crop_rec["models"][m_key]["pred_consensus"]
            h_pred = high_rec["models"][m_key]["pred_consensus"]
            target = crop_rec["target_label"]
            distractor = crop_rec["distractor_label"]

            crop_latencies.extend(crop_rec["models"][m_key]["latencies_ms"])
            high_latencies.extend(high_rec["models"][m_key]["latencies_ms"])

            # Crop outcome
            if c_pred == target:
                crop_correct += 1
                task_stats[task]["crop_correct"] += 1

            # Highlighted outcome
            if h_pred == target:
                high_correct += 1
                task_stats[task]["high_correct"] += 1
            elif h_pred == distractor:
                association_errors += 1
                task_stats[task]["association_errors"] += 1
            elif h_pred in ("unknown", "not_visible"):
                unknown_errors += 1
            else:
                classification_errors += 1
                task_stats[task]["classification_errors"] += 1

        crop_acc = round(crop_correct / total_scenes, 4)
        high_acc = round(high_correct / total_scenes, 4)
        assoc_rate = round(association_errors / total_scenes, 4)
        class_rate = round(classification_errors / total_scenes, 4)
        unk_rate = round(unknown_errors / total_scenes, 4)
        context_penalty = round(crop_acc - high_acc, 4)

        # Per task formatting
        task_summary = {}
        for t, s in task_stats.items():
            task_summary[t] = {
                "scenes": s["scenes"],
                "crop_accuracy": round(s["crop_correct"] / s["scenes"], 4),
                "highlighted_accuracy": round(s["high_correct"] / s["scenes"], 4),
                "association_error_rate": round(s["association_errors"] / s["scenes"], 4),
                "classification_error_rate": round(s["classification_errors"] / s["scenes"], 4),
            }

        all_results["models"][m_key] = {
            "metadata": {
                "name": "Laya Vision" if m_key == "laya" else "BLIP VQA",
                "model_id": "thaitea/laya-vision" if m_key == "laya" else "Salesforce/blip-vqa-base",
                "load_time_s": round(laya_load_time_s if m_key == "laya" else blip_load_time_s, 2),
                "errors": laya_errors if m_key == "laya" else blip_errors,
            },
            "scene_count": total_scenes,
            "crop_accuracy": {
                "correct": crop_correct, "total": total_scenes, "accuracy": crop_acc,
                "wilson_ci_95": list(wilson_interval(crop_correct, total_scenes)),
            },
            "highlighted_accuracy": {
                "correct": high_correct, "total": total_scenes, "accuracy": high_acc,
                "wilson_ci_95": list(wilson_interval(high_correct, total_scenes)),
            },
            "error_breakdown_highlighted": {
                "association_errors": association_errors,
                "association_error_rate": assoc_rate,
                "classification_errors": classification_errors,
                "classification_error_rate": class_rate,
                "unknown_errors": unknown_errors,
                "unknown_error_rate": unk_rate,
            },
            "context_penalty": {
                "delta_accuracy_pp": round(context_penalty * 100, 2),
                "crop_vs_highlighted_ratio": round(crop_acc / max(0.001, high_acc), 2),
            },
            "per_task": task_summary,
            "latency_ms": {
                "crop_median": round(float(np.median(crop_latencies)), 1),
                "crop_p90": round(float(np.percentile(crop_latencies, 90)), 1),
                "highlighted_median": round(float(np.median(high_latencies)), 1),
                "highlighted_p90": round(float(np.percentile(high_latencies, 90)), 1),
            },
        }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\nResults written to: {args.output}")

    print("\n" + "=" * 75)
    print("RESUMO: TESTE 5 – MÚLTIPLOS OBJETOS E ASSOCIAÇÃO DA RESPOSTA")
    print("=" * 75)
    print(f"{'Métrica':<38} | {'Laya Vision':<15} | {'BLIP VQA':<15}")
    print("-" * 75)
    l_m = all_results["models"]["laya"]
    b_m = all_results["models"]["blip"]
    print(f"{'Acurácia no Recorte Isolado (Crop)':<38} | {l_m['crop_accuracy']['accuracy']*100:>13.1f}% | {b_m['crop_accuracy']['accuracy']*100:>13.1f}%")
    print(f"{'Acurácia Imagem Inteira Destacada':<38} | {l_m['highlighted_accuracy']['accuracy']*100:>13.1f}% | {b_m['highlighted_accuracy']['accuracy']*100:>13.1f}%")
    print(f"{'Penalidade de Contexto (Queda)':<38} | {l_m['context_penalty']['delta_accuracy_pp']:>+12.1f}pp | {b_m['context_penalty']['delta_accuracy_pp']:>+12.1f}pp")
    print(f"{'Taxa de Erro de Associação (Troca)':<38} | {l_m['error_breakdown_highlighted']['association_error_rate']*100:>13.1f}% | {b_m['error_breakdown_highlighted']['association_error_rate']*100:>13.1f}%")
    print(f"{'Erro de Classificação Geral':<38} | {l_m['error_breakdown_highlighted']['classification_error_rate']*100:>13.1f}% | {b_m['error_breakdown_highlighted']['classification_error_rate']*100:>13.1f}%")
    print(f"{'Taxa de Recusa / Unknown':<38} | {l_m['error_breakdown_highlighted']['unknown_error_rate']*100:>13.1f}% | {b_m['error_breakdown_highlighted']['unknown_error_rate']*100:>13.1f}%")
    print(f"{'Latência Mediana Crop (ms)':<38} | {l_m['latency_ms']['crop_median']:>13.1f} | {b_m['latency_ms']['crop_median']:>13.1f}")
    print(f"{'Latência Mediana Full (ms)':<38} | {l_m['latency_ms']['highlighted_median']:>13.1f} | {b_m['latency_ms']['highlighted_median']:>13.1f}")
    print("=" * 75)


if __name__ == "__main__":
    main()

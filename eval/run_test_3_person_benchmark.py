"""Empirical Benchmark Runner for Test 3: Visible Person Attributes.

Adheres strictly to the empirical test plan:
- 3 directly observable attributes:
  1. helmet (yes, no, not_visible)
  2. carrying (backpack, bag_or_box, none, not_visible)
  3. clothing (safety_vest, jacket_coat, shirt_tshirt, not_visible)
- 60 curated evaluation images (including controls without people and difficult conditions)
- Model load time measured separately
- Warm-up execution excluded from latency stats
- 3 full repetitions across all images and questions
- Metrics computed per question and per class, plus difficulty breakdown.
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

MANIFEST_PATH = ROOT / "eval/datasets/test_3_person_attributes_manifest.json"
DEFAULT_OUTPUT = ROOT / "eval/results/test_3_person_attributes_results.json"

LAYA_QUESTIONS = {
    "helmet": {
        "type": "choice",
        "instructions": "Is the prominent person wearing a helmet or hard hat?",
        "criteria": {
            "yes": "Yes, wearing helmet, hard hat, or motorcycle helmet",
            "no": "No, bare head or regular cap/hat",
            "not_visible": "Head not visible, occluded, or no person",
        },
    },
    "carrying": {
        "type": "choice",
        "instructions": "Is the prominent person carrying a bag, backpack, or visible object?",
        "criteria": {
            "backpack": "Carrying a backpack on back",
            "bag_or_box": "Carrying a bag, briefcase, box, or suitcase",
            "none": "Hands free, not carrying visible items",
            "not_visible": "Body occluded, unclear, or no person",
        },
    },
    "clothing": {
        "type": "choice",
        "instructions": "What upper-body clothing is the prominent person wearing?",
        "criteria": {
            "safety_vest": "High-visibility fluorescent safety vest",
            "jacket_coat": "Jacket, coat, suit, or outerwear",
            "shirt_tshirt": "T-shirt, shirt, or light top",
            "not_visible": "Upper body not visible, occluded, or no person",
        },
    },
}

BLIP_PROMPTS = {
    "helmet": "Is the person wearing a helmet or hard hat? Options: yes, no, or not visible.",
    "carrying": "Is the person carrying a backpack, bag, box, or nothing? Options: backpack, bag, box, nothing, or not visible.",
    "clothing": "What upper clothing is the person wearing: safety vest, jacket, shirt, or not visible?",
}


def _normalise_blip(question_key: str, answer: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", answer.lower()).strip()
    words = set(cleaned.split())

    if question_key == "helmet":
        if any(w in words or w in cleaned for w in ("yes", "helmet", "hard hat", "hardhat")):
            return "yes"
        if any(w in words or w in cleaned for w in ("no", "bare", "hat", "cap")):
            return "no"
        return "not_visible"

    if question_key == "carrying":
        if any(w in words or w in cleaned for w in ("backpack", "rucksack")):
            return "backpack"
        if any(w in words or w in cleaned for w in ("bag", "box", "briefcase", "suitcase", "package")):
            return "bag_or_box"
        if any(w in words or w in cleaned for w in ("nothing", "none", "no")):
            return "none"
        return "not_visible"

    if question_key == "clothing":
        if any(w in words or w in cleaned for w in ("safety vest", "vest", "reflective", "hi-vis", "yellow vest")):
            return "safety_vest"
        if any(w in words or w in cleaned for w in ("jacket", "coat", "suit", "blazer", "sweater", "outerwear")):
            return "jacket_coat"
        if any(w in words or w in cleaned for w in ("shirt", "t-shirt", "tshirt", "polo", "top")):
            return "shirt_tshirt"
        return "not_visible"

    return "not_visible"


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
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output JSON path")
    args = parser.parse_args()

    import torch
    torch.set_num_threads(args.threads)

    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)

    items = manifest["items"]
    total_images = len(items)

    print(f"=== TEST 3: EMPIRICAL BENCHMARK - VISIBLE PERSON ATTRIBUTES ===")
    print(f"Dataset items: {total_images} (eval split)")
    print(f"Repetitions per image: {args.repetitions}")
    print(f"Attributes tested: helmet, carrying, clothing")
    print(f"CPU threads: {args.threads}")

    all_results: dict[str, Any] = {
        "test": "Test 3: Visible Person Attributes (Atributos Visíveis de Pessoas)",
        "attributes": manifest["attributes"],
        "device": "cpu",
        "precision": "float32",
        "threads": args.threads,
        "repetitions": args.repetitions,
        "image_count": total_images,
        "models": {},
        "per_image_results": [],
    }

    records_by_id = {
        it["id"]: {
            "id": it["id"],
            "filename": it["filename"],
            "rel_path": it["rel_path"],
            "expected": it["expected"],
            "title": it["title"],
            "difficult": it["difficult"],
            "difficulty_tags": it["difficulty_tags"],
            "models": {
                "laya": {"questions": {q: {"raw": [], "normalized": [], "probs": None} for q in LAYA_QUESTIONS}, "latencies_ms": [], "errors": []},
                "blip": {"questions": {q: {"raw": [], "normalized": []} for q in LAYA_QUESTIONS}, "latencies_ms": [], "errors": []},
            },
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

    print("[Candidate 1] Running warm-up...")
    _ = laya_model.predict(warmup_image, LAYA_QUESTIONS)
    print("[Candidate 1] Warm-up complete.")

    laya_latencies_all = []
    laya_errors = 0

    print(f"[Candidate 1] Running {args.repetitions} repetitions across {total_images} images...")
    for rep in range(1, args.repetitions + 1):
        print(f"  -> Repetition {rep}/{args.repetitions}...")
        for idx, it in enumerate(items, 1):
            m_rec = records_by_id[it["id"]]["models"]["laya"]
            img_path = ROOT / it["rel_path"]
            try:
                with Image.open(img_path) as s:
                    img = s.convert("RGB")
                t_inf0 = time.perf_counter()
                out = laya_model.predict(img, LAYA_QUESTIONS)
                lat_ms = (time.perf_counter() - t_inf0) * 1000

                answers = out.get("answers", {})
                for q_key in LAYA_QUESTIONS:
                    q_ans = answers.get(q_key, {})
                    ch = str(q_ans.get("choice", "not_visible")).lower()
                    m_rec["questions"][q_key]["raw"].append(ch)
                    m_rec["questions"][q_key]["normalized"].append(ch)
                    if m_rec["questions"][q_key]["probs"] is None:
                        m_rec["questions"][q_key]["probs"] = q_ans.get("probabilities", {})

                m_rec["latencies_ms"].append(round(lat_ms, 2))
                laya_latencies_all.append(lat_ms)
            except Exception as exc:
                laya_errors += 1
                m_rec["errors"].append(str(exc))
                for q_key in LAYA_QUESTIONS:
                    m_rec["questions"][q_key]["raw"].append("ERROR")
                    m_rec["questions"][q_key]["normalized"].append("not_visible")
                m_rec["latencies_ms"].append(0.0)

            if idx % 20 == 0 or idx == total_images:
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

    print("[Candidate 2] Running warm-up...")
    with torch.inference_mode():
        in_t = blip_proc(images=[warmup_image]*3, text=list(BLIP_PROMPTS.values()), return_tensors="pt", padding=True).to("cpu")
        _ = blip_model.generate(**in_t, max_new_tokens=10, do_sample=False)
    print("[Candidate 2] Warm-up complete.")

    blip_latencies_all = []
    blip_errors = 0

    print(f"[Candidate 2] Running {args.repetitions} repetitions across {total_images} images...")
    blip_q_keys = list(BLIP_PROMPTS.keys())
    blip_q_prompts = list(BLIP_PROMPTS.values())

    for rep in range(1, args.repetitions + 1):
        print(f"  -> Repetition {rep}/{args.repetitions}...")
        for idx, it in enumerate(items, 1):
            m_rec = records_by_id[it["id"]]["models"]["blip"]
            img_path = ROOT / it["rel_path"]
            try:
                with Image.open(img_path) as s:
                    img = s.convert("RGB")
                t_inf0 = time.perf_counter()
                with torch.inference_mode():
                    in_t = blip_proc(images=[img]*3, text=blip_q_prompts, return_tensors="pt", padding=True).to("cpu")
                    tokens = blip_model.generate(**in_t, max_new_tokens=10, do_sample=False)
                lat_ms = (time.perf_counter() - t_inf0) * 1000

                for q_k, tok in zip(blip_q_keys, tokens):
                    raw_txt = blip_proc.decode(tok, skip_special_tokens=True).strip()
                    norm = _normalise_blip(q_k, raw_txt)
                    m_rec["questions"][q_k]["raw"].append(raw_txt)
                    m_rec["questions"][q_k]["normalized"].append(norm)

                m_rec["latencies_ms"].append(round(lat_ms, 2))
                blip_latencies_all.append(lat_ms)
            except Exception as exc:
                blip_errors += 1
                m_rec["errors"].append(str(exc))
                for q_k in blip_q_keys:
                    m_rec["questions"][q_k]["raw"].append("ERROR")
                    m_rec["questions"][q_k]["normalized"].append("not_visible")
                m_rec["latencies_ms"].append(0.0)

            if idx % 20 == 0 or idx == total_images:
                print(f"     Progresso: {idx}/{total_images} imagens processadas (rep {rep})")

    del blip_proc, blip_model
    gc.collect()

    # ----------------------------------------------------
    # METRICS CALCULATION AND ANALYSIS
    # ----------------------------------------------------
    print("\nComputing comprehensive empirical metrics for person attributes...")
    all_results["per_image_results"] = list(records_by_id.values())

    model_metadata = {
        "laya": {
            "name": "Laya Vision",
            "model_id": "thaitea/laya-vision",
            "revision": "f2fe3c12cb6d04c59d8a190250bf3fb40fc828dc",
            "format": "typed_choice_multitask",
            "load_time_s": round(laya_load_time_s, 2),
            "errors": laya_errors,
        },
        "blip": {
            "name": "BLIP VQA",
            "model_id": "Salesforce/blip-vqa-base",
            "revision": "787b3d35d57e49572baabd22884b3d5a05acf072",
            "format": "open_vqa_batched",
            "prompts": BLIP_PROMPTS,
            "load_time_s": round(blip_load_time_s, 2),
            "errors": blip_errors,
        },
    }

    q_keys = list(LAYA_QUESTIONS.keys())

    for m_key in ["laya", "blip"]:
        q_metrics = {}
        total_questions = 0
        total_correct = 0

        std_correct = 0
        std_total = 0
        diff_correct = 0
        diff_total = 0

        for q_k in q_keys:
            tax = manifest["attributes"][q_k]["taxonomy"]
            class_stats = {c: {"tp": 0, "fp": 0, "fn": 0, "total": 0} for c in tax}
            cm = {act: {pred: 0 for pred in tax + ["other", "error"]} for act in tax}
            q_corr = 0
            q_tot = 0

            for rec in all_results["per_image_results"]:
                exp = rec["expected"][q_k]
                m_q_data = rec["models"][m_key]["questions"][q_k]
                preds = m_q_data["normalized"]
                pred = max(set(preds), key=preds.count) if preds else "not_visible"
                m_q_data["pred_consensus"] = pred
                is_match = (pred == exp)
                m_q_data["is_correct"] = is_match

                q_tot += 1
                total_questions += 1
                if is_match:
                    q_corr += 1
                    total_correct += 1

                if rec["difficult"]:
                    diff_total += 1
                    if is_match: diff_correct += 1
                else:
                    std_total += 1
                    if is_match: std_correct += 1

                if exp in cm:
                    cm[exp][pred] = cm[exp].get(pred, 0) + 1

                if exp in class_stats:
                    class_stats[exp]["total"] += 1
                    if is_match:
                        class_stats[exp]["tp"] += 1
                    else:
                        class_stats[exp]["fn"] += 1

                if pred in class_stats and not is_match:
                    class_stats[pred]["fp"] += 1

            per_class = {}
            for c, s in class_stats.items():
                tp, fp, fn, n = s["tp"], s["fp"], s["fn"], s["total"]
                prec = round(tp / (tp + fp), 4) if (tp + fp) else 0.0
                rec = round(tp / (tp + fn), 4) if (tp + fn) else 0.0
                f1 = round(2 * prec * rec / (prec + rec), 4) if (prec + rec) else 0.0
                per_class[c] = {"total": n, "correct": tp, "accuracy": round(tp / n, 4) if n else 0.0,
                                "precision": prec, "recall": rec, "f1_score": f1}

            w_lo, w_hi = wilson_interval(q_corr, q_tot)
            q_metrics[q_k] = {
                "correct": q_corr,
                "total": q_tot,
                "accuracy": round(q_corr / q_tot, 4) if q_tot else 0.0,
                "wilson_ci_95": [w_lo, w_hi],
                "per_class": per_class,
                "confusion_matrix": cm,
            }

        # Rejection on negative controls (where expected for all 3 is not_visible)
        control_items = [r for r in all_results["per_image_results"] if all(v == "not_visible" for v in r["expected"].values())]
        ctrl_q_total = len(control_items) * 3
        ctrl_q_correct = sum(
            sum(1 for q in q_keys if r["models"][m_key]["questions"][q]["pred_consensus"] == "not_visible")
            for r in control_items
        )
        correct_rejection_rate = round(ctrl_q_correct / ctrl_q_total, 4) if ctrl_q_total else 0.0

        lats = laya_latencies_all if m_key == "laya" else blip_latencies_all
        all_results["models"][m_key] = {
            "metadata": model_metadata[m_key],
            "overall_accuracy": {
                "correct": total_correct,
                "total": total_questions,
                "accuracy": round(total_correct / total_questions, 4) if total_questions else 0.0,
                "wilson_ci_95": list(wilson_interval(total_correct, total_questions)),
                "standard_images_accuracy": round(std_correct / std_total, 4) if std_total else 0.0,
                "difficult_images_accuracy": round(diff_correct / diff_total, 4) if diff_total else 0.0,
            },
            "by_question": q_metrics,
            "refusal_analysis": {
                "negative_control_images": len(control_items),
                "control_total_queries": ctrl_q_total,
                "control_correct_not_visible": ctrl_q_correct,
                "correct_rejection_rate": correct_rejection_rate,
            },
            "latency_ms_per_image": {
                "mean": round(float(np.mean(lats)), 1) if lats else 0.0,
                "median": round(float(np.median(lats)), 1) if lats else 0.0,
                "p90": round(float(np.percentile(lats, 90)), 1) if lats else 0.0,
                "p95": round(float(np.percentile(lats, 95)), 1) if lats else 0.0,
                "min": round(float(np.min(lats)), 1) if lats else 0.0,
                "max": round(float(np.max(lats)), 1) if lats else 0.0,
            },
        }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nBenchmark completed successfully! Results written to: {out_path}")

    # Summary table
    print("\n" + "=" * 68)
    print("RESUMO FINAL: TESTE 3 - ATRIBUTOS VISÍVEIS DE PESSOAS")
    print("=" * 68)
    print(f"{'Métrica':<38} | {'Laya Vision':<12} | {'BLIP VQA':<12}")
    print("-" * 68)
    l_o = all_results['models']['laya']['overall_accuracy']
    b_o = all_results['models']['blip']['overall_accuracy']
    l_lat = all_results['models']['laya']['latency_ms_per_image']
    b_lat = all_results['models']['blip']['latency_ms_per_image']
    l_ref = all_results['models']['laya']['refusal_analysis']
    b_ref = all_results['models']['blip']['refusal_analysis']

    print(f"{'Acurácia Global (180 questões)':<38} | {l_o['accuracy']*100:>10.2f}% | {b_o['accuracy']*100:>10.2f}%")
    for q in q_keys:
        la = all_results['models']['laya']['by_question'][q]['accuracy']
        ba = all_results['models']['blip']['by_question'][q]['accuracy']
        q_label = f" -> Acurácia: {q.title()}"
        print(f"{q_label:<38} | {la*100:>10.2f}% | {ba*100:>10.2f}%")

    print(f"{'Rejeição Correta (controles vazios)':<38} | {l_ref['correct_rejection_rate']*100:>10.2f}% | {b_ref['correct_rejection_rate']*100:>10.2f}%")
    print(f"{'Tempo de Carregamento':<38} | {all_results['models']['laya']['metadata']['load_time_s']:>10.2f}s | {all_results['models']['blip']['metadata']['load_time_s']:>10.2f}s")
    print(f"{'Latência Mediana por imagem (3 q)':<38} | {l_lat['median']:>10.1f}ms | {b_lat['median']:>10.1f}ms")
    print(f"{'Latência p90 por imagem (3 q)':<38} | {l_lat['p90']:>10.1f}ms | {b_lat['p90']:>10.1f}ms")
    print(f"{'Erros de inferência':<38} | {all_results['models']['laya']['metadata']['errors']:>12} | {all_results['models']['blip']['metadata']['errors']:>12}")
    print("=" * 68)


if __name__ == "__main__":
    main()

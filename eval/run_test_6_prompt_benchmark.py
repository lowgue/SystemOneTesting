"""Empirical Benchmark Runner for Test 6: Prompt and Output Format Comparison.

Compares:
  - Generative model (BLIP VQA):
      1. short_open
      2. short_options
      3. long_descriptions
      4. structured_json
  - Decision model (Laya Vision):
      1. native_short
      2. native_descriptive

Measures:
  - Accuracy
  - Option Adherence (% conforming strictly to allowed taxonomy)
  - Invalid / Malformed Rate (% unparsable or broken JSON)
  - Ambiguity Rate (% mentioning multiple conflicting colors)
  - Latency (Median, p90, p95)
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

MANIFEST_PATH = ROOT / "eval/datasets/test_6_prompt_formats_manifest.json"
DEFAULT_OUTPUT = ROOT / "eval/results/test_6_prompt_formats_results.json"

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
}


def _normalise_blip_color(answer: str) -> tuple[str, bool, bool]:
    """Normalises raw text answer and returns (normalized_label, is_ambiguous, is_adherent)."""
    cleaned = re.sub(r"[^a-z0-9]+", " ", answer.lower()).strip()
    words = set(cleaned.split())

    matched_labels = []
    for label, terms in COLOR_ANSWERS.items():
        if any(term in words or term in cleaned for term in terms):
            matched_labels.append(label)

    # Check ambiguity: more than 1 distinct color class detected in the answer
    is_ambiguous = len(set(matched_labels)) > 1
    is_adherent = len(matched_labels) == 1

    if len(matched_labels) == 1:
        return matched_labels[0], False, True
    elif len(matched_labels) > 1:
        return matched_labels[0], True, False
    else:
        return "other" if cleaned else "unknown", False, False


def _parse_json_response(raw_text: str) -> tuple[str, bool, bool, bool]:
    """Parses structured JSON response: returns (color, is_valid_json, is_ambiguous, is_adherent)."""
    # Attempt to extract JSON substring if wrapped in markdown or extra text
    raw_clean = raw_text.strip()
    match = re.search(r"\{.*?\}", raw_clean, re.DOTALL)
    candidate = match.group(0) if match else raw_clean

    try:
        data = json.loads(candidate)
        if isinstance(data, dict) and "color" in data:
            c_val = str(data["color"])
            norm, ambig, adh = _normalise_blip_color(c_val)
            return norm, True, ambig, adh
        return "other", True, False, False
    except Exception:
        # Failed JSON parsing
        norm, ambig, _ = _normalise_blip_color(raw_clean)
        return norm, False, ambig, False


def _apply_laya_transformers_compat_patch():
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

    images = manifest["images"]
    n_images = len(images)
    blip_prompts = manifest["blip_prompts"]
    laya_criteria = manifest["laya_criteria"]

    print(f"=== TEST 6: EMPIRICAL BENCHMARK - PROMPT & OUTPUT FORMAT COMPARISON ===")
    print(f"Total evaluation images: {n_images} (balanced across {len(manifest['taxonomy'])} colors)")
    print(f"Repetitions per item: {args.repetitions}")
    print(f"CPU threads: {args.threads}")

    all_results: dict[str, Any] = {
        "test": "Test 6: Prompt and Output Format Comparison",
        "image_count": n_images,
        "taxonomy": manifest["taxonomy"],
        "threads": args.threads,
        "repetitions": args.repetitions,
        "formats_tested": {
            "blip": list(blip_prompts.keys()),
            "laya": list(laya_criteria.keys()),
        },
        "models": {},
        "per_image_results": [],
    }

    # Initialize per-image tracking
    per_image_data: dict[str, dict] = {}
    for img_info in images:
        per_image_data[img_info["id"]] = {
            "id": img_info["id"],
            "source_id": img_info["source_id"],
            "rel_path": img_info["rel_path"],
            "expected_color": img_info["expected_color"],
            "difficult": img_info["difficult"],
            "blip_formats": {},
            "laya_formats": {},
        }

    warmup_img_path = ROOT / images[0]["rel_path"]
    with Image.open(warmup_img_path) as s:
        warmup_image = s.convert("RGB")

    # ──────────────────────────────────────────────────────────────────
    # 1. EVALUATE BLIP VQA ACROSS 4 PROMPT FORMATS
    # ──────────────────────────────────────────────────────────────────
    print("\n[Model: BLIP VQA] Loading weights...")
    t0 = time.perf_counter()
    blip_proc, blip_model = load_blip_model("cpu")
    blip_load_time_s = time.perf_counter() - t0
    print(f"[Model: BLIP VQA] Loaded in {blip_load_time_s:.2f}s")

    blip_format_stats: dict[str, dict] = {}

    for fmt_name, prompt_text in blip_prompts.items():
        print(f"\n  -> Running BLIP format: '{fmt_name}'...")
        print(f"     Prompt: \"{prompt_text[:65]}...\"")

        # Warmup
        with torch.inference_mode():
            in_t = blip_proc(images=warmup_image, text=prompt_text, return_tensors="pt")
            _ = blip_model.generate(**in_t, max_new_tokens=25, do_sample=False)

        latencies_all = []
        fmt_correct_images = 0
        fmt_adherent_count = 0
        fmt_invalid_count = 0
        fmt_ambiguous_count = 0

        for idx, img_info in enumerate(images, 1):
            img_path = ROOT / img_info["rel_path"]
            with Image.open(img_path) as s:
                img = s.convert("RGB")

            raw_reps = []
            norm_reps = []
            lat_reps = []
            is_valid_json_reps = []
            is_ambiguous_reps = []
            is_adherent_reps = []

            for _ in range(args.repetitions):
                t_inf0 = time.perf_counter()
                with torch.inference_mode():
                    in_t = blip_proc(images=img, text=prompt_text, return_tensors="pt")
                    tok = blip_model.generate(**in_t, max_new_tokens=25, do_sample=False)
                lat_ms = (time.perf_counter() - t_inf0) * 1000

                raw_text = blip_proc.decode(tok[0], skip_special_tokens=True).strip()

                if fmt_name == "structured_json":
                    norm, valid_json, ambig, adh = _parse_json_response(raw_text)
                    is_valid_json_reps.append(valid_json)
                else:
                    norm, ambig, adh = _normalise_blip_color(raw_text)
                    is_valid_json_reps.append(True)

                raw_reps.append(raw_text)
                norm_reps.append(norm)
                lat_reps.append(round(lat_ms, 1))
                latencies_all.append(lat_ms)
                is_ambiguous_reps.append(ambig)
                is_adherent_reps.append(adh)

            # Consensus across repetitions
            consensus_pred = max(set(norm_reps), key=norm_reps.count) if norm_reps else "unknown"
            is_correct = (consensus_pred == img_info["expected_color"])
            if is_correct:
                fmt_correct_images += 1

            if any(is_adherent_reps):
                fmt_adherent_count += 1
            if fmt_name == "structured_json" and not any(is_valid_json_reps):
                fmt_invalid_count += 1
            if any(is_ambiguous_reps):
                fmt_ambiguous_count += 1

            per_image_data[img_info["id"]]["blip_formats"][fmt_name] = {
                "raw_answers": raw_reps,
                "normalized": norm_reps,
                "consensus": consensus_pred,
                "is_correct": is_correct,
                "is_adherent": bool(any(is_adherent_reps)),
                "is_ambiguous": bool(any(is_ambiguous_reps)),
                "is_valid_json": bool(all(is_valid_json_reps)) if fmt_name == "structured_json" else True,
                "latencies_ms": lat_reps,
                "latency_median_ms": round(float(np.median(lat_reps)), 1),
            }

            if idx % 15 == 0 or idx == n_images:
                print(f"     Progress: {idx}/{n_images} images")

        acc = round(fmt_correct_images / n_images, 4)
        adh_rate = round(fmt_adherent_count / n_images, 4)
        amb_rate = round(fmt_ambiguous_count / n_images, 4)
        inv_rate = round(fmt_invalid_count / n_images, 4)
        ci = list(wilson_interval(fmt_correct_images, n_images))

        blip_format_stats[fmt_name] = {
            "prompt": prompt_text,
            "accuracy": acc,
            "wilson_ci_95": ci,
            "correct_images": fmt_correct_images,
            "total_images": n_images,
            "adherence_rate": adh_rate,
            "ambiguous_rate": amb_rate,
            "invalid_rate": inv_rate,
            "latency_ms": {
                "median": round(float(np.median(latencies_all)), 1),
                "p90": round(float(np.percentile(latencies_all, 90)), 1),
                "p95": round(float(np.percentile(latencies_all, 95)), 1),
            },
        }

    del blip_proc, blip_model
    gc.collect()

    # ──────────────────────────────────────────────────────────────────
    # 2. EVALUATE LAYA VISION ACROSS 2 CRITERIA SCHEMAS
    # ──────────────────────────────────────────────────────────────────
    print("\n[Model: Laya Vision] Loading weights...")
    t0 = time.perf_counter()
    laya_model = load_laya_model()
    laya_load_time_s = time.perf_counter() - t0
    print(f"[Model: Laya Vision] Loaded in {laya_load_time_s:.2f}s")

    laya_format_stats: dict[str, dict] = {}

    for crit_name, crit_dict in laya_criteria.items():
        print(f"\n  -> Running Laya criteria format: '{crit_name}'...")
        q_spec = {
            "color": {
                "type": "choice",
                "instructions": "What is the predominant color of this vehicle?",
                "criteria": crit_dict,
            }
        }

        # Warmup
        _ = laya_model.predict(warmup_image, q_spec)

        latencies_all = []
        fmt_correct_images = 0

        for idx, img_info in enumerate(images, 1):
            img_path = ROOT / img_info["rel_path"]
            with Image.open(img_path) as s:
                img = s.convert("RGB")

            raw_reps = []
            norm_reps = []
            lat_reps = []
            conf_reps = []

            for _ in range(args.repetitions):
                t_inf0 = time.perf_counter()
                out = laya_model.predict(img, q_spec)
                lat_ms = (time.perf_counter() - t_inf0) * 1000

                ans = out.get("answers", {}).get("color", {})
                choice = str(ans.get("choice", "unknown")).lower()
                conf = float(ans.get("confidence", 0.0))

                raw_reps.append(choice)
                norm_reps.append(choice)
                lat_reps.append(round(lat_ms, 1))
                latencies_all.append(lat_ms)
                conf_reps.append(round(conf, 4))

            consensus_pred = max(set(norm_reps), key=norm_reps.count) if norm_reps else "unknown"
            is_correct = (consensus_pred == img_info["expected_color"])
            if is_correct:
                fmt_correct_images += 1

            per_image_data[img_info["id"]]["laya_formats"][crit_name] = {
                "raw_answers": raw_reps,
                "normalized": norm_reps,
                "consensus": consensus_pred,
                "confidence_avg": round(float(np.mean(conf_reps)), 4),
                "is_correct": is_correct,
                "is_adherent": True,  # Constrained output is always adherent
                "is_ambiguous": False,
                "is_valid_schema": True,
                "latencies_ms": lat_reps,
                "latency_median_ms": round(float(np.median(lat_reps)), 1),
            }

            if idx % 15 == 0 or idx == n_images:
                print(f"     Progress: {idx}/{n_images} images")

        acc = round(fmt_correct_images / n_images, 4)
        ci = list(wilson_interval(fmt_correct_images, n_images))

        laya_format_stats[crit_name] = {
            "accuracy": acc,
            "wilson_ci_95": ci,
            "correct_images": fmt_correct_images,
            "total_images": n_images,
            "adherence_rate": 1.0,  # 100% adherence guaranteed by decision head
            "ambiguous_rate": 0.0,
            "invalid_rate": 0.0,
            "latency_ms": {
                "median": round(float(np.median(latencies_all)), 1),
                "p90": round(float(np.percentile(latencies_all, 90)), 1),
                "p95": round(float(np.percentile(latencies_all, 95)), 1),
            },
        }

    del laya_model
    gc.collect()

    all_results["per_image_results"] = list(per_image_data.values())
    all_results["models"] = {
        "blip": {
            "name": "BLIP VQA",
            "model_id": "Salesforce/blip-vqa-base",
            "type": "generative_vqa",
            "load_time_s": round(blip_load_time_s, 2),
            "formats": blip_format_stats,
        },
        "laya": {
            "name": "Laya Vision",
            "model_id": "thaitea/laya-vision",
            "type": "decision_model",
            "load_time_s": round(laya_load_time_s, 2),
            "formats": laya_format_stats,
        },
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\nResults successfully written to: {args.output}")

    # Summary table
    print("\n" + "=" * 85)
    print("RESUMO: TESTE 6 – COMPARAÇÃO DE PROMPTS E FORMATO DE SAÍDA")
    print("=" * 85)
    print(f"{'Modelo / Formato':<32} | {'Acurácia':<10} | {'Aderência':<10} | {'Ambíguo':<10} | {'Latência (ms)':<12}")
    print("-" * 85)

    for fmt, s in blip_format_stats.items():
        name = f"BLIP: {fmt}"
        print(f"{name:<32} | {s['accuracy']*100:>8.1f}% | {s['adherence_rate']*100:>8.1f}% | {s['ambiguous_rate']*100:>8.1f}% | {s['latency_ms']['median']:>10.1f} ms")

    print("-" * 85)
    for fmt, s in laya_format_stats.items():
        name = f"Laya: {fmt}"
        print(f"{name:<32} | {s['accuracy']*100:>8.1f}% | {s['adherence_rate']*100:>8.1f}% | {s['ambiguous_rate']*100:>8.1f}% | {s['latency_ms']['median']:>10.1f} ms")

    print("=" * 85)


if __name__ == "__main__":
    main()

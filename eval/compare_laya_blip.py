"""Compare Laya Vision and BLIP VQA on all annotated local car images.

Run from any directory with:
    python eval/compare_laya_blip.py [--progress-file PATH]

One model is loaded at a time. Intermediate progress is written atomically when
--progress-file is supplied, so a local browser page can poll it while running.
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
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "laya-vision"))
sys.path.insert(0, str(ROOT / "eval"))

from PIL import Image

from download_stanford_cars_sample import MAKES, _canonical

OLD_IMAGES = ROOT / "eval/images/cars"
STANFORD_IMAGES = ROOT / "eval/images/stanford_cars"
STANFORD_MANIFEST = ROOT / "eval/datasets/stanford_cars_sample.json"
COLOR_IMAGES = ROOT / "eval/images/color_test"
COLOR_MANIFEST = ROOT / "eval/datasets/car_color_test.json"
REAL_METADATA = ROOT / "eval/datasets/real_cars_metadata.json"

# Seven hand-labeled examples include color and two no-car images.
CURATED_EXPECTED: dict[str, dict[str, Any]] = {
    "toyota_red.jpg": {"tem_carro": True, "marca": "toyota", "cor": "red", "tipo": "sedan"},
    "ford_blue.jpg": {"tem_carro": True, "marca": "ford", "cor": "blue", "tipo": "hatch"},
    "honda_white.jpg": {"tem_carro": True, "marca": "honda", "cor": "white", "tipo": "sedan"},
    "gray_car.jpg": {"tem_carro": True, "marca": "unknown", "cor": "gray_silver", "tipo": "sedan"},
    "black_car.jpg": {"tem_carro": True, "marca": "unknown", "cor": "black", "tipo": "suv"},
    "no_car_landscape.jpg": {"tem_carro": False, "marca": "unknown", "cor": "unknown", "tipo": "unknown"},
    "no_car_field.jpg": {"tem_carro": False, "marca": "unknown", "cor": "unknown", "tipo": "unknown"},
}

BASE_QUESTIONS = {
    "tem_carro": {
        "type": "noul",
        "instructions": "Is there a car in the image? Answer yes or no.",
    },
    "marca": {
        "type": "choice",
        "instructions": "What is the make of the most prominent car?",
    },
    "cor": {
        "type": "choice",
        "instructions": "What is the predominant color of the most prominent car?",
        "criteria": {
            "black": "Black", "white": "White", "gray_silver": "Gray or silver",
            "red": "Red", "blue": "Blue", "green": "Green", "yellow": "Yellow",
            "orange": "Orange", "brown": "Brown", "purple": "Purple", "pink": "Pink",
            "other": "Another identifiable color", "unknown": "No car or color not identifiable",
        },
    },
    "tipo": {
        "type": "choice",
        "instructions": "What body style is the most prominent vehicle?",
    },
}
BLIP_QUESTIONS = {key: value["instructions"] for key, value in BASE_QUESTIONS.items()}

BODY_LABELS = {
    "sedan": "Sedan", "suv": "SUV", "hatch": "Hatchback", "truck": "Pickup truck",
    "van": "Van or minivan", "coupe": "Coupe", "convertible": "Convertible",
    "wagon": "Wagon", "other": "Another body style", "unknown": "No car or not identifiable",
}
BODY_ANSWERS = {
    "sedan": ("sedan",), "suv": ("suv", "crossover"),
    "hatch": ("hatchback", "hatch"), "truck": ("truck", "pickup", "crew cab", "club cab"),
    "van": ("minivan", "van"), "coupe": ("coupe",),
    "convertible": ("convertible",), "wagon": ("wagon",),
}
COLOR_ANSWERS = {
    "black": ("black",), "white": ("white",), "gray_silver": ("gray", "grey", "silver"),
    "red": ("red",), "blue": ("blue",), "green": ("green",), "yellow": ("yellow",),
    "orange": ("orange",), "brown": ("brown",), "purple": ("purple",), "pink": ("pink",),
    "other": ("teal", "turquoise", "gold", "beige", "bronze", "maroon"),
}

MAKE_CANDIDATES = ["toyota", "ford", "chevrolet", "honda", "bmw", "mercedes_benz",
                   "audi", "volkswagen", "nissan", "jeep"]


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


MAKE_LABELS = {_slug(make): make for make in MAKES}


def load_examples(images_dir: Path = OLD_IMAGES, stanford_dir: Path = STANFORD_IMAGES,
                  stanford_manifest: Path = STANFORD_MANIFEST) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    included: set[str] = set()

    def add(filename: str, folder: Path, expected: dict[str, Any], source: str, title: str) -> None:
        path = folder / filename
        if path.is_file() and filename not in included:
            included.add(filename)
            examples.append({
                "filename": filename, "path": path, "expected": expected,
                "source": source, "title": title,
            })

    for filename, expected in CURATED_EXPECTED.items():
        add(filename, images_dir, expected, "curated", filename)

    if REAL_METADATA.is_file():
        for item in json.loads(REAL_METADATA.read_text(encoding="utf-8")):
            make = _slug(item["make"])
            body = item.get("body_type", "other").lower()
            if body in {"hatchback", "hatch"}:
                body = "hatch"
            elif body in {"minivan"}:
                body = "van"
            if body not in BODY_LABELS:
                body = "other"
            add(item["filename"], images_dir, {
                "tem_carro": True, "marca": make, "tipo": body,
            }, "stanford_existing", item.get("full_name", item["filename"]))

    if stanford_manifest.is_file():
        manifest = json.loads(stanford_manifest.read_text(encoding="utf-8"))
        for item in manifest.get("items", []):
            body = item.get("body_type", "other")
            if body not in BODY_LABELS:
                body = "other"
            add(item["filename"], stanford_dir, {
                "tem_carro": True, "marca": item["make"], "tipo": body,
            }, "stanford_test", item.get("class_name", item["filename"]))

    if COLOR_MANIFEST.is_file():
        manifest = json.loads(COLOR_MANIFEST.read_text(encoding="utf-8"))
        for item in manifest.get("items", []):
            add(item["filename"], COLOR_IMAGES, {"cor": item["color"]},
                "car_color", f"Carro {item['color']} · {item.get('source_color', item['color'])}")
    return examples


def _apply_laya_transformers_compat_patch() -> None:
    import torch
    import laya.vlm as vlm

    def encode_images_compat(self, pixel_values, pixel_attention_mask=None):
        mask = pixel_attention_mask[None] if pixel_attention_mask is not None else None
        result = self._image_features(pixel_values[None], mask)
        return torch.stack(result) if isinstance(result, list) else result

    vlm.VLMDecisionModel.encode_images = encode_images_compat


def load_laya():
    _apply_laya_transformers_compat_patch()
    import laya
    return laya.load_vlm("thaitea/laya-vision")


def load_blip(device: str):
    from transformers import AutoModelForVisualQuestionAnswering, AutoProcessor
    processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = AutoModelForVisualQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    model.to(device)
    model.eval()
    return processor, model


def _questions_for(expected: dict[str, Any], make_labels: dict[str, str]) -> dict[str, Any]:
    """Use the same four base questions and fixed option counts for every image."""
    questions = {key: dict(value) for key, value in BASE_QUESTIONS.items()}
    target_make = expected.get("marca")
    make_keys = list(MAKE_CANDIDATES)
    if target_make and target_make not in {"other", "unknown"} and target_make not in make_keys:
        make_keys[-1] = target_make
    make_criteria = {key: make_labels.get(key, key.replace("_", " ").title()) for key in make_keys}
    make_criteria.update({"other": "Another identifiable make", "unknown": "No car or make not identifiable"})
    questions["marca"]["criteria"] = make_criteria
    questions["cor"]["criteria"] = dict(BASE_QUESTIONS["cor"]["criteria"])
    questions["tipo"]["criteria"] = dict(BODY_LABELS)
    return questions


def _clean_answer(answer: str) -> tuple[str, set[str]]:
    text = re.sub(r"[^a-z0-9]+", " ", answer.lower()).strip()
    return text, set(text.split())


def _normalise_text(answer: str, question: str, make_labels: dict[str, str]) -> str:
    text, words = _clean_answer(answer)
    if question == "tem_carro":
        if words.intersection({"no", "not", "none"}):
            return "false"
        if words.intersection({"yes", "car", "vehicle", "automobile"}):
            return "true"
        return "unknown"

    if question == "marca":
        if not text or words.intersection({"unknown", "none", "unidentified", "unclear"}):
            return "unknown"
        normalized = _slug(text)
        for slug, display in make_labels.items():
            if slug in normalized or _slug(display) in normalized:
                return slug
        return "other"

    if question == "cor":
        for label, terms in COLOR_ANSWERS.items():
            if any(term in words for term in terms):
                return label
    if question == "tipo":
        for label, terms in BODY_ANSWERS.items():
            if any(term in text for term in terms):
                return label
    if not text or words.intersection({"unknown", "none", "cannot", "unclear", "unidentifiable"}):
        return "unknown"
    return "unknown"


def _laya_predictions(model, image: Image.Image, expected: dict[str, Any],
                      make_labels: dict[str, str]) -> tuple[dict[str, str], float]:
    questions = _questions_for(expected, make_labels)
    started = time.perf_counter()
    result = model.predict(image, questions)
    elapsed_ms = (time.perf_counter() - started) * 1000
    answers = result.get("answers", {})
    predictions: dict[str, str] = {}
    for key in questions:
        answer = answers.get(key, {})
        if key == "tem_carro":
            predictions[key] = "true" if answer.get("noul", 0.0) >= 0.5 else "false"
        else:
            predictions[key] = str(answer.get("choice", "unknown")).lower()
    return predictions, elapsed_ms


def _blip_predictions(processor, model, device: str, image: Image.Image,
                      expected: dict[str, Any], make_labels: dict[str, str]
                      ) -> tuple[dict[str, str], dict[str, str], float]:
    import torch
    predictions: dict[str, str] = {}
    raw_answers: dict[str, str] = {}
    questions = _questions_for(expected, make_labels)
    prompts = []
    keys = []
    for key, definition in questions.items():
        choices = definition.get("criteria")
        prompt = definition["instructions"]
        if choices:
            options = "; ".join(f"{option}: {label}" for option, label in choices.items())
            prompt += f" Options: {options}. Choose one option and answer with its label."
        prompts.append(prompt)
        keys.append(key)
    started = time.perf_counter()
    with torch.inference_mode():
        inputs = processor(images=[image] * len(prompts), text=prompts,
                           return_tensors="pt", padding=True).to(device)
        tokens = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        for key, token in zip(keys, tokens):
            answer = processor.decode(token, skip_special_tokens=True).strip()
            raw_answers[key] = answer
            predictions[key] = _normalise_text(answer, key, make_labels)
    return predictions, raw_answers, (time.perf_counter() - started) * 1000


def _summary(rows: list[dict[str, Any]], model_name: str) -> dict[str, Any]:
    metrics: dict[str, dict[str, int]] = {}
    latencies: list[float] = []
    for row in rows:
        latencies.append(row.get("latency_ms", 0.0))
        if row.get("error"):
            continue
        for key, expected in row["expected"].items():
            metrics.setdefault(key, {"correct": 0, "total": 0})
            metrics[key]["total"] += 1
            metrics[key]["correct"] += row["predictions"].get(key) == str(expected).lower()
    total = sum(value["total"] for value in metrics.values())
    correct = sum(value["correct"] for value in metrics.values())
    per_question = {
        key: {
            "correct": value["correct"], "total": value["total"],
            "accuracy": round(value["correct"] / value["total"], 4) if value["total"] else None,
        }
        for key, value in metrics.items()
    }
    ordered = sorted(latencies)
    median = ordered[len(ordered) // 2] if ordered else None
    return {
        "model": model_name,
        "correct": correct,
        "total": total,
        "accuracy": round(correct / total, 4) if total else None,
        "accuracy_by_question": per_question,
        "mean_latency_ms_per_image": round(sum(latencies) / len(latencies), 1) if latencies else None,
        "median_latency_ms_per_image": round(median, 1) if median is not None else None,
    }


def _write_progress(path: Path | None, state: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     delete=False, suffix=".tmp") as tmp:
        json.dump(state, tmp, ensure_ascii=False)
        temp_path = Path(tmp.name)
    os.replace(temp_path, path)


def run_comparison(examples: list[dict[str, Any]], threads: int = 8, device: str = "cpu",
                   progress_file: Path | None = None) -> dict[str, Any]:
    import torch
    torch.set_num_threads(threads)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA foi solicitado, mas não está disponível.")

    make_labels = dict(MAKE_LABELS)
    for example in examples:
        slug = example["expected"].get("marca")
        if slug and slug not in {"other", "unknown"}:
            make_labels.setdefault(slug, slug.replace("_", " ").title())

    total = len(examples)
    started = time.time()
    results: dict[str, list[dict[str, Any]]] = {"laya": [], "blip": []}
    live: dict[str, Any] = {
        "state": "running", "phase": "laya", "phase_label": "Laya Vision",
        "phase_completed": 0, "phase_total": total,
        "overall_completed": 0, "overall_total": total * 2,
        "dataset_total": total, "elapsed_seconds": 0,
        "current": None, "results": results, "summaries": [],
    }
    _write_progress(progress_file, live)

    def add_row(phase: str, index: int, item: dict[str, Any],
                predictions: dict[str, str], latency: float,
                raw_answers: dict[str, str] | None = None, error: str | None = None) -> None:
        row: dict[str, Any] = {
            "filename": item["filename"], "title": item["title"], "source": item["source"],
            "expected": item["expected"], "predictions": predictions,
            "latency_ms": round(latency, 1),
        }
        if raw_answers is not None:
            row["raw_answers"] = raw_answers
        if error:
            row["error"] = error
        results[phase].append(row)
        live.update({
            "phase": phase, "phase_label": "Laya Vision" if phase == "laya" else "BLIP VQA",
            "phase_completed": index, "phase_total": total,
            "overall_completed": (0 if phase == "laya" else total) + index,
            "elapsed_seconds": round(time.time() - started, 1),
            "current": row, "results": results,
            "summaries": [
                _summary(results["laya"], "Laya Vision") if results["laya"] else None,
                _summary(results["blip"], "BLIP VQA") if results["blip"] else None,
            ],
        })
        _write_progress(progress_file, live)

    print("Carregando Laya Vision...", flush=True)
    model = load_laya()
    # Warm-up is deliberately excluded from recorded image latencies.
    with Image.open(examples[0]["path"]) as source:
        _laya_predictions(model, source.convert("RGB"), examples[0]["expected"], make_labels)
    for index, item in enumerate(examples, 1):
        try:
            with Image.open(item["path"]) as source:
                image = source.convert("RGB")
            pred, latency = _laya_predictions(model, image, item["expected"], make_labels)
            add_row("laya", index, item, pred, latency)
        except Exception as exc:
            add_row("laya", index, item, {}, 0.0, error=f"{type(exc).__name__}: {exc}")
        print(f"Laya {index}/{total}: {item['filename']}", flush=True)
    del model
    gc.collect()

    live.update({"phase": "blip", "phase_label": "BLIP VQA", "phase_completed": 0})
    _write_progress(progress_file, live)
    print("Carregando BLIP VQA...", flush=True)
    processor, model = load_blip(device)
    with Image.open(examples[0]["path"]) as source:
        _blip_predictions(processor, model, device, source.convert("RGB"),
                          examples[0]["expected"], make_labels)
    for index, item in enumerate(examples, 1):
        try:
            with Image.open(item["path"]) as source:
                image = source.convert("RGB")
            pred, raw, latency = _blip_predictions(processor, model, device, image,
                                                    item["expected"], make_labels)
            add_row("blip", index, item, pred, latency, raw)
        except Exception as exc:
            add_row("blip", index, item, {}, 0.0, error=f"{type(exc).__name__}: {exc}")
        print(f"BLIP {index}/{total}: {item['filename']}", flush=True)
    del model, processor
    gc.collect()

    final = {
        "dataset": "Stanford Cars + local real cars and controls + 330 balanced color-labeled cars",
        "dataset_total": total,
        "device": device,
        "torch_threads": threads,
        "questions": BLIP_QUESTIONS,
        "summaries": [
            _summary(results["laya"], "thaitea/laya-vision"),
            _summary(results["blip"], "Salesforce/blip-vqa-base"),
        ],
        "results": results,
        "option_counts": {"tem_carro": 2, "marca": 12, "cor": 13, "tipo": len(BODY_LABELS)},
        "note": "All four base questions are sent to both models for every image. Only fields with ground-truth labels are scored. Every choice question has the same candidate count for Laya and BLIP.",
    }
    live.update({
        "state": "complete", "phase": "complete", "phase_label": "Concluído",
        "phase_completed": total, "overall_completed": total * 2,
        "elapsed_seconds": round(time.time() - started, 1),
        "summaries": final["summaries"], "results": results,
        "final": final,
    })
    _write_progress(progress_file, live)
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=OLD_IMAGES)
    parser.add_argument("--stanford-images", type=Path, default=STANFORD_IMAGES)
    parser.add_argument("--manifest", type=Path, default=STANFORD_MANIFEST)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--out", type=Path, default=ROOT / "eval/results/laya-blip-car-color.json")
    parser.add_argument("--progress-file", type=Path)
    args = parser.parse_args()

    examples = load_examples(args.images, args.stanford_images, args.manifest)
    if not examples:
        parser.error("Nenhuma imagem anotada encontrada; execute eval/download_stanford_cars_sample.py primeiro.")
    print(f"Benchmark em {len(examples)} imagens (4 tarefas onde há rótulos).", flush=True)
    result = run_comparison(examples, args.threads, args.device, args.progress_file)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\nResumo")
    for summary in result["summaries"]:
        print(f"  {summary['model']}: {summary['correct']}/{summary['total']} "
              f"({summary['accuracy'] * 100:.1f}%), mediana "
              f"{summary['median_latency_ms_per_image']:.0f} ms/imagem")
        for question, values in summary["accuracy_by_question"].items():
            print(f"    {question}: {values['correct']}/{values['total']} "
                  f"({values['accuracy'] * 100:.1f}%)")
    print(f"Resultados: {args.out}")


if __name__ == "__main__":
    main()

"""Build dataset for Test 4: Robustness to Image Quality.

Takes a subset of images from Tests 1-3 (with known ground truth) and applies
6 degradation strata to each:
  1. original      – Baseline control (no change)
  2. low_light     – Simulated low-light (brightness × 0.35)
  3. blur          – Gaussian blur (radius 3.5)
  4. low_res       – Downscale to 1/4 then upscale back (nearest-neighbour)
  5. jpeg_compress – Heavy JPEG compression (quality=12)
  6. occlusion     – Gray rectangle covering ~40% of the image

Selects ~20 images from each prior test (60 total source images) and generates
6 variants each = 360 evaluation images.

The manifest records original ground truth, source test, and degradation type.
"""
from __future__ import annotations

import io
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "eval" / "images" / "robustness"
MANIFEST_PATH = ROOT / "eval" / "datasets" / "test_4_robustness_manifest.json"

DEGRADATION_STRATA = [
    "original",
    "low_light",
    "blur",
    "low_res",
    "jpeg_compress",
    "occlusion",
]

# How many images to pick from each prior test
SAMPLE_PER_TEST = 20


def apply_degradation(img: Image.Image, stratum: str) -> Image.Image:
    """Apply a single degradation to a copy of the image."""
    img = img.copy()

    if stratum == "original":
        return img

    elif stratum == "low_light":
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(0.35)

    elif stratum == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=3.5))

    elif stratum == "low_res":
        w, h = img.size
        small = img.resize((max(1, w // 4), max(1, h // 4)), Image.NEAREST)
        return small.resize((w, h), Image.NEAREST)

    elif stratum == "jpeg_compress":
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=12)
        buf.seek(0)
        return Image.open(buf).copy()

    elif stratum == "occlusion":
        draw = ImageDraw.Draw(img)
        w, h = img.size
        # Cover ~40% of the image with a gray block in a random-ish position
        bw = int(w * 0.45)
        bh = int(h * 0.45)
        x0 = int(w * 0.30)
        y0 = int(h * 0.30)
        draw.rectangle([x0, y0, x0 + bw, y0 + bh], fill=(128, 128, 128))
        return img

    else:
        raise ValueError(f"Unknown stratum: {stratum}")


def sample_from_test1(n: int) -> list[dict[str, Any]]:
    """Sample n images from Test 1 (vehicle color) manifest."""
    manifest_path = ROOT / "eval" / "datasets" / "test_1_vehicle_color_manifest.json"
    with open(manifest_path) as f:
        m = json.load(f)

    eval_items = [it for it in m["items"] if it["split"] == "eval" and it["color"] != "unknown"]

    # Stratified sample: pick ~proportionally from each color
    by_color: dict[str, list] = {}
    for it in eval_items:
        by_color.setdefault(it["color"], []).append(it)

    selected = []
    per_color = max(1, n // len(by_color))
    for color, items in sorted(by_color.items()):
        random.shuffle(items)
        for it in items[:per_color]:
            selected.append({
                "source_test": "test_1_vehicle_color",
                "source_id": it["id"],
                "rel_path": it["rel_path"],
                "task": "color",
                "question_en": "What is the predominant color of the most prominent car?",
                "expected": it["color"],
                "taxonomy": list(m["taxonomy"]),
            })
    random.shuffle(selected)
    return selected[:n]


def sample_from_test2(n: int) -> list[dict[str, Any]]:
    """Sample n images from Test 2 (vehicle type) manifest."""
    manifest_path = ROOT / "eval" / "datasets" / "test_2_vehicle_type_manifest.json"
    with open(manifest_path) as f:
        m = json.load(f)

    eval_items = [it for it in m["items"] if it["split"] == "eval" and it["expected_type"] != "unknown"]

    by_type: dict[str, list] = {}
    for it in eval_items:
        by_type.setdefault(it["expected_type"], []).append(it)

    selected = []
    per_type = max(1, n // len(by_type))
    for vtype, items in sorted(by_type.items()):
        random.shuffle(items)
        for it in items[:per_type]:
            selected.append({
                "source_test": "test_2_vehicle_type",
                "source_id": it["id"],
                "rel_path": it["rel_path"],
                "task": "vehicle_type",
                "question_en": m["question_en"],
                "expected": it["expected_type"],
                "taxonomy": list(m["taxonomy"]),
            })
    random.shuffle(selected)
    return selected[:n]


def sample_from_test3(n: int) -> list[dict[str, Any]]:
    """Sample n images from Test 3 (person attributes) manifest."""
    manifest_path = ROOT / "eval" / "datasets" / "test_3_person_attributes_manifest.json"
    with open(manifest_path) as f:
        m = json.load(f)

    eval_items = [it for it in m["items"] if it["split"] == "eval"]

    random.shuffle(eval_items)
    selected = []
    for it in eval_items[:n]:
        # Use helmet as the primary question for robustness comparison
        selected.append({
            "source_test": "test_3_person_attributes",
            "source_id": it["id"],
            "rel_path": it["rel_path"],
            "task": "helmet",
            "question_en": m["attributes"]["helmet"]["question_en"],
            "expected": it["expected"]["helmet"],
            "taxonomy": m["attributes"]["helmet"]["taxonomy"],
        })
    return selected


def main() -> None:
    random.seed(42)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Sampling source images from Tests 1-3...")
    sources: list[dict[str, Any]] = []
    sources.extend(sample_from_test1(SAMPLE_PER_TEST))
    sources.extend(sample_from_test2(SAMPLE_PER_TEST))
    sources.extend(sample_from_test3(SAMPLE_PER_TEST))

    print(f"  Total source images: {len(sources)}")
    print(f"  From test_1: {sum(1 for s in sources if s['source_test'] == 'test_1_vehicle_color')}")
    print(f"  From test_2: {sum(1 for s in sources if s['source_test'] == 'test_2_vehicle_type')}")
    print(f"  From test_3: {sum(1 for s in sources if s['source_test'] == 'test_3_person_attributes')}")

    manifest_items = []
    errors = []

    for idx, src in enumerate(sources, 1):
        src_path = ROOT / src["rel_path"]
        if not src_path.exists():
            print(f"  WARNING: Source image not found: {src_path}")
            errors.append(src["source_id"])
            continue

        try:
            with Image.open(src_path) as orig_img:
                orig_img = orig_img.convert("RGB")

                for stratum in DEGRADATION_STRATA:
                    degraded = apply_degradation(orig_img, stratum)

                    out_name = f"{src['source_id']}_{stratum}.jpg"
                    out_path = OUTPUT_DIR / out_name
                    degraded.save(out_path, "JPEG", quality=85)

                    rel = str(out_path.relative_to(ROOT))
                    manifest_items.append({
                        "id": f"rob_{src['source_id']}_{stratum}",
                        "filename": out_name,
                        "rel_path": rel,
                        "source_test": src["source_test"],
                        "source_id": src["source_id"],
                        "task": src["task"],
                        "question_en": src["question_en"],
                        "expected": src["expected"],
                        "taxonomy": src["taxonomy"],
                        "stratum": stratum,
                        "split": "eval",
                    })
        except Exception as exc:
            print(f"  ERROR processing {src['source_id']}: {exc}")
            errors.append(src["source_id"])

        if idx % 10 == 0 or idx == len(sources):
            print(f"  Processed {idx}/{len(sources)} source images "
                  f"({len(manifest_items)} degraded images generated)")

    manifest = {
        "benchmark": "Test 4: Robustness to Image Quality (Robustez à Qualidade da Imagem)",
        "description": "Same visual questions as Tests 1-3, applied to degraded versions of source images.",
        "strata": DEGRADATION_STRATA,
        "strata_descriptions": {
            "original": "Baseline control – no degradation applied",
            "low_light": "Brightness reduced to 35% (simulates low-light conditions)",
            "blur": "Gaussian blur with radius 3.5 (simulates motion blur or defocus)",
            "low_res": "Downscaled to 1/4 resolution and upscaled back with nearest-neighbour",
            "jpeg_compress": "JPEG compression at quality=12 (heavy artefacts)",
            "occlusion": "Gray rectangle covering ~40% of the image area",
        },
        "source_tests": ["test_1_vehicle_color", "test_2_vehicle_type", "test_3_person_attributes"],
        "source_images_total": len(sources),
        "eval_count": len(manifest_items),
        "errors": errors,
        "items": manifest_items,
    }

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nManifest written: {MANIFEST_PATH}")
    print(f"Total degraded images: {len(manifest_items)}")
    print(f"Strata: {DEGRADATION_STRATA}")
    if errors:
        print(f"Errors ({len(errors)} sources skipped): {errors}")


if __name__ == "__main__":
    main()

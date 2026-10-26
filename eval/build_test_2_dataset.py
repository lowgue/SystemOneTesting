"""Build a rigorous, balanced dataset for Test 2: Vehicle Type.

Taxonomy (11 classes, 10 images per class in eval = 110 images):
1. sedan
2. hatchback
3. suv
4. pickup
5. van
6. coupe
7. convertible
8. truck (commercial heavy truck)
9. bus
10. motorcycle
11. unknown (negative control scenes without vehicles)

Sources:
- Local real cars & Stanford Cars sample
- Hugging Face Stanford Cars rows API
- Wikimedia Commons (buses, heavy trucks, motorcycles, minivans)
- Curated negative controls
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
TYPE_IMAGES_DIR = ROOT / "eval/images/vehicle_type"
MANIFEST_PATH = ROOT / "eval/datasets/test_2_vehicle_type_manifest.json"
CONTROLS_DIR = ROOT / "eval/images/unknown_controls"
STANFORD_SAMPLE_PATH = ROOT / "eval/datasets/stanford_cars_sample.json"
REAL_METADATA_PATH = ROOT / "eval/datasets/real_cars_metadata.json"

TAXONOMY = [
    "sedan", "hatchback", "suv", "pickup", "van",
    "coupe", "convertible", "truck", "bus", "motorcycle", "unknown"
]

STANFORD_BODY_PATTERNS = [
    ("convertible", r"\bconvertible\b"),
    ("hatchback", r"\bhatchback\b"),
    ("suv", r"\bsuv\b"),
    ("sedan", r"\bsedan\b"),
    ("coupe", r"\bcoupe\b"),
    ("van", r"\bminivan\b|\bvan\b"),
    ("pickup", r"\bpickup\b|\bcrew cab\b|\bclub cab\b|\bregular cab\b"),
]


def _get_json(url: str) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "SystemOneTesting/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _download_image(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 1000:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "SystemOneTesting/1.0"})
        with urllib.request.urlopen(req, timeout=20) as resp, open(dest, "wb") as f:
            f.write(resp.read())
        return True
    except Exception as e:
        if dest.exists():
            dest.unlink()
        return False


def fetch_commons_images(category: str, needed: int, prefix: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    url = (
        f"https://commons.wikimedia.org/w/api.php?action=query&generator=categorymembers&"
        f"gcmtitle=Category:{urllib.parse.quote(category)}&gcmtype=file&gcmlimit=40&"
        f"prop=imageinfo&iiprop=url&iiurlwidth=800&format=json"
    )
    try:
        data = _get_json(url)
        pages = data.get("query", {}).get("pages", {})
        for idx, (pid, pdata) in enumerate(pages.items(), 1):
            if len(results) >= needed:
                break
            info = pdata.get("imageinfo", [{}])[0]
            thumb = info.get("thumburl")
            if not thumb or not thumb.split("?")[0].lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            fname = f"{prefix}_{len(results)+1:02d}.jpg"
            dest = TYPE_IMAGES_DIR / fname
            if _download_image(thumb, dest):
                results.append({
                    "filename": fname,
                    "rel_path": f"eval/images/vehicle_type/{fname}",
                    "source": "wikimedia_commons",
                    "source_title": pdata.get("title", ""),
                })
    except Exception as e:
        print(f"Error fetching {category}: {e}")
    return results


def analyze_image_difficulty(path: Path, yolo: YOLO, expected_type: str) -> tuple[bool, list[str]]:
    if expected_type == "unknown":
        return False, []
    try:
        with Image.open(path) as img:
            img_rgb = img.convert("RGB")
            w, h = img_rgb.size
            arr = np.array(img_rgb)
            gray = np.mean(arr, axis=2)
            mean_b = np.mean(gray)
            std_b = np.std(gray)
            p10 = np.percentile(gray, 10)
            p98 = np.percentile(gray, 98)

        tags: list[str] = []
        if mean_b < 45:
            tags.append("low_light")
        if std_b > 68 and p10 < 22:
            tags.append("heavy_shadow")
        if p98 > 252 and std_b > 55:
            tags.append("high_reflection")

        res = yolo(img_rgb, verbose=False)[0]
        # Check all vehicle classes: 2:car, 3:motorcycle, 5:bus, 7:truck
        veh = [b for b in res.boxes if int(b.cls[0]) in (2, 3, 5, 7)]
        if len(veh) >= 2:
            tags.append("multiple_vehicles")
        elif len(veh) == 1:
            box = veh[0].xyxy[0].tolist()
            box_area = (box[2] - box[0]) * (box[3] - box[1])
            if box_area / (w * h) < 0.22:
                tags.append("small_or_distant")
        elif len(veh) == 0:
            tags.append("partial_or_atypical_view")

        return (len(tags) > 0), tags
    except Exception:
        return False, []


def main() -> None:
    TYPE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading YOLO for difficulty tagging...")
    yolo = YOLO(ROOT / "yolo11n.pt")

    items_by_class: dict[str, list[dict[str, Any]]] = {c: [] for c in TAXONOMY}

    # 1. Real cars from existing metadata
    if REAL_METADATA_PATH.exists():
        with open(REAL_METADATA_PATH, encoding="utf-8") as f:
            real_cars = json.load(f)
        for rc in real_cars:
            b = rc.get("body_type")
            if b == "hatch":
                b = "hatchback"
            if b in items_by_class and len(items_by_class[b]) < 10:
                src_path = ROOT / "eval/images/cars" / rc["filename"]
                if src_path.exists():
                    items_by_class[b].append({
                        "filename": rc["filename"],
                        "rel_path": f"eval/images/cars/{rc['filename']}",
                        "source": "real_cars_metadata",
                        "title": rc.get("full_name", rc["filename"]),
                    })

    # 2. Stanford cars sample
    if STANFORD_SAMPLE_PATH.exists():
        with open(STANFORD_SAMPLE_PATH, encoding="utf-8") as f:
            sc_sample = json.load(f)
        for sc in sc_sample.get("items", []):
            b = sc.get("body_type")
            if b == "hatch":
                b = "hatchback"
            if b in items_by_class and len(items_by_class[b]) < 10:
                src_path = ROOT / "eval/images/stanford_cars" / sc["filename"]
                if src_path.exists():
                    items_by_class[b].append({
                        "filename": sc["filename"],
                        "rel_path": f"eval/images/stanford_cars/{sc['filename']}",
                        "source": "stanford_cars_sample",
                        "title": sc.get("class_name", sc["filename"]),
                    })

    # 3. Fetch additional classes from Hugging Face Stanford Cars rows API
    needed_from_stanford = [c for c in ["sedan", "hatchback", "suv", "pickup", "van", "coupe", "convertible"]
                           if len(items_by_class[c]) < 10]
    if needed_from_stanford:
        print(f"Fetching additional rows for {needed_from_stanford} from Stanford Cars API...")
        for offset in range(0, 500, 100):
            if all(len(items_by_class[c]) >= 10 for c in needed_from_stanford):
                break
            url = f"https://datasets-server.huggingface.co/rows?dataset=naufalso/stanford_cars&config=default&split=test&offset={offset}&length=100"
            try:
                data = _get_json(url)
                for row_entry in data.get("rows", []):
                    row = row_entry["row"]
                    name = row["car_name"]
                    src_url = row.get("image_path", {}).get("src")
                    if not src_url:
                        continue
                    # Match body
                    target_body = None
                    for b_label, pat in STANFORD_BODY_PATTERNS:
                        if re.search(pat, name, re.IGNORECASE):
                            target_body = b_label
                            break
                    if target_body and len(items_by_class[target_body]) < 10:
                        idx_num = len(items_by_class[target_body]) + 1
                        fname = f"{target_body}_stanford_{idx_num:02d}.jpg"
                        dest = TYPE_IMAGES_DIR / fname
                        if _download_image(src_url, dest):
                            items_by_class[target_body].append({
                                "filename": fname,
                                "rel_path": f"eval/images/vehicle_type/{fname}",
                                "source": "stanford_cars_api",
                                "title": name,
                            })
            except Exception as e:
                print(f"Error fetching offset {offset}: {e}")

    # 4. Fetch missing categories from Wikimedia Commons
    commons_map = {
        "bus": "Buses",
        "truck": "Trucks",
        "motorcycle": "Motorcycles",
        "van": "Minivans",
        "pickup": "Pickup_trucks",
        "hatchback": "Hatchbacks",
    }
    for c, cat in commons_map.items():
        needed = 10 - len(items_by_class[c])
        if needed > 0:
            print(f"Fetching {needed} {c} from Commons Category:{cat}...")
            fetched = fetch_commons_images(cat, needed, f"type_{c}")
            items_by_class[c].extend(fetched)

    # 5. Negative controls (unknown)
    control_files = sorted([f for f in CONTROLS_DIR.iterdir() if f.name.endswith(".jpg") and not f.name.startswith(".")])[:10]
    for cfile in control_files:
        items_by_class["unknown"].append({
            "filename": cfile.name,
            "rel_path": f"eval/images/unknown_controls/{cfile.name}",
            "source": "curated_negative_controls",
            "title": f"No vehicle control: {cfile.name}",
        })

    # Assemble manifest items
    manifest_items = []
    print("\nAssembling and analyzing difficulty for 110 evaluation items...")
    for c in TAXONOMY:
        c_items = items_by_class[c][:10]
        print(f"[{c}] total gathered: {len(c_items)}/10")
        for idx, it in enumerate(c_items, 1):
            full_path = ROOT / it["rel_path"]
            is_diff, tags = analyze_image_difficulty(full_path, yolo, c)
            item_entry = {
                "id": f"type_{c}_{idx:02d}",
                "filename": it["filename"],
                "rel_path": it["rel_path"],
                "expected_type": c,
                "split": "eval",
                "source": it["source"],
                "title": it.get("title", it["filename"]),
                "difficult": is_diff,
                "difficulty_tags": tags,
            }
            manifest_items.append(item_entry)

    manifest = {
        "benchmark": "Test 2: Vehicle Type (Tipo de Veículo)",
        "question_pt": "Qual é o tipo do veículo principal visível?",
        "question_en": "What body type is this vehicle: sedan, hatchback, suv, pickup truck, van, coupe, convertible, truck, bus, or motorcycle?",
        "taxonomy": TAXONOMY,
        "eval_count": len(manifest_items),
        "items": manifest_items,
    }

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nManifest generated successfully at: {MANIFEST_PATH}")
    print(f"Total items in Test 2 eval set: {len(manifest_items)} (10 per class x 11 classes)")


if __name__ == "__main__":
    main()

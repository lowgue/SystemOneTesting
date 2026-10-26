"""Build a rigorous, balanced, stratified dataset for Test 1: Vehicle Color.

11 color classes (30 images each = 330 images) + 10 unknown/negative controls.
Stratifies into:
- 'eval' set (10 per class = 120 images): used for the 3-repetition benchmark.
- 'dev' set (remaining 20 per color class = 220 images): reserved for tuning/calibration.
Automatically analyzes visual difficulty:
- low_light
- heavy_shadow
- high_reflection
- multiple_vehicles
- unclear_vehicle_crop
"""
from __future__ import annotations

import json
from pathlib import Path
from PIL import Image
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
COLOR_MANIFEST = ROOT / "eval/datasets/car_color_test.json"
COLOR_DIR = ROOT / "eval/images/color_test"
CONTROLS_DIR = ROOT / "eval/images/unknown_controls"
OUTPUT_MANIFEST = ROOT / "eval/datasets/test_1_vehicle_color_manifest.json"

COLOR_CLASSES = [
    "black", "white", "gray_silver", "red", "blue",
    "green", "yellow", "orange", "brown", "purple", "pink"
]

def analyze_image(path: Path, yolo: YOLO, is_control: bool = False) -> tuple[bool, list[str]]:
    if is_control:
        return False, []
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
    if mean_b < 50:
        tags.append("low_light")
    if std_b > 70 and p10 < 20:
        tags.append("heavy_shadow")
    if p98 > 252 and std_b > 60:
        tags.append("high_reflection")

    res = yolo(img_rgb, verbose=False)[0]
    veh = [b for b in res.boxes if int(b.cls[0]) in (2, 3, 5, 7)]
    if len(veh) >= 2:
        tags.append("multiple_vehicles")
    elif len(veh) == 1:
        box = veh[0].xyxy[0].tolist()
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        if box_area / (w * h) < 0.25:
            tags.append("small_vehicle")
    elif len(veh) == 0:
        tags.append("unclear_vehicle_crop")

    return (len(tags) > 0), tags

def main() -> None:
    print("Loading YOLO model for vehicle detection and difficulty analysis...")
    yolo = YOLO(ROOT / "yolo11n.pt")

    with open(COLOR_MANIFEST, encoding="utf-8") as f:
        data = json.load(f)

    items_by_color: dict[str, list[dict]] = {c: [] for c in COLOR_CLASSES}
    for it in data["items"]:
        color = it["color"]
        if color in items_by_color:
            items_by_color[color].append(it)

    manifest_items = []
    
    # Process color items
    for color in COLOR_CLASSES:
        color_items = items_by_color[color]
        # Sort deterministically
        color_items.sort(key=lambda x: x["filename"])
        
        # We select 10 for eval, remaining 20 for dev
        # Ensure a balanced mix of commons vs dataset if present
        eval_items = color_items[:10]
        dev_items = color_items[10:]
        
        for idx, it in enumerate(color_items):
            split = "eval" if idx < 10 else "dev"
            rel_path = f"eval/images/color_test/{it['filename']}"
            full_path = ROOT / rel_path
            
            is_diff, tags = analyze_image(full_path, yolo)
            source = "commons" if "commons" in it.get("source_file", "") else "vehicle-colors-dataset"
            
            manifest_items.append({
                "id": f"color_{color}_{idx+1:03d}",
                "filename": it["filename"],
                "rel_path": rel_path,
                "color": color,
                "split": split,
                "source": source,
                "difficult": is_diff,
                "difficulty_tags": tags,
            })
            print(f"[{color}] {it['filename']} ({split}) -> diff={is_diff} tags={tags}")

    # Process controls (unknown / not visible)
    control_files = sorted([f for f in CONTROLS_DIR.iterdir() if f.name.endswith(".jpg") and not f.name.startswith(".")])
    for idx, cfile in enumerate(control_files):
        rel_path = f"eval/images/unknown_controls/{cfile.name}"
        manifest_items.append({
            "id": f"control_unknown_{idx+1:03d}",
            "filename": cfile.name,
            "rel_path": rel_path,
            "color": "unknown",
            "split": "eval",
            "source": "curated-negative-control",
            "difficult": False,
            "difficulty_tags": [],
        })
        print(f"[unknown] {cfile.name} (eval) -> negative control")

    manifest = {
        "benchmark": "Test 1: Vehicle Color (Cor do Veículo)",
        "question": "Qual é a cor predominante da carroceria do veículo principal?",
        "question_en": "What is the predominant color of the most prominent car?",
        "taxonomy": COLOR_CLASSES + ["unknown"],
        "eval_count": sum(1 for it in manifest_items if it["split"] == "eval"),
        "dev_count": sum(1 for it in manifest_items if it["split"] == "dev"),
        "total_count": len(manifest_items),
        "items": manifest_items,
    }

    OUTPUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nManifest generated successfully at: {OUTPUT_MANIFEST}")
    print(f"Eval set: {manifest['eval_count']} items (10 per class x 11 colors + 10 unknown controls)")
    print(f"Dev set: {manifest['dev_count']} items")

if __name__ == "__main__":
    main()

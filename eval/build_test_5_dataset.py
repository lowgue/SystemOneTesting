"""Build dataset for Test 5: Multiple Objects and Response Association.

Curates multi-object scenes (multiple vehicles, vehicle + person, multiple persons)
with clearly defined Target Object (with ground truth A) and Distractor Object (with ground truth B != A).

For each scene, creates two visual representations:
  1. `crop`: Isolated bounding box crop of the Target Object (detector output).
  2. `full_highlighted`: Full image with the Target Object highlighted with a prominent
     red bounding box (visual prompt for grounding).

Evaluates whether the model:
  - Correctly describes the target object (Acc)
  - Suffers from association error (predicts distractor's attribute instead of target)
  - Suffers from general classification error (predicts neither)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_IMG_DIR = ROOT / "eval/images/multi_object"
MANIFEST_PATH = ROOT / "eval/datasets/test_5_multi_object_manifest.json"

# Curated multi-object scenes with validated bounding boxes, target and distractor labels
SCENES = [
    # ── Group 1: Multi-vehicle Color (15 scenes) ─────────────────────────
    {
        "id": "scene_col_01",
        "src_path": "eval/images/color_test/color_orange_029.jpg",
        "task": "color",
        "target_bbox": [560, 235, 959, 526],
        "target_label": "white",
        "distractor_bbox": [244, 227, 759, 573],
        "distractor_label": "orange",
        "description": "White car on right (target) vs prominent orange car in center (distractor)",
    },
    {
        "id": "scene_col_02",
        "src_path": "eval/images/color_test/color_orange_016.jpg",
        "task": "color",
        "target_bbox": [442, 64, 958, 254],
        "target_label": "gray_silver",
        "distractor_bbox": [72, 114, 862, 515],
        "distractor_label": "orange",
        "description": "Silver/gray car in upper background (target) vs orange car in foreground (distractor)",
    },
    {
        "id": "scene_col_03",
        "src_path": "eval/images/color_test/color_pink_015.jpg",
        "task": "color",
        "target_bbox": [100, 2, 291, 200],
        "target_label": "blue",
        "distractor_bbox": [164, 55, 900, 458],
        "distractor_label": "pink",
        "description": "Blue car in upper left (target) vs large pink car in center (distractor)",
    },
    {
        "id": "scene_col_04",
        "src_path": "eval/images/color_test/color_purple_021.jpg",
        "task": "color",
        "target_bbox": [558, 640, 2108, 1310],
        "target_label": "black",
        "distractor_bbox": [221, 1174, 2644, 2905],
        "distractor_label": "purple",
        "description": "Black car in background (target) vs purple car in foreground (distractor)",
    },
    {
        "id": "scene_col_05",
        "src_path": "eval/images/color_test/color_brown_027.jpg",
        "task": "color",
        "target_bbox": [0, 38, 311, 266],
        "target_label": "white",
        "distractor_bbox": [114, 38, 642, 427],
        "distractor_label": "brown",
        "description": "White vehicle on left (target) vs large brown car in center (distractor)",
    },
    {
        "id": "scene_col_06",
        "src_path": "eval/images/color_test/color_orange_012.jpg",
        "task": "color",
        "target_bbox": [250, 209, 672, 376],
        "target_label": "black",
        "distractor_bbox": [438, 227, 959, 495],
        "distractor_label": "orange",
        "description": "Black car in middle (target) vs orange car on right (distractor)",
    },
    {
        "id": "scene_col_07",
        "src_path": "eval/images/color_test/color_pink_006.jpg",
        "task": "color",
        "target_bbox": [624, 491, 957, 920],
        "target_label": "white",
        "distractor_bbox": [44, 480, 856, 1138],
        "distractor_label": "pink",
        "description": "White car on right (target) vs pink car on left (distractor)",
    },
    {
        "id": "scene_col_08",
        "src_path": "eval/images/color_test/color_orange_009.jpg",
        "task": "color",
        "target_bbox": [514, 14, 908, 291],
        "target_label": "blue",
        "distractor_bbox": [96, 101, 891, 555],
        "distractor_label": "orange",
        "description": "Blue car in background (target) vs large orange car in foreground (distractor)",
    },
    {
        "id": "scene_col_09",
        "src_path": "eval/images/color_test/color_pink_010.jpg",
        "task": "color",
        "target_bbox": [0, 98, 253, 340],
        "target_label": "white",
        "distractor_bbox": [98, 181, 959, 695],
        "distractor_label": "pink",
        "description": "White vehicle on left (target) vs pink truck in center (distractor)",
    },
    {
        "id": "scene_col_10",
        "src_path": "eval/images/color_test/color_purple_026.jpg",
        "task": "color",
        "target_bbox": [233, 46, 368, 100],
        "target_label": "white",
        "distractor_bbox": [40, 62, 447, 238],
        "distractor_label": "purple",
        "description": "White car in background (target) vs purple car in foreground (distractor)",
    },
    {
        "id": "scene_col_11",
        "src_path": "eval/images/color_test/color_brown_017.jpg",
        "task": "color",
        "target_bbox": [0, 100, 280, 300],
        "target_label": "white",
        "distractor_bbox": [150, 50, 900, 550],
        "distractor_label": "brown",
        "description": "White vehicle on left (target) vs brown vehicle in center (distractor)",
    },
    {
        "id": "scene_col_12",
        "src_path": "eval/images/color_test/color_orange_001.jpg",
        "task": "color",
        "target_bbox": [0, 124, 219, 292],
        "target_label": "blue",
        "distractor_bbox": [35, 183, 925, 519],
        "distractor_label": "orange",
        "description": "Blue car on left (target) vs orange car in foreground (distractor)",
    },
    {
        "id": "scene_col_13",
        "src_path": "eval/images/color_test/color_purple_011.jpg",
        "task": "color",
        "target_bbox": [0, 129, 188, 292],
        "target_label": "red",
        "distractor_bbox": [49, 68, 907, 540],
        "distractor_label": "purple",
        "description": "Red car in background (target) vs purple truck in foreground (distractor)",
    },
    {
        "id": "scene_col_14",
        "src_path": "eval/images/color_test/color_pink_027.jpg",
        "task": "color",
        "target_bbox": [0, 136, 282, 379],
        "target_label": "black",
        "distractor_bbox": [200, 50, 900, 500],
        "distractor_label": "pink",
        "description": "Black car on left (target) vs pink vehicle in center (distractor)",
    },
    {
        "id": "scene_col_15",
        "src_path": "eval/images/cars/real_volkswagen_2.jpg",
        "task": "color",
        "target_bbox": [372, 91, 639, 267],
        "target_label": "blue",
        "distractor_bbox": [28, 58, 626, 373],
        "distractor_label": "white",
        "description": "Blue car on right (target) vs white truck on left (distractor)",
    },

    # ── Group 2: Multi-vehicle Type (10 scenes) ──────────────────────────
    {
        "id": "scene_typ_01",
        "src_path": "eval/images/color_test/color_orange_015.jpg",
        "task": "vehicle_type",
        "target_bbox": [0, 1, 397, 320],
        "target_label": "bus",
        "distractor_bbox": [59, 118, 839, 771],
        "distractor_label": "sedan",
        "description": "Bus on left (target) vs sedan in center (distractor)",
    },
    {
        "id": "scene_typ_02",
        "src_path": "eval/images/cars/real_volkswagen_2.jpg",
        "task": "vehicle_type",
        "target_bbox": [372, 91, 639, 267],
        "target_label": "sedan",
        "distractor_bbox": [28, 58, 626, 373],
        "distractor_label": "truck",
        "description": "Sedan car on right (target) vs large truck on left (distractor)",
    },
    {
        "id": "scene_typ_03",
        "src_path": "eval/images/vehicle_type/type_bus_01.jpg",
        "task": "vehicle_type",
        "target_bbox": [880, 433, 924, 469],
        "target_label": "sedan",
        "distractor_bbox": [232, 307, 529, 556],
        "distractor_label": "bus",
        "description": "Small sedan on right (target) vs large bus in center (distractor)",
    },
    {
        "id": "scene_typ_04",
        "src_path": "eval/images/color_test/color_pink_010.jpg",
        "task": "vehicle_type",
        "target_bbox": [0, 98, 253, 340],
        "target_label": "sedan",
        "distractor_bbox": [98, 181, 959, 695],
        "distractor_label": "truck",
        "description": "Sedan car on left (target) vs large truck in center (distractor)",
    },
    {
        "id": "scene_typ_05",
        "src_path": "eval/images/color_test/color_purple_011.jpg",
        "task": "vehicle_type",
        "target_bbox": [0, 129, 188, 292],
        "target_label": "sedan",
        "distractor_bbox": [49, 68, 907, 540],
        "distractor_label": "truck",
        "description": "Sedan on left (target) vs truck in foreground (distractor)",
    },
    {
        "id": "scene_typ_06",
        "src_path": "eval/images/cars/real_audi_2.jpg",
        "task": "vehicle_type",
        "target_bbox": [42, 55, 231, 147],
        "target_label": "sedan",
        "distractor_bbox": [201, 27, 258, 155],
        "distractor_label": "truck",
        "description": "Audi sedan in center (target) vs white truck behind it (distractor)",
    },
    {
        "id": "scene_typ_07",
        "src_path": "eval/images/vehicle_type/type_bus_03.jpg",
        "task": "vehicle_type",
        "target_bbox": [0, 315, 42, 403],
        "target_label": "sedan",
        "distractor_bbox": [45, 61, 871, 571],
        "distractor_label": "bus",
        "description": "Car on far left edge (target) vs large bus dominating scene (distractor)",
    },
    {
        "id": "scene_typ_08",
        "src_path": "eval/images/color_test/color_brown_027.jpg",
        "task": "vehicle_type",
        "target_bbox": [0, 38, 311, 266],
        "target_label": "truck",
        "distractor_bbox": [114, 38, 642, 427],
        "distractor_label": "sedan",
        "description": "White truck on left (target) vs brown sedan in center (distractor)",
    },
    {
        "id": "scene_typ_09",
        "src_path": "eval/images/vehicle_type/type_bus_08.jpg",
        "task": "vehicle_type",
        "target_bbox": [750, 350, 850, 450],
        "target_label": "sedan",
        "distractor_bbox": [50, 50, 750, 550],
        "distractor_label": "bus",
        "description": "Car adjacent to bus (target) vs main bus (distractor)",
    },
    {
        "id": "scene_typ_10",
        "src_path": "eval/images/color_test/color_orange_023.jpg",
        "task": "vehicle_type",
        "target_bbox": [704, 414, 956, 519],
        "target_label": "sedan",
        "distractor_bbox": [245, 315, 548, 542],
        "distractor_label": "truck",
        "description": "Sedan car on right (target) vs truck in center (distractor)",
    },

    # ── Group 3: Multi-person / Person vs Vehicle (5 scenes) ─────────────
    {
        "id": "scene_per_01",
        "src_path": "eval/images/person_attributes/helmet_yes_04.jpg",
        "task": "helmet",
        "target_bbox": [58, 23, 227, 415],
        "target_label": "yes",
        "distractor_bbox": [361, 48, 588, 452],
        "distractor_label": "no",
        "description": "Person on left wearing helmet (target) vs person on right with bare head (distractor)",
    },
    {
        "id": "scene_per_02",
        "src_path": "eval/images/person_attributes/backpack_07.jpg",
        "task": "carrying",
        "target_bbox": [552, 633, 807, 1341],
        "target_label": "backpack",
        "distractor_bbox": [324, 643, 594, 1270],
        "distractor_label": "none",
        "description": "Person on right carrying backpack (target) vs person on left with no bag (distractor)",
    },
    {
        "id": "scene_per_03",
        "src_path": "eval/images/person_attributes/safety_vest_01.jpg",
        "task": "clothing",
        "target_bbox": [98, 248, 892, 1176],
        "target_label": "safety_vest",
        "distractor_bbox": [0, 141, 285, 358],
        "distractor_label": "jacket_coat",
        "description": "Person wearing safety vest in foreground (target) vs background cars/objects (distractor)",
    },
    {
        "id": "scene_per_04",
        "src_path": "eval/images/person_attributes/pedestrian_12.jpg",
        "task": "helmet",
        "target_bbox": [504, 850, 558, 1011],
        "target_label": "no",
        "distractor_bbox": [564, 856, 616, 1011],
        "distractor_label": "no",
        "description": "Pedestrian on left without helmet (target) vs pedestrian on right (distractor)",
    },
    {
        "id": "scene_per_05",
        "src_path": "eval/images/cars/real_toyota_2.jpg",
        "task": "color",
        "target_bbox": [22, 31, 417, 305],
        "target_label": "white",
        "distractor_bbox": [56, 2, 153, 134],
        "distractor_label": "blue",
        "description": "White car (target) vs person in blue shirt standing next to car (distractor)",
    },
]

TASK_PROMPTS = {
    "color": {
        "crop": "What is the predominant color of this vehicle?",
        "highlighted": "In the highlighted object inside the red box, what is the predominant color of the vehicle?",
    },
    "vehicle_type": {
        "crop": "What body type is this vehicle: sedan, hatchback, suv, pickup truck, van, coupe, convertible, truck, bus, or motorcycle?",
        "highlighted": "In the highlighted object inside the red box, what body type is this vehicle: sedan, hatchback, suv, pickup truck, van, coupe, convertible, truck, bus, or motorcycle?",
    },
    "helmet": {
        "crop": "Is the person wearing a helmet or hard hat?",
        "highlighted": "In the highlighted object inside the red box, is the person wearing a helmet or hard hat?",
    },
    "carrying": {
        "crop": "Is the person carrying a backpack, bag, box, or nothing?",
        "highlighted": "In the highlighted object inside the red box, is the person carrying a backpack, bag, box, or nothing?",
    },
    "clothing": {
        "crop": "What upper-body clothing is the person wearing: safety_vest, jacket_coat, shirt_tshirt, or not_visible?",
        "highlighted": "In the highlighted object inside the red box, what upper-body clothing is the person wearing?",
    },
}


def main():
    OUTPUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
    manifest_items = []
    print(f"Building Test 5 dataset: {len(SCENES)} multi-object scenes...")

    for sc in SCENES:
        src = ROOT / sc["src_path"]
        if not src.exists():
            print(f"  WARNING: missing {src}")
            continue

        with Image.open(src) as img:
            img = img.convert("RGB")
            w, h = img.size

            # 1. Crop
            x1, y1, x2, y2 = sc["target_bbox"]
            # Pad by 5px clamped
            cx1 = max(0, x1 - 5)
            cy1 = max(0, y1 - 5)
            cx2 = min(w, x2 + 5)
            cy2 = min(h, y2 + 5)
            crop_img = img.crop((cx1, cy1, cx2, cy2))

            crop_name = f"{sc['id']}_crop.jpg"
            crop_path = OUTPUT_IMG_DIR / crop_name
            crop_img.save(crop_path, "JPEG", quality=90)

            # 2. Full Highlighted (draw bright red box around target)
            high_img = img.copy()
            draw = ImageDraw.Draw(high_img)
            # Outline with 6px width
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=6)

            high_name = f"{sc['id']}_full_highlighted.jpg"
            high_path = OUTPUT_IMG_DIR / high_name
            high_img.save(high_path, "JPEG", quality=90)

        # Manifest item for CROP
        manifest_items.append({
            "id": f"{sc['id']}_crop",
            "scene_id": sc["id"],
            "presentation": "crop",
            "task": sc["task"],
            "rel_path": str(crop_path.relative_to(ROOT)),
            "target_label": sc["target_label"],
            "distractor_label": sc["distractor_label"],
            "question_en": TASK_PROMPTS[sc["task"]]["crop"],
            "description": sc["description"],
            "target_bbox": sc["target_bbox"],
            "distractor_bbox": sc["distractor_bbox"],
        })

        # Manifest item for FULL HIGHLIGHTED
        manifest_items.append({
            "id": f"{sc['id']}_full_highlighted",
            "scene_id": sc["id"],
            "presentation": "full_highlighted",
            "task": sc["task"],
            "rel_path": str(high_path.relative_to(ROOT)),
            "target_label": sc["target_label"],
            "distractor_label": sc["distractor_label"],
            "question_en": TASK_PROMPTS[sc["task"]]["highlighted"],
            "description": sc["description"],
            "target_bbox": sc["target_bbox"],
            "distractor_bbox": sc["distractor_bbox"],
        })

    manifest = {
        "benchmark": "Test 5: Multiple Objects and Response Association",
        "description": "Evaluates response association and distractor intrusion when comparing detector crops vs full scene with visual highlight.",
        "scene_count": len(SCENES),
        "total_items": len(manifest_items),
        "tasks_breakdown": {
            "color": sum(1 for it in manifest_items if it["task"] == "color") // 2,
            "vehicle_type": sum(1 for it in manifest_items if it["task"] == "vehicle_type") // 2,
            "person_attributes": sum(1 for it in manifest_items if it["task"] in ("helmet", "carrying", "clothing")) // 2,
        },
        "items": manifest_items,
    }

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Manifest written to {MANIFEST_PATH}")
    print(f"Generated {len(manifest_items)} items ({len(SCENES)} scenes × 2 presentations: crop & full_highlighted).")


if __name__ == "__main__":
    main()

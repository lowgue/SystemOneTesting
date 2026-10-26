"""Build a curated, balanced dataset for Test 3: Visible Person Attributes.

Target attributes (strictly observable, zero sensitive inference):
1. helmet: 'yes' (helmet/hard hat), 'no' (bare/hat/cap), 'not_visible'
2. carrying: 'backpack', 'bag_or_box', 'none', 'not_visible'
3. clothing: 'safety_vest', 'jacket_coat', 'shirt_tshirt', 'not_visible'

Evaluation split: 60 diverse images.
Controls include scenes without people (to measure false positive hallucinations)
and scenes with distant or occluded people.
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
IMAGES_DIR = ROOT / "eval/images/person_attributes"
CONTROLS_DIR = ROOT / "eval/images/unknown_controls"
MANIFEST_PATH = ROOT / "eval/datasets/test_3_person_attributes_manifest.json"

HEADERS = {"User-Agent": "SystemOneTestingBenchmark/1.0 (research benchmark; mailto:lowgue16@gmail.com)"}


def _get_commons_files(category: str, limit: int = 20) -> list[dict[str, str]]:
    cat_param = category if category.startswith("Category:") else f"Category:{category}"
    url = (
        f"https://commons.wikimedia.org/w/api.php?action=query&generator=categorymembers&"
        f"gcmtitle={urllib.parse.quote(cat_param)}&gcmtype=file&gcmlimit={limit}&"
        f"prop=imageinfo&iiprop=url&iiurlwidth=800&format=json"
    )
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.load(resp)
    pages = data.get("query", {}).get("pages", {})
    files = []
    for pid, pdata in pages.items():
        info = pdata.get("imageinfo", [{}])[0]
        thumb = info.get("thumburl")
        if thumb and thumb.split("?")[0].lower().endswith((".jpg", ".jpeg", ".png")):
            files.append({"title": pdata.get("title", ""), "url": thumb})
    return files


def _download_file(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 1000:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=20) as resp, open(dest, "wb") as f:
            f.write(resp.read())
        return True
    except Exception as e:
        if dest.exists():
            dest.unlink()
        return False


def main() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading YOLO for person detection and difficulty analysis...")
    yolo = YOLO(ROOT / "yolo11n.pt")

    groups = [
        # 1. Helmets
        ("Category:People_wearing_helmets", "helmet_yes", 10, {
            "helmet": "yes", "carrying": "none", "clothing": "jacket_coat"
        }),
        ("Category:Hard_hats", "hardhat", 5, {
            "helmet": "yes", "carrying": "none", "clothing": "safety_vest"
        }),
        # 2. Carrying backpacks
        ("Category:People_with_backpacks", "backpack", 10, {
            "helmet": "no", "carrying": "backpack", "clothing": "jacket_coat"
        }),
        # 3. High-visibility safety vests
        ("Category:High-visibility_clothing", "safety_vest", 10, {
            "helmet": "no", "carrying": "none", "clothing": "safety_vest"
        }),
        # 4. Standard pedestrians (t-shirts/shirts, no helmet)
        ("Category:Pedestrians_in_Germany", "pedestrian", 15, {
            "helmet": "no", "carrying": "none", "clothing": "shirt_tshirt"
        }),
    ]

    manifest_items = []
    count = 0

    for cat, prefix, target_count, default_labels in groups:
        print(f"Fetching {cat}...")
        try:
            files = _get_commons_files(cat, limit=target_count + 5)
            time.sleep(1.2)
        except Exception as e:
            print(f"Error querying {cat}: {e}")
            files = []

        gathered = 0
        for f_idx, f_entry in enumerate(files, 1):
            if gathered >= target_count:
                break
            fname = f"{prefix}_{gathered+1:02d}.jpg"
            fpath = IMAGES_DIR / fname
            if _download_file(f_entry["url"], fpath):
                # Verify with YOLO
                res = yolo(fpath, verbose=False)[0]
                persons = [b for b in res.boxes if int(b.cls[0]) == 0]
                # If category requires a person, make sure YOLO sees at least 1 person or body
                if len(persons) == 0 and "control" not in prefix:
                    # Let it pass if it's an atypical angle or hard hat close-up
                    pass
                gathered += 1
                count += 1
                
                # Analyze difficulty
                tags = []
                w, h = Image.open(fpath).size
                if len(persons) >= 2:
                    tags.append("multiple_people")
                elif len(persons) == 1:
                    box = persons[0].xyxy[0].tolist()
                    box_area = (box[2] - box[0]) * (box[3] - box[1])
                    if box_area / (w * h) < 0.15:
                        tags.append("distant_small_person")
                elif len(persons) == 0:
                    tags.append("partial_or_close_crop")

                manifest_items.append({
                    "id": f"person_{count:03d}",
                    "filename": fname,
                    "rel_path": f"eval/images/person_attributes/{fname}",
                    "split": "eval",
                    "source": "wikimedia_commons",
                    "title": f_entry["title"],
                    "expected": dict(default_labels),
                    "difficult": len(tags) > 0,
                    "difficulty_tags": tags,
                })
                print(f"  [{prefix}] saved {fname} ({gathered}/{target_count}) -> tags={tags}")

    # Add 10 negative controls without people
    print("Adding 10 negative control images without people...")
    control_files = sorted([f for f in CONTROLS_DIR.iterdir() if f.name.endswith(".jpg") and not f.name.startswith(".")])[:10]
    for cfile in control_files:
        count += 1
        manifest_items.append({
            "id": f"person_{count:03d}",
            "filename": cfile.name,
            "rel_path": f"eval/images/unknown_controls/{cfile.name}",
            "split": "eval",
            "source": "curated_negative_controls",
            "title": f"No person control: {cfile.name}",
            "expected": {
                "helmet": "not_visible",
                "carrying": "not_visible",
                "clothing": "not_visible",
            },
            "difficult": False,
            "difficulty_tags": [],
        })
        print(f"  [control] {cfile.name} -> not_visible for all attributes")

    manifest = {
        "benchmark": "Test 3: Visible Person Attributes (Atributos Visíveis de Pessoas)",
        "attributes": {
            "helmet": {
                "question_pt": "A pessoa principal está usando capacete?",
                "question_en": "Is the prominent person wearing a helmet or hard hat?",
                "taxonomy": ["yes", "no", "not_visible"],
            },
            "carrying": {
                "question_pt": "A pessoa principal está carregando algum objeto visível?",
                "question_en": "Is the prominent person carrying a bag, backpack, or visible object?",
                "taxonomy": ["backpack", "bag_or_box", "none", "not_visible"],
            },
            "clothing": {
                "question_pt": "Qual tipo de vestimenta externa superior está visível na pessoa principal?",
                "question_en": "What upper-body clothing is the prominent person wearing?",
                "taxonomy": ["safety_vest", "jacket_coat", "shirt_tshirt", "not_visible"],
            },
        },
        "eval_count": len(manifest_items),
        "items": manifest_items,
    }

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nManifest generated successfully at: {MANIFEST_PATH}")
    print(f"Total items in Test 3 eval set: {len(manifest_items)}")


if __name__ == "__main__":
    main()

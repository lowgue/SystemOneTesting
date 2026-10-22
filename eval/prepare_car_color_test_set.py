"""Select a balanced, car-only color test set from the public source archive.

The source archive must first be extracted; see README.md. This script keeps
30 examples for each color class used by the Laya-vs-BLIP benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "eval/images/color_test"
MANIFEST = ROOT / "eval/datasets/car_color_test.json"
SEED = 20261022
SOURCE_GROUPS = {
    "black": ["black"],
    "white": ["white"],
    "gray_silver": ["grey"],
    "red": ["red"],
    "blue": ["blue"],
    "brown": ["brown"], "green": ["green"], "orange": ["orange"],
    "pink": ["pink"], "purple": ["purple"], "yellow": ["yellow"],
}
VEHICLE_CLASSES = (2, 3, 5, 7)  # COCO: car, motorcycle, bus, truck
# Manual visual review exclusions: small motorcycles and three-wheeled taxis
# that the lightweight detector had mistaken for cars.
REJECTED_NON_CARS = {
    "06b28c66-1150-4df9-8d99-e5b33a8618b0.jpg",
    "ce980c18-5fa0-4f20-ae85-ce08f63ee29b.jpg",
    "c1963dd0-46b3-47da-9640-20c0111cb9e5.jpg",
    "98.png",
    "0d5123a7-4b3f-4b41-8328-f2c9083c9090.jpg",
    "14.png",
    "23e46c2a-2b15-4e11-a8bf-58899a15123f.jpg",
    "4ec6239f-50ee-4d65-9328-a761c489742f.jpg",
    "b182ba59-f2b1-4dba-84db-e66ca7e9bbe0.jpg",
    "12b142b3-cfc6-49e4-bc89-58b0a169d936.jpg",
    "5ec3b195-790e-4717-a998-4e60d988f46c.jpg",
    "d6fd7d15-4c61-4548-b085-f466d6364558.jpg",
    "8516c4c1-ae5c-49d2-9ad4-eb6a137d8992.jpg",
    "244b7abf-8841-4464-8221-7809c16d62b1.jpg",
    "51.png",
    "76.png",
    "50538043-92fe-4ac3-8f50-dc6321b753b3.jpg",
    "650c6d9c-3ac6-4d4c-b548-9d1b67d40b96.jpg",
    "41.png",
    "8b08e5cd-c10f-457b-bb64-9356271b53f0.jpg",
    "4571962c-cdd5-46b3-bead-ee6355d2e995.jpg",
    "4195777a-c38d-45da-8801-b88e228281c5.jpg",
    "136372ae-40ce-4a8b-b2db-c260e01652fe.jpg",
    "a1dfc680-5cfd-4f2c-a7c0-722e4dede81d.jpg",
    "3d43edcc-fef6-4590-a638-c9c2ec286d34.jpg",
    "105.png",
    "59.png",
    "4.png",
    "260d4d16-1e55-45f5-a9ea-8083131d4945.jpg",
    "007f1426-fea9-402d-9bcc-28062d319abd.jpg",
    "c9a8e94f-ca7e-43be-94fd-1f6031c89ad0.jpg",
    "115.png",
    "03c0dc7b-b681-4db3-9250-f70401d002ad.jpg",
    "7a62649e-ef22-4989-93f2-6e656c608b0d.jpg",
}


def _commons_json(params: dict) -> dict:
    query = urllib.parse.urlencode({**params, "format": "json"})
    request = urllib.request.Request(
        f"https://commons.wikimedia.org/w/api.php?{query}",
        headers={"User-Agent": "SystemOneTestingBenchmark/1.0 (local research benchmark)"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def _download_commons_supplement(color: str, folder: Path, minimum: int = 50) -> None:
    """Supplement sparse source groups with Commons-licensed photos of that color."""
    source = folder / "commons"
    source.mkdir(parents=True, exist_ok=True)
    metadata_path = source / "sources.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    if len(list(source.glob("*.jpg"))) >= minimum:
        return
    continuation = None
    while len(list(source.glob("*.jpg"))) < 60:
        params = {
            "action": "query", "generator": "categorymembers",
            "gcmtitle": f"Category:{color.title()} automobiles", "gcmtype": "file", "gcmlimit": 50,
            "prop": "imageinfo", "iiprop": "url|extmetadata", "iiurlwidth": 800,
        }
        if continuation:
            params.update(continuation)
        page = _commons_json(params)
        for item in page.get("query", {}).get("pages", {}).values():
            info = item.get("imageinfo", [{}])[0]
            thumb = info.get("thumburl") or info.get("url")
            if not thumb:
                continue
            filename = re.sub(r"[^a-zA-Z0-9._-]+", "_", item["title"].removeprefix("File:"))
            path = source / (filename[:100] + ".jpg")
            if not path.exists():
                request = urllib.request.Request(thumb, headers={"User-Agent": "SystemOneTestingBenchmark/1.0"})
                try:
                    with urllib.request.urlopen(request, timeout=30) as response:
                        path.write_bytes(response.read())
                except Exception:
                    continue
            ext = info.get("extmetadata", {})
            metadata[path.name] = {
                "source_url": info.get("descriptionurl"),
                "license": re.sub("<[^>]+>", "", ext.get("LicenseShortName", {}).get("value", "")),
                "author": re.sub("<[^>]+>", "", ext.get("Artist", {}).get("value", "")),
            }
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        continuation = page.get("continue")
        if not continuation:
            break


def average_hash(path: Path) -> int:
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("L").resize((8, 8)), dtype=np.float32)
    bits = pixels >= pixels.mean()
    return sum(int(bit) << index for index, bit in enumerate(bits.flatten()))


def car_detection(result) -> bool:
    if not len(result.boxes):
        return False
    boxes = result.boxes
    order = boxes.conf.argsort(descending=True)
    for idx in order.tolist():
        if int(boxes.cls[idx]) not in VEHICLE_CLASSES:
            continue
        if int(boxes.cls[idx]) != 2 or float(boxes.conf[idx]) < 0.25:
            continue
        x1, y1, x2, y2 = boxes.xyxy[idx].tolist()
        image_area = max(1, result.orig_shape[0] * result.orig_shape[1])
        if (x2 - x1) * (y2 - y1) / image_area >= 0.005:
            return True
        return False
    return False


def prepare(source_dir: Path, model_path: str, seed: int, per_color: int) -> list[dict]:
    import torch

    torch.set_num_threads(min(8, torch.get_num_threads()))
    detector = YOLO(model_path)
    randomizer = random.Random(seed)
    for color in ("orange", "brown", "pink", "purple"):
        _download_commons_supplement(color, source_dir / color)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for file in OUTPUT_DIR.iterdir():
        if file.is_file():
            file.unlink()
    manifest_items: list[dict] = []

    for color, source_colors in SOURCE_GROUPS.items():
        candidates = []
        for source_color in source_colors:
            folder = source_dir / source_color
            if not folder.is_dir():
                raise FileNotFoundError(f"Pasta de cor ausente: {folder}")
            for path in folder.rglob("*"):
                if path.name in REJECTED_NON_CARS:
                    continue
                if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                try:
                    with Image.open(path) as image:
                        image.load()
                    candidates.append((path, source_color))
                except Exception:
                    continue
        randomizer.shuffle(candidates)

        seen_hashes: list[int] = []
        seen_bytes: set[str] = set()
        accepted = 0
        for offset in range(0, len(candidates), 64):
            batch = candidates[offset:offset + 64]
            results = detector.predict(
                [str(path) for path, _ in batch], imgsz=320, batch=16,
                device="cpu", classes=list(VEHICLE_CLASSES), conf=0.20,
                verbose=False,
            )
            for (path, source_color), result in zip(batch, results):
                if not car_detection(result):
                    continue
                digest = hashlib.md5(path.read_bytes()).hexdigest()
                if digest in seen_bytes:
                    continue
                try:
                    image_hash = average_hash(path)
                except Exception:
                    continue
                if any((image_hash ^ previous).bit_count() <= 4 for previous in seen_hashes):
                    continue
                seen_bytes.add(digest)
                seen_hashes.append(image_hash)
                accepted += 1
                filename = f"color_{color}_{accepted:03d}{path.suffix.lower()}"
                destination = OUTPUT_DIR / filename
                with Image.open(path) as image:
                    image.convert("RGB").save(destination, quality=95)
                item = {
                    "filename": filename, "color": color,
                    "source_color": source_color, "source_file": path.name,
                }
                supplemental = path.parent / "sources.json"
                if supplemental.is_file():
                    source_item = json.loads(supplemental.read_text(encoding="utf-8")).get(path.name)
                    if source_item:
                        item.update(source_item)
                manifest_items.append(item)
                if accepted == per_color:
                    break
            print(f"{color}: {accepted}/{per_color}", flush=True)
            MANIFEST.parent.mkdir(parents=True, exist_ok=True)
            MANIFEST.write_text(json.dumps({
                "dataset": "WandererGuy/vehicle-color-classification-dataset",
                "source_url": "https://huggingface.co/datasets/WandererGuy/vehicle-color-classification-dataset",
                "seed": seed, "samples_per_color": per_color,
                "color_classes": list(SOURCE_GROUPS),
                    "car_filter": "Ultralytics YOLO11n, COCO car class, confidence >= 0.25 and box >= 0.5% of image; near-duplicate average hash distance > 4",
                "manual_non_car_exclusions": sorted(REJECTED_NON_CARS),
                "items": manifest_items,
            }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            if accepted == per_color:
                break
        if accepted < per_color:
            raise RuntimeError(f"{color}: encontrei apenas {accepted}/{per_color} imagens de carro.")
    return manifest_items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True,
                        help="Pasta 'dataset' extraída do arquivo público")
    parser.add_argument("--model", default="yolo11n.pt", help="Pesos YOLO11n")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--per-color", type=int, default=30)
    args = parser.parse_args()
    items = prepare(args.source_dir, args.model, args.seed, args.per_color)
    print(f"Conjunto pronto: {len(items)} imagens em {OUTPUT_DIR}")
    print(f"Manifesto: {MANIFEST}")


if __name__ == "__main__":
    main()

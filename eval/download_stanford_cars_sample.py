"""Download a reproducible, label-balanced Stanford Cars test subset.

Images are fetched from the public Hugging Face Dataset Viewer rows API so this
script downloads only the selected image files, not the multi-gigabyte dataset.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATASET = "naufalso/stanford_cars"
SPLIT = "test"
ROWS_API = "https://datasets-server.huggingface.co/rows"
IMAGE_DIR = ROOT / "eval/images/stanford_cars"
MANIFEST = ROOT / "eval/datasets/stanford_cars_sample.json"
EXISTING_METADATA = ROOT / "eval/datasets/real_cars_metadata.json"
SEED = 20261022

MAKES = sorted([
    "AM General", "Aston Martin", "Land Rover", "Rolls-Royce", "Mercedes-Benz",
    "Acura", "Audi", "BMW", "Bentley", "Bugatti", "Buick", "Cadillac",
    "Chevrolet", "Chrysler", "Daewoo", "Dodge", "Eagle", "FIAT", "Ferrari",
    "Fisker", "Ford", "GMC", "Geo", "HUMMER", "Honda", "Hyundai", "Infiniti",
    "Isuzu", "Jaguar", "Jeep", "Lamborghini", "Lincoln", "Maybach", "Mazda",
    "McLaren", "Mitsubishi", "Nissan", "Plymouth", "Porsche", "Ram", "Scion",
    "smart", "Spyker", "Suzuki", "Tesla", "Toyota", "Volkswagen", "Volvo",
], key=len, reverse=True)
BODY_PATTERNS = [
    ("convertible", r"\bconvertible\b"),
    ("hatch", r"\bhatchback\b"),
    ("suv", r"\bsuv\b"),
    ("sedan", r"\bsedan\b"),
    ("coupe", r"\bcoupe\b"),
    ("wagon", r"\bwagon\b"),
    ("van", r"\bminivan\b|\bvan\b"),
    ("truck", r"\bpickup\b|\bcrew cab\b|\bclub cab\b|\bregular cab\b"),
]


def _get_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "SystemOneTesting/1.0"})
    with urllib.request.urlopen(request, timeout=45) as response:
        return json.loads(response.read().decode("utf-8"))


def _make_and_body(class_name: str) -> tuple[str, str, str]:
    for make in MAKES:
        if class_name.lower().startswith(make.lower() + " "):
            rest = class_name[len(make):].strip()
            slug = re.sub(r"[^a-z0-9]+", "_", make.lower()).strip("_")
            for body, pattern in BODY_PATTERNS:
                if re.search(pattern, rest, flags=re.IGNORECASE):
                    return slug, make, body
            return slug, make, "other"
    raise ValueError(f"Não consegui extrair a marca do rótulo: {class_name}")


def _canonical(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _fetch_rows(offset: int, length: int = 100) -> list[dict[str, Any]]:
    query = urllib.parse.urlencode({
        "dataset": DATASET,
        "config": "default",
        "split": SPLIT,
        "offset": offset,
        "length": length,
    })
    return _get_json(f"{ROWS_API}?{query}")["rows"]


def choose_rows(count: int, per_make: int, seed: int) -> list[dict[str, Any]]:
    with EXISTING_METADATA.open(encoding="utf-8") as handle:
        existing = {_canonical(item["full_name"]) for item in json.load(handle)}

    # A few metadata pages are enough for a diverse, reproducible sample; no
    # parquet shard or full Stanford Cars archive is downloaded.
    offsets = list(range(0, 1000, 100))
    pages: list[list[dict[str, Any]]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        pages = list(pool.map(_fetch_rows, offsets))
    candidates = [row for page in pages for row in page]
    random.Random(seed).shuffle(candidates)

    chosen: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    seen_classes: set[str] = set()
    for entry in candidates:
        row = entry["row"]
        class_name = row["car_name"]
        canon = _canonical(class_name)
        if canon in existing or canon in seen_classes:
            continue
        try:
            make_key, make, body_type = _make_and_body(class_name)
        except ValueError:
            continue
        if counts.get(make_key, 0) >= per_make:
            continue
        if not row.get("image_path", {}).get("src"):
            continue
        seen_classes.add(canon)
        counts[make_key] = counts.get(make_key, 0) + 1
        chosen.append({
            "row_idx": entry["row_idx"],
            "class_name": class_name,
            "make": make_key,
            "make_display": make,
            "body_type": body_type,
            "image_url": row["image_path"]["src"],
        })
        if len(chosen) == count:
            return chosen
    raise RuntimeError(
        f"Só encontrei {len(chosen)} imagens válidas; tente aumentar as páginas ou mudar --per-make."
    )


def _download_one(item: dict[str, Any]) -> dict[str, Any]:
    filename = f"stanford_test_{item['row_idx']:05d}.jpg"
    destination = IMAGE_DIR / filename
    if not destination.exists():
        request = urllib.request.Request(item["image_url"], headers={"User-Agent": "SystemOneTesting/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            image_bytes = response.read()
        from PIL import Image
        import io

        with Image.open(io.BytesIO(image_bytes)) as image:
            image.convert("RGB").save(destination, format="JPEG", quality=95)
    return {
        "filename": filename,
        "row_idx": item["row_idx"],
        "class_name": item["class_name"],
        "make": item["make"],
        "make_display": item["make_display"],
        "body_type": item["body_type"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--per-make", type=int, default=2)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    selected = choose_rows(args.count, args.per_make, args.seed)
    print(f"Baixando {len(selected)} imagens do split de teste Stanford Cars...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        items = list(pool.map(_download_one, selected))

    manifest = {
        "dataset": DATASET,
        "split": SPLIT,
        "seed": args.seed,
        "image_count": len(items),
        "source_url": f"https://huggingface.co/datasets/{DATASET}",
        "tasks_with_ground_truth": ["car_presence (all yes)", "make", "body_type"],
        "color_is_not_annotated": True,
        "items": items,
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    makers = sorted({item["make_display"] for item in items})
    print(f"Salvas em: {IMAGE_DIR}")
    print(f"Manifesto: {MANIFEST}")
    print(f"Marcas cobertas ({len(makers)}): {', '.join(makers)}")


if __name__ == "__main__":
    main()

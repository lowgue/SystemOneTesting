"""Download a reproducible, diverse Stanford Cars test subset.

Images are fetched from the public Hugging Face Dataset Viewer rows API. Only
the selected images are downloaded; the source dataset is not mirrored locally.
Pass ``--runner-ready`` to write a dataset for ``vision_bench`` under
``.artifacts/datasets``. The default output preserves the historical format.
"""

from __future__ import annotations

import argparse
import concurrent.futures
from functools import partial
import hashlib
import json
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DATASET = "naufalso/stanford_cars"
SPLIT = "test"
ROWS_API = "https://datasets-server.huggingface.co/rows"
IMAGE_DIR = ROOT / "benchmarks/datasets/vehicles/images/stanford_cars"
MANIFEST = ROOT / "benchmarks/datasets/historical/vehicles/stanford_cars_sample.json"
OUTPUT_DIR = ROOT / ".artifacts/datasets/stanford_cars_diverse"
EXISTING_METADATA = ROOT / "benchmarks/datasets/vehicles/real_cars_metadata.json"
SEED = 20261022
PAGE_SIZE = 100
MAX_WORKERS = 8
PAGE_WORKERS = 4
REQUEST_ATTEMPTS = 4
TASK_FIELDS = ["make", "body"]

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
    """Send a retryable JSON request and decode its response body."""
    request = urllib.request.Request(url, headers={"User-Agent": "SystemOneTesting/1.0"})
    for attempt in range(REQUEST_ATTEMPTS):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            if error.code not in {429, 502, 503, 504}:
                raise
            if attempt + 1 == REQUEST_ATTEMPTS:
                raise
            retry_after = error.headers.get("Retry-After")
            delay = int(retry_after) if retry_after and retry_after.isdigit() else 5 * 2**attempt
            time.sleep(min(delay, 60))
        except (urllib.error.URLError, TimeoutError):
            if attempt + 1 == REQUEST_ATTEMPTS:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("Dataset request exhausted its retries")


def _make_and_body(class_name: str) -> tuple[str, str, str]:
    """Extract vehicle make and body labels from a dataset example."""
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
    """Map a source label to its canonical dataset label."""
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _fetch_rows(offset: int, length: int = PAGE_SIZE) -> dict[str, Any]:
    """Read paginated dataset records from the configured Hub source."""
    query = urllib.parse.urlencode({
        "dataset": DATASET,
        "config": "default",
        "split": SPLIT,
        "offset": offset,
        "length": length,
    })
    return _get_json(f"{ROWS_API}?{query}")


def choose_rows(
    count: int,
    per_make: int,
    per_class: int,
    seed: int,
    scan_limit: int,
) -> list[dict[str, Any]]:
    """Select a reproducible, balanced sample of dataset rows."""
    if count < 1 or per_make < 1 or per_class < 1 or scan_limit < 1:
        raise ValueError("count, per-make, per-class and scan-limit must be positive")

    with EXISTING_METADATA.open(encoding="utf-8") as handle:
        existing = {_canonical(item["full_name"]) for item in json.load(handle)}

    first_page = _fetch_rows(0)
    total_rows = first_page["num_rows_total"]
    scan_end = min(total_rows, scan_limit)
    offsets = range(PAGE_SIZE, scan_end, PAGE_SIZE)
    with concurrent.futures.ThreadPoolExecutor(max_workers=PAGE_WORKERS) as pool:
        pages = list(pool.map(_fetch_rows, offsets))
    candidates = [*first_page["rows"], *(row for page in pages for row in page["rows"])]
    random.Random(seed).shuffle(candidates)

    chosen: list[dict[str, Any]] = []
    make_counts: dict[str, int] = {}
    class_counts: dict[str, int] = {}
    for entry in candidates:
        row = entry["row"]
        class_name = row["car_name"]
        canon = _canonical(class_name)
        if canon in existing:
            continue
        try:
            make_key, make, body_type = _make_and_body(class_name)
        except ValueError:
            continue
        if make_counts.get(make_key, 0) >= per_make:
            continue
        if class_counts.get(canon, 0) >= per_class:
            continue
        if not row.get("image_path", {}).get("src"):
            continue
        make_counts[make_key] = make_counts.get(make_key, 0) + 1
        class_counts[canon] = class_counts.get(canon, 0) + 1
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
        f"Only found {len(chosen)} valid images; lower --count or increase "
        "--scan-limit, --per-make or --per-class."
    )


def _download_one(item: dict[str, Any], image_dir: Path) -> dict[str, Any]:
    """Download one dataset image and return its metadata."""
    filename = f"stanford_test_{item['row_idx']:05d}.jpg"
    destination = image_dir / filename
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
        "sha256": _sha256(destination),
    }


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one downloaded image."""
    digest = hashlib.sha256()
    with path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _make_manifest(items: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    """Build the runner manifest with partial ground truth for vehicle tasks."""
    events = []
    for item in items:
        events.append({
            "id": item["filename"],
            "domain": "vehicle",
            "rel_path": f"images/{item['filename']}",
            "sha256": item["sha256"],
            "task_fields": TASK_FIELDS,
            "annotations": {
                "make": item["make"],
                "body": item["body_type"],
            },
        })
    return {
        "schema_version": 1,
        "name": "stanford-cars-diverse",
        "label_status": "exploratory_huggingface_labels",
        "source": DATASET,
        "split": SPLIT,
        "seed": seed,
        "image_count": len(events),
        "task_fields": TASK_FIELDS,
        "note": (
            "Labels are inherited from the source dataset. Make and body type "
            "are scored; color is not annotated."
        ),
        "items": events,
    }


def main() -> None:
    """Download selected images and write a validated-runner manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--per-make", type=int, default=2)
    parser.add_argument("--per-class", type=int, default=1)
    parser.add_argument("--scan-limit", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--runner-ready",
        action="store_true",
        help="Write images and a manifest for vision_bench under .artifacts.",
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or OUTPUT_DIR
    image_dir = output_dir / "images" if args.runner_ready else IMAGE_DIR
    manifest_path = output_dir / "manifest.json" if args.runner_ready else MANIFEST
    image_dir.mkdir(parents=True, exist_ok=True)

    selected = choose_rows(
        args.count,
        args.per_make,
        args.per_class,
        args.seed,
        args.scan_limit,
    )
    print(f"Downloading {len(selected)} images from Stanford Cars test split...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        download_one = partial(_download_one, image_dir=image_dir)
        items = list(pool.map(download_one, selected))

    if args.runner_ready:
        manifest = _make_manifest(items, args.seed)
    else:
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
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    makers = sorted({item["make_display"] for item in items})
    models = len({item["class_name"] for item in items})
    print(f"Images: {image_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Coverage: {len(makers)} makes and {models} car models")
    print(f"Makes covered: {', '.join(makers)}")


if __name__ == "__main__":
    main()

"""Convert historical task manifests into normalized dataset bundles."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_DATASETS = ROOT / "benchmarks/datasets/historical"
HISTORICAL_CONTRACT_PATH = ROOT / "benchmarks/contracts/historical_tasks_v1.json"
BASE_CONTRACT_PATH = ROOT / "benchmarks/contracts/description_ollama_v1.json"

DESTINATIONS = {
    "color-all": "vehicles/color-all.json",
    "color-dev": "vehicles/color-dev.json",
    "color-eval": "vehicles/color-eval.json",
    "vehicle-types": "vehicles/vehicle-types.json",
    "stanford": "vehicles/stanford.json",
    "people": "people/people.json",
    "robustness": "robustness/robustness.json",
    "multi-object": "multi_object/multi-object.json",
    "prompt-formats": "prompts/prompt-formats.json",
    "complete": "mixed/complete.json",
}

FIELD_BY_TASK = {
    "vehicle_type": "body",
    "color": "color",
    "helmet": "helmet",
    "carrying": "carrying",
    "clothing": "clothing",
}
LABEL_ALIASES = {"hatchback": "hatch"}
PRESERVED_ITEM_FIELDS = (
    "split",
    "stratum",
    "presentation",
    "difficult",
    "difficulty_tags",
    "source",
    "source_id",
    "scene_id",
    "annotation",
)
TASKS = (
    (1, "color", "color-all", "Cores — coleção completa (inclui desenvolvimento)"),
    (2, "body", "vehicle-types", "Tipos de veículo"),
    (3, "person", "people", "Pessoas: capacete, objeto e roupa"),
    (4, "robustness", "robustness", "Robustez: iluminação, desfoque e oclusão"),
    (5, "multi", "multi-object", "Múltiplos objetos: recorte e contexto"),
    (6, "prompts", "prompt-formats", "Comparação histórica de prompts: cores"),
    (7, "complete", "complete", "Descrição completa: veículos e pessoas"),
)


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from disk using UTF-8 encoding."""
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    """Write a consistently formatted JSON object and create its parent folder."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _build_historical_contract() -> dict[str, Any]:
    """Create the historical taxonomy contract without changing the current contract."""
    contract = _read_json(BASE_CONTRACT_PATH)
    contract["name"] = "historical-tasks-v1"
    contract["label_policy"] = (
        "Historical task taxonomy: hatchback canonicalized as hatch; pickup remains "
        "distinct from truck. New runs are not pooled with historical results."
    )
    body_choices = contract["domains"]["vehicle"]["questions"]["body"]["criteria"]
    body_choices.update(pickup="Pickup truck", bus="Bus", motorcycle="Motorcycle")
    contract["aliases"]["body"] = {
        "pickup": ["pickup", "pickup truck"],
        "truck": ["truck", "lorry"],
        "hatch": ["hatch", "hatchback"],
        "bus": ["bus"],
        "motorcycle": ["motorcycle", "motorbike"],
    }
    return contract


def _labels_for_row(
    row: dict[str, Any],
    kind: str,
) -> tuple[str, dict[str, str]]:
    """Select the domain and canonical labels represented by one source row."""
    task = row.get("task")
    domain = row.get(
        "domain",
        "person" if kind == "person" or task in {"helmet", "carrying", "clothing"} else "vehicle",
    )

    if kind == "color":
        return domain, {"color": row["color"]}
    if kind == "body":
        return domain, {"body": row["expected_type"]}
    if kind == "prompts":
        return domain, {"color": row["expected_color"]}
    if kind in {"robustness", "multi"}:
        field = FIELD_BY_TASK[task]
        return domain, {field: row.get("target_label", row.get("expected"))}
    if kind == "stanford":
        return domain, {"make": row["make"], "body": row["body_type"]}
    return domain, row["expected"]


def _transform_item(
    row: dict[str, Any],
    kind: str,
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Validate a historical annotation and normalize its image record."""
    domain, labels = _labels_for_row(row, kind)
    labels = {key: LABEL_ALIASES.get(value, value) for key, value in labels.items()}
    questions = contract["domains"][domain]["questions"]

    if "make" in labels and labels["make"] not in questions["make"]["criteria"]:
        labels["make"] = "other"
    for field, value in labels.items():
        if value not in questions[field]["criteria"]:
            raise ValueError((row.get("id"), field, value))

    relative_path = row.get("rel_path") or (
        "benchmarks/datasets/vehicles/images/stanford_cars/" + row["filename"]
    )
    image_path = ROOT / relative_path
    if not image_path.is_file():
        raise ValueError(f"Missing image: {relative_path}")

    item = {
        "id": row.get("id", image_path.name),
        "domain": domain,
        "filename": image_path.name,
        "rel_path": relative_path,
        "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
        "annotations": labels,
    }
    if set(labels) == set(questions):
        item["expected"] = labels
        item.pop("annotations")

    for field in PRESERVED_ITEM_FIELDS:
        if field in row:
            item[field] = row[field]
    item["group_id"] = row.get("scene_id", row.get("source_id", item["sha256"]))

    task = row.get("task")
    if task in FIELD_BY_TASK:
        item["task_field"] = FIELD_BY_TASK[task]
    if row.get("question_en"):
        item["question"] = row["question_en"]
    if kind == "person":
        item["task_fields"] = ["helmet", "carrying", "clothing"]
    return item


def _save_dataset(
    key: str,
    name: str,
    rows: Iterable[dict[str, Any]],
    source_manifest: Path,
    kind: str,
    contract: dict[str, Any],
) -> Path:
    """Validate source rows and write a versioned historical dataset document."""
    items = [_transform_item(row, kind, contract) for row in rows]
    source_relative_path = source_manifest.relative_to(ROOT).as_posix()
    source_digest = hashlib.sha256(source_manifest.read_bytes()).hexdigest()
    document = {
        "schema_version": 1,
        "name": name,
        "label_status": "exploratory_historical_unadjudicated",
        "source_manifest": source_relative_path,
        "source_manifest_sha256": source_digest,
        "required_contract": "historical",
        "note": (
            "Gabaritos históricos exploratórios. Imagens derivadas compartilham grupo; "
            "eventos não equivalem a imagens independentes. hatchback → hatch; "
            "pickup e truck preservados."
        ),
        "items": items,
    }
    destination = HISTORICAL_DATASETS / DESTINATIONS[key]
    _write_json(destination, document)
    return destination


def _manifest_for_task(number: int) -> Path:
    """Find the source manifest for one numbered historical task."""
    pattern = f"test_{number}_*manifest.json"
    matches = sorted(HISTORICAL_DATASETS.rglob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one source manifest for task {number}; found {len(matches)}"
        )
    return matches[0]


def _prepare_task_datasets(contract: dict[str, Any]) -> None:
    """Convert task manifests and split the vehicle-color set by evaluation use."""
    for number, kind, key, name in TASKS:
        manifest_path = _manifest_for_task(number)
        manifest = _read_json(manifest_path)
        rows = manifest.get("items", manifest.get("images", []))
        if manifest.get("question_en"):
            rows = [
                {**row, "question_en": row.get("question_en", manifest["question_en"])}
                for row in rows
            ]

        _save_dataset(key, name, rows, manifest_path, kind, contract)
        if number == 1:
            for split, title in (
                ("eval", "Cores — avaliação"),
                ("dev", "Cores — desenvolvimento"),
            ):
                split_rows = [row for row in rows if row["split"] == split]
                _save_dataset(
                    f"color-{split}",
                    title,
                    split_rows,
                    manifest_path,
                    kind,
                    contract,
                )


def _prepare_stanford_dataset(contract: dict[str, Any]) -> None:
    """Normalize the selected Stanford Cars sample as a historical dataset."""
    manifest_path = HISTORICAL_DATASETS / "vehicles/stanford_cars_sample.json"
    manifest = _read_json(manifest_path)
    _save_dataset(
        "stanford",
        "Stanford Cars: marca e carroceria",
        manifest["items"],
        manifest_path,
        "stanford",
        contract,
    )


def main() -> None:
    """Write the historical contract and all normalized historical datasets."""
    contract = _build_historical_contract()
    _write_json(HISTORICAL_CONTRACT_PATH, contract)
    _prepare_task_datasets(contract)
    _prepare_stanford_dataset(contract)


if __name__ == "__main__":
    main()

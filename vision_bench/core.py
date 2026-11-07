"""Shared input contract, strict parsing and verified dataset I/O."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "benchmarks/contracts/description_v1.json"


def sha256(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for a file on disk."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(obj: Any) -> str:
    """Return a stable SHA-256 digest for JSON-compatible configuration data."""
    serialized = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode()).hexdigest()


def load_json(path: str | Path) -> Any:
    """Read and decode one UTF-8 JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, obj: Any) -> None:
    """Write formatted JSON through a temporary file to avoid partial output."""
    path = Path(path)
    temporary_path = path.with_suffix(".tmp")
    contents = json.dumps(obj, ensure_ascii=False, indent=2) + "\n"
    temporary_path.write_text(contents, encoding="utf-8")
    temporary_path.replace(path)


def resolve_image(root: str | Path, rel: str) -> Path:
    """Resolve a dataset image path while keeping it inside the data root."""
    root = Path(root).resolve()
    path = (root / rel).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Image escapes data root: {rel}")
    return path


def _validate_annotations(item: dict[str, Any], domain: dict[str, Any]) -> None:
    """Validate optional partial annotations against a domain's allowed fields."""
    annotations = item.get("annotations")
    if annotations is not None:
        fields = domain["questions"]
        if (
            not isinstance(annotations, dict)
            or not annotations
            or not set(annotations) <= set(fields)
        ):
            raise ValueError(f"Invalid annotation fields: {item['id']}")
        if any(
            value not in fields[key]["criteria"]
            for key, value in annotations.items()
        ):
            raise ValueError(f"Invalid annotation label: {item['id']}")


def _validate_expected(item: dict[str, Any], domain: dict[str, Any]) -> None:
    """Validate a complete expected-answer map when the manifest provides one."""
    expected = item.get("expected")
    if expected is not None:
        questions = domain["questions"]
        if set(expected) != set(questions):
            raise ValueError(f"Incomplete expected fields: {item['id']}")
        if any(
            value not in questions[key]["criteria"]
            for key, value in expected.items()
        ):
            raise ValueError(f"Invalid expected label: {item['id']}")


def validate_manifest(
    path: str | Path,
    data_root: str | Path,
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Validate dataset IDs, domains, labels, image paths, and image hashes.

    The manifest is returned unchanged after validation. Invalid or incomplete
    image data raises ``ValueError`` before a benchmark starts.
    """
    document = load_json(path)
    items = document.get("items", [])
    if not items:
        raise ValueError("Manifest has no items")

    seen_ids = set()
    for item in items:
        if item["id"] in seen_ids:
            raise ValueError(f"Duplicate event ID: {item['id']}")
        seen_ids.add(item["id"])

        domain = contract["domains"].get(item["domain"])
        if domain is None:
            raise ValueError(f"Unsupported domain: {item['domain']}")

        image_path = resolve_image(data_root, item["rel_path"])
        if not image_path.is_file():
            raise ValueError(
                f"Missing image: {item['rel_path']}; import the dataset bundle "
                "or use benchmarks/datasets/vehicles/portable.json"
            )
        if not item.get("sha256") or sha256(image_path) != item["sha256"]:
            raise ValueError(f"Image checksum mismatch: {item['id']}")

        _validate_annotations(item, domain)
        _validate_expected(item, domain)

    return document


def normalize_text(
    contract: dict[str, Any],
    domain: str,
    key: str,
    raw: Any,
) -> str:
    """Map a free-text model answer to one label in the configured taxonomy.

    Ambiguous, empty, or out-of-taxonomy answers become ``parse_error``. The
    returned label is not evidence that the model's visual claim is correct.
    """
    text = re.sub(r"[^a-z0-9]+", " ", str(raw).lower()).strip()
    if not text:
        return "parse_error"

    if text in {
        "unknown",
        "not visible",
        "unclear",
        "cannot tell",
        "not identifiable",
        "unidentifiable",
    }:
        return contract["absence"][domain] if key != "presence" else "parse_error"

    if key == "presence":
        if text in {"yes", "no"}:
            return text
        if re.match(r"^(no|not|there is no|there are no)\b", text):
            return "no"
        if re.match(r"^(yes|there is a|there are)\b", text):
            return "yes"
        return "parse_error"

    if key == "helmet" and re.match(r"^(no|not)\b", text):
        return "no"
    if key == "carrying" and re.match(r"^(no|not carrying|nothing|none)\b", text):
        return "none"

    choices = contract["domains"][domain]["questions"][key]["criteria"]

    def clean(value: str) -> str:
        """Normalize a candidate label before comparing it with the response."""
        return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()

    if (
        key == "body"
        and "pickup" in choices
        and text in {"pickup", "pickup truck"}
    ):
        return "pickup"

    aliases = {
        choice: [clean(choice), clean(label)]
        for choice, label in choices.items()
        if choice not in {"unknown", "not_visible"}
    }
    aliases.update(contract.get("aliases", {}).get(key, {}))

    def has_term(term: str) -> bool:
        """Check whether a normalized label appears as a complete phrase."""
        return f" {clean(term)} " in f" {text} "

    if key == "helmet" and (has_term("hard hat") or has_term("hardhat")):
        return "yes"

    hits = {
        choice
        for choice, terms in aliases.items()
        if any(has_term(term) for term in terms)
    }
    return next(iter(hits)) if len(hits) == 1 else "parse_error"


def finalize(
    contract: dict[str, Any],
    domain: str,
    answers: dict[str, Any],
) -> dict[str, Any]:
    """Validate model choices and apply the contract's absence rule.

    The result keeps both raw normalized choices and post-absence choices so
    reports can explain how the final description was formed.
    """
    fields = contract["domains"][domain]["questions"]
    predicted = {key: str(answers.get(key, "parse_error")) for key in fields}
    before_absence_gate = dict(predicted)

    if predicted.get("presence") == "no":
        predicted.update(
            {
                key: contract["absence"][domain]
                for key in predicted
                if key != "presence"
            }
        )

    valid = all(
        value in fields[key]["criteria"] for key, value in predicted.items()
    )
    facts = (
        {
            key: value
            for key, value in predicted.items()
            if key != "presence"
            and value not in {"unknown", "not_visible", "parse_error"}
        }
        if predicted.get("presence") == "yes"
        else {}
    )

    labels = [f"{key}={value}" for key, value in facts.items()]
    description = (
        (
            "Veículo" if domain == "vehicle" else "Pessoa"
        )
        + (": " + ", ".join(labels) if labels else "")
        + "."
        if predicted.get("presence") == "yes"
        else None
    )
    return dict(
        predicted=predicted,
        before_absence_gate=before_absence_gate,
        valid=valid,
        description=description,
        facts=facts,
    )


def pack_dataset(
    manifest: str | Path,
    data_root: str | Path,
    output: str | Path,
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Create a verified ZIP containing a manifest and each unique image once."""
    document = validate_manifest(manifest, data_root, contract)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    transformed_items = []
    written_names = set()
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in document["items"]:
            transformed_item = dict(item)
            source_path = resolve_image(data_root, transformed_item["rel_path"])
            archive_name = (
                "images/"
                + transformed_item["sha256"]
                + source_path.suffix.lower()
            )
            transformed_item["rel_path"] = archive_name
            transformed_items.append(transformed_item)
            if archive_name not in written_names:
                archive.write(source_path, archive_name)
                written_names.add(archive_name)

        copied_document = {
            **document,
            "items": transformed_items,
            "source_manifest_sha256": sha256(manifest),
        }
        archive.writestr(
            "manifest.json",
            json.dumps(copied_document, ensure_ascii=False, indent=2) + "\n",
        )

    return dict(
        events=len(transformed_items),
        unique_images=len(written_names),
        sha256=sha256(output),
    )


def unpack_dataset(
    bundle: str | Path,
    destination: str | Path,
    contract: dict[str, Any],
) -> Path:
    """Verify a dataset bundle before extracting it into a new directory."""
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError("Destination exists; use a new directory")

    with zipfile.ZipFile(bundle) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or "manifest.json" not in names:
            raise ValueError("Invalid or duplicate archive entries")

        for member in archive.infolist():
            resolve_image(destination, member.filename)
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Symlinks are not allowed in dataset bundles")

        document = json.loads(archive.read("manifest.json"))
        required_names = {"manifest.json"} | {
            item["rel_path"] for item in document["items"]
        }
        if set(names) != required_names:
            raise ValueError("Archive contents differ from manifest")

        # Verify every image before writing anything.
        for item in document["items"]:
            image_hash = hashlib.sha256(archive.read(item["rel_path"])).hexdigest()
            if image_hash != item["sha256"]:
                raise ValueError("Dataset archive checksum mismatch")

        destination.mkdir(parents=True)
        for name in names:
            path = resolve_image(destination, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.read(name))

    validate_manifest(destination / "manifest.json", destination, contract)
    return destination / "manifest.json"

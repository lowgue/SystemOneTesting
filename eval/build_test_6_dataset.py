"""Build dataset for Test 6: Prompt and Output Format Comparison.

Selects 44 balanced vehicle images from Test 1 (4 per color across 11 colors).
Defines exact prompt formulations for:
  - Generative model (BLIP VQA):
      1. short_open: Short question without choices
      2. short_options: Question with short list of choices
      3. long_descriptions: Question with descriptive explanations of choices
      4. structured_json: Request for structured JSON output validated against a schema
  - Decision model (Laya Vision):
      1. native_short: Native choice criteria with standard labels
      2. native_descriptive: Native choice criteria with detailed visual definitions
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_1_MANIFEST = ROOT / "eval/datasets/test_1_vehicle_color_manifest.json"
MANIFEST_PATH = ROOT / "eval/datasets/test_6_prompt_formats_manifest.json"

TAXONOMY = [
    "black", "white", "gray_silver", "red", "blue", "green",
    "yellow", "orange", "brown", "purple", "pink"
]

# Prompts for Generative VLM (BLIP)
BLIP_PROMPTS = {
    "short_open": "What is the predominant color of this vehicle?",
    "short_options": "Is this vehicle black, white, gray, red, blue, green, yellow, orange, brown, purple, or pink?",
    "long_descriptions": (
        "Choose the predominant body color from the following options: "
        "black (deep dark tone), white (pure bright), gray or silver (metallic gray), "
        "red (crimson or cherry), blue (navy or light blue), green (forest or lime), "
        "yellow (bright yellow), orange (amber), brown (earth tone), purple (violet), or pink. "
        "What is the color?"
    ),
    "structured_json": (
        'Answer with a JSON object in the format {"color": "<color_name>", "confidence": "<high|medium|low>"}. '
        'What is the predominant vehicle color?'
    ),
}

# Criteria for Native Decision Model (Laya)
LAYA_CRITERIA_SHORT = {
    "black": "Black",
    "white": "White",
    "gray_silver": "Gray or silver",
    "red": "Red",
    "blue": "Blue",
    "green": "Green",
    "yellow": "Yellow",
    "orange": "Orange",
    "brown": "Brown",
    "purple": "Purple",
    "pink": "Pink",
    "unknown": "Color not identifiable / not visible",
}

LAYA_CRITERIA_DESCRIPTIVE = {
    "black": "Black or deep dark tone",
    "white": "White or off-white bright tone",
    "gray_silver": "Gray or silver metallic tone",
    "red": "Red, crimson, or cherry tone",
    "blue": "Blue, navy, or sky blue tone",
    "green": "Green, forest, or lime tone",
    "yellow": "Yellow or bright golden tone",
    "orange": "Orange or amber tone",
    "brown": "Brown, bronze, or earth tone",
    "purple": "Purple, violet, or magenta tone",
    "pink": "Pink or rose tone",
    "unknown": "Color not identifiable / not visible",
}


def main():
    with open(TEST_1_MANIFEST, encoding="utf-8") as f:
        m1 = json.load(f)

    # Pick 4 images per color from Test 1 eval set
    eval_items = [it for it in m1["items"] if it["split"] == "eval" and it["color"] in TAXONOMY]

    by_color: dict[str, list] = {}
    for it in eval_items:
        by_color.setdefault(it["color"], []).append(it)

    selected_images = []
    for c in TAXONOMY:
        items = by_color.get(c, [])
        for it in items[:4]:
            selected_images.append({
                "id": f"prompt_eval_{it['id']}",
                "source_id": it["id"],
                "rel_path": it["rel_path"],
                "expected_color": it["color"],
                "difficult": it.get("difficult", False),
            })

    manifest = {
        "benchmark": "Test 6: Prompt and Output Format Comparison",
        "description": "Compares prompt variations (short open, short options, long descriptions, structured JSON) for generative models vs native typed choice for decision models.",
        "image_count": len(selected_images),
        "taxonomy": TAXONOMY,
        "blip_prompts": BLIP_PROMPTS,
        "laya_criteria": {
            "native_short": LAYA_CRITERIA_SHORT,
            "native_descriptive": LAYA_CRITERIA_DESCRIPTIVE,
        },
        "images": selected_images,
    }

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Test 6 dataset manifest written to {MANIFEST_PATH}")
    print(f"Total evaluation images: {len(selected_images)} (4 per color across {len(TAXONOMY)} colors)")


if __name__ == "__main__":
    main()

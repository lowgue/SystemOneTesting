"""Ask Laya Vision typed questions about one image."""

import json
import sys

import laya
from PIL import Image


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python3 examples/laya_vision.py path/to/image.jpg")

    image_path = sys.argv[1]
    try:
        image = Image.open(image_path).convert("RGB")
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Could not open image {image_path}: {exc}") from exc

    # This is the visual Laya checkpoint, loaded directly from Hugging Face.
    agent = laya.load_vlm("thaitea/laya-vision")
    result = agent.predict(
        {"image": image},
        {
            "is_car": {
                "type": "noul",
                "instructions": "Is there at least one car visible in the image?",
            },
            "make": {
                "type": "choice",
                "instructions": "What make is the most prominent visible car? Choose unknown if it cannot be identified.",
                "criteria": {
                    "toyota": "Toyota",
                    "ford": "Ford",
                    "honda": "Honda",
                    "other": "A different identifiable make",
                    "unknown": "The make is not visually identifiable",
                },
            },
            "color": {
                "type": "choice",
                "instructions": "What is the apparent color of the most prominent visible car? Choose unknown if there is no car or its color cannot be identified.",
                "criteria": {
                    "black": "Black",
                    "white": "White",
                    "gray_silver": "Gray or silver",
                    "red": "Red",
                    "blue": "Blue",
                    "other": "Another identifiable color",
                    "unknown": "The color cannot be identified",
                },
            },
        },
    )
    print(json.dumps(result["answers"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

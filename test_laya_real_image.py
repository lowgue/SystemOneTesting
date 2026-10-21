"""Run a single Laya Vision image decision locally."""

import argparse
import json
import time

from PIL import Image


QUESTIONS = {
    "tem_carro": {
        "type": "noul",
        "instructions": "Is there a car in this image?",
    },
    "marca": {
        "type": "choice",
        "instructions": "What is the make of the most prominent car?",
        "criteria": {
            "audi": "Audi",
            "bmw": "BMW",
            "ford": "Ford",
            "toyota": "Toyota",
            "other": "Another make",
        },
    },
}


def apply_transformers_compat_patch() -> None:
    """Handle the list-valued image feature output in newer Transformers."""
    import torch
    import laya.vlm as vlm

    def encode_images_compat(self, pixel_values, pixel_attention_mask=None):
        mask = pixel_attention_mask[None] if pixel_attention_mask is not None else None
        result = self._image_features(pixel_values[None], mask)
        return torch.stack(result) if isinstance(result, list) else result

    vlm.VLMDecisionModel.encode_images = encode_images_compat


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", default="eval/images/cars/real_audi_1.jpg")
    args = parser.parse_args()

    apply_transformers_compat_patch()
    import laya

    print("Carregando thaitea/laya-vision...")
    agent = laya.load_vlm("thaitea/laya-vision")
    with Image.open(args.image) as source:
        image = source.convert("RGB")

    print(f"Modelo: Laya Vision | Imagem: {args.image}")
    start = time.perf_counter()
    result = agent.predict(image, QUESTIONS)
    print(f"Latência: {(time.perf_counter() - start) * 1000:.0f} ms")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

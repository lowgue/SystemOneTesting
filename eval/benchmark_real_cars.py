"""Benchmark Laya Vision on real photos from the Stanford Cars dataset."""

import json
import time
from pathlib import Path
from PIL import Image

def apply_transformers_compat_patch():
    import torch
    import laya.vlm as _vlm

    def _encode_images_compat(self, pixel_values, pixel_attention_mask=None):
        pam = pixel_attention_mask[None] if pixel_attention_mask is not None else None
        result = self._image_features(pixel_values[None], pam)
        if isinstance(result, list):
            result = torch.stack(result)
        return result

    _vlm.VLMDecisionModel.encode_images = _encode_images_compat


def main():
    metadata_path = Path("eval/datasets/real_cars_metadata.json")
    if not metadata_path.exists():
        print("Execute primeiro eval/extract_real_cars.py")
        return

    with open(metadata_path, encoding="utf-8") as f:
        cars = json.load(f)

    apply_transformers_compat_patch()
    import laya
    print("Carregando thaitea/laya-vision...")
    agent = laya.load_vlm("thaitea/laya-vision")

    questions = {
        "tem_carro": {
            "type": "noul",
            "instructions": "Is there at least one car (automobile) clearly visible in this image?",
        },
        "marca": {
            "type": "choice",
            "instructions": "What make is the most prominent car visible?",
            "criteria": {
                "toyota": "Toyota",
                "ford": "Ford",
                "honda": "Honda",
                "chevrolet": "Chevrolet / GM",
                "volkswagen": "Volkswagen",
                "bmw": "BMW",
                "audi": "Audi",
                "dodge": "Dodge",
                "jeep": "Jeep",
                "hyundai": "Hyundai",
                "other": "A different identifiable make",
                "unknown": "No car present or make not identifiable",
            },
        },
        "tipo": {
            "type": "choice",
            "instructions": "What body style is the most prominent vehicle?",
            "criteria": {
                "sedan": "Sedan (4-door car with trunk)",
                "suv": "SUV or crossover",
                "hatch": "Hatchback",
                "truck": "Pickup truck or cab",
                "coupe": "Coupe or sports car",
                "convertible": "Convertible",
                "van": "Van or minivan",
                "other": "Other body style",
            },
        },
    }

    print(f"\nIniciando benchmark em {len(cars)} fotos reais de carros...\n")
    correct_presence = 0
    correct_make = 0
    correct_body = 0
    total = len(cars)
    latencies = []

    for i, car in enumerate(cars, 1):
        img_path = Path("eval/images/cars") / car["filename"]
        img = Image.open(img_path).convert("RGB")

        t0 = time.perf_counter()
        res = agent.predict(img, questions)
        lat = round((time.perf_counter() - t0) * 1000, 1)
        latencies.append(lat)

        ans = res.get("answers", {})
        has_car_p = ans.get("tem_carro", {}).get("noul", 0.0)
        pred_make = ans.get("marca", {}).get("choice", "")
        pred_body = ans.get("tipo", {}).get("choice", "")

        exp_make = car["make"]
        exp_body = car["body_type"]

        ok_presence = has_car_p >= 0.5
        ok_make = pred_make == exp_make
        ok_body = pred_body == exp_body

        if ok_presence:
            correct_presence += 1
        if ok_make:
            correct_make += 1
        if ok_body:
            correct_body += 1

        print(f"[{i:02d}/{total}] {car['full_name']:35s} | Presença: {'✓' if ok_presence else '✗'} ({has_car_p*100:.0f}%) | Marca: {pred_make:10s} (esp: {exp_make:10s}) {'✓' if ok_make else '✗'} | Tipo: {pred_body:10s} ({lat}ms)")

    mean_lat = sum(latencies) / len(latencies)
    print("\n" + "=" * 65)
    print("  RESULTADOS NO DATASET DE CARROS REAIS (Stanford Cars)")
    print("=" * 65)
    print(f"  Detecção de Presença (noul) : {correct_presence}/{total} ({correct_presence/total*100:.1f}%)")
    print(f"  Classificação de Marca      : {correct_make}/{total} ({correct_make/total*100:.1f}%)")
    print(f"  Estilo de Carroceria        : {correct_body}/{total} ({correct_body/total*100:.1f}%)")
    print(f"  Latência Média              : {mean_lat:.0f} ms/imagem (CPU)")
    print("=" * 65)


if __name__ == "__main__":
    main()

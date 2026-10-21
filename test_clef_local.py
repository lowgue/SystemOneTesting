"""Run Clef-Flash Q4_K_M through the local llama-server System One API."""

import argparse
import base64
import io
import json
import os
import time
import urllib.error
import urllib.request

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", default="eval/images/cars/real_audi_1.jpg")
    parser.add_argument(
        "--url",
        default=os.environ.get(
            "CLEF_LOCAL_SYSTEMONE_URL", "http://127.0.0.1:11437/v1/systemone"
        ),
    )
    args = parser.parse_args()

    with Image.open(args.image) as image:
        buffer = io.BytesIO()
        image_rgb = image.convert("RGB")
        image_rgb.thumbnail((2048, 2048))
        image_rgb.save(buffer, format="JPEG", quality=85, optimize=True)
    image_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    payload = {
        "state": "Classifique este veículo.",
        "images": [f"data:image/jpeg;base64,{image_b64}"],
        "questions": QUESTIONS,
    }
    request = urllib.request.Request(
        args.url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    print(f"Modelo: Clef-Flash Q4_K_M (local) | Imagem: {args.image}")
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"API retornou HTTP {exc.code}: {exc.read().decode('utf-8')}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"Não foi possível acessar {args.url}: {exc.reason}") from exc

    print(f"Latência: {(time.perf_counter() - start) * 1000:.0f} ms")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

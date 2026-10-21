"""Run a single Clef-Flash image decision through Cloudflare Workers AI."""

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
    args = parser.parse_args()
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    api_token = os.environ.get("CLOUDFLARE_API_TOKEN")
    if not account_id or not api_token:
        raise SystemExit(
            "Defina CLOUDFLARE_ACCOUNT_ID e CLOUDFLARE_API_TOKEN no ambiente."
        )

    with Image.open(args.image) as image:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=92)
    image_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    payload = {
        "model": "clef-flash",
        "state": "Classifique este veículo.",
        "images": [f"data:image/jpeg;base64,{image_b64}"],
        "questions": QUESTIONS,
    }
    request = urllib.request.Request(
        f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/@cf/cloudflare/clef-flash",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    print(f"Modelo: Clef-Flash (Cloudflare Workers AI) | Imagem: {args.image}")
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"API retornou HTTP {exc.code}: {exc.read().decode('utf-8')}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"Não foi possível acessar Cloudflare Workers AI: {exc.reason}") from exc

    print(f"Latência: {(time.perf_counter() - start) * 1000:.0f} ms")
    result = response_data.get("result", response_data)
    if response_data.get("success") is False:
        raise SystemExit(json.dumps(response_data.get("errors", response_data), ensure_ascii=False))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

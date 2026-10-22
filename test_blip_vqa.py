"""Ask Salesforce BLIP VQA one question about a local image."""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "eval"))

from PIL import Image

from compare_laya_blip import load_blip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", default=str(ROOT / "eval/images/cars/real_audi_1.jpg"))
    parser.add_argument("--question", default="What color is the car?")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    import torch
    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA foi solicitado, mas não está disponível.")

    print("Carregando Salesforce/blip-vqa-base...")
    processor, model = load_blip(args.device)
    with Image.open(args.image) as source:
        image = source.convert("RGB")
    inputs = processor(image, args.question, return_tensors="pt").to(args.device)

    start = time.perf_counter()
    with torch.inference_mode():
        tokens = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    elapsed_ms = (time.perf_counter() - start) * 1000
    answer = processor.decode(tokens[0], skip_special_tokens=True).strip()

    print(f"Imagem: {args.image}")
    print(f"Pergunta: {args.question}")
    print(f"Resposta: {answer}")
    print(f"Latência de inferência: {elapsed_ms:.1f} ms ({args.device})")


if __name__ == "__main__":
    main()

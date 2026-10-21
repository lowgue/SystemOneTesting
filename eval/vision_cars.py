"""Evaluation harness for Laya Vision on car images.

Loads thaitea/laya-vision from HuggingFace, runs typed questions
(noul / choice) over a set of annotated images and reports accuracy,
confidence and latency.

Usage:
    python eval/vision_cars.py [--images eval/images/cars] [--out eval/results/]
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

IMAGE_QUESTIONS = {
    "tem_carro": {
        "type": "noul",
        "instructions": "Is there at least one car (automobile) clearly visible in this image?",
    },
    "marca": {
        "type": "choice",
        "instructions": (
            "What make is the most prominent car visible? "
            "Choose 'unknown' if no car is present or the make cannot be identified."
        ),
        "criteria": {
            "toyota": "Toyota",
            "ford": "Ford",
            "honda": "Honda",
            "chevrolet": "Chevrolet / GM",
            "volkswagen": "Volkswagen",
            "other": "A different identifiable make",
            "unknown": "No car present or make not identifiable",
        },
    },
    "cor": {
        "type": "choice",
        "instructions": (
            "What is the predominant color of the most prominent car? "
            "Choose 'unknown' if no car is present or the color cannot be identified."
        ),
        "criteria": {
            "black": "Black",
            "white": "White",
            "gray_silver": "Gray or silver",
            "red": "Red",
            "blue": "Blue",
            "green": "Green",
            "yellow": "Yellow",
            "other": "Another identifiable color",
            "unknown": "No car or color not identifiable",
        },
    },
    "tipo": {
        "type": "choice",
        "instructions": "What body style is the most prominent vehicle?",
        "criteria": {
            "sedan": "Sedan (4-door car with trunk)",
            "suv": "SUV or crossover",
            "hatch": "Hatchback",
            "truck": "Pickup truck",
            "van": "Van or minivan",
            "other": "Other body style",
            "unknown": "No car or body style not identifiable",
        },
    },
}

# Ground truth for each image filename
ANNOTATIONS: dict[str, dict[str, Any]] = {
    "toyota_red.jpg":     {"tem_carro": True,  "marca": "toyota",  "cor": "red",        "tipo": "sedan"},
    "ford_blue.jpg":      {"tem_carro": True,  "marca": "ford",    "cor": "blue",       "tipo": "hatch"},
    "honda_white.jpg":    {"tem_carro": True,  "marca": "honda",   "cor": "white",      "tipo": "sedan"},
    "gray_car.jpg":       {"tem_carro": True,  "marca": "unknown", "cor": "gray_silver","tipo": "sedan"},
    "black_car.jpg":      {"tem_carro": True,  "marca": "unknown", "cor": "black",      "tipo": "suv"},
    "no_car_landscape.jpg": {"tem_carro": False, "marca": "unknown","cor": "unknown",   "tipo": "unknown"},
    "no_car_field.jpg":   {"tem_carro": False, "marca": "unknown", "cor": "unknown",    "tipo": "unknown"},
}


@dataclass
class QuestionResult:
    question: str
    qtype: str
    prediction: str
    expected: str | None
    correct: bool | None
    confidence: float | None
    p_correct: float | None  # for noul only
    probabilities: dict | None


@dataclass
class ImageResult:
    filename: str
    latency_ms: float
    questions: list[QuestionResult] = field(default_factory=list)
    error: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_prediction(ans: dict) -> str:
    t = ans.get("type")
    if t == "noul":
        return "true" if ans["noul"] >= 0.5 else "false"
    if t == "choice":
        return str(ans.get("choice", "unknown"))
    return ""


def is_correct(pred: str, expected: Any, qtype: str) -> bool:
    if qtype == "noul":
        pred_pos = pred == "true"
        exp_pos = bool(expected) if not isinstance(expected, str) else expected.lower() in {"true","yes","1"}
        return pred_pos == exp_pos
    return pred.lower() == str(expected).lower()


def p_correct_noul(ans: dict, expected: Any) -> float:
    exp_pos = bool(expected) if not isinstance(expected, str) else expected.lower() in {"true","yes","1"}
    p = ans["noul"]
    return p if exp_pos else 1.0 - p


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

def evaluate_image(agent, img_path: Path, annotations: dict | None) -> ImageResult:
    from PIL import Image as PILImage

    try:
        img = PILImage.open(img_path).convert("RGB")
    except Exception as exc:
        return ImageResult(filename=img_path.name, latency_ms=0, error=str(exc))

    t0 = time.perf_counter()
    try:
        result = agent.predict({"image": img}, IMAGE_QUESTIONS)
    except Exception as exc:
        return ImageResult(filename=img_path.name, latency_ms=0, error=str(exc))
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    answers = result.get("answers", {})
    q_results = []
    for qkey, ans in answers.items():
        qtype = ans.get("type", "")
        pred = extract_prediction(ans)
        exp = annotations.get(qkey) if annotations else None
        correct = is_correct(pred, exp, qtype) if exp is not None else None

        p_c = None
        if qtype == "noul" and exp is not None:
            p_c = round(p_correct_noul(ans, exp), 4)

        probs = ans.get("probabilities")
        conf = ans.get("confidence")

        q_results.append(QuestionResult(
            question=qkey, qtype=qtype, prediction=pred,
            expected=str(exp) if exp is not None else None,
            correct=correct, confidence=conf,
            p_correct=p_c, probabilities=probs,
        ))

    return ImageResult(filename=img_path.name, latency_ms=latency_ms, questions=q_results)


def print_report(results: list[ImageResult]) -> dict:
    total_q, correct_q = 0, 0
    latencies = []
    p_corrects = []

    print(f"\n{'═'*65}")
    print("  AVALIAÇÃO VISUAL – Laya Vision (carros)")
    print(f"{'═'*65}")

    for r in results:
        print(f"\n  📷 {r.filename}  (latência: {r.latency_ms} ms)")
        if r.error:
            print(f"    ERRO: {r.error}")
            continue
        latencies.append(r.latency_ms)
        for q in r.questions:
            status = "✓" if q.correct is True else ("✗" if q.correct is False else "·")
            pc_str = f"  p={q.p_correct:.3f}" if q.p_correct is not None else ""
            conf_str = f"  conf={q.confidence:.2f}" if q.confidence is not None else ""
            print(f"    {status} {q.question:12s}: {q.prediction:15s} (esperado: {q.expected}){pc_str}{conf_str}")
            if q.probabilities:
                top = sorted(q.probabilities.items(), key=lambda x: -x[1])[:3]
                top_str = "  |  ".join(f"{k}={v:.3f}" for k,v in top)
                print(f"      top-3: {top_str}")
            if q.correct is not None:
                total_q += 1
                if q.correct:
                    correct_q += 1
            if q.p_correct is not None:
                p_corrects.append(q.p_correct)

    accuracy = correct_q / total_q if total_q else None
    mean_lat = sum(latencies) / len(latencies) if latencies else None
    mean_pc  = sum(p_corrects) / len(p_corrects) if p_corrects else None

    print(f"\n{'─'*65}")
    print(f"  Imagens avaliadas  : {len(results)}")
    print(f"  Questões avaliadas : {total_q}")
    print(f"  Corretas           : {correct_q}")
    if accuracy is not None:
        print(f"  Acurácia global    : {accuracy*100:.1f}%")
    if mean_pc is not None:
        print(f"  Média p(correto)   : {mean_pc:.4f}")
    if mean_lat:
        print(f"  Latência média     : {mean_lat:.0f} ms/imagem")
    print()

    return {
        "total_images": len(results),
        "total_questions": total_q,
        "correct": correct_q,
        "accuracy": round(accuracy, 4) if accuracy else None,
        "mean_p_correct": round(mean_pc, 4) if mean_pc else None,
        "mean_latency_ms": round(mean_lat, 1) if mean_lat else None,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _apply_transformers_compat_patch() -> None:
    """Monkey-patch VLMDecisionModel.encode_images for transformers >=5.x.

    Newer versions of transformers return a list from get_image_features instead
    of a single tensor. This patch stacks the list back into a tensor so the rest
    of VLMAgent.predict works unchanged.
    """
    import torch as _torch
    import laya.vlm as _vlm

    _orig = _vlm.VLMDecisionModel.encode_images

    def _encode_images_compat(self, pixel_values, pixel_attention_mask=None):
        pam = pixel_attention_mask[None] if pixel_attention_mask is not None else None
        result = self._image_features(pixel_values[None], pam)
        if isinstance(result, list):
            result = _torch.stack(result)
        return result

    _vlm.VLMDecisionModel.encode_images = _encode_images_compat


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Laya Vision on car images")
    parser.add_argument("--images", type=Path, default=Path("eval/images/cars"))
    parser.add_argument("--out", type=Path, default=Path("eval/results"))
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    if not args.images.is_dir():
        print(f"Pasta de imagens não encontrada: {args.images}", file=sys.stderr)
        sys.exit(1)

    print("Carregando thaitea/laya-vision (pode levar 1-2 min na 1ª vez)...")
    t_load = time.perf_counter()
    import laya
    _apply_transformers_compat_patch()
    agent = laya.load_vlm("thaitea/laya-vision")
    print(f"Modelo carregado em {(time.perf_counter()-t_load):.1f}s\n")

    image_paths = sorted(p for p in args.images.iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png",".webp"})
    print(f"Encontradas {len(image_paths)} imagens em '{args.images}'\n")

    all_results = []
    for i, img_path in enumerate(image_paths, 1):
        ann = ANNOTATIONS.get(img_path.name)
        print(f"[{i}/{len(image_paths)}] {img_path.name} ...", end=" ", flush=True)
        r = evaluate_image(agent, img_path, ann)
        all_results.append(r)
        if r.error:
            print(f"ERRO: {r.error}")
        else:
            n_ok = sum(1 for q in r.questions if q.correct is True)
            n_ev = sum(1 for q in r.questions if q.correct is not None)
            print(f"{n_ok}/{n_ev} corretas  {r.latency_ms} ms")

    summary = print_report(all_results)

    if not args.no_save:
        args.out.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_file = args.out / f"vision-cars-{ts}.json"
        payload = {
            "run": f"vision-cars-{ts}",
            "model": "thaitea/laya-vision",
            "timestamp": ts,
            "summary": summary,
            "results": [asdict(r) for r in all_results],
        }
        out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        print(f"Resultados salvos em: {out_file}")


if __name__ == "__main__":
    main()

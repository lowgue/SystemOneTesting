"""Web server providing a browser interface for Clef-Flash System One decisions.

Usage:
    python web/server.py [--port 7860]
"""

import argparse
import base64
import io
import json
import mimetypes
import os
import sys
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "vendor" / "laya-vision"))

CARS_DIR = PROJECT_ROOT / "eval" / "images" / "cars"
WEB_DIR = PROJECT_ROOT / "web"

# Keep the browser request within the tested CPU inference budget.
DEFAULT_QUESTIONS = {
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
            "bmw": "BMW",
            "audi": "Audi",
            "dodge": "Dodge",
            "jeep": "Jeep",
            "hyundai": "Hyundai",
            "other": "A different identifiable make",
            "unknown": "No car present or make not identifiable",
        },
    },
}

PRESET_NAMES = {
    "no_car_landscape.jpg": "Paisagem (Sem Carro)",
    "no_car_field.jpg": "Campo (Sem Carro)",
}

METADATA_FILE = PROJECT_ROOT / "eval" / "datasets" / "real_cars_metadata.json"
if METADATA_FILE.exists():
    try:
        with open(METADATA_FILE, encoding="utf-8") as f:
            for item in json.load(f):
                PRESET_NAMES[item["filename"]] = f"📷 {item['full_name']}"
    except Exception:
        pass

CF_ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
CF_API_TOKEN = os.environ.get("CLOUDFLARE_API_TOKEN", "")
CF_CLEF_URL = (
    f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}"
    "/ai/run/@cf/cloudflare/clef-flash"
)
LOCAL_CLEF_URL = os.environ.get(
    "CLEF_LOCAL_SYSTEMONE_URL", "http://127.0.0.1:11437/v1/systemone"
)
AGENT = None
INFERENCE_LOCK = threading.Lock()


def call_cloudflare_clef(image: Image.Image, questions: dict[str, Any]) -> dict[str, Any]:
    if not CF_ACCOUNT_ID or not CF_API_TOKEN:
        raise RuntimeError(
            "Configure CLOUDFLARE_ACCOUNT_ID e CLOUDFLARE_API_TOKEN para usar Clef-Flash na Cloudflare."
        )

    image_buffer = io.BytesIO()
    image = image.copy()
    image.thumbnail((2048, 2048))
    image.save(image_buffer, format="JPEG", quality=85, optimize=True)
    image_data = base64.b64encode(image_buffer.getvalue()).decode("ascii")
    payload = json.dumps({
        "model": "clef-flash",
        "state": "Classifique os elementos visíveis nesta imagem.",
        "images": [f"data:image/jpeg;base64,{image_data}"],
        "questions": questions,
    }).encode("utf-8")
    request = urllib.request.Request(
        CF_CLEF_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {CF_API_TOKEN}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        result = json.loads(response.read().decode("utf-8"))
    if result.get("success") is False:
        errors = result.get("errors") or [{"message": "Cloudflare Workers AI retornou um erro."}]
        raise RuntimeError(errors[0].get("message", str(errors[0])))
    return result.get("result", result)


def call_local_clef(image: Image.Image, questions: dict[str, Any]) -> dict[str, Any]:
    image_buffer = io.BytesIO()
    image = image.copy()
    image.thumbnail((2048, 2048))
    image.save(image_buffer, format="JPEG", quality=85, optimize=True)
    image_data = base64.b64encode(image_buffer.getvalue()).decode("ascii")
    payload = json.dumps({
        "state": "Classifique os elementos visíveis nesta imagem.",
        "images": [f"data:image/jpeg;base64,{image_data}"],
        "questions": questions,
    }).encode("utf-8")
    request = urllib.request.Request(
        LOCAL_CLEF_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.loads(response.read().decode("utf-8"))
MODEL_BACKEND = "laya"


def apply_transformers_compat_patch():
    """Compatibility fix for transformers >= 5.x pooler list output."""
    import torch
    import laya.vlm as _vlm

    def _encode_images_compat(self, pixel_values, pixel_attention_mask=None):
        pam = pixel_attention_mask[None] if pixel_attention_mask is not None else None
        result = self._image_features(pixel_values[None], pam)
        if isinstance(result, list):
            result = torch.stack(result)
        return result

    _vlm.VLMDecisionModel.encode_images = _encode_images_compat


class DecisionServerHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Concise logging
        sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {args[0]} {args[1]}\n")

    def send_json(self, data: dict, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        url_path = self.path.split("?")[0]

        # 1. Main frontend page
        if url_path in ("/", "/index.html"):
            index_path = WEB_DIR / "index.html"
            if not index_path.exists():
                self.send_error(404, "index.html not found")
                return
            content = index_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return

        # 2. List presets
        if url_path == "/api/status":
            names = {
                "laya": "Laya Vision (rápido)",
                "clef": "Clef-Flash (Cloudflare)",
                "clef-local": "Clef-Flash local (Q4)",
            }
            display_name = names[MODEL_BACKEND]
            self.send_json({"model": MODEL_BACKEND, "display_name": display_name})
            return

        if url_path == "/api/presets":
            presets = []
            if CARS_DIR.exists():
                # Put real cars first, then landscapes
                files = sorted(CARS_DIR.iterdir(), key=lambda p: (not p.name.startswith("real_"), p.name))
                for p in files:
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                        title = PRESET_NAMES.get(p.name, p.stem.replace("_", " ").title())
                        presets.append({
                            "filename": p.name,
                            "title": title,
                            "url": f"/api/image/{p.name}"
                        })
            self.send_json(presets)
            return

        # 3. Serve preset image
        if url_path.startswith("/api/image/"):
            img_name = os.path.basename(url_path)
            img_path = CARS_DIR / img_name
            if not img_path.exists():
                self.send_error(404, "Imagem não encontrada")
                return

            mime_type, _ = mimetypes.guess_type(str(img_path))
            mime_type = mime_type or "image/jpeg"
            content = img_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mime_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "public, max-age=3600")
            self.end_headers()
            self.wfile.write(content)
            return

        self.send_error(404, "Not Found")

    def do_POST(self):
        url_path = self.path.split("?")[0]

        if url_path == "/api/predict":
            content_len = int(self.headers.get("Content-Length", 0))
            if content_len == 0:
                self.send_json({"error": "Corpo da requisição vazio"}, status=400)
                return

            try:
                body = json.loads(self.rfile.read(content_len).decode("utf-8"))
            except Exception as e:
                self.send_json({"error": f"JSON inválido: {e}"}, status=400)
                return

            # Determine image input
            pil_img = None
            if "image_data" in body and body["image_data"]:
                data_uri = body["image_data"]
                if "," in data_uri:
                    header, b64_data = data_uri.split(",", 1)
                else:
                    b64_data = data_uri
                try:
                    raw_bytes = base64.b64decode(b64_data)
                    pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                except Exception as e:
                    self.send_json({"error": f"Falha ao decodificar imagem base64: {e}"}, status=400)
                    return
            elif "filename" in body and body["filename"]:
                safe_name = os.path.basename(body["filename"])
                target_path = CARS_DIR / safe_name
                if not target_path.exists():
                    self.send_json({"error": f"Arquivo {safe_name} não encontrado no servidor"}, status=404)
                    return
                try:
                    pil_img = Image.open(target_path).convert("RGB")
                except Exception as e:
                    self.send_json({"error": f"Falha ao abrir imagem {safe_name}: {e}"}, status=400)
                    return
            else:
                self.send_json({"error": "Envie 'filename' ou 'image_data'"}, status=400)
                return

            questions = body.get("questions") or DEFAULT_QUESTIONS

            if not INFERENCE_LOCK.acquire(blocking=False):
                self.send_json({"error": "Uma avaliação já está em andamento. Aguarde e tente novamente."}, status=503)
                return

            t0 = time.perf_counter()
            try:
                model = MODEL_BACKEND
                if model == "laya":
                    if AGENT is None:
                        self.send_json({"error": "Laya Vision ainda está carregando."}, status=503)
                        return
                    res = AGENT.predict(pil_img, questions)
                elif model == "clef":
                    res = call_cloudflare_clef(pil_img, questions)
                elif model == "clef-local":
                    res = call_local_clef(pil_img, questions)
                else:
                    self.send_json({"error": "Modelo desconhecido."}, status=400)
                    return
                latency_ms = round((time.perf_counter() - t0) * 1000, 1)

                output = {
                    "model": {
                        "laya": "Laya Vision",
                        "clef": "Clef-Flash (Cloudflare Workers AI)",
                        "clef-local": "Clef-Flash local Q4_K_M",
                    }[model],
                    "latency_ms": latency_ms,
                    "answers": res.get("answers", {}),
                    "usage": res.get("usage", {}),
                    "provenance": res.get("provenance", {}),
                }
                self.send_json(output)
            except Exception as exc:
                sys.stderr.write(f"Erro de inferência: {exc}\n")
                self.send_json({"error": str(exc)}, status=500)
            finally:
                INFERENCE_LOCK.release()
            return

        self.send_error(404, "Not Found")


def run_server(host: str = "0.0.0.0", port: int = 7860, model: str = "laya"):
    global AGENT, MODEL_BACKEND
    MODEL_BACKEND = model
    print(f"═"*60)
    print("  Inicializando Vision Playground")
    print(f"═"*60)
    if model == "laya":
        apply_transformers_compat_patch()
        import laya
        print("Carregando Laya Vision para inferência rápida...")
        AGENT = laya.load_vlm("thaitea/laya-vision")
        print("Laya Vision pronto.")
    else:
        if model == "clef":
            if not CF_ACCOUNT_ID or not CF_API_TOKEN:
                print("AVISO: configure CLOUDFLARE_ACCOUNT_ID e CLOUDFLARE_API_TOKEN antes de avaliar.")
            print("Usando Clef-Flash hospedado no Cloudflare Workers AI.")
        else:
            print(f"Usando Clef-Flash local em {LOCAL_CLEF_URL}.")

    server_address = (host, port)
    httpd = ThreadingHTTPServer(server_address, DecisionServerHandler)
    print(f"🚀 Servidor ativo e pronto para uso!")
    print(f"👉 Abra no navegador: http://localhost:{port}")
    print(f"═"*60)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor encerrado pelo usuário.")
        httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--model", choices=("laya", "clef", "clef-local"), default="laya")
    args = parser.parse_args()
    run_server(host=args.host, port=args.port, model=args.model)

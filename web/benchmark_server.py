"""Local live benchmark dashboard; starts comparing Laya and BLIP on launch."""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
from compare_laya_blip import load_examples

PAGE = Path(__file__).resolve().parent / "benchmark.html"
PROGRESS_FILE = ROOT / "eval/results/benchmark-live-progress.json"
RESULT_FILE = ROOT / "eval/results/laya-blip-car-color.json"
LOG_FILE = ROOT / "eval/results/benchmark-live.log"
OLD_IMAGES = ROOT / "eval/images/cars"
STANFORD_IMAGES = ROOT / "eval/images/stanford_cars"
COLOR_IMAGES = ROOT / "eval/images/color_test"
MANIFEST = ROOT / "eval/datasets/stanford_cars_sample.json"
SAMPLE_LOCK = threading.Lock()
BENCHMARK_PROCESS: subprocess.Popen | None = None


def _image_map() -> dict[str, Path]:
    paths = {p.name: p for p in OLD_IMAGES.iterdir() if p.is_file()}
    if COLOR_IMAGES.is_dir():
        paths.update({p.name: p for p in COLOR_IMAGES.iterdir() if p.is_file()})
    if MANIFEST.is_file():
        data = json.loads(MANIFEST.read_text(encoding="utf-8"))
        for item in data.get("items", []):
            paths[item["filename"]] = STANFORD_IMAGES / item["filename"]
    return paths


def _status() -> dict:
    if PROGRESS_FILE.is_file():
        try:
            result = json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            result = {"state": "starting"}
    else:
        result = {"state": "starting"}
    process = BENCHMARK_PROCESS
    if process is not None:
        code = process.poll()
        if code is not None and result.get("state") not in {"complete", "failed"}:
            result["state"] = "failed"
            result["error"] = f"O processo de avaliação terminou com código {code}. Veja {LOG_FILE}."
    return result


def _start_benchmark(restart: bool = False, threads: int = 8) -> dict:
    global BENCHMARK_PROCESS
    with SAMPLE_LOCK:
        if BENCHMARK_PROCESS is not None and BENCHMARK_PROCESS.poll() is None:
            if not restart:
                return {"started": False, "reason": "already_running"}
            BENCHMARK_PROCESS.terminate()
            try:
                BENCHMARK_PROCESS.wait(timeout=5)
            except subprocess.TimeoutExpired:
                BENCHMARK_PROCESS.kill()
                BENCHMARK_PROCESS.wait()

        examples = ROOT / "eval/compare_laya_blip.py"
        if not MANIFEST.is_file():
            return {"started": False, "error": "Dataset ausente. Execute eval/download_stanford_cars_sample.py."}

        PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
        try:
            dataset_total = len(load_examples())
        except Exception as exc:
            return {"started": False, "error": f"Não consegui ler o conjunto local: {exc}"}
        initial = {
            "state": "starting", "phase": "startup", "phase_label": "Iniciando",
            "phase_completed": 0, "phase_total": dataset_total,
            "overall_completed": 0, "overall_total": dataset_total * 2,
            "dataset_total": dataset_total, "elapsed_seconds": 0, "current": None,
            "results": {"laya": [], "blip": []}, "summaries": [],
        }
        PROGRESS_FILE.write_text(json.dumps(initial), encoding="utf-8")
        python = ROOT / ".venv/bin/python"
        if not python.exists():
            python = Path(sys.executable)
        with LOG_FILE.open("w", encoding="utf-8") as log:
            BENCHMARK_PROCESS = subprocess.Popen([
                str(python), str(examples), "--threads", str(threads), "--device", "cpu",
                "--out", str(RESULT_FILE), "--progress-file", str(PROGRESS_FILE),
            ], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        return {"started": True, "pid": BENCHMARK_PROCESS.pid}


class BenchmarkHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {args[0]} {args[1]}\n")

    def _send(self, body: bytes, content_type: str, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def _send_json(self, result: dict, status: int = 200):
        self._send(json.dumps(result, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8", status)

    def do_GET(self):
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            self._send(PAGE.read_bytes(), "text/html; charset=utf-8")
            return
        if path == "/api/benchmark/status":
            self._send_json(_status())
            return
        if path.startswith("/images/"):
            name = unquote(path.removeprefix("/images/"))
            target = _image_map().get(name)
            if target is None or not target.is_file():
                self._send_json({"error": "Imagem não encontrada"}, 404)
                return
            suffix = target.suffix.lower()
            kind = "image/png" if suffix == ".png" else "image/jpeg"
            self._send(target.read_bytes(), kind)
            return
        self._send_json({"error": "Not found"}, 404)

    def do_POST(self):
        if urlparse(self.path).path not in {"/api/benchmark/restart", "/api/benchmark/start"}:
            self._send_json({"error": "Not found"}, 404)
            return
        try:
            length = min(int(self.headers.get("Content-Length", "0")), 4096)
            body = json.loads(self.rfile.read(length) or b"{}")
            threads = max(1, min(int(body.get("threads", 8)), 12))
        except Exception:
            self._send_json({"error": "JSON inválido"}, 400)
            return
        result = _start_benchmark(restart=urlparse(self.path).path.endswith("restart"), threads=threads)
        self._send_json(result, 200 if "error" not in result else 500)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--no-open", action="store_true", help="não abrir o navegador automaticamente")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), BenchmarkHandler)
    started = _start_benchmark(threads=args.threads)
    if "error" in started:
        server.server_close()
        parser.error(started["error"])
    url = f"http://localhost:{args.port}/"
    print(f"Dashboard do benchmark: {url}", flush=True)
    print("O teste inicia automaticamente; o progresso aparece no navegador.", flush=True)
    if not args.no_open:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Encerrando dashboard...", flush=True)
    finally:
        server.server_close()
        if BENCHMARK_PROCESS is not None and BENCHMARK_PROCESS.poll() is None:
            BENCHMARK_PROCESS.terminate()
            try:
                BENCHMARK_PROCESS.wait(timeout=5)
            except subprocess.TimeoutExpired:
                BENCHMARK_PROCESS.kill()


if __name__ == "__main__":
    main()

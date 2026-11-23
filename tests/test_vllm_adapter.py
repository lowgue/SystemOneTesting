"""Test the vLLM chat adapter and its optional concurrent benchmark path."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vision_bench.cli import main
from vision_bench.core import DEFAULT_CONTRACT, sha256


class VLLMAdapterTests(unittest.TestCase):
    """Cover OpenAI chat formatting, answer validation, and request batching."""

    def test_parallel_run_uses_vllm_requests_and_records_batch_metrics(self) -> None:
        """Verify parallel vLLM calls overlap and retain per-image result rows."""
        observed = {"active": 0, "maximum": 0}
        lock = threading.Lock()

        class Handler(BaseHTTPRequestHandler):
            """Track concurrent calls and return a minimal chat completion."""

            def log_message(self, *_args: object) -> None:
                """Keep local HTTP fixture logs out of test output."""

            def do_POST(self) -> None:
                """Record the chat request and emit a parseable model response."""
                length = int(self.headers["Content-Length"])
                request = json.loads(self.rfile.read(length))
                self.server.requests.append(request)
                with lock:
                    observed["active"] += 1
                    observed["maximum"] = max(observed["maximum"], observed["active"])
                time.sleep(0.05)
                with lock:
                    observed["active"] -= 1
                payload = json.dumps(
                    {"choices": [{"message": {"content": "{}"}}]}
                ).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.requests = []
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                items = []
                for index in range(3):
                    image_path = root / f"image-{index}.png"
                    Image.new("RGB", (16, 16), "white").save(image_path)
                    items.append(
                        {
                            "id": f"image-{index}",
                            "domain": "vehicle",
                            "rel_path": image_path.name,
                            "sha256": sha256(image_path),
                        }
                    )
                manifest = root / "manifest.json"
                manifest.write_text(json.dumps({"items": items}), encoding="utf-8")
                config = root / "models.json"
                config.write_text(
                    json.dumps(
                        {
                            "models": [
                                {
                                    "id": "fixture-vllm",
                                    "adapter": "vllm_openai",
                                    "url": (
                                        f"http://127.0.0.1:{server.server_port}"
                                        "/v1/chat/completions"
                                    ),
                                    "model": "fixture-model",
                                }
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
                output = root / "run"
                status = main(
                    [
                        "run",
                        "--config",
                        str(config),
                        "--manifest",
                        str(manifest),
                        "--contract",
                        str(DEFAULT_CONTRACT),
                        "--data-root",
                        str(root),
                        "--output",
                        str(output),
                        "--repetitions",
                        "1",
                        "--concurrency",
                        "2",
                    ]
                )
                self.assertEqual(status, 0)
                rows = [
                    json.loads(line)
                    for line in (output / "predictions.jsonl").read_text().splitlines()
                ]
                self.assertGreaterEqual(observed["maximum"], 2)
                self.assertEqual(len(rows), 3)
                self.assertTrue(
                    all(
                        row["resource_scope"] == "client_batch_shared"
                        for row in rows
                    )
                )
                self.assertTrue(all(row["batch_size"] == 2 for row in rows[:2]))
                self.assertTrue(
                    all(
                        "image_url" in str(request["messages"])
                        for request in server.requests
                    )
                )
                self.assertTrue(
                    all(
                        request["response_format"] == {"type": "json_object"}
                        for request in server.requests
                    )
                )
                report = root / "report"
                main(["report", str(output), "--output", str(report)])
                metrics = json.loads((report / "metrics.json").read_text())
                model_metrics = metrics[0]
                self.assertEqual(
                    model_metrics["throughput_mode"], "concurrent_batches"
                )
                expected_batch_seconds = sum(
                    row["batch_elapsed_ms"]
                    for index, row in enumerate(rows)
                    if row["batch_id"] not in {
                        earlier["batch_id"] for earlier in rows[:index]
                    }
                ) / 1000
                self.assertAlmostEqual(
                    model_metrics["measured_s"], expected_batch_seconds
                )
                self.assertGreater(model_metrics["throughput_requests_s"], 0)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()

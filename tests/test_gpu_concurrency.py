"""Automatic vLLM batch sizing from reported GPU memory."""

from __future__ import annotations

import subprocess
from io import BytesIO
from unittest import TestCase
from unittest.mock import patch

from vision_bench.gpu import resolve_concurrency, vllm_cache_metrics


class ResolveConcurrencyTests(TestCase):
    """Check explicit batches, automatic sizing, and safe fallbacks."""

    def test_explicit_positive_integer_is_preserved(self) -> None:
        """An explicit batch size is returned unchanged, even above the auto cap."""
        self.assertEqual(resolve_concurrency("12", max_concurrency=4), 12)

    def test_invalid_batch_size_is_rejected(self) -> None:
        """Zero, negative values, and non-numeric text are rejected."""
        for value in (0, -1, "many"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    resolve_concurrency(value)

    def test_auto_uses_free_gpu_memory_and_honors_cap(self) -> None:
        """Automatic sizing divides free VRAM by the per-sequence budget and caps it."""
        query_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="24000, 2000\n", stderr=""
        )
        with (
            patch("vision_bench.gpu.shutil.which", return_value="/usr/bin/nvidia-smi"),
            patch("vision_bench.gpu.subprocess.run", return_value=query_result),
        ):
            self.assertEqual(
                resolve_concurrency("auto", max_concurrency=8), 8
            )

    def test_auto_falls_back_without_nvidia_smi(self) -> None:
        """A machine without nvidia-smi gets the small safe fallback batch."""
        with patch("vision_bench.gpu.shutil.which", return_value=None):
            self.assertEqual(resolve_concurrency("auto", max_concurrency=16), 4)

    def test_auto_falls_back_when_query_fails(self) -> None:
        """A failed GPU query cannot abort the benchmark setup."""
        with (
            patch("vision_bench.gpu.shutil.which", return_value="/usr/bin/nvidia-smi"),
            patch(
                "vision_bench.gpu.subprocess.run",
                side_effect=subprocess.SubprocessError("driver failure"),
            ),
        ):
            self.assertEqual(resolve_concurrency("auto"), 4)

    def test_vllm_cache_metrics_reads_latest_gauges(self) -> None:
        """The latest running and cache-pressure gauges are extracted from /metrics."""
        text = (
            "# HELP vllm:num_requests_running Running requests\n"
            "vllm:num_requests_running{model_name=\"clef\"} 2.0\n"
            "vllm:num_requests_running{model_name=\"clef\"} 4.0\n"
            "vllm:kv_cache_usage_perc{model_name=\"clef\"} 0.5\n"
        )
        with patch("vision_bench.gpu.urllib.request.urlopen", return_value=BytesIO(text.encode())):
            metrics = vllm_cache_metrics("http://127.0.0.1:8000/metrics")
        self.assertEqual(
            metrics,
            {"num_requests_running": 4.0, "kv_cache_usage_perc": 0.5},
        )

    def test_auto_reduces_batch_when_kv_cache_is_pressured(self) -> None:
        """Live vLLM pressure reduces auto concurrency and keeps one slot free."""
        text = (
            "vllm:num_requests_running 28.0\n"
            "vllm:kv_cache_usage_perc 0.875\n"
        )
        with patch("vision_bench.gpu.urllib.request.urlopen", return_value=BytesIO(text.encode())):
            self.assertEqual(
                resolve_concurrency(
                    "auto",
                    max_concurrency=64,
                    metrics_url="http://127.0.0.1:8000/metrics",
                    server_max_num_seqs=32,
                ),
                3,
            )

    def test_auto_uses_server_sequence_cap_for_idle_external_server(self) -> None:
        """A warmed external server uses its configured sequence cap when idle."""
        text = "vllm:num_requests_running 0.0\nvllm:kv_cache_usage_perc 0.0\n"
        with patch("vision_bench.gpu.urllib.request.urlopen", return_value=BytesIO(text.encode())):
            self.assertEqual(
                resolve_concurrency(
                    "auto",
                    max_concurrency=64,
                    metrics_url="http://127.0.0.1:8000/metrics",
                    server_max_num_seqs=32,
                ),
                32,
            )

    def test_non_positive_server_sequence_cap_is_rejected(self) -> None:
        """An invalid server sequence cap is rejected before any benchmark starts."""
        with self.assertRaises(ValueError):
            resolve_concurrency("auto", server_max_num_seqs=0)

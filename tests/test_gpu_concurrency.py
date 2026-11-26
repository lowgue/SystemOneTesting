"""Automatic vLLM batch sizing from reported GPU memory."""

from __future__ import annotations

import subprocess
from unittest import TestCase
from unittest.mock import patch

from vision_bench.gpu import resolve_concurrency


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

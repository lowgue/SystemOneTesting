"""CPU and GPU telemetry collected during concurrent benchmark requests."""

from __future__ import annotations

import subprocess
from unittest import TestCase
from unittest.mock import patch

from vision_bench.runner import Resources, gpu_sample


class ResourceSamplingTests(TestCase):
    """Check parsing and aggregation of client CPU and server GPU samples."""

    def test_gpu_sample_parses_utilization_and_memory(self) -> None:
        """One nvidia-smi row becomes utilization percentage and used MiB."""
        result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="42, 12345\n", stderr=""
        )
        with (
            patch("vision_bench.runner.shutil.which", return_value="/usr/bin/nvidia-smi"),
            patch("vision_bench.runner.subprocess.run", return_value=result),
        ):
            self.assertEqual(gpu_sample(), (42.0, 12345.0))

    def test_gpu_sample_fails_softly(self) -> None:
        """A malformed nvidia-smi response does not abort the benchmark."""
        result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="unknown\n", stderr=""
        )
        with (
            patch("vision_bench.runner.shutil.which", return_value="/usr/bin/nvidia-smi"),
            patch("vision_bench.runner.subprocess.run", return_value=result),
        ):
            self.assertEqual(gpu_sample(), (None, None))

    def test_resource_summary_ignores_missing_gpu_samples(self) -> None:
        """CPU always has a scalar summary; GPU fields stay optional."""
        resource = Resources(sample_gpu=False)
        resource.samples = [
            dict(
                ts_rel_ms=0.0,
                cpu_pct_total=10.0,
                cpu_cores_equiv=0.1,
                gpu_util_pct=20.0,
                gpu_mem_used_mib=1000.0,
            ),
            dict(
                ts_rel_ms=50.0,
                cpu_pct_total=30.0,
                cpu_cores_equiv=0.3,
                gpu_util_pct=None,
                gpu_mem_used_mib=None,
            ),
        ]
        resource._summarize()
        self.assertEqual(resource.cpu_pct_mean, 20.0)
        self.assertEqual(resource.cpu_pct_peak, 30.0)
        self.assertEqual(resource.gpu_util_mean, 20.0)
        self.assertEqual(resource.gpu_util_peak, 20.0)
        self.assertEqual(resource.gpu_mem_peak_mib, 1000.0)

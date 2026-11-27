"""Cleanup of vLLM worker process trees after a benchmark."""

from __future__ import annotations

import importlib.util
import io
import os
from pathlib import Path
import signal
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, patch


ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "test_all", ROOT / "scripts" / "test_all.py"
)
test_all = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(test_all)


class StopProcessTreeTests(TestCase):
    """Check that temporary servers never leave GPU workers behind."""

    def test_graceful_group_shutdown_does_not_escalate(self) -> None:
        """A group that exits after SIGTERM is not killed with SIGKILL."""
        process = MagicMock()
        process.pid = 4242
        process.wait.return_value = 0
        calls = []

        with (
            patch.object(test_all.os, "killpg", side_effect=lambda *args: calls.append(args[1])),
            patch.object(test_all, "_process_group_exists", return_value=False),
        ):
            test_all.stop_process_tree(process)

        self.assertEqual(calls, [signal.SIGTERM])
        self.assertEqual(process.wait.call_count, 1)

    def test_surviving_workers_are_killed(self) -> None:
        """Workers that keep the process group alive receive SIGKILL."""
        process = MagicMock()
        process.pid = 4242
        process.wait.side_effect = [
            subprocess.TimeoutExpired(cmd="vllm", timeout=30),
            None,
        ]
        calls = []

        with (
            patch.object(test_all.os, "killpg", side_effect=lambda *args: calls.append(args[1])),
            patch.object(test_all, "_process_group_exists", return_value=True),
            patch.object(
                test_all.time,
                "monotonic",
                side_effect=[0, 0, 10, 10, 10, 10, 10, 10],
            ),
            patch.object(test_all.time, "sleep"),
        ):
            test_all.stop_process_tree(process)

        self.assertEqual(calls, [signal.SIGTERM, signal.SIGKILL])
        self.assertEqual(process.wait.call_count, 2)


class GpuAvailabilityTests(TestCase):
    """Check the preflight guard against other GPU services."""

    def test_busy_gpu_aborts_with_actionable_message(self) -> None:
        """A busy GPU stops the run before any vLLM process starts."""
        stderr = io.StringIO()
        with (
            patch.object(
                test_all, "gpu_memory_usage_mib", return_value=(23568, 7610)
            ),
            patch("sys.stderr", stderr),
        ):
            with self.assertRaises(SystemExit) as context:
                test_all.check_gpu_availability(False)

        self.assertEqual(context.exception.code, 2)
        self.assertIn("ollama", stderr.getvalue())
        self.assertIn("--allow-busy-gpu", stderr.getvalue())

    def test_allow_busy_gpu_continues(self) -> None:
        """An operator can explicitly opt into sharing an occupied GPU."""
        with (
            patch.object(
                test_all, "gpu_memory_usage_mib", return_value=(23568, 7610)
            ),
            patch("sys.stderr", io.StringIO()),
        ):
            test_all.check_gpu_availability(True)


class ExternalServerTests(TestCase):
    """Check that an already warmed vLLM server is reused without takeover."""

    def test_metrics_url_from_endpoint(self) -> None:
        """The Prometheus endpoint keeps only the served host and port."""
        self.assertEqual(
            test_all.metrics_url_from_endpoint("http://127.0.0.1:8123/v1/systemone"),
            "http://127.0.0.1:8123/metrics",
        )

    def test_benchmark_model_reuses_external_server(self) -> None:
        """External mode neither starts nor stops a server nor rewrites its URL."""
        entry = {
            "id": "clef-flash-vllm",
            "url_env": "VLLM_SYSTEMONE_URL",
            "route": "systemone",
            "contract": "description_ollama_v1.json",
        }
        url = "http://127.0.0.1:8123/v1/systemone"
        os.environ["VLLM_SYSTEMONE_URL"] = url
        args = SimpleNamespace(repetitions=1, vllm_executable=Path("/missing/vllm"))

        def complete_run(command: list[str]) -> None:
            """Create the status file that the benchmark expects after a run."""
            output = Path(command[-1])
            output.mkdir(parents=True, exist_ok=True)
            (output / "status.json").write_text('{"status":"complete"}', encoding="utf-8")

        with TemporaryDirectory() as directory:
            with (
                patch.object(test_all, "write_model_config"),
                patch.object(test_all, "start_vllm_server") as start,
                patch.object(test_all, "stop_server") as stop,
                patch.object(test_all, "start_gpu_monitor", return_value=None),
                patch.object(test_all, "run", side_effect=complete_run),
            ):
                output = test_all.benchmark_model(
                    args, entry, {}, 22, Path(directory), external=True
                )

            self.assertTrue(output.is_dir())
            start.assert_not_called()
            stop.assert_not_called()

        self.assertEqual(os.environ["VLLM_SYSTEMONE_URL"], url)

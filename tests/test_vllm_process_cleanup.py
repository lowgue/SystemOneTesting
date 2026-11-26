"""Cleanup of vLLM worker process trees after a benchmark."""

from __future__ import annotations

import importlib.util
import io
from pathlib import Path
import signal
import subprocess
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

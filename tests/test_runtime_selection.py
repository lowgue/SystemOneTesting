"""Tests for automatic CUDA selection and CPU thread sizing."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from vision_bench.runtime import resolve_device, resolve_thread_count


class RuntimeSelectionTests(unittest.TestCase):
    """Check hardware selection without requiring a local GPU."""

    def test_auto_device_uses_first_cuda_device_when_available(self) -> None:
        """Select CUDA device zero when Torch reports a visible GPU."""
        torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 2)
        )
        with patch("vision_bench.runtime.importlib.import_module", return_value=torch):
            self.assertEqual(resolve_device("auto", has_local_torch_models=True), "cuda:0")

    def test_auto_device_falls_back_to_cpu_without_cuda(self) -> None:
        """Use CPU automatically when Torch has no visible CUDA device."""
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with patch("vision_bench.runtime.importlib.import_module", return_value=torch):
            self.assertEqual(resolve_device("auto", has_local_torch_models=True), "cpu")

    def test_explicit_cuda_fails_when_cuda_is_unavailable(self) -> None:
        """Explain an explicit CUDA request that cannot be satisfied."""
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with patch("vision_bench.runtime.importlib.import_module", return_value=torch):
            with self.assertRaisesRegex(RuntimeError, "não detectou uma GPU"):
                resolve_device("cuda", has_local_torch_models=True)

    def test_invalid_cuda_device_name_is_rejected(self) -> None:
        """Reject malformed CUDA device names before consulting Torch."""
        with self.assertRaisesRegex(ValueError, "Dispositivo inválido"):
            resolve_device("cuda:wrong", has_local_torch_models=True)

    def test_auto_threads_use_physical_core_count(self) -> None:
        """Avoid counting simultaneous multithreading as extra Torch cores."""
        with patch("vision_bench.runtime.psutil.cpu_count", side_effect=[8, 16]):
            self.assertEqual(resolve_thread_count("auto"), 8)

    def test_auto_threads_fall_back_to_logical_count(self) -> None:
        """Use logical CPU count when physical core information is unavailable."""
        with patch("vision_bench.runtime.psutil.cpu_count", side_effect=[None, 12]):
            self.assertEqual(resolve_thread_count("auto"), 12)


if __name__ == "__main__":
    unittest.main()

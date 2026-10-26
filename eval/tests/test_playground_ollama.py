"""Regression checks for errors returned by the local decision server."""
import importlib.util
import io
from pathlib import Path
import unittest
import urllib.error
from unittest.mock import patch
from PIL import Image

spec = importlib.util.spec_from_file_location('playground', Path(__file__).resolve().parents[2] / 'web/server.py')
playground = importlib.util.module_from_spec(spec)
spec.loader.exec_module(playground)


class OllamaErrors(unittest.TestCase):
    def failure(self, body):
        error = urllib.error.HTTPError('http://localhost', 500, 'error', {}, io.BytesIO(body))
        with patch.object(playground.urllib.request, 'urlopen', side_effect=error):
            with self.assertRaises(RuntimeError) as caught:
                playground.call_local_clef(Image.new('RGB', (8, 8)), {}, ollama=True)
        return str(caught.exception)

    def test_memory_error_preserves_allocation_details(self):
        message = self.failure(b'{"error":"insufficient memory: attempted to allocate 12634 MB"}')
        self.assertIn('Memória insuficiente', message)
        self.assertIn('12634 MB', message)

    def test_other_failure_preserves_server_reason(self):
        self.assertIn('model runner failed', self.failure(b'{"error":"model runner failed"}'))

    def test_plain_text_error(self):
        self.assertIn('backend unavailable', self.failure(b'backend unavailable'))

    def test_connection_error(self):
        with patch.object(playground.urllib.request, 'urlopen', side_effect=urllib.error.URLError('refused')):
            with self.assertRaisesRegex(RuntimeError, 'Ollama está ativo'):
                playground.call_local_clef(Image.new('RGB', (8, 8)), {}, ollama=True)

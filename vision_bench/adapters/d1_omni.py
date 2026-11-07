"""LiquidAI d1-omni adapter using the model's typed image-question method."""

from __future__ import annotations

import time
from typing import Any

from PIL import Image

from .transformers import _TorchAdapter

class D1Omni(_TorchAdapter):
    """Use LiquidAI's typed image-question method without generating tokens."""

    def __init__(
        self,
        spec: dict[str, Any],
        contract: dict[str, Any],
        device: str = "cpu",
        offline: bool = False,
        **_: Any,
    ) -> None:
        """Load the configured d1-omni model with a device-appropriate dtype."""
        super().__init__(spec, contract, device)
        import torch
        from transformers import AutoModel

        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.model = (
            AutoModel.from_pretrained(
                spec["model"],
                revision=spec["revision"],
                trust_remote_code=True,
                torch_dtype=dtype,
                local_files_only=offline,
            )
            .to(device)
            .eval()
        )
        self._torch = torch

    def predict(
        self, image: Image.Image, domain: str
    ) -> tuple[dict[str, str], dict[str, Any]]:
        """Run one typed request and distribute its total latency across fields."""
        from ..protocols import questions

        question_map = questions(self.spec, self.contract, domain)
        self.synchronize()
        started_at = time.perf_counter()
        raw = self.model.system_one(
            state="Describe only the visible attributes of the target in this image.",
            questions=question_map,
            images=image,
        )
        self.synchronize()
        elapsed_ms = (time.perf_counter() - started_at) * 1_000

        response_answers = raw.get("answers", {})
        answers = {
            key: str(response_answers.get(key, {}).get("choice", "parse_error"))
            for key in self.contract["domains"][domain]["questions"]
        }
        field_latency = elapsed_ms / len(question_map)
        self.last_metrics = {
            "inference_calls": 1,
            "attribute_ms": {key: field_latency for key in question_map},
        }
        return answers, raw

    def close(self) -> None:
        """Release model memory and clear the CUDA allocator when applicable."""
        del self.model
        import gc

        gc.collect()
        if self.device.startswith("cuda"):
            self._torch.cuda.empty_cache()

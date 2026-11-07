"""Local Torch adapter base and the Transformers BLIP implementation."""

from __future__ import annotations

import json
import time
from typing import Any

from PIL import Image

from ..core import normalize_text
from .base import MockAdapter

class _TorchAdapter(MockAdapter):
    """Share device synchronization and model settings for local Torch adapters."""

    def __init__(
        self,
        spec: dict[str, Any],
        contract: dict[str, Any],
        device: str,
    ) -> None:
        """Store shared configuration without importing Torch at module load."""
        super().__init__(spec, contract)
        self.spec = spec
        self.device = device

    def synchronize(self) -> None:
        """Wait for queued CUDA work so measured durations include completed work."""
        if self.device.startswith("cuda"):
            import torch

            torch.cuda.synchronize(self.device)

class BLIP(_TorchAdapter):
    """Answer typed visual questions with the Transformers BLIP VQA model."""

    def __init__(
        self,
        spec: dict[str, Any],
        contract: dict[str, Any],
        device: str = "cpu",
        offline: bool = False,
        **_: Any,
    ) -> None:
        """Load the configured processor and model on the requested device."""
        super().__init__(spec, contract, device)
        from transformers import AutoModelForVisualQuestionAnswering, AutoProcessor

        model_options = {
            "revision": spec["revision"],
            "local_files_only": offline,
        }
        self.processor = AutoProcessor.from_pretrained(spec["model"], **model_options)
        self.model = (
            AutoModelForVisualQuestionAnswering.from_pretrained(
                spec["model"], **model_options
            )
            .to(device)
            .eval()
        )

    def predict(
        self, image: Image.Image, domain: str
    ) -> tuple[dict[str, str], dict[str, Any]]:
        """Run prompts in configured batches and normalize each decoded answer."""
        import torch

        from ..protocols import prompts

        prompt_map = prompts(self.spec, self.contract, domain)
        keys = list(prompt_map)
        raw_answers: dict[str, str] = {}
        timings: dict[str, float] = {}
        groups = [[key] for key in keys] if self.spec.get("profile") == "short_serial" else [keys]

        for group in groups:
            self.synchronize()
            started_at = time.perf_counter()
            inputs = self.processor(
                images=[image] * len(group),
                text=[prompt_map[key] for key in group],
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            max_tokens = 64 if self.spec.get("profile") == "structured_json" else 12
            with torch.inference_mode():
                tokens = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                )
            self.synchronize()
            decoded = self.processor.batch_decode(tokens, skip_special_tokens=True)
            raw_answers.update(zip(group, decoded, strict=True))
            if len(group) == 1:
                timings[group[0]] = (time.perf_counter() - started_at) * 1_000

        answers = {
            key: self._normalize_answer(key, value, domain)
            for key, value in raw_answers.items()
        }
        self.last_metrics = {
            "inference_calls": len(groups),
            "attribute_ms": timings,
        }
        return answers, raw_answers

    def _normalize_answer(self, key: str, value: str, domain: str) -> str:
        """Parse strict JSON profiles or normalize a free-text model response."""
        if self.spec.get("profile") != "structured_json":
            return normalize_text(self.contract, domain, key, value)

        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            return "parse_error"
        if not isinstance(parsed, dict) or set(parsed) != {key}:
            return "parse_error"

        answer = str(parsed[key])
        valid_choices = self.contract["domains"][domain]["questions"][key]["criteria"]
        return answer if answer in valid_choices else "parse_error"

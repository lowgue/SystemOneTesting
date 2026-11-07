"""Laya Vision adapter and its Transformers image-feature compatibility patch."""

from __future__ import annotations

import logging
import sys
import time
from typing import Any

from PIL import Image

from ..core import ROOT
from .transformers import _TorchAdapter

def _encode_laya_images(
    model: Any,
    pixel_values: Any,
    pixel_attention_mask: Any = None,
) -> Any:
    """Adapt Laya image features to the Transformers batch shape it expects."""
    import torch

    attention_mask = (
        pixel_attention_mask[None]
        if pixel_attention_mask is not None
        else None
    )
    features = model._image_features(pixel_values[None], attention_mask)
    return torch.stack(features) if isinstance(features, list) else features

class Laya(_TorchAdapter):
    """Run Laya Vision with its native typed-question interface."""

    def __init__(
        self,
        spec: dict[str, Any],
        contract: dict[str, Any],
        device: str = "cpu",
        offline: bool = False,
        **_: Any,
    ) -> None:
        """Load Laya and install the image-batch compatibility patch it requires."""
        super().__init__(spec, contract, device)
        sys.path.insert(0, str(ROOT / "modelos/laya/laya_vision"))
        import laya
        import laya.vlm as vlm
        from transformers.utils import logging as transformers_logging

        self._configure_logging(transformers_logging)
        vlm.VLMDecisionModel.encode_images = _encode_laya_images
        self.model = laya.load_vlm(
            spec["model"],
            revision=spec["revision"],
            device=device,
        )
        if max_length := spec.get("head_max_len"):
            # Input-budget changes are explicit configuration, never silent truncation.
            self.model.cfg["head_max_len"] = max_length

    @staticmethod
    def _configure_logging(transformers_logging: Any) -> None:
        """Suppress only the known multimodal encoder warning from Transformers."""
        class MultimodalEncoderFilter(logging.Filter):
            """Filter repetitive encoder-output diagnostics from model startup."""

            def filter(self, record: logging.LogRecord) -> bool:
                """Drop log records that only mention encoder output internals."""
                return "mm_encoder_outputs" not in record.getMessage()

        library_logger = transformers_logging._get_library_root_logger()
        warning_filter = MultimodalEncoderFilter()
        library_logger.addFilter(warning_filter)
        for handler in library_logger.handlers:
            handler.addFilter(warning_filter)

    def predict(
        self, image: Image.Image, domain: str
    ) -> tuple[dict[str, str], dict[str, Any]]:
        """Send configured native questions and retain response timing metadata."""
        from ..protocols import questions

        question_map = questions(self.spec, self.contract, domain)
        response_answers: dict[str, Any] = {}
        timings: dict[str, float] = {}
        groups = (
            [{key: question} for key, question in question_map.items()]
            if self.spec.get("profile") == "native_serial"
            else [question_map]
        )

        for group in groups:
            self.synchronize()
            started_at = time.perf_counter()
            response = self.model.predict(image, group, strict=True)
            self.synchronize()
            response_answers.update(response.get("answers", {}))
            if len(group) == 1:
                key = next(iter(group))
                timings[key] = (time.perf_counter() - started_at) * 1_000

        raw = {"answers": response_answers}
        answers = {
            key: str(response_answers.get(key, {}).get("choice", "parse_error"))
            for key in self.contract["domains"][domain]["questions"]
        }
        self.last_metrics = {
            "inference_calls": len(groups),
            "attribute_ms": timings,
        }
        return answers, raw

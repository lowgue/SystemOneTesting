"""OpenAI-compatible multimodal adapter for vLLM chat completions."""

from __future__ import annotations

import base64
from io import BytesIO
import json
import os
from typing import Any
import urllib.error
import urllib.request

from PIL import Image

from .base import MockAdapter
from .http import endpoint


class VLLMOpenAI(MockAdapter):
    """Send contract-driven image questions to a vLLM chat endpoint."""

    def __init__(
        self,
        spec: dict[str, Any],
        contract: dict[str, Any],
        **options: Any,
    ) -> None:
        """Validate and store endpoint, model, and optional token configuration."""
        super().__init__(spec, contract)
        self.spec = spec
        self.url = endpoint(spec)
        model_variable = spec.get("model_env")
        model = (
            os.environ.get(model_variable, "")
            if model_variable
            else spec.get("model")
        )
        if not isinstance(model, str) or not model:
            raise ValueError("Set the configured vLLM served model name")
        self.model = model
        token_variable = spec.get("token_env")
        self.token = os.environ.get(token_variable, "") if token_variable else ""
        if token_variable and not self.token:
            raise ValueError(f"Set {token_variable} for this backend")
        self.timeout = spec.get("timeout_s", 300)
        self.last_metrics = {"inference_calls": 0, "attribute_ms": {}}

    def predict(
        self, image: Image.Image, domain: str
    ) -> tuple[dict[str, str], dict[str, Any]]:
        """Submit one image and normalize its JSON field choices."""
        from ..protocols import questions

        image_buffer = BytesIO()
        image.save(image_buffer, format="PNG")
        image_data = base64.b64encode(image_buffer.getvalue()).decode("ascii")
        model_questions = questions(self.spec, self.contract, domain)
        prompt = self._prompt(model_questions)
        payload = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": self.spec.get("max_tokens", 256),
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_data}"
                            },
                        },
                    ],
                }
            ],
        }
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError(
                f"vLLM HTTP {error.code}; verify model and endpoint"
            ) from None
        except urllib.error.URLError:
            raise RuntimeError(
                "Could not connect to the configured vLLM endpoint"
            ) from None

        content = self._response_content(raw)
        answers = self._parse_answers(content, model_questions)
        self.last_metrics = {"inference_calls": 1, "attribute_ms": {}}
        return answers, raw

    @staticmethod
    def _prompt(model_questions: dict[str, Any]) -> str:
        """Build a strict JSON choice prompt from the selected contract fields."""
        fields = {
            key: {
                "instructions": question["instructions"],
                "allowed_choices": list(question["criteria"]),
            }
            for key, question in model_questions.items()
        }
        return (
            "Describe only visible attributes in the image. For each field, choose "
            "exactly one allowed choice. Return only a JSON object mapping every "
            "field name to its chosen choice key. Do not add explanation. Fields: "
            + json.dumps(fields, ensure_ascii=False)
        )

    @staticmethod
    def _response_content(raw: Any) -> str:
        """Validate and extract the first OpenAI-compatible chat response."""
        try:
            return raw["choices"][0]["message"]["content"]
        except (TypeError, KeyError, IndexError):
            raise ValueError("vLLM returned no chat completion content") from None

    @staticmethod
    def _parse_answers(
        content: str, model_questions: dict[str, Any]
    ) -> dict[str, str]:
        """Accept only JSON choices present in the contract; mark others invalid."""
        try:
            parsed = json.loads(content)
        except (TypeError, json.JSONDecodeError):
            parsed = {}
        if not isinstance(parsed, dict):
            parsed = {}
        return {
            key: (
                parsed[key]
                if isinstance(parsed.get(key), str)
                and parsed[key] in question["criteria"]
                else "parse_error"
            )
            for key, question in model_questions.items()
        }

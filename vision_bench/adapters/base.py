"""Shared adapter interface and deterministic simulation implementation."""

from __future__ import annotations

from typing import Any, Protocol

from PIL import Image

class Adapter(Protocol):
    """Interface required by the benchmark runner for a model adapter."""

    last_metrics: dict[str, Any]

    def predict(self, image: Image.Image, domain: str) -> tuple[dict[str, str], dict[str, Any]]:
        """Return normalized field choices and the backend's raw response."""
        ...

    def synchronize(self) -> None:
        """Wait for asynchronous device work before timing or shutdown."""
        ...

    def close(self) -> None:
        """Release model resources owned by the adapter."""
        ...

class MockAdapter:
    """Return deterministic schema-valid answers for explicit simulation runs."""

    def __init__(
        self,
        spec: dict[str, Any],
        contract: dict[str, Any],
        **_: Any,
    ) -> None:
        """Store the response contract used to create simulated answers."""
        self.contract = contract
        self.last_metrics: dict[str, Any] = {"inference_calls": 0, "attribute_ms": {}}

    def predict(
        self, image: Image.Image, domain: str
    ) -> tuple[dict[str, str], dict[str, Any]]:
        """Choose the first allowed label for every field and mark it simulated."""
        questions = self.contract["domains"][domain]["questions"]
        answers = {
            key: next(iter(question["criteria"]))
            for key, question in questions.items()
        }
        raw = {
            "simulated": True,
            "answers": {
                key: {"choice": choice} for key, choice in answers.items()
            },
        }
        return answers, raw

    def synchronize(self) -> None:
        """Provide the no-op synchronization required by the adapter interface."""

    def close(self) -> None:
        """Provide the no-op cleanup required by the adapter interface."""

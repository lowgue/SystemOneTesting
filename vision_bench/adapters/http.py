"""HTTP endpoint validation and System One request adapter."""

from __future__ import annotations

import base64
from io import BytesIO
import json
import os
import re
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

from PIL import Image

from .base import MockAdapter

def endpoint(spec: dict[str, Any]) -> str:
    """Resolve and validate an HTTP endpoint from a model configuration.

    A configuration may use a literal URL or an environment variable. Cloudflare
    endpoints are composed from an account ID without accepting embedded secrets.
    """
    url = os.environ.get(spec.get("url_env", ""), spec.get("url", ""))
    if spec.get("account_env"):
        account = os.environ.get(spec["account_env"], "")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", account):
            raise ValueError("Set the configured Cloudflare account environment variable")
        url = (
            "https://api.cloudflare.com/client/v4/accounts/"
            f"{account}/ai/run/@cf/cloudflare/clef-flash"
        )

    parsed = urllib.parse.urlsplit(url)
    valid_scheme = parsed.scheme in {"http", "https"}
    has_host = bool(parsed.hostname)
    contains_credentials = bool(parsed.username or parsed.password)
    has_suffix = bool(parsed.query or parsed.fragment)
    if not valid_scheme or not has_host or contains_credentials or has_suffix:
        raise ValueError(
            "Endpoint must be HTTP(S), without embedded credentials, query or fragment"
        )
    return url

class SystemOneHTTP(MockAdapter):
    """Send image questions to a System One compatible HTTP endpoint."""

    def __init__(
        self,
        spec: dict[str, Any],
        contract: dict[str, Any],
        **options: Any,
    ) -> None:
        """Validate endpoint and optional bearer-token configuration."""
        super().__init__(spec, contract)
        self.spec = spec
        self.url = endpoint(spec)
        self.diagnostics = bool(options.get("diagnostics", False))
        token_variable = spec.get("token_env")
        self.token = os.environ.get(token_variable, "") if token_variable else ""
        if token_variable and not self.token:
            raise ValueError(f"Set {token_variable} for this backend")

    def predict(
        self, image: Image.Image, domain: str
    ) -> tuple[dict[str, str], dict[str, Any]]:
        """Encode an image, issue one request, and normalize the returned choices."""
        from ..protocols import questions

        image_buffer = BytesIO()
        image.save(image_buffer, format="PNG")
        encoded_image = base64.b64encode(image_buffer.getvalue()).decode("ascii")
        if self.spec.get("image_encoding") == "base64":
            image_value = encoded_image
        else:
            image_value = f"data:image/png;base64,{encoded_image}"

        payload: dict[str, Any] = {
            "state": "Describe only the visible attributes of the target in this image.",
            "images": [image_value],
            "questions": questions(self.spec, self.contract, domain),
        }
        if model_name := self.spec.get("request_model"):
            payload["model"] = model_name

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
            timeout = self.spec.get("timeout_s", 300)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = json.load(response)
        except urllib.error.HTTPError as error:
            raise self._http_error(error) from None
        except urllib.error.URLError:
            raise RuntimeError(
                "Não foi possível conectar ao servidor. Verifique o serviço "
                "Ollama/llama-server e a configuração do endpoint."
            ) from None

        raw = self._unwrap_response(raw)
        answers = self._extract_answers(raw, domain)
        self.last_metrics = {"inference_calls": 1, "attribute_ms": {}}
        return answers, raw

    def _http_error(self, error: urllib.error.HTTPError) -> RuntimeError:
        """Return detailed errors only for unauthenticated local endpoints."""
        hostname = urllib.parse.urlsplit(self.url).hostname
        local_hosts = {
            "127.0.0.1",
            "localhost",
            "::1",
            "ollama",
            "host.docker.internal",
        }
        if not self.diagnostics or hostname not in local_hosts or self.token:
            return RuntimeError(
                f"HTTP {error.code}; verify System One route, model capability "
                "and context budget"
            )

        details = error.read(16_384).decode("utf-8", errors="replace")
        try:
            details = str(json.loads(details).get("error", details))
        except (ValueError, AttributeError):
            pass
        details = details[:2_000]
        memory_errors = ("insufficient memory", "out of memory", "not enough memory")
        if any(term in details.lower() for term in memory_errors):
            details = f"Memória insuficiente ao carregar o modelo. Detalhes do servidor: {details}"
        return RuntimeError(f"HTTP {error.code}: {details}")

    @staticmethod
    def _unwrap_response(raw: Any) -> dict[str, Any]:
        """Validate the common System One response envelope."""
        if not isinstance(raw, dict) or raw.get("success") is False:
            raise RuntimeError("System One backend reported failure")
        result = raw.get("result", raw)
        if not isinstance(result, dict) or not isinstance(result.get("answers"), dict):
            raise ValueError("Backend returned no answers map")
        return result

    def _extract_answers(self, raw: dict[str, Any], domain: str) -> dict[str, str]:
        """Extract one choice per requested field and reject truncated replies."""
        response_answers = raw["answers"]
        for answer in response_answers.values():
            if isinstance(answer, dict) and answer.get("truncated"):
                raise ValueError("Backend truncated a question or options")

        questions = self.contract["domains"][domain]["questions"]
        return {
            key: str(response_answers.get(key, {}).get("choice", "parse_error"))
            for key in questions
        }

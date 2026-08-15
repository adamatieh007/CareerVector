from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import requests

from careervector.config import DEFAULT_OLLAMA_BASE_URL


class OllamaError(RuntimeError):
    """Base error for local Ollama communication failures."""


class OllamaConnectionError(OllamaError):
    """Raised when CareerVector cannot reach the configured Ollama server."""


class OllamaModelNotFoundError(OllamaError):
    """Raised when the requested local Ollama model is unavailable."""


class OllamaClient:
    """Small dependency-light client for Ollama's local REST API.

    CareerVector intentionally uses the REST API directly instead of adding an
    orchestration framework. That keeps the RAG path easy to inspect, test, and
    explain: retrieve evidence, build a grounded prompt, and call the local LLM.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        *,
        timeout: float = 120.0,
        session: Any | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.session = session or requests.Session()

    def _url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    @staticmethod
    def _error_message(response: Any) -> str:
        try:
            payload = response.json()
        except Exception:
            payload = None
        if isinstance(payload, Mapping) and payload.get("error"):
            return str(payload["error"])
        text = str(getattr(response, "text", "") or "").strip()
        return text or f"HTTP {getattr(response, 'status_code', 'error')}"

    def list_models(self) -> list[str]:
        """Return names of models installed in the local Ollama instance."""
        try:
            response = self.session.get(
                self._url("/api/tags"),
                timeout=min(self.timeout, 5.0),
            )
        except requests.RequestException as exc:
            raise OllamaConnectionError(
                f"Could not reach Ollama at {self.base_url}. Start Ollama and try again."
            ) from exc

        if getattr(response, "status_code", 500) >= 400:
            raise OllamaError(f"Ollama model-list request failed: {self._error_message(response)}")

        try:
            payload = response.json()
        except Exception as exc:
            raise OllamaError("Ollama returned an invalid model-list response.") from exc

        models = payload.get("models", []) if isinstance(payload, Mapping) else []
        names: list[str] = []
        for item in models:
            if not isinstance(item, Mapping):
                continue
            name = item.get("name") or item.get("model")
            if name:
                names.append(str(name))
        return sorted(dict.fromkeys(names))

    def chat(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, str]],
        temperature: float = 0.2,
    ) -> str:
        """Generate one non-streaming chat response with a local Ollama model."""
        payload = {
            "model": model,
            "messages": [dict(message) for message in messages],
            "stream": False,
            "options": {"temperature": float(temperature)},
        }

        try:
            response = self.session.post(
                self._url("/api/chat"),
                json=payload,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise OllamaConnectionError(
                f"Could not reach Ollama at {self.base_url}. Start Ollama and try again."
            ) from exc

        if getattr(response, "status_code", 500) >= 400:
            message = self._error_message(response)
            lowered = message.lower()
            if response.status_code == 404 or "not found" in lowered:
                raise OllamaModelNotFoundError(
                    f"Ollama model '{model}' is not available locally. Run `ollama pull {model}`."
                )
            raise OllamaError(f"Ollama chat request failed: {message}")

        try:
            data = response.json()
            content = data["message"]["content"]
        except (KeyError, TypeError, ValueError) as exc:
            raise OllamaError("Ollama returned an invalid chat response.") from exc

        content = str(content).strip()
        if not content:
            raise OllamaError("Ollama returned an empty response.")
        return content

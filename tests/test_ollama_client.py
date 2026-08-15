import requests
import pytest

from careervector.rag.ollama import (
    OllamaClient,
    OllamaConnectionError,
    OllamaModelNotFoundError,
)


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self):
        self.last_post = None

    def get(self, url, timeout):
        assert url.endswith("/api/tags")
        return FakeResponse(
            payload={"models": [{"name": "gemma3:4b"}, {"model": "qwen3:4b"}]}
        )

    def post(self, url, json, timeout):
        assert url.endswith("/api/chat")
        self.last_post = json
        return FakeResponse(payload={"message": {"content": "Grounded answer [CV1]"}})


def test_ollama_client_lists_models_and_chats() -> None:
    session = FakeSession()
    client = OllamaClient("http://localhost:11434/", session=session)
    assert client.list_models() == ["gemma3:4b", "qwen3:4b"]
    answer = client.chat(
        model="gemma3:4b",
        messages=[{"role": "user", "content": "hello"}],
    )
    assert answer == "Grounded answer [CV1]"
    assert session.last_post["stream"] is False
    assert session.last_post["model"] == "gemma3:4b"


def test_ollama_client_maps_connection_errors() -> None:
    class BrokenSession:
        def get(self, *args, **kwargs):
            raise requests.ConnectionError("offline")

    with pytest.raises(OllamaConnectionError):
        OllamaClient(session=BrokenSession()).list_models()


def test_ollama_client_maps_missing_model() -> None:
    class MissingModelSession:
        def post(self, *args, **kwargs):
            return FakeResponse(status_code=404, payload={"error": "model not found"})

    client = OllamaClient(session=MissingModelSession())
    with pytest.raises(OllamaModelNotFoundError):
        client.chat(model="missing", messages=[{"role": "user", "content": "hi"}])

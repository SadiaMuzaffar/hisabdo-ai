import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import llm_service

client = TestClient(app)


@pytest.fixture(autouse=True)
def no_real_key(monkeypatch):
    """Tests must never call the real AI or need a real key."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_chat_valid_returns_schema():
    r = client.post("/api/chat", json={"conversation_id": "conv_001", "message": "What can you help me with?"})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"conversation_id", "message"}
    assert body["conversation_id"] == "conv_001"
    assert body["message"]


@pytest.mark.parametrize(
    "payload",
    [
        {"conversation_id": "conv_001", "message": ""},
        {"conversation_id": "conv_001", "message": "   "},
        {"conversation_id": "", "message": "hi"},
        {"conversation_id": "conv_001"},
        {},
        {"conversation_id": "conv_001", "message": "x" * 4001},
    ],
)
def test_chat_invalid_input_returns_422(payload):
    assert client.post("/api/chat", json=payload).status_code == 422


def test_llm_failure_is_safe(monkeypatch):
    """If the model call breaks, the API still answers 200 with a safe message, no secrets."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-test")

    def boom(*a, **k):
        raise RuntimeError("secret-should-not-leak fake-key-for-test")

    monkeypatch.setattr("google.genai.Client", boom)
    r = client.post("/api/chat", json={"conversation_id": "c1", "message": "hello"})
    assert r.status_code == 200
    assert r.json()["message"] == llm_service.FALLBACK_REPLY
    assert "fake-key-for-test" not in r.text

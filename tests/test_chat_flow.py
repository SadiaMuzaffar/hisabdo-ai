"""User -> API -> LLM -> response flow, context handling and error handling."""
import json
import logging

import pytest

from app.prompts.system_prompt import SYSTEM_PROMPT
from app.services.chat_service import FALLBACK_MESSAGES
from app.services.llm_service import (
    LLMProviderError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
)


def chat(client, message="What can you help me with?", cid="conv_001"):
    return client.post("/api/chat", json={"conversation_id": cid, "message": message})


# ---------- health ----------
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


# ---------- full flow ----------
def test_user_to_api_to_llm_to_response(client, provider):
    r = chat(client)
    assert r.status_code == 200
    assert r.json() == {
        "conversation_id": "conv_001",
        "message": "Hello from fake LLM",
        "status": "ok",
        "error_code": None,
    }
    assert len(provider.calls) == 1
    system_prompt, messages, timeout = provider.calls[0]
    assert [(m.role, m.content) for m in messages] == [("user", "What can you help me with?")]
    assert timeout == 7.5  # configured timeout reaches the provider


def test_system_prompt_applied_on_every_call(client, provider):
    chat(client, "first")
    chat(client, "second")
    assert len(provider.calls) == 2
    assert all(call[0] == SYSTEM_PROMPT for call in provider.calls)


def test_message_is_trimmed_before_sending(client, provider):
    chat(client, "   padded message   ")
    assert provider.calls[0][1][-1].content == "padded message"


# ---------- conversation context ----------
def test_history_is_preserved_across_turns(client, provider):
    chat(client, "My shop sells shoes.")
    provider.reply = "Noted."
    chat(client, "What do I sell?")
    _, messages, _ = provider.calls[1]
    assert [(m.role, m.content) for m in messages] == [
        ("user", "My shop sells shoes."),
        ("assistant", "Hello from fake LLM"),
        ("user", "What do I sell?"),
    ]


def test_conversations_are_isolated(client, provider):
    chat(client, "secret of A", cid="A")
    chat(client, "hello from B", cid="B")
    _, messages, _ = provider.calls[1]
    assert [m.content for m in messages] == ["hello from B"]


def test_history_is_capped(client, provider):
    for i in range(30):
        chat(client, f"message {i}")
    _, messages, _ = provider.calls[-1]
    assert len(messages) <= 21  # at most 20 stored + the current message
    assert messages[-1].content == "message 29"


# ---------- error handling ----------
@pytest.mark.parametrize(
    "error, code",
    [
        (LLMTimeoutError("t"), "timeout"),
        (LLMRateLimitError("r"), "rate_limited"),
        (LLMProviderError("p"), "provider_error"),
        (LLMResponseError("b"), "bad_response"),
        (RuntimeError("unexpected bug"), "provider_error"),
    ],
)
def test_provider_failures_return_clear_fallback(client, provider, error, code):
    provider.error = error
    r = chat(client)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "fallback"
    assert body["error_code"] == code
    assert body["message"] == FALLBACK_MESSAGES[code]
    assert "Traceback" not in r.text


def test_failed_turn_is_not_stored(client, provider):
    provider.error = LLMTimeoutError("t")
    chat(client, "lost message")
    provider.error = None
    chat(client, "retry message")
    _, messages, _ = provider.calls[-1]
    assert [m.content for m in messages] == ["retry message"]


def test_recovers_after_failure(client, provider):
    provider.error = LLMRateLimitError("r")
    assert chat(client).json()["status"] == "fallback"
    provider.error = None
    assert chat(client).json()["status"] == "ok"


# ---------- validation ----------
@pytest.mark.parametrize(
    "payload",
    [
        {"conversation_id": "c", "message": ""},
        {"conversation_id": "c", "message": "   "},
        {"conversation_id": "", "message": "hi"},
        {"conversation_id": "  ", "message": "hi"},
        {"conversation_id": "c"},
        {"message": "hi"},
        {},
        {"conversation_id": "c", "message": "x" * 4001},
        {"conversation_id": "c" * 101, "message": "hi"},
    ],
)
def test_invalid_requests_return_422_and_never_reach_llm(client, provider, payload):
    assert client.post("/api/chat", json=payload).status_code == 422
    assert provider.calls == []


# ---------- logging ----------
def _events(caplog):
    out = []
    for rec in caplog.records:
        try:
            out.append(json.loads(rec.getMessage()))
        except ValueError:
            pass
    return out


def test_success_log_has_metadata_but_no_message_text(client, caplog):
    caplog.set_level(logging.INFO)
    chat(client, "my private customer note 12345")
    events = [e for e in _events(caplog) if e["event"] == "chat_completed"]
    assert len(events) == 1
    event = events[0]
    assert event["conversation_id"] == "conv_001"
    assert {"provider", "latency_ms", "request_chars", "reply_chars"} <= set(event)
    assert "my private customer note 12345" not in caplog.text
    assert "Hello from fake LLM" not in caplog.text


def test_failure_log_has_error_code_but_no_secrets(client, provider, caplog):
    caplog.set_level(logging.INFO)
    provider.error = LLMProviderError("boom key=sk-SECRET-12345")
    chat(client, "hello")
    events = [e for e in _events(caplog) if e["event"] == "chat_failed"]
    assert len(events) == 1
    assert events[0]["error_code"] == "provider_error"
    assert "sk-SECRET-12345" not in caplog.text


def test_unexpected_exception_is_logged_without_its_message(client, provider, caplog):
    caplog.set_level(logging.INFO)
    provider.error = RuntimeError("leaks sk-SECRET-98765")
    r = chat(client, "hello")
    assert r.json()["error_code"] == "provider_error"
    assert "sk-SECRET-98765" not in caplog.text


def test_unhandled_error_returns_generic_500(provider):
    from fastapi.testclient import TestClient

    from app.main import app
    from app.services.chat_service import get_chat_service

    class Broken:
        def handle(self, *a, **k):
            raise RuntimeError("internal detail sk-SECRET-777")

    app.dependency_overrides[get_chat_service] = lambda: Broken()
    try:
        r = TestClient(app, raise_server_exceptions=False).post(
            "/api/chat", json={"conversation_id": "c", "message": "hi"}
        )
    finally:
        app.dependency_overrides.clear()
    assert r.status_code == 500
    assert r.json() == {"detail": "Internal server error"}
    assert "sk-SECRET-777" not in r.text

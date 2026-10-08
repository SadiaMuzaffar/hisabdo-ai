"""Provider layer: error translation, factory and conversation store."""
import httpx
import pytest
from google import genai
from google.genai import errors

from app.config import Settings
from app.services.conversation_store import ConversationStore
from app.services.llm_service import (
    ChatMessage,
    GeminiProvider,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
    MockProvider,
    get_provider,
)

MESSAGES = [ChatMessage("user", "hi")]


def settings(**kw):
    base = dict(
        llm_provider="auto",
        gemini_api_key=None,
        gemini_model="test-model",
        llm_timeout_seconds=5.0,
        history_max_messages=20,
    )
    base.update(kw)
    return Settings(**base)


class FakeModels:
    def __init__(self, behavior):
        self.behavior = behavior
        self.kwargs = None

    def generate_content(self, **kwargs):
        self.kwargs = kwargs
        if isinstance(self.behavior, Exception):
            raise self.behavior
        return self.behavior


class FakeClient:
    last = None

    def __init__(self, behavior, **kwargs):
        self.models = FakeModels(behavior)
        self.init_kwargs = kwargs
        FakeClient.last = self


def patch_client(monkeypatch, behavior):
    monkeypatch.setattr(genai, "Client", lambda **kw: FakeClient(behavior, **kw))


class Resp:
    def __init__(self, text):
        self.text = text


def api_error(code, status):
    return errors.ClientError(code, {"error": {"message": "x", "status": status}})


def test_gemini_success_sends_system_prompt_and_history(monkeypatch):
    patch_client(monkeypatch, Resp("  answer  "))
    history = [ChatMessage("user", "a"), ChatMessage("assistant", "b"), ChatMessage("user", "c")]
    out = GeminiProvider("key", "test-model").generate("SYSTEM", history, 3.0)
    assert out == "answer"
    kwargs = FakeClient.last.models.kwargs
    assert kwargs["model"] == "test-model"
    assert kwargs["config"].system_instruction == "SYSTEM"
    assert [c.role for c in kwargs["contents"]] == ["user", "model", "user"]
    assert FakeClient.last.init_kwargs["http_options"].timeout == 3000  # seconds -> ms


@pytest.mark.parametrize(
    "behavior, expected",
    [
        (api_error(429, "RESOURCE_EXHAUSTED"), LLMRateLimitError),
        (api_error(504, "DEADLINE_EXCEEDED"), LLMTimeoutError),
        (api_error(403, "PERMISSION_DENIED"), LLMProviderError),
        (api_error(404, "NOT_FOUND"), LLMProviderError),
        (httpx.ReadTimeout("slow"), LLMTimeoutError),
        (ConnectionError("down"), LLMProviderError),
        (Resp(None), LLMResponseError),
        (Resp("   "), LLMResponseError),
    ],
)
def test_gemini_errors_are_translated(monkeypatch, behavior, expected):
    patch_client(monkeypatch, behavior)
    with pytest.raises(expected):
        GeminiProvider("key", "m").generate("S", MESSAGES, 3.0)


def test_error_message_never_contains_key_or_body(monkeypatch):
    err = errors.ClientError(403, {"error": {"message": "bad key sk-LEAK-555", "status": "PERMISSION_DENIED"}})
    patch_client(monkeypatch, err)
    with pytest.raises(LLMProviderError) as info:
        GeminiProvider("sk-LEAK-555", "m").generate("S", MESSAGES, 3.0)
    assert "sk-LEAK-555" not in str(info.value)
    assert "403" in str(info.value)


def test_factory_chooses_provider():
    assert isinstance(get_provider(settings()), MockProvider)  # auto, no key
    assert isinstance(get_provider(settings(gemini_api_key="k")), GeminiProvider)  # auto, key
    assert isinstance(get_provider(settings(llm_provider="mock", gemini_api_key="k")), MockProvider)
    assert isinstance(get_provider(settings(llm_provider="gemini")), MockProvider)  # no key -> safe mock
    assert isinstance(get_provider(settings(llm_provider="nonsense")), MockProvider)


def test_mock_provider_echoes_last_user_message():
    msgs = [ChatMessage("user", "one"), ChatMessage("assistant", "x"), ChatMessage("user", "two")]
    assert MockProvider().generate("S", msgs, 1.0) == "[mock] You said: two"


def test_store_caps_messages_and_conversations():
    store = ConversationStore(max_messages=4, max_conversations=2)
    for i in range(5):
        store.append_exchange("a", f"u{i}", f"r{i}")
    assert [m.content for m in store.history("a")] == ["u3", "r3", "u4", "r4"]
    store.append_exchange("b", "u", "r")
    store.append_exchange("c", "u", "r")  # evicts the least recently used ("a")
    assert store.history("a") == []
    assert store.history("c") != []

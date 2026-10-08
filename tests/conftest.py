import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.prompts.system_prompt import SYSTEM_PROMPT
from app.services.chat_service import ChatService, get_chat_service
from app.services.conversation_store import ConversationStore
from app.services.llm_service import LLMProvider


class FakeProvider(LLMProvider):
    """Records every call so tests can inspect what reached the LLM."""

    name = "fake"

    def __init__(self, reply="Hello from fake LLM", error=None):
        self.reply = reply
        self.error = error
        self.calls = []  # list of (system_prompt, messages, timeout)

    def generate(self, system_prompt, messages, timeout):
        self.calls.append((system_prompt, list(messages), timeout))
        if self.error is not None:
            raise self.error
        return self.reply


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Tests never need, read or print a real key."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)


@pytest.fixture
def provider():
    return FakeProvider()


@pytest.fixture
def client(provider):
    service = ChatService(provider, ConversationStore(max_messages=20), system_prompt=SYSTEM_PROMPT, timeout=7.5)
    app.dependency_overrides[get_chat_service] = lambda: service
    yield TestClient(app, raise_server_exceptions=False)
    app.dependency_overrides.clear()

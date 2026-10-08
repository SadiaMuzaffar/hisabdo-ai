"""Provider-agnostic LLM interface.

The chat service only knows the `LLMProvider` interface and the `LLMError`
family. Each provider translates its own SDK errors into these, so error
handling does not depend on which model vendor is configured.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

from app.config import Settings

logger = logging.getLogger(__name__)

Role = Literal["user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str


class LLMError(Exception):
    """Base error. `code` is a stable, safe-to-log identifier."""

    code = "provider_error"


class LLMTimeoutError(LLMError):
    code = "timeout"


class LLMRateLimitError(LLMError):
    code = "rate_limited"


class LLMProviderError(LLMError):
    code = "provider_error"


class LLMResponseError(LLMError):
    """The provider answered, but the answer was empty or malformed."""

    code = "bad_response"


class LLMProvider(ABC):
    name: str

    @abstractmethod
    def generate(self, system_prompt: str, messages: list[ChatMessage], timeout: float) -> str:
        """Return the assistant reply text, or raise an LLMError subclass."""


class MockProvider(LLMProvider):
    """Used when no API key is configured, so the service always runs."""

    name = "mock"

    def generate(self, system_prompt: str, messages: list[ChatMessage], timeout: float) -> str:
        last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        return f"[mock] You said: {last_user}"


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, api_key: str, model: str):
        self._api_key = api_key
        self._model = model

    def generate(self, system_prompt: str, messages: list[ChatMessage], timeout: float) -> str:
        import httpx
        from google import genai
        from google.genai import errors, types

        try:
            client = genai.Client(
                api_key=self._api_key,
                http_options=types.HttpOptions(timeout=int(timeout * 1000)),  # milliseconds
            )
            contents = [
                types.Content(
                    role="user" if m.role == "user" else "model",
                    parts=[types.Part(text=m.content)],
                )
                for m in messages
            ]
            response = client.models.generate_content(
                model=self._model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                ),
            )
        except errors.APIError as exc:
            code = getattr(exc, "code", None)
            status = getattr(exc, "status", None)
            # Only the type, HTTP code and status go into the message: no body, no key.
            detail = f"{type(exc).__name__} code={code} status={status}"
            if code == 429:
                raise LLMRateLimitError(detail) from None
            if code in (408, 504):
                raise LLMTimeoutError(detail) from None
            raise LLMProviderError(detail) from None
        except (httpx.TimeoutException, TimeoutError) as exc:
            raise LLMTimeoutError(type(exc).__name__) from None
        except Exception as exc:  # noqa: BLE001 - never let an SDK error escape raw
            raise LLMProviderError(type(exc).__name__) from None

        text = getattr(response, "text", None)
        if not isinstance(text, str) or not text.strip():
            raise LLMResponseError("empty or malformed response")
        return text.strip()


def get_provider(settings: Settings) -> LLMProvider:
    choice = settings.llm_provider
    if choice == "mock":
        return MockProvider()
    if choice in ("gemini", "auto"):
        if settings.gemini_api_key:
            return GeminiProvider(settings.gemini_api_key, settings.gemini_model)
        if choice == "gemini":
            logger.warning("LLM_PROVIDER=gemini but GEMINI_API_KEY is not set; using mock provider")
        return MockProvider()
    logger.warning("Unknown LLM_PROVIDER value; using mock provider")
    return MockProvider()

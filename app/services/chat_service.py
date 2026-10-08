"""Chat service: validate -> build payload -> call LLM -> normalize -> persist -> respond."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

from app.config import get_settings
from app.logging_config import log_event
from app.prompts.system_prompt import SYSTEM_PROMPT
from app.services import context_service, rag_service
from app.services.conversation_store import ConversationStore
from app.services.llm_service import (
    ChatMessage,
    LLMError,
    LLMProvider,
    get_provider,
)

logger = logging.getLogger(__name__)

FALLBACK_MESSAGES = {
    "timeout": "The AI service took too long to respond. Please try again.",
    "rate_limited": "The AI service is busy right now. Please try again in a moment.",
    "provider_error": "Sorry, the AI service is temporarily unavailable.",
    "bad_response": "Sorry, I could not produce a valid reply. Please try again.",
}


class InvalidChatRequest(ValueError):
    """The request is empty or blank after cleaning."""


@dataclass(frozen=True)
class ChatResult:
    conversation_id: str
    message: str
    status: str  # "ok" or "fallback"
    error_code: str | None = None


class ChatService:
    def __init__(
        self,
        provider: LLMProvider,
        store: ConversationStore,
        system_prompt: str = SYSTEM_PROMPT,
        timeout: float = 20.0,
        context_builder: Callable[[str], dict] = context_service.build_context,
        retriever: Callable[[str], list[dict]] = rag_service.retrieve,
    ):
        self._provider = provider
        self._store = store
        self._system_prompt = system_prompt
        self._timeout = timeout
        self._context_builder = context_builder
        self._retriever = retriever

    def handle(self, conversation_id: str, message: str) -> ChatResult:
        # 1. Validate
        cid = (conversation_id or "").strip()
        text = (message or "").strip()
        if not cid or not text:
            raise InvalidChatRequest("conversation_id and message must not be blank")

        started = time.perf_counter()

        # 2. Build the LLM payload: system prompt + history + current message
        context = self._context_builder(cid)
        chunks = self._retriever(text)
        system_prompt = self._compose_system_prompt(context, chunks)
        history = self._store.history(cid)
        payload = [*history, ChatMessage("user", text)]

        # 3 + 4. Call the provider and normalize the reply
        try:
            reply = self._provider.generate(system_prompt, payload, self._timeout)
        except LLMError as exc:
            return self._fallback(cid, exc.code, started, len(history), len(text))
        except Exception:  # noqa: BLE001 - a provider bug must not become a 500
            return self._fallback(cid, "provider_error", started, len(history), len(text), unexpected=True)

        # 5. Persist the finished turn and log safe metadata only
        self._store.append_exchange(cid, text, reply)
        log_event(
            logger,
            "chat_completed",
            conversation_id=cid,
            provider=self._provider.name,
            latency_ms=self._elapsed_ms(started),
            history_messages=len(history),
            request_chars=len(text),
            reply_chars=len(reply),
            used_kb=bool(chunks),
        )
        return ChatResult(cid, reply, "ok")

    def _fallback(
        self, cid: str, code: str, started: float, history_len: int, request_chars: int, unexpected: bool = False
    ) -> ChatResult:
        # 6. Clear fallback; the failed turn is not stored, so history stays consistent
        log_event(
            logger,
            "chat_failed",
            level=logging.ERROR if unexpected else logging.WARNING,
            conversation_id=cid,
            provider=self._provider.name,
            error_code=code,
            unexpected=unexpected,
            latency_ms=self._elapsed_ms(started),
            history_messages=history_len,
            request_chars=request_chars,
        )
        message = FALLBACK_MESSAGES.get(code, FALLBACK_MESSAGES["provider_error"])
        return ChatResult(cid, message, "fallback", code)

    def _compose_system_prompt(self, context: dict, chunks: list[dict]) -> str:
        parts = [self._system_prompt]
        extra = {k: v for k, v in (context or {}).items() if k != "conversation_id"}
        if extra:
            parts.append(f"CONTEXT:\n{extra}")
        if chunks:
            lines = "\n".join(f"- ({c.get('source', 'unknown')}) {c.get('text', '')}" for c in chunks)
            parts.append(f"KNOWLEDGE (use only this for factual claims):\n{lines}")
        return "\n\n".join(parts)

    @staticmethod
    def _elapsed_ms(started: float) -> int:
        return int((time.perf_counter() - started) * 1000)


@lru_cache(maxsize=1)
def get_chat_service() -> ChatService:
    settings = get_settings()
    return ChatService(
        provider=get_provider(settings),
        store=ConversationStore(max_messages=settings.history_max_messages),
        timeout=settings.llm_timeout_seconds,
    )

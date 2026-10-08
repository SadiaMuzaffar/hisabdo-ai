"""In-memory conversation history.

Prototype storage: history is lost on restart. The interface (`history`,
`append_exchange`) is what a database-backed store would implement later.
"""
from __future__ import annotations

import threading
from collections import OrderedDict

from app.services.llm_service import ChatMessage


class ConversationStore:
    def __init__(self, max_messages: int = 20, max_conversations: int = 1000):
        self._max_messages = max(2, max_messages)
        self._max_conversations = max_conversations
        self._data: OrderedDict[str, list[ChatMessage]] = OrderedDict()
        self._lock = threading.Lock()

    def history(self, conversation_id: str) -> list[ChatMessage]:
        with self._lock:
            return list(self._data.get(conversation_id, []))

    def append_exchange(self, conversation_id: str, user_text: str, assistant_text: str) -> None:
        """Store a completed turn. Failed turns are never stored."""
        with self._lock:
            messages = self._data.pop(conversation_id, [])
            messages.append(ChatMessage("user", user_text))
            messages.append(ChatMessage("assistant", assistant_text))
            self._data[conversation_id] = messages[-self._max_messages:]
            while len(self._data) > self._max_conversations:
                self._data.popitem(last=False)  # drop the least recently used conversation

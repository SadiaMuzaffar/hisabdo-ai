"""Evaluation / safe operational logging.

Logs only non-sensitive metadata (ids, lengths, outcome). Never logs API keys
or full message content.
"""
import logging

logger = logging.getLogger("hisabdo.evaluation")


def record_interaction(conversation_id: str, message_length: int, used_kb: bool, ok: bool) -> None:
    logger.info(
        "chat conversation_id=%s msg_len=%d used_kb=%s ok=%s",
        conversation_id,
        message_length,
        used_kb,
        ok,
    )

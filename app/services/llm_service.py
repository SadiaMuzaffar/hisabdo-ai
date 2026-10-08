import logging
import os

from dotenv import load_dotenv

from app.prompts.system_prompt import SYSTEM_PROMPT

load_dotenv()
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-flash-latest"
FALLBACK_REPLY = "Sorry, the AI service is temporarily unavailable."


def _build_prompt(message: str, context: dict | None, chunks: list[dict] | None) -> str:
    parts = []
    if context:
        parts.append(f"CONTEXT:\n{context}")
    if chunks:
        knowledge = "\n".join(f"- ({c.get('source', 'unknown')}) {c.get('text', '')}" for c in chunks)
        parts.append(f"KNOWLEDGE:\n{knowledge}")
    parts.append(f"USER MESSAGE:\n{message}")
    return "\n\n".join(parts)


def generate_reply(message: str, context: dict | None = None, chunks: list[dict] | None = None) -> tuple[str, bool]:
    """Return (reply_text, ok).

    - No API key set  -> mock reply (so the prototype always runs).
    - Model call fails -> safe fallback text, only the error type/code is logged.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return f"[mock] You said: {message}", True

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=os.getenv("GEMINI_MODEL", DEFAULT_MODEL),
            contents=_build_prompt(message, context, chunks),
            config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
        )
        return (response.text or "I could not generate a reply."), True
    except Exception as exc:  # noqa: BLE001
        # Log error type, HTTP code and status only. Never log the key.
        logger.error(
            "LLM call failed: %s code=%s status=%s",
            type(exc).__name__,
            getattr(exc, "code", ""),
            getattr(exc, "status", ""),
        )
        return FALLBACK_REPLY, False

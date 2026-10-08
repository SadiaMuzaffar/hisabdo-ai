"""Runtime settings, read from environment variables (and a local .env file)."""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"


def _float_env(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, default))
        return value if value > 0 else default
    except ValueError:
        return default


def _int_env(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, default))
        return value if value > 0 else default
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    llm_provider: str  # "gemini", "mock" or "auto" (gemini when a key exists, else mock)
    gemini_api_key: str | None
    gemini_model: str
    llm_timeout_seconds: float
    history_max_messages: int


def get_settings() -> Settings:
    return Settings(
        llm_provider=os.getenv("LLM_PROVIDER", "auto").strip().lower(),
        gemini_api_key=(os.getenv("GEMINI_API_KEY") or "").strip() or None,
        gemini_model=os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip(),
        llm_timeout_seconds=_float_env("LLM_TIMEOUT_SECONDS", 20.0),
        history_max_messages=_int_env("HISTORY_MAX_MESSAGES", 20),
    )

"""Structured (JSON) logging helpers.

Rule: log ids, sizes, timings and error codes only. Never log API keys,
message text, or provider response bodies.
"""
import json
import logging


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(message)s")


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO, **fields) -> None:
    logger.log(level, json.dumps({"event": event, **fields}, default=str))

import logging

from fastapi import FastAPI

from app.api.chat import router as chat_router

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="HisabDo AI Service",
    description="AI Business Assistant backend (initial prototype).",
    version="0.1.0",
)
app.include_router(chat_router)


@app.get("/health", tags=["system"])
def health():
    """Service health check."""
    return {"status": "ok"}

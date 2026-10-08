import logging

from fastapi import FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.chat import router as chat_router
from app.logging_config import configure_logging, log_event

configure_logging()
logger = logging.getLogger("hisabdo.api")

app = FastAPI(
    title="HisabDo AI Service",
    description="AI Business Assistant backend (chat service + LLM integration).",
    version="0.2.0",
)
app.include_router(chat_router)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    # Log where it failed, never the submitted values.
    fields = [".".join(str(p) for p in e.get("loc", ())) for e in exc.errors()]
    log_event(logger, "request_invalid", level=logging.WARNING, path=request.url.path, fields=fields)
    return await request_validation_exception_handler(request, exc)


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception):
    log_event(logger, "unhandled_error", level=logging.ERROR, path=request.url.path, error_type=type(exc).__name__)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/health", tags=["system"])
def health():
    """Service health check."""
    return {"status": "ok"}

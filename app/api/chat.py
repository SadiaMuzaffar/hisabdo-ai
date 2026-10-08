from dataclasses import asdict
from typing import Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field, field_validator

from app.services.chat_service import ChatService, get_chat_service

router = APIRouter(prefix="/api", tags=["chat"])


class ChatRequest(BaseModel):
    conversation_id: str = Field(..., min_length=1, max_length=100)
    message: str = Field(..., min_length=1, max_length=4000)

    @field_validator("conversation_id", "message")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be blank")
        return v


class ChatResponse(BaseModel):
    conversation_id: str
    message: str
    status: Literal["ok", "fallback"] = "ok"
    error_code: str | None = None


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, service: ChatService = Depends(get_chat_service)) -> ChatResponse:
    """Send a message. Provider failures return 200 with status "fallback", never a stack trace."""
    return ChatResponse(**asdict(service.handle(req.conversation_id, req.message)))

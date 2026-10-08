from fastapi import APIRouter
from pydantic import BaseModel, Field, field_validator

from app.services import context_service, evaluation_service, llm_service, rag_service

router = APIRouter(prefix="/api", tags=["chat"])


class ChatRequest(BaseModel):
    conversation_id: str = Field(..., min_length=1, max_length=100)
    message: str = Field(..., min_length=1, max_length=4000)

    @field_validator("message")
    @classmethod
    def message_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message must not be blank")
        return v


class ChatResponse(BaseModel):
    conversation_id: str
    message: str


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """Chat flow: validate -> context -> KB/RAG -> LLM -> log -> respond."""
    context = context_service.build_context(req.conversation_id)
    chunks = rag_service.retrieve(req.message)
    reply, ok = llm_service.generate_reply(req.message, context, chunks)
    evaluation_service.record_interaction(req.conversation_id, len(req.message), bool(chunks), ok)
    return ChatResponse(conversation_id=req.conversation_id, message=reply)

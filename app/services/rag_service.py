"""RAG retrieval service (placeholder).

Planned: embed the question, search the vector store, return the top chunks
with source metadata. The prototype returns no chunks, so the assistant
answers without KB content and must not invent facts.
"""


def retrieve(question: str, top_k: int = 4) -> list[dict]:
    return []

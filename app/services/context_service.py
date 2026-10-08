"""Context Builder.

Prototype: returns only the conversation id. Later this will collect user,
business, customer, task and conversation context, and check authorization
before anything is sent to the model.
"""


def build_context(conversation_id: str) -> dict:
    return {"conversation_id": conversation_id}

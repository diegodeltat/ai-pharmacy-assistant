from typing import TypedDict


class AgentState(TypedDict, total=False):
    user_id: str
    question: str
    intent: str
    response: str
    safety_blocked: bool
    sources: list[dict]
    warnings: list[str]
    rag_query: str

    last_intent: str
    last_commune: str
    last_medication_question: str
    last_pharmacies: list[dict]

from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    user_id: str = Field(
        min_length=1,
        description="Identificador del usuario.",
    )

    pregunta: str = Field(
        min_length=1,
        max_length=2000,
        description="Pregunta enviada al asistente.",
    )

    rerank: bool | None = Field(
        default=None,
        description=(
            "Activa o desactiva reranking para esta consulta. Si se omite, "
            "usa RAG_RERANK_ENABLED."
        ),
    )


class SourceCitation(BaseModel):
    source_type: Literal["minsal", "rag"]
    title: str
    reference: str
    score: float | None = None
    live_data: bool | None = None


class ChatResponse(BaseModel):
    user_id: str
    respuesta: str
    intent: str
    safety_blocked: bool
    sources: list[SourceCitation] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    rerank_enabled: bool = False
    rerank_applied: bool = False


class PharmacyRecord(BaseModel):
    nombre: str
    comuna: str
    direccion: str
    horario: str
    telefono: str
    fecha: str


class RagSearchResult(BaseModel):
    content: str
    title: str
    reference: str
    score: float
    metadata: dict = Field(default_factory=dict)

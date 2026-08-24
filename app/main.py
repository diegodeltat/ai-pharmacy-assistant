"""API pública del asistente de farmacias y fichas educativas."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from app.agent.graph import build_graph
from app.agent.state import AgentState
from app.config import get_settings
from app.memory.checkpointer import checkpointer_context
from app.models import ChatRequest, ChatResponse
from app.safety.output_guardrail import apply_output_guardrail
from app.tools.minsal_tool import close_http_client


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with checkpointer_context() as checkpointer:
        app.state.graph = build_graph(checkpointer)

        try:
            yield
        finally:
            await close_http_client()


app = FastAPI(
    title="Asistente Farmacias IA",
    description=(
        "Asistente informativo para consultar farmacias de turno y fichas "
        "educativas de medicamentos."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health_check() -> dict:
    settings = get_settings()

    return {
        "status": "ok",
        "service": "asistente-farmacias-ia",
        "rag_configured": settings.rag_configured,
        "rag_rerank_enabled": settings.rag_rerank_enabled,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    http_request: Request,
) -> ChatResponse:

    settings = get_settings()
    initial_state: AgentState = {
        "user_id": request.user_id,
        "question": request.pregunta,
        "safety_blocked": False,
        "sources": [],
        "warnings": [],
        "rerank_enabled": (
            settings.rag_rerank_enabled
            if request.rerank is None
            else request.rerank
        ),
        "rerank_applied": False,
    }

    config = {
        "configurable": {
            "thread_id": request.user_id,
        }
    }

    try:
        result = await http_request.app.state.graph.ainvoke(
            initial_state,
            config=config,
        )

    except Exception as error:
        logger.exception("Error no controlado al ejecutar el grafo")

        raise HTTPException(
            status_code=503,
            detail="El asistente no está disponible temporalmente.",
        ) from error

    response, output_blocked = apply_output_guardrail(
        result["response"]
    )

    warnings = list(
        dict.fromkeys(result.get("warnings", []))
    )

    return ChatResponse(
        user_id=result["user_id"],
        respuesta=response,
        intent="safety" if output_blocked else result["intent"],
        safety_blocked=(
            result.get("safety_blocked", False)
            or output_blocked
        ),
        sources=(
            []
            if output_blocked
            else result.get("sources", [])
        ),
        warnings=warnings,
        rerank_enabled=result.get("rerank_enabled", False),
        rerank_applied=result.get("rerank_applied", False),
    )

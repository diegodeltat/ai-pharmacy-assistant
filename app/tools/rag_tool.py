"""Tool RAG para responder exclusivamente desde el corpus educativo."""

import asyncio
from functools import lru_cache

from langchain_openai import ChatOpenAI

from app.config import get_settings
from app.models import SourceCitation
from app.rag.retriever import retrieve_medications
from app.rag.vector_store import ensure_rag_configuration
from app.safety.output_guardrail import apply_output_guardrail


RAG_WARNING = (
    "Corpus educativo ficticio; no constituye información clínica ni "
    "reemplaza la evaluación de un profesional de salud."
)

RAG_SYSTEM_PROMPT = """
Eres un asistente informativo sobre fichas educativas de medicamentos.
Responde siempre en español y utiliza exclusivamente el contexto recuperado.
El corpus es ficticio y no es una fuente clínica autoritativa.
No diagnostiques, prescribas, recomiendes medicamentos ni indiques dosis.
No conviertas las concentraciones de una presentación en instrucciones de uso.
Cita cada afirmación mediante las etiquetas [Fuente 1], [Fuente 2], etc.
Si el contexto no permite responder, indica que no se encontró información
suficiente en el corpus educativo. Termina con una advertencia breve que diga
que la información es ficticia y educativa.
""".strip()


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    ensure_rag_configuration()
    settings = get_settings()
    return ChatOpenAI(
        model=settings.chat_model,
        temperature=0,
        api_key=settings.openai_api_key,
    )


def format_context(results: list) -> str:
    blocks = []
    for index, result in enumerate(results, start=1):
        blocks.append(
            f"[Fuente {index}]\n"
            f"Referencia: {result.reference}\n"
            f"Relevancia: {result.score:.4f}\n\n"
            f"{result.content}"
        )
    return "\n\n---\n\n".join(blocks)


async def answer_medication_question(question: str) -> dict:
    try:
        results = await asyncio.to_thread(retrieve_medications, question)
    except RuntimeError as error:
        return {
            "answer": (
                "El módulo de fichas de medicamentos no está disponible en "
                f"este momento. {error}"
            ),
            "sources": [],
            "warnings": [RAG_WARNING],
            "error": True,
        }
    except Exception:
        return {
            "answer": (
                "No fue posible consultar las fichas de medicamentos en este "
                "momento. Intenta nuevamente más tarde."
            ),
            "sources": [],
            "warnings": [RAG_WARNING],
            "error": True,
        }

    if not results:
        return {
            "answer": (
                "No se encontró información suficiente en el corpus "
                "educativo para responder esa consulta."
            ),
            "sources": [],
            "warnings": [RAG_WARNING],
            "error": False,
        }

    prompt = (
        f"Solicitud del usuario:\n{question}\n\n"
        f"Contexto recuperado:\n{format_context(results)}\n\n"
        "Responde en un máximo de tres párrafos y conserva las citas."
    )
    try:
        response = await get_llm().ainvoke(
            [
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )
        answer, blocked = apply_output_guardrail(str(response.content))
    except Exception:
        return {
            "answer": (
                "El proveedor del modelo no pudo generar la respuesta. "
                "Intenta nuevamente más tarde."
            ),
            "sources": [],
            "warnings": [RAG_WARNING],
            "error": True,
        }

    sources = [
        SourceCitation(
            source_type="rag",
            title=result.title,
            reference=result.reference,
            score=result.score,
        ).model_dump()
        for result in results
    ]
    return {
        "answer": answer,
        "sources": [] if blocked else sources,
        "warnings": [RAG_WARNING],
        "error": False,
        "safety_blocked": blocked,
    }

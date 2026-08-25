"""Tool RAG para responder exclusivamente desde el corpus educativo."""

import asyncio
import json
import logging
import re
import time
from functools import lru_cache

from langchain_openai import ChatOpenAI

from app.config import get_settings
from app.models import SourceCitation
from app.rag.entity_resolver import normalize_drug_name, resolve_medication_entity
from app.rag.retriever import retrieve_medications
from app.rag.vector_store import ensure_rag_configuration
from app.safety.output_guardrail import apply_output_guardrail


logger = logging.getLogger("uvicorn.error")


RAG_WARNING = (
    "Corpus educativo ficticio; no constituye información clínica ni "
    "reemplaza la evaluación de un profesional de salud."
)

RAG_CITATION_WARNING = (
    "La respuesta generada no contenía citas verificables y fue reemplazada "
    "por una abstención controlada."
)

RAG_CLARIFICATION = (
    "No encontré una ficha que coincida con un medicamento del corpus. "
    "Indica el nombre exacto del medicamento para consultar su ficha "
    "educativa."
)

CITATION_PATTERN = re.compile(r"\[Fuente\s+(\d+)\]", flags=re.IGNORECASE)

UNSUPPORTED_CORPUS_ATTRIBUTES = {
    "precio": ("precio", "costo", "cuanto cuesta", "cuánto cuesta"),
    "fabricante": ("fabricante", "laboratorio", "manufacturer"),
    "fecha de aprobación": (
        "fecha de aprobacion",
        "fecha de aprobación",
        "approval date",
    ),
    "código NDC": ("ndc",),
    "stock": ("stock", "disponibilidad en farmacia"),
}

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

RERANK_SYSTEM_PROMPT = """
Ordena las fuentes exclusivamente por relevancia para responder la pregunta.
No respondas la pregunta ni agregues conocimiento. Devuelve solamente JSON con
el formato {"ranking": [1, 2, 3]} usando todos los índices recibidos una vez.
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


def citation_indices(answer: str) -> list[int]:
    """Extrae índices de citas preservando su orden y sin duplicados."""

    return list(
        dict.fromkeys(
            int(match)
            for match in CITATION_PATTERN.findall(answer)
        )
    )


def validate_citations(answer: str, source_count: int) -> tuple[bool, list[int]]:
    """Exige al menos una cita y verifica que todos sus índices existan."""

    indices = citation_indices(answer)
    return (
        bool(indices)
        and all(1 <= index <= source_count for index in indices),
        indices,
    )


def remap_citations(answer: str, indices: list[int]) -> str:
    """Renombra citas para que coincidan con la lista de fuentes filtrada."""

    mapping = {
        original: sequential
        for sequential, original in enumerate(indices, start=1)
    }

    def replace(match: re.Match) -> str:
        original = int(match.group(1))
        return f"[Fuente {mapping[original]}]"

    return CITATION_PATTERN.sub(replace, answer)


def is_abstention_answer(answer: str) -> bool:
    """Reconoce una abstención explícita que no necesita citar afirmaciones."""

    normalized = answer.casefold()
    markers = (
        "no se encontró información suficiente",
        "no se encontro informacion suficiente",
        "no es posible responder",
        "no contiene información",
        "no contiene informacion",
    )
    return any(marker in normalized for marker in markers)


def unsupported_attribute(question: str) -> str | None:
    """Detecta campos eliminados deliberadamente del contexto enviado al LLM."""

    normalized = question.casefold()
    for label, markers in UNSUPPORTED_CORPUS_ATTRIBUTES.items():
        if any(marker in normalized for marker in markers):
            return label
    return None


def parse_rerank_order(content: str, result_count: int) -> list[int]:
    """Valida el orden entregado por el modelo y completa índices omitidos."""

    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned).strip()
    payload = json.loads(cleaned)
    raw_ranking = payload.get("ranking", [])
    ranking: list[int] = []
    for value in raw_ranking:
        if (
            isinstance(value, int)
            and 1 <= value <= result_count
            and value not in ranking
        ):
            ranking.append(value)
    ranking.extend(
        index
        for index in range(1, result_count + 1)
        if index not in ranking
    )
    return ranking


async def rerank_results(question: str, results: list) -> tuple[list, bool]:
    """Reordena candidatos con el LLM cuando hay más de un documento."""

    if len(results) <= 1:
        return results, False

    candidates = "\n\n".join(
        f"[{index}] {result.title}\n{result.content[:2000]}"
        for index, result in enumerate(results, start=1)
    )

    settings = get_settings()
    start = time.perf_counter()

    try:
        response = await get_llm().ainvoke(
            [
                {"role": "system", "content": RERANK_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Pregunta:\n{question}\n\nFuentes:\n{candidates}",
                },
            ]
        )

        latency_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "llm_call operation=rag_rerank model=%s "
            "latency_ms=%.1f success=true candidates=%d",
            settings.chat_model,
            latency_ms,
            len(results),
        )

    except Exception:
        latency_ms = (time.perf_counter() - start) * 1000

        logger.exception(
            "llm_call operation=rag_rerank model=%s "
            "latency_ms=%.1f success=false candidates=%d",
            settings.chat_model,
            latency_ms,
            len(results),
        )
        raise

    order = parse_rerank_order(str(response.content), len(results))
    return [results[index - 1] for index in order], True


async def _generate_answer(prompt: str, correction: str = "") -> str:
    messages = [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"{prompt}\n\n{correction}".strip(),
        },
    ]

    settings = get_settings()
    start = time.perf_counter()

    try:
        response = await get_llm().ainvoke(messages)
        latency_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "llm_call operation=rag_generation model=%s "
            "latency_ms=%.1f success=true",
            settings.chat_model,
            latency_ms,
        )

        return str(response.content)

    except Exception:
        latency_ms = (time.perf_counter() - start) * 1000

        logger.exception(
            "llm_call operation=rag_generation model=%s "
            "latency_ms=%.1f success=false",
            settings.chat_model,
            latency_ms,
        )
        raise


async def answer_medication_question(
    question: str,
    rerank_enabled: bool | None = None,
) -> dict:
    settings = get_settings()
    use_rerank = (
        settings.rag_rerank_enabled
        if rerank_enabled is None
        else rerank_enabled
    )
    entity = resolve_medication_entity(question)
    if entity is None:
        return {
            "answer": RAG_CLARIFICATION,
            "sources": [],
            "warnings": [RAG_WARNING],
            "error": False,
            "clarification_required": True,
        }

    missing_attribute = unsupported_attribute(question)
    if missing_attribute:
        return {
            "answer": (
                "No se encontró información suficiente en el corpus educativo "
                f"sobre {missing_attribute} de {entity}. Ese campo no forma "
                "parte de las fichas disponibles."
            ),
            "sources": [],
            "warnings": [RAG_WARNING],
            "error": False,
        }

    try:
        candidate_k = (
            max(settings.rag_top_k * 3, 10)
            if use_rerank
            else settings.rag_top_k
        )
        results = await asyncio.to_thread(
            retrieve_medications,
            question,
            expected_drug_name=entity,
            top_k=candidate_k,
            filter_by_entity=not use_rerank,
        )
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
        logger.exception("Error al recuperar fichas desde Qdrant")
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
                f"No se encontró información suficiente en la ficha de "
                f"{entity} para responder esa consulta."
            ),
            "sources": [],
            "warnings": [RAG_WARNING],
            "error": False,
        }

    rerank_applied = False
    if use_rerank:
        try:
            results, rerank_applied = await rerank_results(question, results)
        except Exception:
            logger.exception("El reranking falló; se conserva el orden vectorial")
            rerank_applied = False
        expected = normalize_drug_name(entity)
        results = [
            result
            for result in results
            if expected
            in {
                normalize_drug_name(result.title),
                normalize_drug_name(str(result.metadata.get("generic_name", ""))),
            }
        ]
        if not results:
            return {
                "answer": (
                    f"No se encontró información suficiente en la ficha de "
                    f"{entity} para responder esa consulta."
                ),
                "sources": [],
                "warnings": [RAG_WARNING],
                "error": False,
                "rerank_applied": rerank_applied,
            }
    results = results[: settings.rag_top_k]

    prompt = (
        f"Solicitud del usuario:\n{question}\n\n"
        f"Medicamento identificado: {entity}\n\n"
        f"Contexto recuperado:\n{format_context(results)}\n\n"
        "Responde en un máximo de tres párrafos y conserva las citas."
    )
    try:
        answer, blocked = apply_output_guardrail(await _generate_answer(prompt))
    except Exception:
        logger.exception("Error al generar la respuesta RAG")
        return {
            "answer": (
                "El proveedor del modelo no pudo generar la respuesta. "
                "Intenta nuevamente más tarde."
            ),
            "sources": [],
            "warnings": [RAG_WARNING],
            "error": True,
        }

    if blocked:
        return {
            "answer": answer,
            "sources": [],
            "warnings": [RAG_WARNING],
            "error": False,
            "safety_blocked": True,
            "rerank_applied": rerank_applied,
        }

    if is_abstention_answer(answer):
        return {
            "answer": answer,
            "sources": [],
            "warnings": [RAG_WARNING],
            "error": False,
            "safety_blocked": False,
            "rerank_applied": rerank_applied,
        }

    citations_valid, cited_indices = validate_citations(answer, len(results))
    if not citations_valid:
        try:
            correction = (
                "La respuesta anterior omitió citas válidas. Regenera la "
                "respuesta y cita cada afirmación factual usando únicamente "
                f"[Fuente 1] hasta [Fuente {len(results)}]."
            )
            answer, blocked = apply_output_guardrail(
                await _generate_answer(prompt, correction)
            )
            citations_valid, cited_indices = validate_citations(
                answer,
                len(results),
            )
        except Exception:
            citations_valid = False

    if blocked:
        return {
            "answer": answer,
            "sources": [],
            "warnings": [RAG_WARNING],
            "error": False,
            "safety_blocked": True,
            "rerank_applied": rerank_applied,
        }

    if is_abstention_answer(answer):
        return {
            "answer": answer,
            "sources": [],
            "warnings": [RAG_WARNING],
            "error": False,
            "safety_blocked": False,
            "rerank_applied": rerank_applied,
        }

    if not citations_valid:
        return {
            "answer": (
                "No fue posible generar una respuesta suficientemente "
                "fundamentada en las fichas recuperadas."
            ),
            "sources": [],
            "warnings": [RAG_WARNING, RAG_CITATION_WARNING],
            "error": False,
            "safety_blocked": False,
            "rerank_applied": rerank_applied,
        }

    all_sources = [
        SourceCitation(
            source_type="rag",
            title=result.title,
            reference=result.reference,
            score=result.score,
        ).model_dump()
        for result in results
    ]
    sources = [all_sources[index - 1] for index in cited_indices]
    answer = remap_citations(answer, cited_indices)
    return {
        "answer": answer,
        "sources": sources,
        "warnings": [RAG_WARNING],
        "error": False,
        "safety_blocked": False,
        "rerank_applied": rerank_applied,
    }

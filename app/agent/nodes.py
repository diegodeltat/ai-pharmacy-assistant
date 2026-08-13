"""Nodos del grafo conversacional."""

import asyncio

from app.agent.router import classify_intent, extract_commune
from app.agent.state import AgentState
from app.models import SourceCitation
from app.safety.input_guardrail import SAFE_REDIRECT
from app.tools.minsal_tool import MINSAL_TURNOS_URL, find_pharmacies_by_commune
from app.tools.rag_tool import answer_medication_question


def classify_node(state: AgentState) -> AgentState:
    question = state["question"]
    intent = classify_intent(question)
    previous_intent = state.get("last_intent", "")
    commune = extract_commune(question)
    rag_query = question

    if intent == "general" and previous_intent in {"pharmacy", "mixed"} and commune:
        intent = "pharmacy"
    elif intent == "general" and previous_intent in {"medication", "mixed"}:
        followup_markers = (
            "efectos",
            "contraindicaciones",
            "mecanismo",
            "y sus",
            "y para",
            "esa ficha",
        )
        if any(marker in question.casefold() for marker in followup_markers):
            intent = "medication"
            previous_question = state.get("last_medication_question", "")
            rag_query = f"{previous_question}\nSeguimiento: {question}".strip()

    return {**state, "intent": intent, "rag_query": rag_query}


async def _pharmacy_result(state: AgentState) -> dict:
    commune = extract_commune(state["question"])
    if not commune:
        return {
            "answer": (
                "Para buscar farmacias de turno necesito que indiques una "
                "comuna. Por ejemplo: '¿Qué farmacias están de turno en "
                "Ñuñoa?'"
            ),
            "sources": [],
            "warnings": [],
            "commune": "",
        }

    result = await find_pharmacies_by_commune(commune)
    if not result["success"] or not result["pharmacies"]:
        return {
            "answer": result["message"],
            "sources": [],
            "warnings": [],
            "commune": commune,
        }

    lines = [f"Farmacias de turno encontradas para {commune}:", ""]
    for pharmacy in result["pharmacies"]:
        lines.extend(
            [
                f"• {pharmacy['nombre']}",
                f"  Dirección: {pharmacy['direccion']}",
                f"  Horario: {pharmacy['horario']}",
                f"  Teléfono: {pharmacy['telefono']}",
                f"  Fecha informada: {pharmacy['fecha']}",
                "",
            ]
        )
    lines.append("Fuente: Ministerio de Salud de Chile (MINSAL).")
    lines.append(
        "MINSAL informa locales y turnos; no confirma stock, precio ni "
        "disponibilidad de medicamentos."
    )

    warnings = []
    if not result["live_data"]:
        warnings.append(
            "Se muestran datos de respaldo, no datos en vivo. "
            f"Captura: {result['captured_at']}."
        )
    source = SourceCitation(
        source_type="minsal",
        title=f"Farmacias de turno en {commune}",
        reference=(
            f"{MINSAL_TURNOS_URL} — fecha {result['captured_at']}"
        ),
        live_data=result["live_data"],
    ).model_dump()
    return {
        "answer": "\n".join(lines),
        "sources": [source],
        "warnings": warnings,
        "commune": commune,
    }


async def pharmacy_node(state: AgentState) -> AgentState:
    result = await _pharmacy_result(state)
    return {
        **state,
        "response": result["answer"],
        "sources": result["sources"],
        "warnings": result["warnings"],
        "safety_blocked": False,
        "last_intent": "pharmacy",
        "last_commune": result["commune"],
    }


async def medication_node(state: AgentState) -> AgentState:
    question = state.get("rag_query", state["question"])
    result = await answer_medication_question(question)
    return {
        **state,
        "response": result["answer"],
        "sources": result["sources"],
        "warnings": result["warnings"],
        "safety_blocked": result.get("safety_blocked", False),
        "last_intent": "medication",
        "last_medication_question": question,
    }


async def mixed_node(state: AgentState) -> AgentState:
    pharmacy_result, rag_result = await asyncio.gather(
        _pharmacy_result(state),
        answer_medication_question(state.get("rag_query", state["question"])),
    )
    response = (
        f"Farmacias de turno\n\n{pharmacy_result['answer']}\n\n"
        f"Ficha educativa\n\n{rag_result['answer']}"
    )
    return {
        **state,
        "response": response,
        "sources": pharmacy_result["sources"] + rag_result["sources"],
        "warnings": pharmacy_result["warnings"] + rag_result["warnings"],
        "safety_blocked": rag_result.get("safety_blocked", False),
        "last_intent": "mixed",
        "last_commune": pharmacy_result["commune"],
        "last_medication_question": state["question"],
    }


def safety_node(state: AgentState) -> AgentState:
    return {
        **state,
        "response": SAFE_REDIRECT,
        "sources": [],
        "warnings": [],
        "safety_blocked": True,
        "last_intent": "safety",
    }


def general_node(state: AgentState) -> AgentState:
    return {
        **state,
        "response": (
            "Puedo buscar farmacias de turno con datos de MINSAL y explicar "
            "información general de fichas educativas de medicamentos con "
            "fuentes. No entrego diagnósticos, tratamientos ni dosis."
        ),
        "sources": [],
        "warnings": [],
        "safety_blocked": False,
        "last_intent": "general",
    }

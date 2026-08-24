"""Nodos del grafo conversacional."""

import asyncio
import re

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
    pharmacy_followup_markers = (
        "dirección",
        "direccion",
        "horario",
        "teléfono",
        "telefono",
        "farmacia",
        "local",
        "esa",
        "ese",
        "cuál",
        "cual",
        "y cuál",
        "y cual",
    )

    if (
        intent == "general"
        and previous_intent in {"pharmacy", "mixed"}
        and (
            commune
            or any(
                marker in question.casefold()
                for marker in pharmacy_followup_markers
            )
        )
    ):

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
    return {
        **state,
        "intent": intent,
        "rag_query": rag_query,
    }

def _closing_minutes(horario: str) -> int:
    """Convierte la hora de cierre en minutos para poder comparar horarios."""
    matches = re.findall(r"(\d{1,2}):(\d{2})", horario)

    if len(matches) < 2:
        return -1

    hour, minute = map(int, matches[-1])
    total = hour * 60 + minute

    # Si el turno termina al día siguiente, se suma un día completo.
    text = horario.casefold()
    if "día siguiente" in text or "dia siguiente" in text:
        total += 24 * 60

    return total


def _is_latest_closing_followup(question: str) -> bool:
    text = question.casefold()

    markers = (
        "cierra más tarde",
        "cierra mas tarde",
        "cierra después",
        "cierra despues",
        "cuál cierra más tarde",
        "cual cierra mas tarde",
        "cuál cierra después",
        "cual cierra despues",
        "cuál de esas cierra",
        "cual de esas cierra",
    )

    return any(marker in text for marker in markers)


def _answer_latest_closing_pharmacy(
    state: AgentState,
) -> AgentState | None:
    pharmacies = state.get("last_pharmacies", [])

    if not pharmacies:
        return None

    if not _is_latest_closing_followup(state["question"]):
        return None

    valid_pharmacies = [
        pharmacy
        for pharmacy in pharmacies
        if _closing_minutes(pharmacy.get("horario", "")) >= 0
    ]

    if not valid_pharmacies:
        return {
            **state,
            "response": (
                "No pude comparar los horarios de las farmacias encontradas "
                "porque los datos no contienen horarios válidos."
            ),
            "safety_blocked": False,
            "last_intent": "pharmacy",
        }

    pharmacy = max(
        valid_pharmacies,
        key=lambda item: _closing_minutes(item.get("horario", "")),
    )

    commune = state.get("last_commune", "")

    response = (
        f"De las farmacias encontradas para {commune}, "
        f"la que cierra más tarde es {pharmacy['nombre']}, "
        f"ubicada en {pharmacy['direccion']}. "
        f"Su horario informado es {pharmacy['horario']}."
    )

    return {
        **state,
        "response": response,
        "sources": state.get("sources", []),
        "warnings": state.get("warnings", []),
        "safety_blocked": False,
        "last_intent": "pharmacy",
        "last_commune": commune,
    }


async def _pharmacy_result(state: AgentState) -> dict:
    commune = (
        extract_commune(state["question"])
        or state.get("last_commune", "")
    )

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
            "pharmacies": [],
        }

    result = await find_pharmacies_by_commune(commune)

    if not result["success"] or not result["pharmacies"]:
        return {
            "answer": result["message"],
            "sources": [],
            "warnings": [],
            "commune": commune,
            "pharmacies": [],
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
        "pharmacies": result["pharmacies"],
    }


async def pharmacy_node(state: AgentState) -> AgentState:
    followup_response = _answer_latest_closing_pharmacy(state)

    if followup_response is not None:
        return followup_response

    result = await _pharmacy_result(state)

    return {
        **state,
        "response": result["answer"],
        "sources": result["sources"],
        "warnings": result["warnings"],
        "safety_blocked": False,
        "last_intent": "pharmacy",
        "last_commune": result["commune"],
        "last_pharmacies": result["pharmacies"],
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
        "last_pharmacies": pharmacy_result["pharmacies"],
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
    question = state["question"].casefold()

    commercial_scope_markers = (
        "stock",
        "precio",
        "precios",
        "costo",
        "costos",
        "disponibilidad del medicamento",
        "disponibilidad de medicamento",
    )

    if any(marker in question for marker in commercial_scope_markers):
        response = (
            "No puedo confirmar stock, precio ni disponibilidad comercial "
            "de medicamentos. MINSAL informa locales y turnos de farmacias, "
            "pero no entrega esa información. Sí puedo ayudarte a encontrar "
            "farmacias de turno o explicar información general de una ficha "
            "educativa de medicamentos."
        )
    else:
        response = (
            "Puedo buscar farmacias de turno con datos de MINSAL y explicar "
            "información general de fichas educativas de medicamentos con "
            "fuentes. No entrego diagnósticos, tratamientos ni dosis."
        )

    return {
        **state,
        "response": response,
        "sources": [],
        "warnings": [],
        "safety_blocked": False,
        "last_intent": "general",
    }

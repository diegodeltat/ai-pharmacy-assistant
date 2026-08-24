"""Clasificación determinística de intenciones y extracción de comuna."""

import re

from app.rag.entity_resolver import resolve_medication_entity
from app.safety.input_guardrail import is_unsafe_request, normalize_for_safety


PHARMACY_WORDS = (
    "farmacia",
    "farmacias",
    "de turno",
    "abierta",
    "abiertas",
    "comuna",
)
MEDICATION_WORDS = (
    "medicamento",
    "medicamentos",
    "amoxicilina",
    "ibuprofeno",
    "paracetamol",
    "para que sirve",
    "ficha",
    "vademecum",
    "efectos secundarios",
    "efectos adversos",
    "contraindicaciones",
    "interacciones de",
    "mecanismo de accion",
    "clase farmacologica",
    "farmaco",
    "farmacos",
    "medicina",
    "dosis",
)

OUT_OF_SCOPE_COMMERCIAL_WORDS = (
    "stock",
    "precio",
    "precios",
    "costo",
    "costos",
    "disponibilidad del medicamento",
    "disponibilidad de medicamento",
)


def classify_intent(question: str) -> str:
    if is_unsafe_request(question):
        return "safety"
    text = normalize_for_safety(question)
    # Stock, precio y disponibilidad comercial no forman parte
    # del alcance del asistente.
    if any(word in text for word in OUT_OF_SCOPE_COMMERCIAL_WORDS):
        return "general"
    pharmacy = any(word in text for word in PHARMACY_WORDS)
    medication = (
        any(word in text for word in MEDICATION_WORDS)
        or resolve_medication_entity(question) is not None
    )
    if pharmacy and medication:
        return "mixed"
    if pharmacy:
        return "pharmacy"
    if medication:
        return "medication"
    return "general"


def extract_commune(question: str) -> str:
    match = re.search(
        r"\ben\s+([A-Za-zÁÉÍÓÚáéíóúÑñÜü\s]+?)"
        r"(?:\s+y\s+|,|\?|$)",
        question.strip(),
        flags=re.IGNORECASE,
    )
    return match.group(1).strip() if match else ""

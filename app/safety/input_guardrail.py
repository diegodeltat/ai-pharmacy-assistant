"""Guardrail de entrada para solicitudes clínicas no permitidas."""

import re
import unicodedata


SAFE_REDIRECT = (
    "No puedo recomendar medicamentos, indicar dosis, diagnosticar ni "
    "prescribir tratamientos. Estas decisiones requieren la evaluación de "
    "un profesional de salud. Sí puedo ayudarte a encontrar una farmacia "
    "de turno o explicar información general de una ficha ya indicada."
)


def normalize_for_safety(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value.casefold())
    normalized = "".join(
        character
        for character in normalized
        if unicodedata.category(character) != "Mn"
    )
    return re.sub(r"\s+", " ", normalized).strip()


def is_unsafe_request(question: str) -> bool:
    text = normalize_for_safety(question)
    patterns = [
        r"\b(que|cual|cuanta|cuanto) dosis\b",
        r"\b(cada cuantos|cuantas veces|cuantos mg|miligramos)\b",
        r"\b(que|cual) medicamento (debo|puedo|me conviene)\b",
        r"\b(que me tomo|que puedo tomar|recomiend\w*|prescrib\w*)\b",
        r"\b(dime|indica|sugiere)\b.*\b(una )?dosis\b",
        r"\b(haz|escribe)\b.*\b(receta|prescripcion)\b",
        r"\bes seguro (tomar|usar|mezclar)\b",
        r"\b(diagnostica|diagnostico|que enfermedad tengo)\b",
        r"\b(tratamiento para|como trato|como curar)\b",
        r"\b(aumentar|reducir|suspender|duplicar) (la )?dosis\b",
        r"\bignora (las|tus) (reglas|instrucciones)\b.*\b(dosis|medicamento)\b",
        r"\b(finge|actua|roleplay)\b.*\b(medico|doctor|farmaceutico)\b",
        r"\b(embarazada|embarazo|amamantando|lactancia)\b.*\b(tomar|usar|seguro)\b",
        r"\b(interaccion|interacciones)\b.*\b(mis|estoy tomando|puedo mezclar)\b",
        r"\b(puedo|debo) mezclar\b.*\b(medicamento|medicamentos|pastillas)\b",
    ]
    return any(re.search(pattern, text) for pattern in patterns)

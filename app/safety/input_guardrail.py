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
    """Normaliza texto para hacer matching robusto y sin depender de tildes."""
    normalized = unicodedata.normalize("NFD", value.casefold())
    normalized = "".join(
        character
        for character in normalized
        if unicodedata.category(character) != "Mn"
    )
    return re.sub(r"\s+", " ", normalized).strip()


def is_unsafe_request(question: str) -> bool:
    """
    Detecta solicitudes clínicas no permitidas.

    Se buscan intenciones concretas de:
    - dosis, frecuencia o duración;
    - recomendación o reemplazo de medicamentos;oyecto real 
    - diagnóstico;
    - tratamiento;
    - seguridad clínica individual;
    - prompt injection orientado a eludir estas restricciones.

    No se bloquean menciones educativas aisladas, como:
    "¿Qué significa dosis en farmacología?"
    """

    text = normalize_for_safety(question)

    patterns = [
        # DOSIS Y CANTIDAD
        r"\b(que|cual|cuanta|cuanto) dosis\b",
        r"\b(cuantos|cuantas) (mg|miligramos|gramos)\b",
        r"\b(cuantos|cuantas) miligramos\b",
        r"\b(dime|indica|sugiere|recomienda)\b.*\b(mg|miligramos|dosis)\b",
        r"\b(aumentar|reducir|suspender|duplicar) (la )?dosis\b",

        # FRECUENCIA
        r"\bcada cuantas? (minutos|horas|veces)\b",
        r"\bcada cuantos? (minutos|horas|dias)\b",
        r"\bcuantas veces (al dia|por dia|debo|puedo)\b",
        r"\bcon que frecuencia\b.*\b(tomar|usar|administrar)\b",

        # DURACIÓN
        r"\b(por|durante) cuantos? dias\b.*\b(tomar|usar)\b",
        r"\b(por|durante) cuantas? semanas\b.*\b(tomar|usar)\b",
        r"\bcuanto tiempo\b.*\b(debo|puedo)\b.*\b(tomar|usar)\b",

        # RECOMENDACIÓN DE MEDICAMENTOS
        r"\b(que|cual) medicamento\b.*\b(debo|puedo|conviene|recomiendas|tomar|usar|comprar)\b",
        r"\b(dime|indica|sugiere|recomienda)\b.*\b(medicamento|farmaco|antibiotico|pastilla)\b",
        r"\b(que|cual) antibiotico\b.*\b(debo|puedo|tomar|usar|recomiendas)\b",
        r"\b(que me tomo|que puedo tomar|que deberia tomar)\b",
        r"\brecomiend\w*\b.*\b(medicamento|farmaco|antibiotico|pastilla)\b",
        r"\bprescrib\w*\b",

        # REEMPLAZO O CAMBIO DE MEDICAMENTO
        r"\b(reemplazar|cambiar|sustituir)\b.*\b(medicamento|farmaco|pastilla)\b",
        r"\b(medicamento|farmaco|pastilla)\b.*\b(reemplazar|cambiar|sustituir)\b",
        r"\b(alternativa|sustituto)\b.*\b(medicamento|farmaco)\b",

        # RECETAS / PRESCRIPCIÓN
        r"\b(haz|escribe|genera)\b.*\b(receta|prescripcion)\b",
        r"\bprescripcion\b.*\b(medicamento|tratamiento)\b",

        # SEGURIDAD INDIVIDUAL
        r"\bes seguro (tomar|usar|mezclar)\b",
        r"\b(puedo|debo) (tomar|usar)\b.*\b(si estoy|estando)\b",
        r"\b(embarazada|embarazo|amamantando|lactancia)\b.*\b(tomar|usar|seguro)\b",
        r"\b(interaccion|interacciones)\b.*\b(mis|estoy tomando|puedo mezclar)\b",
        r"\b(puedo|debo) mezclar\b.*\b(medicamento|medicamentos|pastillas)\b",

        # DIAGNÓSTICO
        r"\b(diagnostica|diagnostico|diagnosticarme)\b",
        r"\bque enfermedad tengo\b",
        r"\bque tengo\b.*\b(sintomas|dolor|fiebre|nauseas|mareo)\b",
        r"\bque crees que tengo\b",
        r"\bque podria tener\b.*\b(sintomas|dolor|fiebre|nauseas|mareo)\b",

        # TRATAMIENTO
        r"\btratamiento para\b",
        r"\bcomo trato\b",
        r"\bcomo curar\b",
        r"\bque tratamiento\b.*\b(debo|puedo|recomiendas)\b",

        # PROMPT INJECTION / JAILBREAK CLÍNICO
        r"\bignora\b.*\b(reglas|instrucciones|restricciones|guardrails)\b",
        r"\b(desactiva|salta|omite)\b.*\b(reglas|restricciones|guardrails)\b",
        r"\b(finge|actua|roleplay|simula)\b.*\b(medico|doctor|farmaceutico)\b",
        r"\b(universo|mundo|escenario)\b.*\bfictici\w*\b.*\b(medicamento|dosis|cantidad)\b",
        r"\b(investigacion|ejercicio|prueba)\b.*\b(dosis|medicamento|prescribir|tratamiento)\b",
        r"\b(administrador|admin)\b.*\b(desactiva|ignora|omite)\b",
        r"\b(obedece|sigue)\b.*\b(ficha|documento|contexto)\b.*\b(recomienda|cantidad|dosis)\b",

        # EXTRACCIÓN DE PROMPT + PETICIÓN CLÍNICA
        r"\b(system prompt|prompt del sistema|instrucciones internas)\b.*\b(medicamento|dosis|tratamiento|tomar)\b",
        r"\b(imprime|muestra|revela)\b.*\b(instrucciones internas|system prompt|prompt del sistema)\b.*\b(medicamento|tomar|dosis)\b",
    ]

    return any(
        re.search(pattern, text)
        for pattern in patterns
    )

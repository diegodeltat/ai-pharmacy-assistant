"""Control de salida para evitar recomendaciones clínicas accidentales."""

import re

from app.safety.input_guardrail import SAFE_REDIRECT, normalize_for_safety


def apply_output_guardrail(response: str) -> tuple[str, bool]:
    text = normalize_for_safety(response)
    unsafe_patterns = [
        r"\b(toma|debes tomar|te recomiendo|aumenta|reduce)\b.{0,40}\b(mg|ml|dosis)\b",
        r"\b[0-9]+\s*(mg|ml)\b.{0,30}\b(cada|diario|al dia)\b",
        r"\btu diagnostico es\b",
    ]
    if any(re.search(pattern, text) for pattern in unsafe_patterns):
        return SAFE_REDIRECT, True
    return response, False
